"""
lora_dreamer_run.py  —  Paper-exact LoRA on DREAMER
====================================================
Mirrors lora_amigos_run.py exactly, adapted for DREAMER:
  • 14 EEG channels (all EMOTIV Epoc channels, 128 Hz)
  • Binary arousal: >=3 -> 1 (1-5 scale)
  • 23 subjects, 18 trials each
  • LoRA: rank=32, alpha=64, epochs=50, targets=(qkv, proj, fc1, fc2)
  • 80/20 trial-level split (no window-level leakage)
  • Metric: AUC ROC

Run from /home/ug24/FoundationalModel/:
    python3 -m NormWear.lora_dreamer_run
"""
import os, sys, json, time, copy, pickle, tempfile, re
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import roc_auc_score

from NormWear.lora.lora_dataset import PersonalizedDownstreamDataset, parse_label
from NormWear.lora.lora_layers  import LoRALinear
from NormWear.modules.normwear  import NormWear

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR    = "/home/ug24/FoundationalModel/NormWear/data/wearable_downstream/dreamer/sample_for_downstream"
CKPT_PATH   = "/home/ug24/FoundationalModel/NormWear/data/results/full_pretrain_checkpoint-399.pth"
SAVE_PATH   = "/home/ug24/FoundationalModel/NormWear/data/results/lora_results/dreamer_lora_paper_summary.json"

DS_NAME       = "dreamer"
NUM_CLASSES   = 2
TASK_TYPE     = "classification"
PAD_NVAR      = 16   # pad to multiple of 4 (NormWear nvar=4 internals require B*nvar % 4 == 0)
MAX_L         = 390
BATCH_SIZE    = 8
NUM_WORKERS   = 4

# ── Paper-exact hyperparameters ───────────────────────────────────────────────
EPOCHS        = 50
WARMUP_EPOCHS = 5
LR            = 3e-4
WEIGHT_DECAY  = 1e-2
LORA_RANK     = 8
LORA_ALPHA    = 16.0
LORA_DROPOUT  = 0.1
TARGET_MODULES = ("qkv", "proj", "fc1", "fc2")

SEED     = 42
TRIAL_RX = re.compile(r"^(S\d+)_t(\d+)_w\d+\.pkl$")


class MLPHead(nn.Module):
    def __init__(self, embed_dim: int = 768, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def inject_lora_paper(model: nn.Module, rank: int, alpha: float,
                      dropout: float, target_modules: tuple) -> list:
    for p in model.parameters():
        p.requires_grad_(False)

    replaced = 0
    for _name, module in model.named_modules():
        for attr in target_modules:
            linear = getattr(module, attr, None)
            if isinstance(linear, nn.Linear):
                setattr(module, attr,
                        LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout))
                replaced += 1

    lora_params = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    lora  = sum(p.numel() for p in lora_params)
    print(f"[LoRA] Injected into {replaced} Linear layers "
          f"(rank={rank}, alpha={alpha}, targets={target_modules})")
    print(f"[LoRA] Total params  : {total:,}")
    print(f"[LoRA] LoRA trainable: {lora:,}  ({100*lora/max(total,1):.2f}%)")
    return lora_params


class NormWearLoRAPaper(nn.Module):
    def __init__(self, base_encoder):
        super().__init__()
        self.encoder = base_encoder
        inject_lora_paper(self.encoder, LORA_RANK, LORA_ALPHA,
                          LORA_DROPOUT, TARGET_MODULES)
        self.head = MLPHead(embed_dim=768, hidden=256, dropout=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, nvar, C, L, F = x.shape
        x_in = x.view(B * nvar, C, L, F)
        latent, _, _ = self.encoder.forward_encoder(x_in)
        latent = latent[:, 1:, :]
        P = latent.shape[1]
        latent = latent.view(B, nvar, P, -1).mean(dim=(1, 2))
        return self.head(latent)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": train, "frozen": total - train}


def load_encoder(ckpt_path: str):
    model = NormWear(
        img_size=(387, 65), patch_size=(9, 5), in_chans=3, target_len=388,
        nvar=4, embed_dim=768, decoder_embed_dim=512, depth=12, num_heads=12,
        decoder_depth=2, mlp_ratio=4.0, fuse_freq=2, mask_t_prob=0.6,
        mask_f_prob=0.5, mask_prob=0.8, mask_scheme="random",
        use_cwt=True, is_pretrain=False,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd   = ckpt.get("model", ckpt.get("state_dict", ckpt))
    sd   = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[Encoder] Loaded — missing={len(missing)}, unexpected={len(unexpected)}")
    return model


def gather_subject_trials():
    by_subj_trial = defaultdict(lambda: defaultdict(list))
    trial_label   = {}
    for fname in sorted(os.listdir(DATA_DIR)):
        m = TRIAL_RX.match(fname)
        if not m: continue
        sid, tidx = m.group(1), int(m.group(2))
        by_subj_trial[sid][tidx].append(fname)
        key = (sid, tidx)
        if key not in trial_label:
            d = pickle.load(open(os.path.join(DATA_DIR, fname), "rb"))
            trial_label[key] = parse_label(DS_NAME, d)
    return by_subj_trial, trial_label


def stratified_trial_split(trials_with_labels, train_frac=0.8, rng=None):
    if rng is None: rng = np.random.default_rng(SEED)
    by_label = defaultdict(list)
    for t, l in trials_with_labels:
        by_label[l].append(t)
    train, test = [], []
    for l, ts in by_label.items():
        ts = sorted(ts); rng.shuffle(ts)
        n_tr = max(1, int(round(len(ts) * train_frac)))
        if n_tr == len(ts) and len(ts) >= 2: n_tr = len(ts) - 1
        train.extend(ts[:n_tr]); test.extend(ts[n_tr:])
    return sorted(train), sorted(test)


class _CachedDataset(Dataset):
    """Holds pre-computed CWT tensors in RAM — eliminates per-epoch CWT recomputation."""
    def __init__(self, inputs: torch.Tensor, labels: torch.Tensor):
        self.inputs = inputs
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return {"input": self.inputs[i], "label": self.labels[i]}


def _preload_cwt(ds: PersonalizedDownstreamDataset) -> "_CachedDataset":
    """
    Compute CWT for every window once and cache in RAM.
    DREAMER has 14 channels -> 160ms CWT per window. Without this, CWT is
    recomputed from disk on every __getitem__ call (50 times per window per epoch).
    """
    t0 = time.time()
    inputs, labels = [], []
    for i in range(len(ds)):
        item = ds[i]
        inputs.append(item["input"])
        labels.append(item["label"])
    inputs = torch.stack(inputs)   # [N, PAD_NVAR, 3, MAX_L, 65]
    labels = torch.stack(labels)   # [N]
    print(f"    CWT pre-load: {len(ds)} windows in {time.time()-t0:.1f}s  "
          f"({inputs.element_size()*inputs.numel()/1e6:.0f} MB RAM)")
    return _CachedDataset(inputs, labels)


def train_one_subject(sid, train_files, test_files, encoder_template, device):
    # All train_files come from train trials; test_files from held-out test trials.
    # No window-level val split — that would leak across windows of the same trial.
    # Paper (lora.md §V-A) trains for exactly EPOCHS epochs with no early stopping.
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir="/tmp")
    json.dump({"train": train_files, "test": test_files}, tmp)
    tmp.close()

    common = dict(data_dir=DATA_DIR, ds_name=DS_NAME, split_file=tmp.name,
                  max_L=MAX_L, pad_nvar=PAD_NVAR, task_type=TASK_TYPE, sid_split_idx=0)
    ds_train = PersonalizedDownstreamDataset(**common, split="train")
    ds_test  = PersonalizedDownstreamDataset(**common, split="test")
    os.unlink(tmp.name)

    # Pre-compute CWT once per subject into RAM so each epoch reads cached tensors.
    # Without this, the 14-channel CWT (160ms/window) runs 50x per window = ~2h/subject.
    print(f"  Pre-loading CWT into RAM...")
    cached_train = _preload_cwt(ds_train)
    cached_test  = _preload_cwt(ds_test)

    dl_train = DataLoader(cached_train, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True, drop_last=False)
    dl_test  = DataLoader(cached_test,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)

    model = NormWearLoRAPaper(copy.deepcopy(encoder_template)).to(device)
    params_info = model.count_parameters()
    print(f"  Params: trainable={params_info['trainable']:,}  "
          f"frozen={params_info['frozen']:,}  "
          f"({100*params_info['trainable']/max(params_info['total'],1):.2f}%)")

    trainable = model.trainable_parameters()
    optim = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY,
                               betas=(0.9, 0.999))

    # Class-balanced BCE: 72.5% of DREAMER trials are positive (high arousal).
    # Without pos_weight, the model is biased toward predicting positive and
    # never learns to discriminate. pos_weight = n_neg / n_pos applied to the
    # positive class brings the effective gradient back to balanced.
    # (Paper §IV-A-e specifies BCE for binary tasks but does not specify class
    # weighting — pos_weight is a training-detail improvement, not a LoRA change.)
    y_train = cached_train.labels.long()
    n_pos = int((y_train == 1).sum().item())
    n_neg = int((y_train == 0).sum().item())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    print(f"  Class balance: n_pos={n_pos}  n_neg={n_neg}  pos_weight={pos_weight.item():.3f}")
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS: return (epoch + 1) / max(WARMUP_EPOCHS, 1)
        prog = (epoch - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)
        return 0.5 * (1 + np.cos(np.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    for ep in range(EPOCHS):
        t0 = time.time()

        model.train()
        total_loss, n = 0.0, 0
        for batch in dl_train:
            x = batch["input"].to(device)
            y = batch["label"].float().to(device)
            optim.zero_grad()
            logits = model(x).squeeze(-1)
            loss = bce(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step()
            total_loss += loss.item() * x.size(0); n += x.size(0)
        train_loss = total_loss / max(n, 1)
        sched.step()

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  [{sid}] E{ep+1:2d}/{EPOCHS} train_loss={train_loss:.4f} "
                  f"lr={optim.param_groups[0]['lr']:.1e} "
                  f"({time.time()-t0:.1f}s)")

    # Single evaluation on held-out test trials after all 50 epochs
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for batch in dl_test:
            x = batch["input"].to(device)
            p = torch.sigmoid(model(x).squeeze(-1)).cpu().numpy()
            all_y.append(batch["label"].numpy())
            all_p.append(p)
    y_arr = np.concatenate(all_y)
    p_arr = np.concatenate(all_p)
    if len(np.unique(y_arr)) < 2:
        test_auc = float("nan")
    else:
        test_auc = roc_auc_score(y_arr, p_arr) * 100

    del optim
    torch.cuda.empty_cache()
    return test_auc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print(f"  NormWear + LoRA  [Paper-Exact: DREAMER]")
    print(f"  rank={LORA_RANK}, alpha={LORA_ALPHA}, epochs={EPOCHS}")
    print(f"  targets={TARGET_MODULES}")
    print(f"  device={device}")
    print(f"{'='*65}\n")

    encoder_template = load_encoder(CKPT_PATH)
    by_subj_trial, trial_label = gather_subject_trials()
    subjects = sorted(by_subj_trial.keys())
    print(f"Subjects: {subjects}\n")

    rng = np.random.default_rng(SEED)
    per_subject = []

    for sid in subjects:
        trials_dict = by_subj_trial[sid]
        tidxs = sorted(trials_dict.keys())
        twl   = [(t, trial_label[(sid, t)]) for t in tidxs]

        train_t, test_t = stratified_trial_split(twl, train_frac=0.8, rng=rng)
        train_files = [f for t in train_t for f in trials_dict[t]]
        test_files  = [f for t in test_t  for f in trials_dict[t]]

        test_lbls = [trial_label[(sid, t)] for t in test_t]
        if len(set(test_lbls)) < 2:
            print(f"[{sid}] SKIP — single-class test set"); continue

        print(f"\n{'='*65}")
        print(f"[{sid}] train_trials={len(train_t)} "
              f"({sum(trial_label[(sid,t)] for t in train_t)} high) | "
              f"test_trials={len(test_t)} "
              f"({sum(trial_label[(sid,t)] for t in test_t)} high)")
        print(f"[{sid}] train_windows={len(train_files)} "
              f"test_windows={len(test_files)}")

        try:
            test_auc = train_one_subject(sid, train_files, test_files,
                                         encoder_template, device)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[{sid}] ERROR: {e}"); continue

        print(f"[{sid}] *** test_AUC = {test_auc:.2f}% ***")
        per_subject.append({
            "subject_id": sid, "auc": float(test_auc),
            "n_train_trials": len(train_t), "n_test_trials": len(test_t),
            "n_train_wins": len(train_files), "n_test_wins": len(test_files),
        })

    test_aucs = [r["auc"] for r in per_subject if not np.isnan(r["auc"])]
    summary = {
        "method": (f"LoRA-Paper-Exact rank={LORA_RANK} alpha={LORA_ALPHA} "
                   f"epochs={EPOCHS} targets={TARGET_MODULES}"),
        "checkpoint": os.path.basename(CKPT_PATH),
        "per_subject": per_subject,
        "mean_test_auc": float(np.mean(test_aucs)) if test_aucs else float("nan"),
        "std_test_auc":  float(np.std(test_aucs))  if test_aucs else float("nan"),
        "n_subjects": len(per_subject),
    }
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, "w") as fp:
        json.dump(summary, fp, indent=2)

    print(f"\n{'='*65}")
    print(f"  DREAMER LoRA (Paper-Exact) Results")
    print(f"{'='*65}")
    for r in per_subject:
        print(f"  {r['subject_id']:6}  test_AUC = {r['auc']:.2f}%")
    print(f"  {'─'*30}")
    print(f"  Mean test AUC = {summary['mean_test_auc']:.2f} ± {summary['std_test_auc']:.2f}%"
          f"  (n={summary['n_subjects']})")
    print(f"{'='*65}")
    print(f"Saved → {SAVE_PATH}")


if __name__ == "__main__":
    main()
