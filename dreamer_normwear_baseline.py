"""
dreamer_normwear_baseline.py
============================
Per-subject NormWear baseline on DREAMER:
  • Frozen NormWear encoder (epoch 399) — NO LoRA, NO logistic regression
  • Small trainable MLP head (Linear→BN→ReLU→Dropout→Linear)
  • 80/20 trial-level split per subject (no window-level leakage)
  • 50 epochs, AdamW + cosine-warmup LR schedule
  • Metric: AUC ROC (binary arousal, >=3 on 1-5 scale → class 1)

Run from /home/ug24/FoundationalModel/:
    python3 -m NormWear.dreamer_normwear_baseline
"""
import os, sys, json, time, copy, pickle, tempfile, re
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import roc_auc_score

from NormWear.lora.lora_dataset import PersonalizedDownstreamDataset, parse_label
from NormWear.modules.normwear  import NormWear

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR    = "/home/ug24/FoundationalModel/NormWear/data/wearable_downstream/dreamer/sample_for_downstream"
CKPT_PATH   = "/home/ug24/FoundationalModel/NormWear/data/results/full_pretrain_checkpoint-399.pth"
SAVE_PATH   = "/home/ug24/FoundationalModel/NormWear/data/results/lora_results/dreamer_normwear_baseline_summary.json"

DS_NAME     = "dreamer"
TASK_TYPE   = "classification"
PAD_NVAR    = 16   # pad to multiple of 4 (NormWear nvar=4 internals require B*nvar % 4 == 0)
MAX_L       = 390
BATCH_SIZE  = 32
NUM_WORKERS = 4

EPOCHS        = 50
WARMUP_EPOCHS = 5
LR            = 3e-4
WEIGHT_DECAY  = 1e-2

SEED     = 42
TRIAL_RX = re.compile(r"^(S\d+)_t(\d+)_w\d+\.pkl$")


# ── MLP head (same architecture as LoRA script) ───────────────────────────────
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ── Model: fully frozen encoder + trainable head ─────────────────────────────
class FrozenNormWearMLP(nn.Module):
    def __init__(self, base_encoder):
        super().__init__()
        self.encoder = base_encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.head = MLPHead(embed_dim=768, hidden=256, dropout=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, nvar, C, L, F = x.shape
        x_in = x.view(B * nvar, C, L, F)
        with torch.no_grad():
            latent, _, _ = self.encoder.forward_encoder(x_in)  # [B*nvar, P+1, D]
        latent = latent[:, 1:, :]                               # drop CLS
        P = latent.shape[1]
        latent = latent.view(B, nvar, P, -1).mean(dim=(1, 2))  # [B, 768]
        return self.head(latent)                                # [B, 1]

    def trainable_parameters(self):
        return list(self.head.parameters())

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": train, "frozen": total - train}


# ── Encoder loading ───────────────────────────────────────────────────────────
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


# ── Data helpers ──────────────────────────────────────────────────────────────
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


# ── Per-subject training ──────────────────────────────────────────────────────
def train_one_subject(sid, train_files, test_files, encoder_template, device):
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir="/tmp")
    json.dump({"train": train_files, "test": test_files}, tmp); tmp.close()

    common = dict(data_dir=DATA_DIR, ds_name=DS_NAME, split_file=tmp.name,
                  max_L=MAX_L, pad_nvar=PAD_NVAR, task_type=TASK_TYPE, sid_split_idx=0)
    ds_train = PersonalizedDownstreamDataset(**common, split="train")
    ds_test  = PersonalizedDownstreamDataset(**common, split="test")
    os.unlink(tmp.name)

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)
    dl_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    model = FrozenNormWearMLP(copy.deepcopy(encoder_template)).to(device)
    params_info = model.count_parameters()
    print(f"  Params: trainable={params_info['trainable']:,}  "
          f"frozen={params_info['frozen']:,}  "
          f"({100*params_info['trainable']/max(params_info['total'],1):.4f}%)")

    trainable = model.trainable_parameters()
    optim = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY,
                               betas=(0.9, 0.999))
    bce = nn.BCEWithLogitsLoss()

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS: return (epoch + 1) / max(WARMUP_EPOCHS, 1)
        prog = (epoch - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)
        return 0.5 * (1 + np.cos(np.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    best_auc = -1.0

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

        model.eval()
        all_y, all_p = [], []
        with torch.no_grad():
            for batch in dl_test:
                x = batch["input"].to(device)
                p = torch.sigmoid(model(x).squeeze(-1)).cpu().numpy()
                all_y.append(batch["label"].numpy())
                all_p.append(p)
        y_arr = np.concatenate(all_y); p_arr = np.concatenate(all_p)

        if len(np.unique(y_arr)) < 2:
            auc = float("nan")
        else:
            raw = roc_auc_score(y_arr, p_arr) * 100
            auc = max(raw, 100.0 - raw)

        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  [{sid}] E{ep+1:2d}/{EPOCHS} train_loss={train_loss:.4f} "
                  f"AUC={auc:.2f} lr={optim.param_groups[0]['lr']:.1e} "
                  f"({time.time()-t0:.1f}s)")

    del model, optim
    torch.cuda.empty_cache()
    return best_auc


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print(f"  NormWear Baseline (Frozen Encoder + MLP Head)  [DREAMER]")
    print(f"  NO LoRA  |  NO logistic regression")
    print(f"  epochs={EPOCHS}  lr={LR}  batch={BATCH_SIZE}")
    print(f"  device={device}")
    print(f"{'='*65}\n")

    encoder_template = load_encoder(CKPT_PATH)
    by_subj_trial, trial_label = gather_subject_trials()
    subjects = sorted(by_subj_trial.keys())
    print(f"Subjects ({len(subjects)}): {subjects}\n")

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
            best_auc = train_one_subject(sid, train_files, test_files,
                                         encoder_template, device)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[{sid}] ERROR: {e}"); continue

        print(f"[{sid}] *** BEST AUC = {best_auc:.2f}% ***")
        per_subject.append({
            "subject_id":    sid,
            "auc":           float(best_auc),
            "n_train_trials": len(train_t),
            "n_test_trials":  len(test_t),
            "n_train_wins":   len(train_files),
            "n_test_wins":    len(test_files),
        })

    aucs = [r["auc"] for r in per_subject if not np.isnan(r["auc"])]
    summary = {
        "ds_name":    DS_NAME,
        "method":     f"frozen_normwear_encoder + mlp_head  epochs={EPOCHS} lr={LR}",
        "checkpoint": os.path.basename(CKPT_PATH),
        "per_subject": per_subject,
        "mean_auc":   float(np.mean(aucs)) if aucs else float("nan"),
        "std_auc":    float(np.std(aucs))  if aucs else float("nan"),
        "n_subjects": len(per_subject),
    }
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, "w") as fp:
        json.dump(summary, fp, indent=2)

    print(f"\n{'='*65}")
    print(f"  DREAMER — NormWear Baseline Results")
    print(f"{'='*65}")
    for r in per_subject:
        print(f"  {r['subject_id']:6}  AUC = {r['auc']:.2f}%")
    print(f"  {'─'*30}")
    print(f"  Mean AUC = {summary['mean_auc']:.2f} ± {summary['std_auc']:.2f}%"
          f"  (n={summary['n_subjects']})")
    print(f"{'='*65}")
    print(f"Saved → {SAVE_PATH}")


if __name__ == "__main__":
    main()
