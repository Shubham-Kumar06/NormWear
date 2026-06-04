"""
neuralhead_amigos_run.py  —  Side B control for AMIGOS (parallel to
neuralhead_wesad_run.py).

Identical to lora_amigos_run.py EXCEPT no LoRA injection:
    encoder is fully frozen; only the MLPHead is trainable.

All other training-recipe details (AdamW, cosine LR, warmup, label smoothing,
grad clipping, batch size, 30 epochs, MLPHead architecture, same per-subject
splits) are kept byte-identical to lora_amigos_run.py so the only variable
vs Side C (LoRA) is the presence/absence of LoRA layers.
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

# ── Config (mirrors lora_amigos_run.py) ──────────────────────────────────────
DATA_DIR    = "/home/ug24/FoundationalModel/NormWear/data/wearable_downstream/amigos/sample_for_downstream"
SPLIT_JSON  = "/home/ug24/FoundationalModel/NormWear/data/wearable_downstream/amigos/per_subject_splits.json"
CKPT_PATH   = "/home/ug24/FoundationalModel/NormWear/data/results/normwear_aug_BEST_ep0_81p35.pth"
SAVE_PATH   = "/home/ug24/FoundationalModel/NormWear/data/results/lora_results/amigos_neuralhead_summary_AUG_BEST.json"

DS_NAME       = "amigos"
NUM_CLASSES   = 2
TASK_TYPE     = "classification"
PAD_NVAR      = 4
MAX_L         = 390
BATCH_SIZE    = 32
NUM_WORKERS   = 4

# ── Same training recipe as lora_amigos_run.py ───────────────────────────────
EPOCHS        = 30
WARMUP_EPOCHS = 3
LR            = 3e-4
WEIGHT_DECAY  = 1e-2
SEED          = 42

TRIAL_RX = re.compile(r"^(P\d+)_t(\d+)_w\d+\.pkl$")


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


class NormWearFrozenHead(nn.Module):
    """Frozen encoder + trainable MLP head — Side B of the ablation."""
    def __init__(self, base_encoder):
        super().__init__()
        self.encoder = base_encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.head = MLPHead(embed_dim=768, hidden=256, dropout=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, nvar, C, L, F = x.shape
        x_in = x.view(B * nvar, C, L, F)
        self.encoder.eval()
        with torch.no_grad():
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


def eval_auc(model, dl, device):
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for batch in dl:
            x = batch["input"].to(device)
            p = torch.sigmoid(model(x).squeeze(-1)).cpu().numpy()
            all_y.append(batch["label"].numpy())
            all_p.append(p)
    y_arr = np.concatenate(all_y); p_arr = np.concatenate(all_p)
    if len(np.unique(y_arr)) < 2:
        return float("nan")
    return roc_auc_score(y_arr, p_arr) * 100


def train_one_subject(sid, train_files, test_files, encoder_template, device):
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir="/tmp")
    json.dump({"train": train_files, "test": test_files}, tmp)
    tmp.close()

    common = dict(data_dir=DATA_DIR, ds_name=DS_NAME, split_file=tmp.name,
                  max_L=MAX_L, pad_nvar=PAD_NVAR, task_type=TASK_TYPE, sid_split_idx=0)
    ds_train = PersonalizedDownstreamDataset(**common, split="train")
    ds_test  = PersonalizedDownstreamDataset(**common, split="test")
    os.unlink(tmp.name)

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)
    dl_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    model = NormWearFrozenHead(copy.deepcopy(encoder_template)).to(device)
    info = model.count_parameters()
    print(f"  Params: trainable={info['trainable']:,}  "
          f"frozen={info['frozen']:,}  "
          f"({100*info['trainable']/max(info['total'],1):.2f}%)")

    trainable = model.trainable_parameters()
    optim = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY,
                               betas=(0.9, 0.999))
    smooth = 0.1
    bce = lambda logits, y: nn.BCEWithLogitsLoss()(logits, y * (1 - smooth) + smooth * 0.5)

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

    test_auc = eval_auc(model, dl_test, device)
    del model, optim
    torch.cuda.empty_cache()
    return test_auc


def _resolve_splits():
    if os.path.isfile(SPLIT_JSON):
        with open(SPLIT_JSON) as fp:
            return json.load(fp)
    raise RuntimeError(f"split file not found: {SPLIT_JSON}")


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*65}")
    print(f"  NormWear + Frozen-Encoder + MLPHead  (AMIGOS Side B control)")
    print(f"  epochs={EPOCHS}  recipe matches lora_amigos_run.py")
    print(f"  device={device}")
    print(f"{'='*65}\n")

    encoder_template = load_encoder(CKPT_PATH)
    subject_splits   = _resolve_splits()
    subjects         = sorted(subject_splits.keys())  # AMIGOS IDs are strings (P01, P07, ...)
    print(f"Subjects ({len(subjects)}): {subjects}\n")

    per_subject = []
    for sid in subjects:
        train_files = subject_splits[sid]["train"]
        test_files  = subject_splits[sid]["test"]
        test_lbls = []
        for f in test_files:
            d = pickle.load(open(os.path.join(DATA_DIR, f), "rb"))
            raw = d["label"]
            cls = raw[0]["class"] if isinstance(raw, list) else int(raw)
            test_lbls.append(1 if int(cls) == 1 else 0)
        if len(set(test_lbls)) < 2:
            print(f"[{sid}] SKIP — single-class test set"); continue

        print(f"\n{'='*65}")
        print(f"[{sid}] train_windows={len(train_files)}  test_windows={len(test_files)}")
        try:
            auc = train_one_subject(sid, train_files, test_files, encoder_template, device)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[{sid}] ERROR: {e}"); continue
        print(f"[{sid}] *** test_AUC = {auc:.2f}% ***")
        per_subject.append({"subject_id": sid, "auc": float(auc),
                            "n_train_wins": len(train_files),
                            "n_test_wins":  len(test_files)})

    aucs = [r["auc"] for r in per_subject if not np.isnan(r["auc"])]
    summary = {
        "method":     "frozen_encoder + MLPHead(256) + AdamW+cosine+labelsmooth (same recipe as LoRA, no LoRA layers)",
        "checkpoint": os.path.basename(CKPT_PATH),
        "split_file": os.path.basename(SPLIT_JSON),
        "per_subject": per_subject,
        "mean_auc":   float(np.mean(aucs)) if aucs else float("nan"),
        "std_auc":    float(np.std(aucs))  if aucs else float("nan"),
        "n_subjects": len(per_subject),
    }
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, "w") as fp:
        json.dump(summary, fp, indent=2)

    print(f"\n{'='*65}")
    print(f"  AMIGOS Side B (Frozen + MLPHead, no LoRA) Results")
    print(f"{'='*65}")
    for r in per_subject:
        print(f"  {r['subject_id']}  test_AUC = {r['auc']:.2f}%")
    print(f"  {'─'*30}")
    print(f"  Mean AUC = {summary['mean_auc']:.2f} ± {summary['std_auc']:.2f}%  "
          f"(n={summary['n_subjects']})")
    print(f"\nSaved → {SAVE_PATH}")


if __name__ == "__main__":
    main()
