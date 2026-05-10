"""
baseline_dreamer.py
===================
Per-subject baseline on DREAMER using the FROZEN NormWear encoder
(epoch 399) + a logistic-regression linear probe — NO LoRA.

For each subject:
  1. Load all their .pkl windows.
  2. CWT + NormWear encoder → [768]-dim feature per window.
  3. Stratified 80/20 split by class.
  4. Fit logistic regression on train, evaluate AUC on test.
  5. Print per-subject AUC and final mean.

Run from /home/ug24/FoundationalModel/:
    python3 -m NormWear.baseline_dreamer
"""
import os, sys, json, pickle
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NormWear.lora.lora_dataset import (
    cwt_transform, parse_label, extract_subject_id,
)
from NormWear.modules.normwear import NormWear

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR    = "/home/ug24/FoundationalModel/NormWear/data/wearable_downstream/dreamer/sample_for_downstream"
CKPT_PATH   = "/home/ug24/FoundationalModel/NormWear/data/results/full_pretrain_checkpoint-399.pth"
SAVE_PATH   = "/home/ug24/FoundationalModel/NormWear/data/results/lora_results/dreamer_baseline_summary.json"
DS_NAME     = "dreamer"
PAD_NVAR    = 16   # pad to multiple of 4 (NormWear nvar=4 internals require B*nvar % 4 == 0)
MAX_L       = 390
TRAIN_FRAC  = 0.8
SEED        = 42
BATCH_SIZE  = 32


def load_encoder(ckpt_path: str, device):
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
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


def prep_input(pkl_path: str) -> torch.Tensor:
    """Load one .pkl, CWT transform, pad/trim to (PAD_NVAR, 3, MAX_L, 65)."""
    d = pickle.load(open(pkl_path, "rb"))
    tss = d.get("data", d.get("tss"))
    if isinstance(tss, torch.Tensor): tss = tss.numpy()
    tss = np.array(tss, dtype=np.float32)
    if tss.ndim == 1:    tss = tss[np.newaxis, :]
    elif tss.ndim > 2:   tss = tss.reshape(-1, tss.shape[-1])
    tss = np.nan_to_num(tss, nan=0.0, posinf=0.0, neginf=0.0)

    cwt = cwt_transform(tss)             # [nvar, 3, L_cwt, 65]
    nvar, _, L_cwt, _ = cwt.shape

    if L_cwt < MAX_L:
        pad = torch.zeros(nvar, 3, MAX_L - L_cwt, 65)
        cwt = torch.cat([cwt, pad], dim=2)
    else:
        cwt = cwt[:, :, :MAX_L, :]

    if nvar < PAD_NVAR:
        pad = torch.zeros(PAD_NVAR - nvar, 3, MAX_L, 65)
        cwt = torch.cat([cwt, pad], dim=0)
    else:
        cwt = cwt[:PAD_NVAR, :, :, :]
    return cwt   # [14, 3, MAX_L, 65]


@torch.no_grad()
def encode_batch(encoder, batch_inputs: torch.Tensor, device) -> np.ndarray:
    """[B, 14, 3, MAX_L, 65] -> [B, 768] mean-pooled embedding."""
    B, nvar, C, L, F = batch_inputs.shape
    x_in = batch_inputs.view(B * nvar, C, L, F).to(device)
    latent, _, _ = encoder.forward_encoder(x_in)        # [B*nvar, P+1, 768]
    latent = latent[:, 1:, :]                            # drop CLS
    P = latent.shape[1]
    latent = latent.view(B, nvar, P, -1).mean(dim=(1, 2))   # [B, 768]
    return latent.cpu().numpy()


def gather_subjects(data_dir: str):
    """{subject_id: [(fname, label), ...]}"""
    by_subj = defaultdict(list)
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".pkl"): continue
        d = pickle.load(open(os.path.join(data_dir, fname), "rb"))
        sid = extract_subject_id(d, fname, 0)
        lbl = parse_label(DS_NAME, d)
        by_subj[sid].append((fname, lbl))
    return by_subj


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Baseline] device={device}")

    encoder = load_encoder(CKPT_PATH, device)
    by_subj = gather_subjects(DATA_DIR)
    subjects = sorted(by_subj.keys())
    print(f"[Baseline] {len(subjects)} subjects, "
          f"total samples={sum(len(v) for v in by_subj.values())}")

    rng = np.random.default_rng(SEED)
    per_subject_results = []

    for sid in subjects:
        files_lbls = sorted(by_subj[sid])
        labels     = np.array([l for _, l in files_lbls])
        fnames     = [f for f, _ in files_lbls]
        n_total    = len(fnames)

        # stratified 80/20 split
        train_idx, test_idx = [], []
        for cls in np.unique(labels):
            idxs = np.where(labels == cls)[0]
            rng.shuffle(idxs)
            n_tr = max(1, int(len(idxs) * TRAIN_FRAC))
            train_idx.extend(idxs[:n_tr]); test_idx.extend(idxs[n_tr:])
        train_idx = np.array(train_idx); test_idx = np.array(test_idx)

        if len(test_idx) == 0 or len(np.unique(labels[test_idx])) < 2:
            print(f"[{sid}] SKIP — single-class test set")
            continue

        feats_all = np.empty((n_total, 768), dtype=np.float32)
        for start in range(0, n_total, BATCH_SIZE):
            batch_files = fnames[start:start+BATCH_SIZE]
            batch = torch.stack([prep_input(os.path.join(DATA_DIR, f))
                                 for f in batch_files])
            feats_all[start:start+len(batch_files)] = encode_batch(encoder, batch, device)

        Xtr, Xte = feats_all[train_idx], feats_all[test_idx]
        ytr, yte = labels[train_idx],  labels[test_idx]

        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr_s, ytr)
        prob_te = clf.predict_proba(Xte_s)[:, 1]
        auc = roc_auc_score(yte, prob_te) * 100

        print(f"[{sid}] train={len(train_idx):4d} test={len(test_idx):4d}  "
              f"AUC = {auc:.2f}%")
        per_subject_results.append({"subject_id": sid, "auc": float(auc),
                                     "n_train": int(len(train_idx)),
                                     "n_test":  int(len(test_idx))})

    aucs = [r["auc"] for r in per_subject_results]
    summary = {
        "ds_name":      DS_NAME,
        "method":       "frozen_normwear_encoder + logistic_regression",
        "checkpoint":   os.path.basename(CKPT_PATH),
        "per_subject":  per_subject_results,
        "mean_auc":     float(np.mean(aucs)) if aucs else float("nan"),
        "std_auc":      float(np.std(aucs))  if aucs else float("nan"),
        "n_subjects":   len(per_subject_results),
    }
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\n[Baseline] saved → {SAVE_PATH}")
    print(f"[Baseline] DREAMER Mean AUC = {summary['mean_auc']:.2f} "
          f"± {summary['std_auc']:.2f}%  (n={summary['n_subjects']})")


if __name__ == "__main__":
    main()
