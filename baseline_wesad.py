"""
baseline_wesad.py
=================
Method A for WESAD: NormWear frozen encoder + logistic-regression linear probe.

Uses the SAME per-subject trial-level 80/20 splits as lora_wesad_run.py
(loaded from per_subject_splits.json) so results are directly comparable.

Run from /home/ug24/FoundationalModel/:
    python3 -m NormWear.baseline_wesad
"""
import os, sys, json, pickle

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NormWear.lora.lora_dataset import cwt_transform
from NormWear.modules.normwear import NormWear

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "/home/ug24/FoundationalModel/NormWear/data/wearable_downstream/wesad/sample_for_downstream"
SPLIT_JSON = "/home/ug24/FoundationalModel/NormWear/data/wearable_downstream/wesad/per_subject_splits.json"
CKPT_PATH  = "/home/ug24/FoundationalModel/NormWear/data/results/full_pretrain_checkpoint-399.pth"
SAVE_PATH  = "/home/ug24/FoundationalModel/NormWear/data/results/lora_results/wesad_baseline_summary.json"
DS_NAME    = "wesad"
PAD_NVAR   = 12   # next multiple of 4 >= 10 channels
MAX_L      = 390
BATCH_SIZE = 32


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
    d = pickle.load(open(pkl_path, "rb"))
    tss = d.get("data", d.get("tss"))
    if isinstance(tss, torch.Tensor): tss = tss.numpy()
    tss = np.array(tss, dtype=np.float32)
    if tss.ndim == 1:   tss = tss[np.newaxis, :]
    elif tss.ndim > 2:  tss = tss.reshape(-1, tss.shape[-1])
    tss = np.nan_to_num(tss, nan=0.0, posinf=0.0, neginf=0.0)

    cwt = cwt_transform(tss)
    nvar, _, L_cwt, _ = cwt.shape

    if L_cwt < MAX_L:
        cwt = torch.cat([cwt, torch.zeros(nvar, 3, MAX_L - L_cwt, 65)], dim=2)
    else:
        cwt = cwt[:, :, :MAX_L, :]

    if nvar < PAD_NVAR:
        cwt = torch.cat([cwt, torch.zeros(PAD_NVAR - nvar, 3, MAX_L, 65)], dim=0)
    else:
        cwt = cwt[:PAD_NVAR, :, :, :]
    return cwt   # [PAD_NVAR, 3, MAX_L, 65]


@torch.no_grad()
def encode_batch(encoder, batch: torch.Tensor, device) -> np.ndarray:
    B, nvar, C, L, F = batch.shape
    x_in = batch.view(B * nvar, C, L, F).to(device)
    latent, _, _ = encoder.forward_encoder(x_in)
    latent = latent[:, 1:, :]
    latent = latent.view(B, nvar, latent.shape[1], -1).mean(dim=(1, 2))
    return latent.cpu().numpy()


def get_label(pkl_path: str) -> int:
    d = pickle.load(open(pkl_path, "rb"))
    raw = d["label"]
    cls = raw[0]["class"] if isinstance(raw, list) else int(raw)
    # WESAD binarize: stress=1, non-stress=0
    return 1 if int(cls) == 1 else 0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Baseline-WESAD] device={device}")

    with open(SPLIT_JSON) as fp:
        subject_splits = json.load(fp)
    print(f"[Baseline-WESAD] Loaded splits for {len(subject_splits)} subjects")

    encoder = load_encoder(CKPT_PATH, device)

    per_subject_results = []

    for sid in sorted(subject_splits.keys(), key=lambda x: int(x)):
        train_files = subject_splits[sid]["train"]
        test_files  = subject_splits[sid]["test"]
        all_files   = train_files + test_files
        n_train     = len(train_files)
        n_total     = len(all_files)

        feats = np.empty((n_total, 768), dtype=np.float32)
        for start in range(0, n_total, BATCH_SIZE):
            batch_f = all_files[start:start + BATCH_SIZE]
            batch   = torch.stack([prep_input(os.path.join(DATA_DIR, f)) for f in batch_f])
            feats[start:start + len(batch_f)] = encode_batch(encoder, batch, device)

        labels = np.array([get_label(os.path.join(DATA_DIR, f)) for f in all_files])

        Xtr, Xte = feats[:n_train], feats[n_train:]
        ytr, yte = labels[:n_train], labels[n_train:]

        if len(np.unique(yte)) < 2:
            print(f"[S{sid}] SKIP — single-class test set")
            continue

        scaler = StandardScaler().fit(Xtr)
        clf    = LogisticRegression(max_iter=2000, C=1.0).fit(
                     scaler.transform(Xtr), ytr)
        auc    = roc_auc_score(yte, clf.predict_proba(scaler.transform(Xte))[:, 1]) * 100

        print(f"[S{sid:>3}] train={n_train:5d}  test={len(test_files):5d}  AUC={auc:.2f}%")
        per_subject_results.append({
            "subject_id": sid,
            "auc":        float(auc),
            "n_train":    n_train,
            "n_test":     len(test_files),
        })

    aucs = [r["auc"] for r in per_subject_results]
    summary = {
        "ds_name":     DS_NAME,
        "method":      "frozen_normwear_encoder + logistic_regression (per-subject split)",
        "checkpoint":  os.path.basename(CKPT_PATH),
        "split_file":  os.path.basename(SPLIT_JSON),
        "per_subject": per_subject_results,
        "mean_auc":    float(np.mean(aucs)) if aucs else float("nan"),
        "std_auc":     float(np.std(aucs))  if aucs else float("nan"),
        "n_subjects":  len(per_subject_results),
    }
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, "w") as fp:
        json.dump(summary, fp, indent=2)

    print(f"\n[Baseline-WESAD] Mean AUC = {summary['mean_auc']:.2f} "
          f"± {summary['std_auc']:.2f}%  (n={summary['n_subjects']})")
    print(f"Saved → {SAVE_PATH}")


if __name__ == "__main__":
    main()
