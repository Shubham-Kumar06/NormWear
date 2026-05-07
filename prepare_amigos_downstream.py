"""
prepare_amigos_downstream.py
============================
Convert AMIGOS .mat files into NormWear downstream-format .pkl files,
preserving subject identity (uid) so per-subject LoRA personalization works.

Output layout:
    data/wearable_downstream/amigos/
    ├── sample_for_downstream/
    │   ├── P01_t00_w000.pkl
    │   ├── P01_t00_w001.pkl
    │   └── ...
    └── train_test_split.json   (global 80/20, used by global mode)

Each .pkl payload:
    {
      "uid":           "P01",
      "data":          np.float16 [4, 388],
      "sampling_rate": 128,
      "label":         [{"class": 0 or 1}],   # binary arousal: >=5 -> 1
      "tss":           same as data (alias for compat)
    }

Channel selection (4 of 17): AF3, AF4, ECG_R, GSR
  -- bilateral frontal EEG + cardiac + skin conductance
"""
import os
import glob
import json
import pickle
import random
import numpy as np
import scipy.io as sio
from collections import defaultdict

# ── Config ───────────────────────────────────────────────────────────────────
SRC_ROOT = "/home/ug24/FoundationalModel/NormWear/data/AMIGOS DATASET/PREPROCESSED DATA"
DST_ROOT = "/home/ug24/FoundationalModel/NormWear/data/wearable_downstream/amigos"
DST_SAMPLES = os.path.join(DST_ROOT, "sample_for_downstream")
SPLIT_JSON  = os.path.join(DST_ROOT, "train_test_split.json")

WINDOW_LEN = 388        # samples per window (matches NormWear pretrain)
SAMPLING_RATE = 128     # AMIGOS preprocessed sampling rate (Hz)
CHANNEL_INDICES = [0, 13, 14, 16]   # AF3, AF4, ECG_R, GSR
CHANNEL_NAMES   = ["AF3", "AF4", "ECG_R", "GSR"]
LABEL_COL = 1           # 0=valence, 1=arousal, 2=dominance
LABEL_NAME = "arousal"
LABEL_THRESHOLD = 5.0   # >=5 -> high (1), <5 -> low (0)
SEED = 42

# ── Build dataset ────────────────────────────────────────────────────────────
def main():
    os.makedirs(DST_SAMPLES, exist_ok=True)

    participant_dirs = sorted(glob.glob(os.path.join(SRC_ROOT, "Data_Preprocessed_P*")))
    print(f"Found {len(participant_dirs)} participants")

    all_filenames = []
    label_dist    = defaultdict(int)
    per_subject_counts = defaultdict(int)

    for pdir in participant_dirs:
        pid = os.path.basename(pdir).replace("Data_Preprocessed_", "")  # e.g. "P01"
        mat_path = os.path.join(pdir, f"Data_Preprocessed_{pid}.mat")
        if not os.path.isfile(mat_path):
            print(f"  [SKIP] {pid}: missing .mat")
            continue

        mat = sio.loadmat(mat_path)
        joined_data = mat["joined_data"][0]               # (1, 20) -> 20 trials
        labels      = mat["labels_selfassessment"][0]

        n_trials = len(joined_data)
        for t in range(n_trials):
            sig = joined_data[t]                          # [L, 17]
            lbl = labels[t]                               # [1, 12]

            # skip empty trials (group condition often missing for some subjects)
            if sig.size == 0 or lbl.size == 0:
                continue

            L, C = sig.shape
            if C < 17:
                continue
            if L < WINDOW_LEN:
                continue

            # binary label (per trial, repeated across windows)
            score = float(lbl[0, LABEL_COL])
            cls   = int(score >= LABEL_THRESHOLD)

            # select channels and transpose to [4, L]
            sig4 = sig[:, CHANNEL_INDICES].astype(np.float32).T   # [4, L]

            # per-channel z-score normalization (handles huge GSR/ECG scale gap)
            mu  = sig4.mean(axis=1, keepdims=True)
            std = sig4.std(axis=1, keepdims=True) + 1e-6
            sig4 = (sig4 - mu) / std

            # window into 388-sample chunks
            n_w = L // WINDOW_LEN
            for w in range(n_w):
                seg = sig4[:, w*WINDOW_LEN : (w+1)*WINDOW_LEN].astype(np.float16)
                fname = f"{pid}_t{t:02d}_w{w:03d}.pkl"
                payload = {
                    "uid":           pid,
                    "data":          seg,
                    "tss":           seg,           # alias for downstream loader
                    "sampling_rate": SAMPLING_RATE,
                    "label":         [{"class": cls}],
                }
                with open(os.path.join(DST_SAMPLES, fname), "wb") as fp:
                    pickle.dump(payload, fp)
                all_filenames.append(fname)
                label_dist[cls] += 1
                per_subject_counts[pid] += 1

        print(f"  {pid}: {per_subject_counts[pid]} windows")

    # ── Global 80/20 split (random, used only for global mode) ───────────────
    random.seed(SEED)
    shuffled = sorted(all_filenames)
    random.shuffle(shuffled)
    n_train = int(0.8 * len(shuffled))
    split = {"train": shuffled[:n_train], "test": shuffled[n_train:]}
    with open(SPLIT_JSON, "w") as fp:
        json.dump(split, fp)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Output  : {DST_ROOT}")
    print(f"Total   : {len(all_filenames)} windows")
    print(f"Subjects: {len(per_subject_counts)}")
    print(f"Labels  : low={label_dist[0]}  high={label_dist[1]}  "
          f"(ratio {label_dist[1]/max(sum(label_dist.values()),1):.2%} high)")
    print(f"Channels: {CHANNEL_NAMES}")
    print(f"Label   : {LABEL_NAME} >= {LABEL_THRESHOLD} -> 1, else 0")
    print(f"Split   : {len(split['train'])} train / {len(split['test'])} test")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
