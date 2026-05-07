"""
prepare_dreamer_downstream.py
=============================
Convert DREAMER.mat into NormWear downstream-format .pkl files,
preserving subject identity (uid) so per-subject LoRA personalization works.

Output layout:
    data/wearable_downstream/dreamer/
    ├── sample_for_downstream/
    │   ├── S01_t00_w000.pkl
    │   ├── S01_t00_w001.pkl
    │   └── ...
    └── train_test_split.json   (global 80/20, used by global mode)

Each .pkl payload:
    {
      "uid":           "S01",
      "data":          np.float16 [14, 388],
      "sampling_rate": 128,
      "label":         [{"class": 0 or 1}],   # binary arousal: >=3 -> 1
      "tss":           same as data (alias for compat)
    }

Channel selection: all 14 EEG channels from EMOTIV Epoc headset
  AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
  (ECG excluded — recorded at 256 Hz vs EEG at 128 Hz, not in pretrain)
"""
import os
import json
import pickle
import random
import numpy as np
import scipy.io as sio
from collections import defaultdict

# ── Config ───────────────────────────────────────────────────────────────────
SRC_MAT     = "/home/ug24/FoundationalModel/NormWear/data/dreamer/DREAMER.mat"
DST_ROOT    = "/home/ug24/FoundationalModel/NormWear/data/wearable_downstream/dreamer"
DST_SAMPLES = os.path.join(DST_ROOT, "sample_for_downstream")
SPLIT_JSON  = os.path.join(DST_ROOT, "train_test_split.json")

WINDOW_LEN      = 388       # samples per window (matches NormWear pretrain)
SAMPLING_RATE   = 128       # EEG sampling rate (Hz)
N_CHANNELS      = 14        # all EEG channels
LABEL_THRESHOLD = 3         # >=3 -> high arousal (1), <3 -> low (0)  [1-5 scale]
SEED            = 42

CHANNEL_NAMES = ["AF3", "F7", "F3", "FC5", "T7", "P7",
                 "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]


def main():
    os.makedirs(DST_SAMPLES, exist_ok=True)

    mat = sio.loadmat(SRC_MAT, simplify_cells=False)
    D   = mat["DREAMER"][0, 0]
    subjects_data = D["Data"]                       # shape (1, 23)
    n_subjects    = D["noOfSubjects"][0, 0]
    n_trials      = D["noOfVideoSequences"][0, 0]
    print(f"DREAMER: {n_subjects} subjects, {n_trials} trials each")

    all_filenames       = []
    label_dist          = defaultdict(int)
    per_subject_counts  = defaultdict(int)

    for s_idx in range(n_subjects):
        sid = f"S{s_idx + 1:02d}"
        subj = subjects_data[0, s_idx]

        eeg_stimuli = subj["EEG"][0, 0]["stimuli"][0, 0]   # (18, 1)
        score_arousal = subj["ScoreArousal"][0, 0].flatten()  # (18,)

        for t_idx in range(n_trials):
            eeg_trial = eeg_stimuli[t_idx, 0]  # [L, 14]

            if eeg_trial.size == 0:
                continue

            L, C = eeg_trial.shape
            if C < N_CHANNELS or L < WINDOW_LEN:
                continue

            arousal_score = int(score_arousal[t_idx])
            cls = int(arousal_score >= LABEL_THRESHOLD)

            sig = eeg_trial[:, :N_CHANNELS].astype(np.float32).T  # [14, L]

            # per-channel z-score normalisation
            mu  = sig.mean(axis=1, keepdims=True)
            std = sig.std(axis=1, keepdims=True) + 1e-6
            sig = (sig - mu) / std

            n_w = L // WINDOW_LEN
            for w in range(n_w):
                seg   = sig[:, w * WINDOW_LEN:(w + 1) * WINDOW_LEN].astype(np.float16)
                fname = f"{sid}_t{t_idx:02d}_w{w:03d}.pkl"
                payload = {
                    "uid":           sid,
                    "data":          seg,
                    "tss":           seg,
                    "sampling_rate": SAMPLING_RATE,
                    "label":         [{"class": cls}],
                }
                with open(os.path.join(DST_SAMPLES, fname), "wb") as fp:
                    pickle.dump(payload, fp)
                all_filenames.append(fname)
                label_dist[cls] += 1
                per_subject_counts[sid] += 1

        print(f"  {sid}: {per_subject_counts[sid]} windows")

    # ── Global 80/20 split ────────────────────────────────────────────────────
    random.seed(SEED)
    shuffled = sorted(all_filenames)
    random.shuffle(shuffled)
    n_train = int(0.8 * len(shuffled))
    split   = {"train": shuffled[:n_train], "test": shuffled[n_train:]}
    with open(SPLIT_JSON, "w") as fp:
        json.dump(split, fp)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = sum(label_dist.values())
    print(f"\n{'='*60}")
    print(f"Output  : {DST_ROOT}")
    print(f"Total   : {len(all_filenames)} windows")
    print(f"Subjects: {len(per_subject_counts)}")
    print(f"Labels  : low={label_dist[0]}  high={label_dist[1]}  "
          f"(ratio {label_dist[1]/max(total,1):.2%} high)")
    print(f"Channels: {CHANNEL_NAMES}")
    print(f"Label   : arousal >= {LABEL_THRESHOLD} -> 1, else 0  [1-5 scale]")
    print(f"Split   : {len(split['train'])} train / {len(split['test'])} test")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
