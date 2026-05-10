"""
prepare_amigos_per_subject_splits.py
=====================================
Generate per-subject 80/20 train-test splits for AMIGOS in NormWear
downstream format.

Split strategy
--------------
* Trial-level split (not window-level) to prevent temporal leakage within
  a trial bleeding across train/test.
* Stratified by arousal class so both train and test sets contain both
  low- and high-arousal trials where possible.
* 80 % of trials per subject → train, 20 % → test.
* All windows that belong to a trial inherit its assignment.

Output
------
data/wearable_downstream/amigos/per_subject_splits.json
  {
    "P01": {
      "train": ["P01_t00_w000.pkl", ...],
      "test":  ["P01_t03_w000.pkl", ...]
    },
    ...
  }

Usage
-----
  python3 -m NormWear.prepare_amigos_per_subject_splits
"""

import os
import re
import json
import pickle
from collections import defaultdict

import numpy as np

DATA_DIR   = "/home/ug24/FoundationalModel/NormWear/data/wearable_downstream/amigos/sample_for_downstream"
OUT_JSON   = "/home/ug24/FoundationalModel/NormWear/data/wearable_downstream/amigos/per_subject_splits.json"
TRIAL_RE   = re.compile(r"^(P\d+)_t(\d+)_w(\d+)\.pkl$")
TRAIN_FRAC = 0.80
SEED       = 42


def gather_trials(data_dir: str):
    """
    Returns
    -------
    by_subj_trial : {sid: {trial_idx: [fname, ...]}}
    trial_label   : {(sid, trial_idx): 0 or 1}
    """
    by_subj_trial = defaultdict(lambda: defaultdict(list))
    trial_label   = {}

    for fname in sorted(os.listdir(data_dir)):
        m = TRIAL_RE.match(fname)
        if not m:
            continue
        sid, tidx = m.group(1), int(m.group(2))
        by_subj_trial[sid][tidx].append(fname)

        key = (sid, tidx)
        if key not in trial_label:
            d = pickle.load(open(os.path.join(data_dir, fname), "rb"))
            raw = d["label"]
            # handles [{"class": 0}] or {"class": 0} or plain int
            if isinstance(raw, list):
                cls = int(raw[0]["class"]) if isinstance(raw[0], dict) else int(raw[0])
            elif isinstance(raw, dict):
                cls = int(raw["class"])
            else:
                cls = int(raw)
            trial_label[key] = cls

    return by_subj_trial, trial_label


def stratified_trial_split(trial_label_pairs, train_frac: float, rng):
    """
    Stratified 80/20 trial split.

    Parameters
    ----------
    trial_label_pairs : [(trial_idx, label), ...]
    train_frac        : fraction going to train
    rng               : np.random.Generator

    Returns
    -------
    train_trials, test_trials : list[int], list[int]
    """
    by_class = defaultdict(list)
    for tidx, lbl in trial_label_pairs:
        by_class[lbl].append(tidx)

    train_trials, test_trials = [], []
    for lbl in sorted(by_class.keys()):
        trials = sorted(by_class[lbl])
        rng.shuffle(trials)
        n_tr = int(round(len(trials) * train_frac))
        # guarantee at least 1 test trial per class when possible
        if n_tr == len(trials) and len(trials) >= 2:
            n_tr = len(trials) - 1
        train_trials.extend(trials[:n_tr])
        test_trials.extend(trials[n_tr:])

    return sorted(train_trials), sorted(test_trials)


def main():
    rng = np.random.default_rng(SEED)

    by_subj_trial, trial_label = gather_trials(DATA_DIR)
    subjects = sorted(by_subj_trial.keys())
    print(f"Found {len(subjects)} subjects, "
          f"{sum(len(v) for d in by_subj_trial.values() for v in d.values())} total windows")

    splits = {}
    print(f"\n{'Subject':8} {'Trials':7} {'TrainT':7} {'TestT':7} {'TrainW':7} {'TestW':6} {'Cls(0/1)':10}")
    print("-" * 65)

    for sid in subjects:
        trials_dict = by_subj_trial[sid]
        tidxs       = sorted(trials_dict.keys())
        twl         = [(t, trial_label[(sid, t)]) for t in tidxs]

        train_t, test_t = stratified_trial_split(twl, TRAIN_FRAC, rng)
        train_files = [f for t in train_t for f in sorted(trials_dict[t])]
        test_files  = [f for t in test_t  for f in sorted(trials_dict[t])]

        tr_lbl = [trial_label[(sid, t)] for t in train_t]
        te_lbl = [trial_label[(sid, t)] for t in test_t]

        print(f"{sid:8} {len(tidxs):7d} {len(train_t):7d} {len(test_t):7d} "
              f"{len(train_files):7d} {len(test_files):6d} "
              f"tr:{tr_lbl.count(0)}L/{tr_lbl.count(1)}H "
              f"te:{te_lbl.count(0)}L/{te_lbl.count(1)}H")

        splits[sid] = {"train": train_files, "test": test_files}

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fp:
        json.dump(splits, fp, indent=2)

    total_train = sum(len(v["train"]) for v in splits.values())
    total_test  = sum(len(v["test"])  for v in splits.values())
    print(f"\nSaved → {OUT_JSON}")
    print(f"Total  : {total_train} train windows / {total_test} test windows "
          f"(ratio {total_train/(total_train+total_test):.1%} / "
          f"{total_test/(total_train+total_test):.1%})")


if __name__ == "__main__":
    main()
