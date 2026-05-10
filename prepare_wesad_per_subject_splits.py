"""
prepare_wesad_per_subject_splits.py
=====================================
Generate per-subject 80/20 train-test splits for WESAD in NormWear
downstream format.

Split strategy
--------------
* Trial-level split to prevent temporal leakage.
* Trial key  =  condition * 1000 + segment  (unique per condition/segment pair).
* Labels binarized BEFORE stratification: stress (raw=1) → 1, else → 0.
* 80 % of trials per class per subject → train, 20 % → test.

Output
------
data/wearable_downstream/wesad/per_subject_splits.json
  {
    "2":  {"train": ["2_0_0_1174", ...], "test": ["2_1_5_0000", ...]},
    "3":  {...},
    ...
  }

Usage
-----
  python3 -m NormWear.prepare_wesad_per_subject_splits
"""

import os
import re
import json
import pickle
from collections import defaultdict

import numpy as np

DATA_DIR  = "/home/ug24/FoundationalModel/NormWear/data/wearable_downstream/wesad/sample_for_downstream"
OUT_JSON  = "/home/ug24/FoundationalModel/NormWear/data/wearable_downstream/wesad/per_subject_splits.json"
TRIAL_RE  = re.compile(r"^(\d+)_(\d+)_(\d+)_\d+$")
TRAIN_FRAC = 0.80
SEED       = 42


def binarize(raw_cls: int) -> int:
    return 1 if raw_cls == 1 else 0


def gather_trials(data_dir: str):
    by_subj_trial = defaultdict(lambda: defaultdict(list))
    trial_label   = {}
    for fname in sorted(os.listdir(data_dir)):
        m = TRIAL_RE.match(fname)
        if not m:
            continue
        sid  = m.group(1)
        tidx = int(m.group(2)) * 1000 + int(m.group(3))
        by_subj_trial[sid][tidx].append(fname)
        key = (sid, tidx)
        if key not in trial_label:
            d = pickle.load(open(os.path.join(data_dir, fname), "rb"))
            raw = d["label"]
            cls = raw[0]["class"] if isinstance(raw, list) else int(raw)
            trial_label[key] = binarize(int(cls))
    return by_subj_trial, trial_label


def stratified_trial_split(trial_label_pairs, train_frac: float, rng):
    by_class = defaultdict(list)
    for tidx, lbl in trial_label_pairs:
        by_class[lbl].append(tidx)
    train_trials, test_trials = [], []
    for lbl in sorted(by_class.keys()):
        trials = sorted(by_class[lbl])
        rng.shuffle(trials)
        n_tr = int(round(len(trials) * train_frac))
        if n_tr == len(trials) and len(trials) >= 2:
            n_tr = len(trials) - 1
        train_trials.extend(trials[:n_tr])
        test_trials.extend(trials[n_tr:])
    return sorted(train_trials), sorted(test_trials)


def main():
    rng = np.random.default_rng(SEED)
    by_subj_trial, trial_label = gather_trials(DATA_DIR)
    subjects = sorted(by_subj_trial.keys(), key=lambda x: int(x))

    total_files = sum(
        len(v) for d in by_subj_trial.values() for v in d.values()
    )
    print(f"Found {len(subjects)} subjects, {total_files} total windows")

    splits = {}
    print(f"\n{'Subject':8} {'Trials':7} {'TrainT':7} {'TestT':7} "
          f"{'TrainW':7} {'TestW':6} {'Cls(0/1)':10}")
    print("-" * 65)

    for sid in subjects:
        trials_dict = by_subj_trial[sid]
        twl = [(t, trial_label[(sid, t)]) for t in sorted(trials_dict.keys())]
        train_t, test_t = stratified_trial_split(twl, TRAIN_FRAC, rng)

        train_files = [f for t in train_t for f in sorted(trials_dict[t])]
        test_files  = [f for t in test_t  for f in sorted(trials_dict[t])]

        tr_lbl = [trial_label[(sid, t)] for t in train_t]
        te_lbl = [trial_label[(sid, t)] for t in test_t]

        print(f"S{sid:>5} {len(twl):7d} {len(train_t):7d} {len(test_t):7d} "
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
    print(f"Total  : {total_train} train / {total_test} test  "
          f"({total_train/(total_train+total_test):.1%} / "
          f"{total_test/(total_train+total_test):.1%})")


if __name__ == "__main__":
    main()
