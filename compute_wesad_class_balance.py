"""
Compute per-subject class balance (stress vs non-stress windows) for WESAD,
using the same per_subject_splits.json that drove the LoRA / baseline runs.

Convention from lora_wesad_run.py: class==1 -> stress, else -> non-stress.
"""
import json, os, pickle
from collections import defaultdict
from pathlib import Path

ROOT       = Path("/home/ug24/FoundationalModel/NormWear")
DATA_DIR   = ROOT / "data/wearable_downstream/wesad/sample_for_downstream"
SPLIT_JSON = ROOT / "data/wearable_downstream/wesad/per_subject_splits.json"
OUT_JSON   = ROOT / "data/results/lora_results/wesad_class_balance.json"

def parse_class(d):
    raw = d["label"]
    cls = raw[0]["class"] if isinstance(raw, list) else int(raw)
    return 1 if int(cls) == 1 else 0

splits = json.load(open(SPLIT_JSON))
per_subject = {}
for sid in sorted(splits.keys(), key=int):
    counts = {"train": [0, 0], "test": [0, 0]}   # [non-stress, stress]
    for split in ("train", "test"):
        for fn in splits[sid][split]:
            d = pickle.load(open(DATA_DIR / fn, "rb"))
            counts[split][parse_class(d)] += 1
    per_subject[sid] = {
        "train_total":      sum(counts["train"]),
        "train_non_stress": counts["train"][0],
        "train_stress":     counts["train"][1],
        "test_total":       sum(counts["test"]),
        "test_non_stress":  counts["test"][0],
        "test_stress":      counts["test"][1],
        "test_stress_pct":  round(100 * counts["test"][1] / max(sum(counts["test"]), 1), 1),
    }

json.dump({"per_subject": per_subject}, open(OUT_JSON, "w"), indent=2)

print(f"\n{'SID':<5}{'tr_tot':>8}{'tr_neg':>8}{'tr_pos':>8}"
      f"{'te_tot':>8}{'te_neg':>8}{'te_pos':>8}{'te_pos%':>9}")
print("-" * 60)
for sid, r in per_subject.items():
    print(f"S{sid:<4}{r['train_total']:>8}{r['train_non_stress']:>8}{r['train_stress']:>8}"
          f"{r['test_total']:>8}{r['test_non_stress']:>8}{r['test_stress']:>8}"
          f"{r['test_stress_pct']:>9.1f}")
print(f"\nSaved → {OUT_JSON}")
