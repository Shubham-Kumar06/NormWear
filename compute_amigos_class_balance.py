"""
Per-subject class balance for AMIGOS, using the same per_subject_splits.json
that drove the LoRA / baseline runs. Mirrors compute_wesad_class_balance.py.

AMIGOS task: binary arousal (class==1 -> high arousal).
"""
import json, pickle
from pathlib import Path

ROOT       = Path("/home/ug24/FoundationalModel/NormWear")
DATA_DIR   = ROOT / "data/wearable_downstream/amigos/sample_for_downstream"
SPLIT_JSON = ROOT / "data/wearable_downstream/amigos/per_subject_splits.json"
OUT_JSON   = ROOT / "data/results/lora_results/amigos_class_balance.json"

def parse_class(d):
    raw = d["label"]
    cls = raw[0]["class"] if isinstance(raw, list) else int(raw)
    return 1 if int(cls) == 1 else 0

splits = json.load(open(SPLIT_JSON))
per_subject = {}
for sid in sorted(splits.keys()):
    counts = {"train": [0, 0], "test": [0, 0]}
    for split in ("train", "test"):
        for fn in splits[sid][split]:
            d = pickle.load(open(DATA_DIR / fn, "rb"))
            counts[split][parse_class(d)] += 1
    per_subject[sid] = {
        "train_total":      sum(counts["train"]),
        "train_low":        counts["train"][0],
        "train_high":       counts["train"][1],
        "test_total":       sum(counts["test"]),
        "test_low":         counts["test"][0],
        "test_high":        counts["test"][1],
        "test_high_pct":    round(100 * counts["test"][1] / max(sum(counts["test"]), 1), 1),
    }

json.dump({"per_subject": per_subject}, open(OUT_JSON, "w"), indent=2)

print(f"\n{'SID':<5}{'tr_tot':>8}{'tr_lo':>8}{'tr_hi':>8}"
      f"{'te_tot':>8}{'te_lo':>8}{'te_hi':>8}{'te_hi%':>9}")
print("-" * 60)
for sid, r in per_subject.items():
    print(f"{sid:<5}{r['train_total']:>8}{r['train_low']:>8}{r['train_high']:>8}"
          f"{r['test_total']:>8}{r['test_low']:>8}{r['test_high']:>8}"
          f"{r['test_high_pct']:>9.1f}")
print(f"\nSaved → {OUT_JSON}")
