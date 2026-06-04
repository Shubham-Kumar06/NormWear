"""
Compare per-subject WESAD LoRA results between the 399-ckpt baseline (May)
and the aug-BEST (epoch 0, MEAN 81.35) re-run.
"""
import json
from pathlib import Path

ROOT = Path("/home/ug24/FoundationalModel/NormWear/data/results/lora_results")
OLD = ROOT / "wesad_lora_paper_summary.json"          # 399-ckpt
NEW = ROOT / "wesad_lora_paper_summary_AUG_BEST.json" # aug-BEST

old = json.load(open(OLD))
new = json.load(open(NEW))

old_by_sid = {r["subject_id"]: r["auc"] for r in old["per_subject"]}
new_by_sid = {r["subject_id"]: r["auc"] for r in new["per_subject"]}

sids = sorted(set(old_by_sid) & set(new_by_sid), key=int)

rows = []
wins = losses = ties = 0
for sid in sids:
    o, n = old_by_sid[sid], new_by_sid[sid]
    d = n - o
    if d > 0.5: wins += 1
    elif d < -0.5: losses += 1
    else: ties += 1
    rows.append({"subject": sid, "old_399": round(o, 2),
                 "new_aug_BEST": round(n, 2), "delta": round(d, 2)})

mean_old = sum(r["old_399"] for r in rows) / len(rows)
mean_new = sum(r["new_aug_BEST"] for r in rows) / len(rows)
delta    = mean_new - mean_old

verdict = ("SHIP aug-BEST" if mean_new >= 94
           else "STAGE 2 sweep" if mean_new >= 92
           else "INVESTIGATE (aug-BEST worse for WESAD)")

out = {
    "old_checkpoint": old["checkpoint"],
    "new_checkpoint": new["checkpoint"],
    "config": new["method"],
    "per_subject": rows,
    "mean_old_399":      round(mean_old, 2),
    "mean_new_aug_BEST": round(mean_new, 2),
    "delta_mean":        round(delta, 2),
    "wins_for_new":   wins,
    "losses_for_new": losses,
    "ties":           ties,
    "verdict": verdict,
}
out_path = ROOT / "wesad_lora_comparison_399_vs_AUG_BEST.json"
json.dump(out, open(out_path, "w"), indent=2)

# Pretty print
print(f"\n{'Subject':<8}{'399-ckpt':>12}{'aug-BEST':>12}{'Δ':>10}")
print("-" * 42)
for r in rows:
    mark = "✅" if r["delta"] > 0.5 else ("❌" if r["delta"] < -0.5 else " ·")
    print(f"S{r['subject']:<7}{r['old_399']:>12.2f}{r['new_aug_BEST']:>12.2f}"
          f"{r['delta']:>+10.2f}  {mark}")
print("-" * 42)
print(f"{'MEAN':<8}{mean_old:>12.2f}{mean_new:>12.2f}{delta:>+10.2f}")
print(f"\nWins:{wins}  Losses:{losses}  Ties:{ties}  (|Δ|>0.5)")
print(f"Verdict: {verdict}")
print(f"\nSaved → {out_path}")
