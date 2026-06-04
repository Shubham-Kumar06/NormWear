"""
On the aug-BEST backbone, does LoRA personalization beat a simple frozen-encoder
+ logistic-regression head?

  Side A (no LoRA) : wesad_baseline_summary_AUG_BEST.json
  Side B (LoRA)    : wesad_lora_paper_summary_AUG_BEST.json

Reports per-subject Δ = LoRA − no-LoRA, mean Δ, win/loss/tie counts, verdict.
Also surfaces the 4-cell historical table (399 vs aug-BEST × no-LoRA vs LoRA).
"""
import json
from pathlib import Path

ROOT = Path("/home/ug24/FoundationalModel/NormWear/data/results/lora_results")

P = {
    ("399",     "no_lora"): ROOT / "wesad_baseline_summary.json",
    ("399",     "lora"):    ROOT / "wesad_lora_paper_summary.json",
    ("AUG_BEST","no_lora"): ROOT / "wesad_baseline_summary_AUG_BEST.json",
    ("AUG_BEST","lora"):    ROOT / "wesad_lora_paper_summary_AUG_BEST.json",
}

def load(path):
    d = json.load(open(path))
    return {r["subject_id"]: r["auc"] for r in d["per_subject"]}

results = {k: load(v) for k, v in P.items()}

base = results[("AUG_BEST", "no_lora")]
lora = results[("AUG_BEST", "lora")]
sids = sorted(set(base) & set(lora), key=int)

rows = []
wins = losses = ties = 0
for sid in sids:
    b, l = base[sid], lora[sid]
    d = l - b
    if d > 0.5: wins += 1
    elif d < -0.5: losses += 1
    else: ties += 1
    rows.append({"subject": sid, "no_lora": round(b, 2),
                 "lora": round(l, 2), "delta": round(d, 2)})

mean_base = sum(r["no_lora"] for r in rows) / len(rows)
mean_lora = sum(r["lora"]    for r in rows) / len(rows)
delta     = mean_lora - mean_base

verdict = ("LoRA HELPS"    if delta >  2 and wins >= 10 else
           "LoRA MARGINAL" if abs(delta) <= 2          else
           "LoRA HURTS")

out = {
    "backbone": "normwear_aug_BEST_ep0_81p35.pth",
    "side_A": {"method": "frozen encoder + logistic regression",
               "source": P[("AUG_BEST","no_lora")].name},
    "side_B": {"method": "frozen encoder + LoRA(rank=4 qkv) + linear head",
               "source": P[("AUG_BEST","lora")].name},
    "per_subject": rows,
    "mean_no_lora": round(mean_base, 2),
    "mean_lora":    round(mean_lora, 2),
    "delta_mean":   round(delta, 2),
    "wins_for_lora":   wins,
    "losses_for_lora": losses,
    "ties": ties,
    "verdict": verdict,
}
out_path = ROOT / "wesad_lora_vs_baseline_AUG_BEST.json"
json.dump(out, open(out_path, "w"), indent=2)

# Per-subject table
print(f"\n=== On aug-BEST backbone: LoRA vs frozen-encoder baseline ===\n")
print(f"{'Subject':<8}{'no-LoRA':>12}{'LoRA':>12}{'Δ (gain)':>12}")
print("-" * 44)
for r in rows:
    mark = "✅" if r["delta"] > 0.5 else ("❌" if r["delta"] < -0.5 else " ·")
    print(f"S{r['subject']:<7}{r['no_lora']:>12.2f}{r['lora']:>12.2f}"
          f"{r['delta']:>+12.2f}  {mark}")
print("-" * 44)
print(f"{'MEAN':<8}{mean_base:>12.2f}{mean_lora:>12.2f}{delta:>+12.2f}")
print(f"\nWins:{wins}  Losses:{losses}  Ties:{ties}  (|Δ|>0.5)")
print(f"Verdict: {verdict}")

# 4-cell historical table
print(f"\n=== Cross-reference: 4-cell table (mean AUC, n=15) ===\n")
def mean(d): return sum(d.values()) / len(d)
m_399_nl, m_399_l = mean(results[("399","no_lora")]), mean(results[("399","lora")])
m_AB_nl,  m_AB_l  = mean(base),                       mean(lora)
print(f"               {'no-LoRA':>10}{'LoRA':>10}{'LoRA gain':>12}")
print(f"  399-ckpt   {m_399_nl:>10.2f}{m_399_l:>10.2f}{m_399_l-m_399_nl:>+12.2f}")
print(f"  aug-BEST   {m_AB_nl:>10.2f}{m_AB_l:>10.2f}{m_AB_l-m_AB_nl:>+12.2f}")
print(f"\nSaved → {out_path}")
