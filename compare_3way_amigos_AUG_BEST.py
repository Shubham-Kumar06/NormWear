"""
3-way comparison on aug-BEST backbone for AMIGOS.
Mirror of compare_3way_AUG_BEST.py for AMIGOS data.

  A — frozen encoder + sklearn LogisticRegression  (linear probe)
  B — frozen encoder + MLPHead, same recipe as C  (recipe-only control)
  C — LoRA(rank=8, qkv/proj/fc1/fc2) + MLPHead    (LoRA condition)

Δ(C-B) = LoRA-attributable gain (the rigorous one).
"""
import json
from pathlib import Path
import statistics

ROOT = Path("/home/ug24/FoundationalModel/NormWear/data/results/lora_results")
SRC = {
    "A_linear_probe": ROOT / "amigos_baseline_summary_AUG_BEST.json",
    "B_neural_head":  ROOT / "amigos_neuralhead_summary_AUG_BEST.json",
    "C_lora":         ROOT / "amigos_lora_paper_summary_AUG_BEST.json",
}

def load(p):
    d = json.load(open(p))
    return {r["subject_id"]: r["auc"] for r in d["per_subject"]}

A = load(SRC["A_linear_probe"])
B = load(SRC["B_neural_head"])
C = load(SRC["C_lora"])
sids = sorted(set(A) & set(B) & set(C))

rows = []
for sid in sids:
    a, b, c = A[sid], B[sid], C[sid]
    rows.append({"subject": sid,
                 "A_lin": round(a, 2), "B_head": round(b, 2), "C_lora": round(c, 2),
                 "delta_C_minus_A": round(c - a, 2),
                 "delta_C_minus_B": round(c - b, 2),
                 "delta_B_minus_A": round(b - a, 2)})

mA = statistics.mean(A[s] for s in sids)
mB = statistics.mean(B[s] for s in sids)
mC = statistics.mean(C[s] for s in sids)
sA = statistics.stdev(A[s] for s in sids)
sB = statistics.stdev(B[s] for s in sids)
sC = statistics.stdev(C[s] for s in sids)

wins   = sum(1 for r in rows if r["delta_C_minus_B"] >  0.5)
losses = sum(1 for r in rows if r["delta_C_minus_B"] < -0.5)
ties   = len(rows) - wins - losses

try:
    from scipy.stats import wilcoxon
    diffs = [r["delta_C_minus_B"] for r in rows]
    W, p_val = wilcoxon(diffs)
    p_str = f"W={W:.1f}, p={p_val:.4g}"
except Exception as e:
    p_str = f"(scipy unavailable: {e})"

diffs = [r["delta_C_minus_B"] for r in rows]
d_mean = statistics.mean(diffs)
d_std  = statistics.stdev(diffs)
cohens_d = d_mean / d_std if d_std > 0 else float("inf")

verdict = ("LoRA HELPS (significant)" if d_mean > 1.5 and wins >= 10
           else "LoRA MARGINAL"        if abs(d_mean) <= 1.5
           else "LoRA HURTS")

out = {
    "backbone": "normwear_aug_BEST_ep0_81p35.pth",
    "dataset": "amigos",
    "conditions": {
        "A": "frozen encoder + sklearn LogisticRegression (linear probe)",
        "B": "frozen encoder + MLPHead(256) + same recipe as C (no LoRA)",
        "C": "LoRA(rank=8 alpha=16 targets=qkv,proj,fc1,fc2) + MLPHead(256)",
    },
    "per_subject": rows,
    "mean": {"A": round(mA, 2), "B": round(mB, 2), "C": round(mC, 2)},
    "std":  {"A": round(sA, 2), "B": round(sB, 2), "C": round(sC, 2)},
    "delta_mean": {
        "C_minus_A_combined":  round(mC - mA, 2),
        "C_minus_B_lora_only": round(mC - mB, 2),
        "B_minus_A_recipe":    round(mB - mA, 2),
    },
    "lora_vs_B_paired": {
        "wins": wins, "losses": losses, "ties": ties,
        "wilcoxon": p_str,
        "cohens_d_paired": round(cohens_d, 2),
    },
    "verdict": verdict,
}
out_path = ROOT / "amigos_3way_AUG_BEST.json"
json.dump(out, open(out_path, "w"), indent=2)

print(f"\n=== AMIGOS 3-way comparison on aug-BEST backbone ===\n")
print(f"{'Subject':<7}{'A:linprobe':>12}{'B:head':>10}{'C:LoRA':>10}"
      f"{'C-A':>8}{'C-B':>8}{'B-A':>8}")
print("-" * 63)
for r in rows:
    mark = "✅" if r["delta_C_minus_B"] > 0.5 else ("❌" if r["delta_C_minus_B"] < -0.5 else " ·")
    print(f"{r['subject']:<7}{r['A_lin']:>12.2f}{r['B_head']:>10.2f}{r['C_lora']:>10.2f}"
          f"{r['delta_C_minus_A']:>+8.2f}{r['delta_C_minus_B']:>+8.2f}"
          f"{r['delta_B_minus_A']:>+8.2f}  {mark}")
print("-" * 63)
print(f"{'MEAN':<7}{mA:>12.2f}{mB:>10.2f}{mC:>10.2f}"
      f"{mC-mA:>+8.2f}{mC-mB:>+8.2f}{mB-mA:>+8.2f}")
print(f"{'STD':<7}{sA:>12.2f}{sB:>10.2f}{sC:>10.2f}")

print(f"\n=== Headline numbers ===")
print(f"  Side A (linear probe)   mean = {mA:.2f} ± {sA:.2f}")
print(f"  Side B (neural head)    mean = {mB:.2f} ± {sB:.2f}")
print(f"  Side C (LoRA)           mean = {mC:.2f} ± {sC:.2f}")
print(f"")
print(f"  Δ(C−A) = {mC-mA:+.2f}   combined gain")
print(f"  Δ(C−B) = {mC-mB:+.2f}   LoRA-attributable gain")
print(f"  Δ(B−A) = {mB-mA:+.2f}   recipe-attributable gain")

print(f"\n=== Paired stats (C vs B, n={len(rows)}) ===")
print(f"  Wins for LoRA  : {wins}/{len(rows)}")
print(f"  Losses for LoRA: {losses}/{len(rows)}")
print(f"  Ties           : {ties}/{len(rows)}")
print(f"  Wilcoxon       : {p_str}")
print(f"  Cohen's d      : {cohens_d:.2f}")

print(f"\nVerdict: {verdict}")
print(f"\nSaved → {out_path}")
