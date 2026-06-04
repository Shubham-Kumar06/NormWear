"""
Per-subject AUC figure for the HCI paper:
  grouped bars showing AUC without LoRA (Side A) vs with LoRA (Side C),
  sorted by Side A AUC descending. Δ annotated above the LoRA bar.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT  = Path("/home/ug24/FoundationalModel/NormWear/data/results/lora_results")
OUT   = Path("/home/ug24/FoundationalModel/NormWear/hci_research_paper/figures/wesad_personalization.pdf")

A = json.load(open(ROOT / "wesad_baseline_summary_AUG_BEST.json"))   # frozen + LR
C = json.load(open(ROOT / "wesad_lora_paper_summary_AUG_BEST.json")) # LoRA

A = {r["subject_id"]: r["auc"] for r in A["per_subject"]}
C = {r["subject_id"]: r["auc"] for r in C["per_subject"]}

# Sort by no-LoRA AUC desc, so the figure visually tells the "weak baselines
# benefit most from personalization" story.
sids = sorted(set(A) & set(C), key=lambda s: -A[s])
a_vals = [A[s] for s in sids]
c_vals = [C[s] for s in sids]
deltas = [c - a for a, c in zip(a_vals, c_vals)]

# ACM single-column width ~ 3.5 in
fig, ax = plt.subplots(figsize=(3.5, 2.6))
x = np.arange(len(sids))
w = 0.4

bA = ax.bar(x - w/2, a_vals, w, label="Without LoRA",
            color="#9ECAE1", edgecolor="black", linewidth=0.4)
bC = ax.bar(x + w/2, c_vals, w, label="With LoRA",
            color="#08519C", edgecolor="black", linewidth=0.4)

# Δ annotation above each LoRA bar
for xi, ci, d in zip(x, c_vals, deltas):
    ax.text(xi + w/2, ci + 1.0, f"+{d:.1f}", ha="center", va="bottom",
            fontsize=5.5)

ax.set_xticks(x)
ax.set_xticklabels([f"S{s}" for s in sids], fontsize=7, rotation=0)
ax.set_ylabel("Test AUC-ROC (%)", fontsize=8)
ax.set_xlabel("Subject (sorted by Without-LoRA AUC)", fontsize=8)
ax.set_ylim(50, 105)
ax.tick_params(axis="y", labelsize=7)
ax.legend(fontsize=7, loc="lower left", frameon=False)

# Mean lines
ax.axhline(np.mean(a_vals), color="#9ECAE1", linestyle="--", linewidth=0.8, alpha=0.7)
ax.axhline(np.mean(c_vals), color="#08519C", linestyle="--", linewidth=0.8, alpha=0.7)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT, format="pdf", bbox_inches="tight")
print(f"Saved → {OUT}")
print(f"Means: without={np.mean(a_vals):.2f}  with={np.mean(c_vals):.2f}  Δ={np.mean(c_vals)-np.mean(a_vals):+.2f}")
