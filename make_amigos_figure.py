"""
Per-subject AUC figure for AMIGOS (HCI paper):
mirror of make_wesad_figure.py adapted for AMIGOS's string IDs.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/ug24/FoundationalModel/NormWear/data/results/lora_results")
OUT  = Path("/home/ug24/FoundationalModel/NormWear/hci_research_paper/figures/amigos_personalization.pdf")

A = json.load(open(ROOT / "amigos_baseline_summary_AUG_BEST.json"))
C = json.load(open(ROOT / "amigos_lora_paper_summary_AUG_BEST.json"))

A = {r["subject_id"]: r["auc"] for r in A["per_subject"]}
C = {r["subject_id"]: r["auc"] for r in C["per_subject"]}

sids = sorted(set(A) & set(C), key=lambda s: -A[s])  # by w/o-LoRA AUC desc
a_vals = [A[s] for s in sids]
c_vals = [C[s] for s in sids]
deltas = [c - a for a, c in zip(a_vals, c_vals)]

fig, ax = plt.subplots(figsize=(3.5, 2.6))
x = np.arange(len(sids))
w = 0.4

ax.bar(x - w/2, a_vals, w, label="Without LoRA",
       color="#9ECAE1", edgecolor="black", linewidth=0.4)
ax.bar(x + w/2, c_vals, w, label="With LoRA",
       color="#08519C", edgecolor="black", linewidth=0.4)

# Δ annotations above each LoRA bar (smaller text since labels are P-coded)
for xi, ci, d in zip(x, c_vals, deltas):
    ax.text(xi + w/2, ci + 1.0, f"{d:+.1f}", ha="center", va="bottom", fontsize=5.0)

ax.set_xticks(x)
ax.set_xticklabels(sids, fontsize=6, rotation=45, ha="right")
ax.set_ylabel("Test AUC-ROC (%)", fontsize=8)
ax.set_xlabel("Subject (sorted by Without-LoRA AUC)", fontsize=8)
ax.set_ylim(40, 105)
ax.tick_params(axis="y", labelsize=7)
ax.legend(fontsize=7, loc="lower left", frameon=False)

ax.axhline(np.mean(a_vals), color="#9ECAE1", linestyle="--", linewidth=0.8, alpha=0.7)
ax.axhline(np.mean(c_vals), color="#08519C", linestyle="--", linewidth=0.8, alpha=0.7)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT, format="pdf", bbox_inches="tight")
print(f"Saved → {OUT}")
print(f"Means: without={np.mean(a_vals):.2f}  with={np.mean(c_vals):.2f}  "
      f"Δ={np.mean(c_vals)-np.mean(a_vals):+.2f}")
