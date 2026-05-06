"""
collect_lora_results.py
=======================
Collect all per-subject LoRA results and print a comparison table.
Run from /home/ug24/FoundationalModel/ after all datasets finish:

    python3 collect_lora_results.py
"""

import json
import glob
import os
import numpy as np

RESULTS_DIR = "NormWear/data/results/lora_results"

PAPER_AUC = {
    "wesad":         76.06,
    "ecg_heart_cat": 99.14,
    "uci_har":       98.95,
    "drive_fatigue": 74.29,
    "gameemo":       54.94,
    "Epilepsy":      92.74,
    "PPG_HTN":       62.34,
    "PPG_CVA":       70.63,
    "PPG_CVD":       51.77,
    "PPG_DM":        55.89,
    "emg-tfc":       99.22,
}

print(f"\n{'Dataset':20s} | {'Paper':>8s} | {'LoRA AUC':>9s} | {'Delta':>8s} | {'Std':>6s} | {'N subj':>6s}")
print("-" * 75)

summary_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_per_subject_summary.json")))
if not summary_files:
    print("No summary files found yet. Wait for training to finish.")
else:
    for fpath in summary_files:
        data = json.load(open(fpath))
        ds   = data.get("ds_name", os.path.basename(fpath).replace("_per_subject_summary.json", ""))
        per  = data.get("per_subject", [])
        aucs = [r["auc"] for r in per if "auc" in r and not (r["auc"] != r["auc"])]  # filter NaN
        if not aucs:
            print(f"  {ds:20s} — no valid AUC results yet")
            continue
        mean_auc = np.mean(aucs)
        std_auc  = np.std(aucs)
        paper    = PAPER_AUC.get(ds, 0.0)
        delta    = mean_auc - paper
        n_subj   = len(aucs)
        print(f"  {ds:20s} | {paper:8.2f} | {mean_auc:9.2f} | {delta:+8.2f} | {std_auc:6.2f} | {n_subj:6d}")

print()
