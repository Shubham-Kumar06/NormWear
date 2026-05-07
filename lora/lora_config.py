"""
lora/lora_config.py
===================
Centralized per-dataset configuration for NormWear LoRA personalization.

Place this file at:  NormWear/lora/lora_config.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset LoRA configuration
#
# max_L      : CWT output length = (signal_len - 2).  Set to signal_len for
#              simplicity; the dataset loader pads/trims automatically.
# min_samples: minimum files a subject must have to be included.
#              Reduce for small datasets (PPG_* have only ~657 total files).
# sid_field  : which pkl key to read for subject ID; None → use filename.
# ─────────────────────────────────────────────────────────────────────────────

DATASET_LORA_CONFIG = {
    # ── Classification datasets ───────────────────────────────────────────
    "wesad": {
        "num_classes":  3,
        "task_type":    "classification",
        "max_L":        390,        # signal len ≈ 389
        "min_samples":  10,         # plenty of data per subject
        "lora_rank":    8,
        "lora_alpha":   16,
        "lora_dropout": 0.15,
        "epochs":       10,
        "lr":           3e-4,
        "batch_size":   16,
        # subject ID: first segment of filename  "10_0_0_1174" → "10"
        "sid_split_idx": 0,
    },
    "amigos": {
        "num_classes":  2,           # binary arousal (>=5 high, <5 low)
        "task_type":    "classification",
        "max_L":        390,         # CWT len = 388-2 = 386, padded
        "min_samples":  20,          # smallest subject has 476 windows
        "lora_rank":    8,
        "lora_alpha":   16,
        "lora_dropout": 0.15,
        "epochs":       10,
        "lr":           3e-4,
        "batch_size":   32,
        # uid field is set explicitly in the pkl; sid_split_idx falls back
        # on filename "P01_t00_w000" -> "P01"
        "sid_split_idx": 0,
    },
    "ecg_heart_cat": {
        "num_classes":  2,
        "task_type":    "classification",
        "max_L":        186,        # signal len = 186
        "min_samples":  5,
        "lora_rank":    8,
        "lora_alpha":   16,
        "lora_dropout": 0.15,
        "epochs":       10,
        "lr":           3e-4,
        "batch_size":   32,
        "sid_split_idx": 0,
    },
    "uci_har": {
        "num_classes":  6,
        "task_type":    "classification",
        "max_L":        165,        # signal len = 165
        "min_samples":  5,
        "lora_rank":    8,
        "lora_alpha":   16,
        "lora_dropout": 0.15,
        "epochs":       10,
        "lr":           3e-4,
        "batch_size":   32,
        "sid_split_idx": 0,
    },
    "drive_fatigue": {
        "num_classes":  2,
        "task_type":    "classification",
        "max_L":        390,        # signal len = 388
        "min_samples":  5,
        "lora_rank":    8,
        "lora_alpha":   16,
        "lora_dropout": 0.15,
        "epochs":       10,
        "lr":           3e-4,
        "batch_size":   16,
        "sid_split_idx": 0,
    },
    "gameemo": {
        "num_classes":  4,
        "task_type":    "classification",
        "max_L":        390,        # signal len = 388
        "min_samples":  5,
        "lora_rank":    8,
        "lora_alpha":   16,
        "lora_dropout": 0.15,
        "epochs":       10,
        "lr":           3e-4,
        "batch_size":   32,
        "sid_split_idx": 0,
    },
    "Epilepsy": {
        "num_classes":  2,
        "task_type":    "classification",
        "max_L":        178,        # signal len = 178
        "min_samples":  5,
        "lora_rank":    8,
        "lora_alpha":   16,
        "lora_dropout": 0.15,
        "epochs":       10,
        "lr":           3e-4,
        "batch_size":   64,
        "sid_split_idx": 0,
    },
    # ── PPG datasets — only ~657 total files; lower thresholds ───────────
    "PPG_HTN": {
        "num_classes":  4,
        "task_type":    "classification",
        "max_L":        271,
        "min_samples":  3,
        "lora_rank":    8,
        "lora_alpha":   16,
        "lora_dropout": 0.15,
        "epochs":       10,
        "lr":           3e-4,
        "batch_size":   16,
        "sid_split_idx": 0,
    },
    "PPG_CVA": {
        "num_classes":  2,
        "task_type":    "classification",
        "max_L":        271,
        "min_samples":  3,
        "lora_rank":    8,
        "lora_alpha":   16,
        "lora_dropout": 0.15,
        "epochs":       10,
        "lr":           3e-4,
        "batch_size":   16,
        "sid_split_idx": 0,
    },
    "PPG_CVD": {
        "num_classes":  3,
        "task_type":    "classification",
        "max_L":        271,
        "min_samples":  3,
        "lora_rank":    8,
        "lora_alpha":   16,
        "lora_dropout": 0.15,
        "epochs":       10,
        "lr":           3e-4,
        "batch_size":   16,
        "sid_split_idx": 0,
    },
    "PPG_DM": {
        "num_classes":  2,
        "task_type":    "classification",
        "max_L":        271,
        "min_samples":  3,
        "lora_rank":    8,
        "lora_alpha":   16,
        "lora_dropout": 0.15,
        "epochs":       10,
        "lr":           3e-4,
        "batch_size":   16,
        "sid_split_idx": 0,
    },
    # ── tiny datasets — only 163 files; use global mode instead ──────────
    "emg-tfc": {
        "num_classes":  3,
        "task_type":    "classification",
        "max_L":        390,        # signal len = 1500, will be trimmed
        "min_samples":  3,
        "lora_rank":    8,
        "lora_alpha":   16,
        "lora_dropout": 0.15,
        "epochs":       10,
        "lr":           3e-4,
        "batch_size":   8,
        "sid_split_idx": 0,
    },
    # ── Regression datasets (optional; very small, results may be noisy) ─
    "non_invasive_bp": {
        "num_classes":  1,
        "task_type":    "regression",
        "max_L":        390,
        "min_samples":  3,
        "lora_rank":    8,
        "lora_alpha":   16,
        "lora_dropout": 0.15,
        "epochs":       10,
        "lr":           3e-4,
        "batch_size":   8,
        "sid_split_idx": 0,
    },
    "ppg_hgb": {
        "num_classes":  1,
        "task_type":    "regression",
        "max_L":        390,
        "min_samples":  2,
        "lora_rank":    8,
        "lora_alpha":   16,
        "lora_dropout": 0.15,
        "epochs":       10,
        "lr":           3e-4,
        "batch_size":   4,
        "sid_split_idx": 0,
    },
    "indian-fPCG": {
        "num_classes":  1,
        "task_type":    "regression",
        "max_L":        390,
        "min_samples":  2,
        "lora_rank":    8,
        "lora_alpha":   16,
        "lora_dropout": 0.15,
        "epochs":       10,
        "lr":           3e-4,
        "batch_size":   4,
        "sid_split_idx": 0,
    },
}


def get_config(ds_name: str) -> dict:
    """Return dataset config, falling back to sensible defaults."""
    defaults = {
        "num_classes":   2,
        "task_type":     "classification",
        "max_L":         390,
        "min_samples":   5,
        "lora_rank":     8,
        "lora_alpha":    16,
        "lora_dropout":  0.15,
        "epochs":        10,
        "lr":            3e-4,
        "batch_size":    16,
        "sid_split_idx": 0,
    }
    cfg = defaults.copy()
    cfg.update(DATASET_LORA_CONFIG.get(ds_name, {}))
    return cfg
