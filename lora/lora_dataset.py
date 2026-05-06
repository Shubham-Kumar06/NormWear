"""
lora/lora_dataset.py  (UPDATED — handles all downstream datasets)
==================================================================
Key fixes vs. original:
  1. extract_subject_id()  — tries uid field, then filename prefix; also
     handles filenames that start with letters (e.g. "A01_...").
  2. _parse_label_generic() — handles every label format found in the
     downstream datasets (list-of-dicts, plain int, numpy, string class).
  3. cwt_transform()        — gracefully handles 1-D and >2-D inputs.
  4. __getitem__()          — tries "tss", "data", "signal" keys in order.
  5. diagnose_subjects()    — new helper: print how many files each subject
     has before you run training (call once to debug a new dataset).

Place this file at:  NormWear/lora/lora_dataset.py
"""

import os
import json
import pickle
from typing import Optional, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

# ─────────────────────────────────────────────────────────────────────────────
# CWT helpers
# ─────────────────────────────────────────────────────────────────────────────

try:
    from scipy.signal import fftconvolve
except ImportError:
    raise ImportError("scipy is required: pip install scipy")


def _ricker(points: int, a: float) -> np.ndarray:
    A = 2.0 / (np.sqrt(3.0 * a) * (np.pi ** 0.25))
    t = np.arange(points) - (points - 1.0) / 2.0
    return A * (1.0 - (t / a) ** 2) * np.exp(-0.5 * (t / a) ** 2)


def _cwt_ricker(signal_2d: np.ndarray, lf: float = 0.1, hf: float = 64) -> np.ndarray:
    scales = np.arange(lf, hf + 1)
    C, L = signal_2d.shape
    coefs = np.zeros((C, len(scales), L), dtype=np.float32)
    for si, s in enumerate(scales):
        w = _ricker(min(10 * int(s) + 1, L), s)
        for c in range(C):
            coefs[c, si] = fftconvolve(signal_2d[c], w, mode="same").astype(np.float32)
    return coefs


def cwt_transform(tss: np.ndarray) -> torch.Tensor:
    """
    [nvar, L]  →  [nvar, 3, L-2, 65]  (float32 tensor)
    Handles any L ≥ 3.
    """
    # ensure 2-D
    if tss.ndim == 1:
        tss = tss[np.newaxis, :]
    if tss.ndim != 2:
        tss = tss.reshape(-1, tss.shape[-1])

    nvar, L = tss.shape
    if L < 3:
        # pad to at least 3
        pad = np.zeros((nvar, 3 - L), dtype=np.float32)
        tss = np.concatenate([tss, pad], axis=1)
        L = 3

    d1 = tss[:, 1:] - tss[:, :-1]   # [nvar, L-1]
    d2 = d1[:, 1:] - d1[:, :-1]     # [nvar, L-2]
    x0 = tss[:, 2:]                  # [nvar, L-2]
    d1t = d1[:, 1:]                  # [nvar, L-2]

    sig3 = np.stack([x0, d1t, d2], axis=1)  # [nvar, 3, L-2]
    Lc = sig3.shape[2]

    all_cwt = []
    for v in range(nvar):
        all_cwt.append(_cwt_ricker(sig3[v], lf=0.1, hf=64))  # [3, 65, Lc]

    cwt_arr = np.stack(all_cwt, axis=0)  # [nvar, 3, 65, Lc]
    return torch.from_numpy(cwt_arr).permute(0, 1, 3, 2)  # [nvar, 3, Lc, 65]


# ─────────────────────────────────────────────────────────────────────────────
# Label helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_label_generic(d: dict, is_regression: bool = False):
    """
    Handles every label format seen in the downstream datasets:
      • [{"class": 0}]          list of dicts  (standard format)
      • {"class": 0}            plain dict
      • 0  /  0.5               plain int / float
      • np.array(0)             numpy scalar
      • torch.tensor(0)         torch scalar
      • "0"                     string digit
    Returns int for classification, float for regression.
    Falls back to 0 with a warning if nothing matches.
    """
    raw = d.get("label", d.get("labels", None))
    if raw is None:
        return 0.0 if is_regression else 0

    # ── list of dicts: [{"class": 0}] ────────────────────────────────────
    if isinstance(raw, (list, tuple)):
        if len(raw) == 0:
            return 0.0 if is_regression else 0
        item = raw[0]
        if isinstance(item, dict):
            # try known keys in priority order
            for key in ("class", "label", "reg", "value"):
                if key in item:
                    v = item[key]
                    if isinstance(v, (np.ndarray, torch.Tensor)):
                        v = float(v.flat[0]) if hasattr(v, 'flat') else float(v)
                    return float(v) if is_regression else int(float(v))
            # fallback: first value
            v = list(item.values())[0]
            return float(v) if is_regression else int(float(v))
        # list of scalars
        return float(item) if is_regression else int(float(item))

    # ── plain dict: {"class": 0} ─────────────────────────────────────────
    if isinstance(raw, dict):
        for key in ("class", "label", "reg", "value"):
            if key in raw:
                v = raw[key]
                if isinstance(v, (np.ndarray, torch.Tensor)):
                    v = float(v.flat[0]) if hasattr(v, 'flat') else float(v)
                return float(v) if is_regression else int(float(v))
        v = list(raw.values())[0]
        return float(v) if is_regression else int(float(v))

    # ── numpy array / torch tensor ────────────────────────────────────────
    if isinstance(raw, np.ndarray):
        return float(raw.flat[0]) if is_regression else int(float(raw.flat[0]))
    if isinstance(raw, torch.Tensor):
        return float(raw.item()) if is_regression else int(raw.item())

    # ── plain scalar / string ─────────────────────────────────────────────
    try:
        return float(raw) if is_regression else int(float(raw))
    except (TypeError, ValueError):
        return 0.0 if is_regression else 0


# dataset → (label_key, is_regression)
_LABEL_META: Dict[str, bool] = {
    "PPG_HTN": False, "PPG_DM": False, "PPG_CVA": False, "PPG_CVD": False,
    "ecg_heart_cat": False, "emg-tfc": False, "gameemo": False,
    "Epilepsy": False, "drive_fatigue": False, "uci_har": False,
    "wesad": False,
    "non_invasive_bp": True, "ppg_hgb": True, "indian-fPCG": True,
}


def parse_label(ds_name: str, data_dict: dict):
    is_reg = _LABEL_META.get(ds_name, False)
    return _parse_label_generic(data_dict, is_regression=is_reg)


# ─────────────────────────────────────────────────────────────────────────────
# Subject-ID extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_subject_id(data_dict: dict, fname: str, split_idx: int = 0) -> str:
    """
    Strategy (in order):
      1. Non-empty 'uid' field in the pkl.
      2. Non-empty 'subject_id' / 'subject' field.
      3. First (split_idx) segment of the filename split by '_'.
         Works for:  "10_0_0_1174"  → "10"
                     "A01_rest_001" → "A01"
                     "subject3_..."→ "subject3"
      4. Full filename as last resort (each file = own "subject").
    """
    for key in ("uid", "subject_id", "subject"):
        v = data_dict.get(key, "")
        if isinstance(v, (int, float)):
            v = str(int(v))
        if v and str(v).strip():
            return str(v).strip()

    base = fname.replace(".pkl", "")
    parts = base.split("_")
    if len(parts) > split_idx:
        return parts[split_idx]
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic helper — call once to understand a new dataset's structure
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_subjects(data_dir: str, ds_name: str, split_idx: int = 0, n_show: int = 5):
    """
    Print subject distribution for a dataset without loading CWT.
    Use this to verify subject IDs are extracted correctly before training.

    Example:
        diagnose_subjects("NormWear/data/wearable_downstream/ecg_heart_cat/sample_for_downstream",
                          "ecg_heart_cat")
    """
    from collections import defaultdict, Counter

    all_files = sorted(
        f for f in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, f)) and not f.endswith(".json")
    )
    print(f"\n[Diagnose] {ds_name}: {len(all_files)} total files")
    print(f"  First {min(n_show, len(all_files))} filenames:")
    for f in all_files[:n_show]:
        print(f"    {f}")

    by_subject = defaultdict(list)
    label_counter: Counter = Counter()
    for fname in all_files[:200]:  # sample first 200 for speed
        fpath = os.path.join(data_dir, fname)
        try:
            d = pickle.load(open(fpath, "rb"))
        except Exception:
            continue
        sid = extract_subject_id(d, fname, split_idx)
        by_subject[sid].append(fname)
        try:
            lbl = parse_label(ds_name, d)
            label_counter[lbl] += 1
        except Exception:
            pass

    print(f"\n  Subjects found (from first 200 files): {len(by_subject)}")
    for sid, files in sorted(by_subject.items())[:10]:
        print(f"    subject={sid:>8}  files={len(files)}")
    print(f"  Label distribution (first 200): {dict(label_counter)}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PersonalizedDownstreamDataset(Dataset):
    """
    Downstream dataset for LoRA personalization.

    Args:
        data_dir    : path to  .../wearable_downstream/{ds_name}/sample_for_downstream/
        ds_name     : dataset name string (used for label parsing)
        split_file  : optional path to train_test_split.json
        split       : 'train' or 'test'
        subject_ids : if not None, only load these subject IDs
        max_L       : time dimension after CWT (pad / trim)
        pad_nvar    : pad number of variables to this
        task_type   : 'classification' or 'regression'
        sid_split_idx : which underscore-segment of filename = subject ID
    """

    def __init__(
        self,
        data_dir:       str,
        ds_name:        str,
        split_file:     Optional[str] = None,
        split:          str           = "train",
        subject_ids:    Optional[List[str]] = None,
        max_L:          int  = 390,
        pad_nvar:       int  = 4,
        transform             = None,
        task_type:      str   = "classification",
        sid_split_idx:  int   = 0,
    ):
        self.data_dir      = data_dir
        self.ds_name       = ds_name
        self.max_L         = max_L
        self.pad_nvar      = pad_nvar
        self.transform     = transform
        self.task_type     = task_type
        self.sid_split_idx = sid_split_idx

        # ── collect file list ──────────────────────────────────────────────
        all_raw = os.listdir(data_dir)
        all_files = sorted(
            f for f in all_raw
            if os.path.isfile(os.path.join(data_dir, f)) and not f.endswith(".json")
        )

        # filter by split
        if split_file and os.path.isfile(split_file):
            with open(split_file) as fp:
                splits = json.load(fp)
            allowed = set(splits.get(split, []))
            all_files = [
                f for f in all_files
                if f in allowed or f.replace(".pkl", "") in allowed
                or os.path.splitext(f)[0] in allowed
            ]

        # ── build sample list ──────────────────────────────────────────────
        self.samples: List[dict] = []

        for fname in all_files:
            fpath = os.path.join(data_dir, fname)
            try:
                d = pickle.load(open(fpath, "rb"))
            except Exception:
                continue

            sid = extract_subject_id(d, fname, sid_split_idx)

            if subject_ids is not None and sid not in subject_ids:
                continue

            try:
                label = parse_label(ds_name, d)
            except Exception:
                continue

            self.samples.append({"path": fpath, "label": label, "subject_id": sid})

        n_subjects = len(set(s["subject_id"] for s in self.samples))
        print(
            f"[Dataset] {ds_name}/{split}: {len(self.samples)} samples "
            f"from {n_subjects} subjects"
            + (f" (filter: {subject_ids})" if subject_ids else "")
        )

    # ── subject utilities ──────────────────────────────────────────────────

    def get_subjects(self) -> List[str]:
        return sorted(set(s["subject_id"] for s in self.samples))

    def get_indices_for_subject(self, subject_id: str) -> List[int]:
        return [i for i, s in enumerate(self.samples) if s["subject_id"] == subject_id]

    # ── PyTorch interface ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        meta = self.samples[idx]

        d = pickle.load(open(meta["path"], "rb"))

        # ── load raw time series ───────────────────────────────────────────
        tss = None
        for key in ("tss", "data", "signal", "samples"):
            v = d.get(key)
            if v is not None:
                tss = v
                break
        if tss is None:
            raise KeyError(f"No data key (tss/data/signal) in {meta['path']}")

        if isinstance(tss, torch.Tensor):
            tss = tss.numpy()
        tss = np.array(tss, dtype=np.float32)

        # ensure 2-D [nvar, L]
        if tss.ndim == 1:
            tss = tss[np.newaxis, :]
        elif tss.ndim > 2:
            tss = tss.reshape(-1, tss.shape[-1])

        # replace NaN / Inf
        tss = np.nan_to_num(tss, nan=0.0, posinf=0.0, neginf=0.0)

        # ── CWT ───────────────────────────────────────────────────────────
        cwt = cwt_transform(tss)      # [nvar, 3, L_cwt, 65]
        nvar, _, L_cwt, _ = cwt.shape

        # ── pad / trim time ────────────────────────────────────────────────
        if L_cwt < self.max_L:
            pad = torch.zeros(nvar, 3, self.max_L - L_cwt, 65)
            cwt = torch.cat([cwt, pad], dim=2)
        else:
            cwt = cwt[:, :, :self.max_L, :]

        # ── pad / trim nvar ────────────────────────────────────────────────
        if nvar < self.pad_nvar:
            pad = torch.zeros(self.pad_nvar - nvar, 3, self.max_L, 65)
            cwt = torch.cat([cwt, pad], dim=0)
        else:
            cwt = cwt[:self.pad_nvar, :, :, :]

        if self.transform is not None:
            cwt = self.transform(cwt)

        # ── label ──────────────────────────────────────────────────────────
        if self.task_type == "classification":
            label = torch.tensor(meta["label"], dtype=torch.long)
        else:
            label = torch.tensor(meta["label"], dtype=torch.float32)

        return {"input": cwt, "label": label, "subject_id": meta["subject_id"]}


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader builders
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    ds_train: PersonalizedDownstreamDataset,
    ds_test:  PersonalizedDownstreamDataset,
    batch_size:  int = 32,
    num_workers: int = 4,
) -> tuple:
    train_loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    test_loader = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


def build_subject_loaders(
    ds_train: PersonalizedDownstreamDataset,
    ds_test:  PersonalizedDownstreamDataset,
    subject_id: str,
    batch_size:  int = 16,
    num_workers: int = 2,
) -> tuple:
    train_idx = ds_train.get_indices_for_subject(subject_id)
    test_idx  = ds_test.get_indices_for_subject(subject_id)
    train_loader = DataLoader(
        ds_train, batch_size=batch_size,
        sampler=SubsetRandomSampler(train_idx),
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    test_loader = DataLoader(
        ds_test, batch_size=batch_size,
        sampler=SubsetRandomSampler(test_idx),
        num_workers=num_workers, pin_memory=True,
    )
    print(f"[Subject loader] {subject_id}: train={len(train_idx)}, test={len(test_idx)}")
    return train_loader, test_loader
