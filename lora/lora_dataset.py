"""
Dataset utilities for LoRA personalization on NormWear downstream tasks.

Each downstream PKL file has the same structure as the pretrain data:
    {
        'tss'   : np.ndarray [nvar, L]   float32  — raw time series
        'sensor': list[str]              — sensor names
        'label' : dict  (task-specific labels)
        'fn'    : str   — filename / subject ID
        'subject_id': str  (optional — may be encoded in 'fn')
    }

PersonalizedDownstreamDataset loads a downstream dataset, optionally filters
by subject, and returns (cwt_input, label) pairs ready for LoRA fine-tuning.
"""

import os
import json
import pickle
import re
from typing import Optional, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

# ─────────────────────────────────────────────────────────────────────────────
# CWT helpers (same as downstream_pipeline/model_apis.py but self-contained)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from scipy.signal import fftconvolve
except ImportError:
    from numpy.fft import fft, ifft  # fallback — not used

def _ricker(points: int, a: float) -> np.ndarray:
    """Ricker (Mexican hat) wavelet."""
    A  = 2.0 / (np.sqrt(3.0 * a) * (np.pi ** 0.25))
    t  = np.linspace(-points // 2, points // 2, points)
    s  = A * (1.0 - (t / a) ** 2) * np.exp(-0.5 * (t / a) ** 2)
    return s


def _cwt_ricker(signal_2d: np.ndarray, lf: float = 0.1, hf: float = 64) -> np.ndarray:
    """
    CWT using Ricker wavelets via FFT convolution.

    Args:
        signal_2d : [C, L]  (channels × time)
        lf, hf    : lowest / highest scale

    Returns:
        coefs : [C, n_scales, L]
    """
    scales  = np.arange(lf, hf + 1)
    n_scales = len(scales)
    C, L    = signal_2d.shape
    coefs   = np.zeros((C, n_scales, L), dtype=np.float32)

    for si, s in enumerate(scales):
        w_len = min(10 * int(s) + 1, L)
        w     = _ricker(w_len, s)
        for c in range(C):
            conv = fftconvolve(signal_2d[c], w, mode="same")
            coefs[c, si] = conv.astype(np.float32)

    return coefs


def cwt_transform(tss: np.ndarray) -> torch.Tensor:
    """
    Full on-the-fly CWT transform matching NormWear's pretrain pipeline.

    Args:
        tss : [nvar, L]  raw time series (L ≈ 388)

    Returns:
        x   : [nvar, 3, L-2, 65]  CWT scalogram (float32 tensor)
    """
    nvar, L = tss.shape

    # first and second derivatives
    d1 = tss[:, 1:]  - tss[:, :-1]     # [nvar, L-1]
    d2 = d1[:,  1:]  - d1[:,  :-1]     # [nvar, L-2]

    # trim all to same length L-2
    x0 = tss[:, 2:]     # [nvar, L-2]
    d1 = d1[:, 1:]      # [nvar, L-2]

    # stack: [nvar, 3, L-2]
    sig3 = np.stack([x0, d1, d2], axis=1)

    nvar_, C, Lc = sig3.shape
    # CWT per variable: [nvar, 3, 65, Lc]
    all_cwt = []
    for v in range(nvar_):
        cwt_v = _cwt_ricker(sig3[v], lf=0.1, hf=64)   # [3, 65, Lc]
        all_cwt.append(cwt_v)

    cwt_arr = np.stack(all_cwt, axis=0)                # [nvar, 3, 65, Lc]
    cwt_t   = torch.from_numpy(cwt_arr).permute(0, 1, 3, 2)  # [nvar, 3, Lc, 65]
    return cwt_t


# ─────────────────────────────────────────────────────────────────────────────
# Label helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_label_generic(d, is_regression=False):
    """Handles list-of-dicts, plain int, plain float, tensor, array."""
    import numpy as np
    l = d.get("label")
    if l is None:
        return 0.0 if is_regression else 0
    # list of dicts: [{"class": 0}] or [{"label": 0}]
    if isinstance(l, list):
        if len(l) == 0:
            return 0.0 if is_regression else 0
        item = l[0]
        if isinstance(item, dict):
            for key in ("class", "label", "value"):
                if key in item:
                    v = item[key]
                    return float(v) if is_regression else int(float(v))
            return float(list(item.values())[0]) if is_regression                    else int(float(list(item.values())[0]))
        return float(item) if is_regression else int(float(item))
    # numpy array or torch tensor
    if hasattr(l, "numpy"):
        l = l.numpy()
    if hasattr(l, "__len__"):
        l = np.array(l).flatten()
        return float(l[0]) if is_regression else int(float(l[0]))
    return float(l) if is_regression else int(float(l))


LABEL_PARSERS: Dict[str, callable] = {
    # classification
    "PPG_HTN":       lambda d: _parse_label_generic(d),
    "PPG_DM":        lambda d: _parse_label_generic(d),
    "PPG_CVA":       lambda d: _parse_label_generic(d),
    "PPG_CVD":       lambda d: _parse_label_generic(d),
    "ecg_heart_cat": lambda d: _parse_label_generic(d),
    "emg-tfc":       lambda d: _parse_label_generic(d),
    "gameemo":       lambda d: _parse_label_generic(d),
    "Epilepsy":      lambda d: _parse_label_generic(d),
    "drive_fatigue": lambda d: _parse_label_generic(d),
    "uci_har":       lambda d: _parse_label_generic(d),
    "wesad":         lambda d: _parse_label_generic(d),
    # regression
    "non_invasive_bp": lambda d: _parse_label_generic(d, is_regression=True),
    "ppg_hgb":         lambda d: _parse_label_generic(d, is_regression=True),
    "indian-fPCG":     lambda d: _parse_label_generic(d, is_regression=True),
}


def parse_label(ds_name: str, data_dict: dict):
    """Extract the label from a data_dict for a given downstream dataset."""
    parser = LABEL_PARSERS.get(ds_name)
    if parser is None:
        # fallback: try common keys
        for key in ("label", "valence", "arousal"):
            if key in data_dict:
                return data_dict[key]
        raise KeyError(f"No label parser for '{ds_name}' and no 'label' key found.")
    return parser(data_dict)


def extract_subject_id(data_dict: dict, fn: str) -> str:
    """
    Try to extract a subject ID from the data dict or filename.
    Falls back to the full filename string.
    """
    if "subject_id" in data_dict:
        return str(data_dict["subject_id"])
    if "subject" in data_dict:
        return str(data_dict["subject"])
    # filename format: "SUBJECT_conditionX_conditionY_offset" e.g. "10_0_0_1174"
    # subject ID is the first segment before the first underscore
    parts = fn.replace(".pkl", "").split("_")
    if parts:
        return parts[0]
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PersonalizedDownstreamDataset(Dataset):
    """
    Downstream dataset for LoRA personalization.

    Args:
        data_dir       : path to the downstream dataset folder
                         (e.g. NormWear/data/wearable_downstream/wesad/)
        ds_name        : dataset name string (used for label parsing)
        split_file     : path to train_test_split.json (optional).
                         If given, only files in `split` are loaded.
        split          : 'train' or 'test'
        subject_ids    : if not None, only load samples from these subject IDs
        max_L          : maximum time length (pad/trim to this)
        pad_nvar       : pad number of variables to this (must match model)
        transform      : optional callable applied to the CWT tensor
        task_type      : 'classification' or 'regression'
    """

    def __init__(
        self,
        data_dir:     str,
        ds_name:      str,
        split_file:   Optional[str] = None,
        split:        str           = "train",
        subject_ids:  Optional[List[str]] = None,
        max_L:        int  = 388,
        pad_nvar:     int  = 4,
        transform             = None,
        task_type:    str   = "classification",
    ):
        self.data_dir   = data_dir
        self.ds_name    = ds_name
        self.max_L      = max_L
        self.pad_nvar   = pad_nvar
        self.transform  = transform
        self.task_type  = task_type

        # ── collect file list ──────────────────────────────────────────
        # files may have .pkl extension or no extension
        all_raw = os.listdir(data_dir)
        all_files = sorted(
            f for f in all_raw
            if f.endswith(".pkl") or (os.path.isfile(os.path.join(data_dir, f)) and "." not in f)
        )

        # filter by split if a split file is provided
        if split_file and os.path.isfile(split_file):
            with open(split_file) as fp:
                splits = json.load(fp)
            allowed = set(splits.get(split, []))
            all_files = [f for f in all_files
                         if f in allowed or f.replace(".pkl", "") in allowed]

        # ── build sample index with subject IDs ───────────────────────
        self.samples: List[dict] = []   # [{path, label, subject_id}, ...]

        for fname in all_files:
            fpath = os.path.join(data_dir, fname)
            try:
                with open(fpath, "rb") as fp:
                    d = pickle.load(fp)
            except Exception:
                continue

            # extract subject ID directly from filename (most reliable)
            # filename format: "SUBJECT_condA_condB_offset" e.g. "10_0_0_1174"
            sid = fname.replace(".pkl", "").split("_")[0]

            # filter by subject
            if subject_ids is not None and sid not in subject_ids:
                continue

            try:
                label = parse_label(ds_name, d)
            except (KeyError, TypeError):
                continue

            self.samples.append({
                "path":       fpath,
                "label":      label,
                "subject_id": sid,
            })

        print(
            f"[Dataset] {ds_name}/{split}: {len(self.samples)} samples "
            f"from {len(self.get_subjects())} subjects"
            + (f" (filtered: {subject_ids})" if subject_ids else "")
        )

    # ── subject utilities ────────────────────────────────────────────

    def get_subjects(self) -> List[str]:
        """Return sorted list of unique subject IDs present in the dataset."""
        return sorted(set(s["subject_id"] for s in self.samples))

    def get_indices_for_subject(self, subject_id: str) -> List[int]:
        """Return sample indices belonging to *subject_id*."""
        return [i for i, s in enumerate(self.samples) if s["subject_id"] == subject_id]

    # ── PyTorch Dataset interface ─────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        meta = self.samples[idx]

        with open(meta["path"], "rb") as fp:
            d = pickle.load(fp)

        tss = d.get("tss", d.get("data", d.get("signal", None)))
        if tss is None:
            raise KeyError(f"No 'tss'/'data'/'signal' key in {meta['path']}")

        # ensure numpy float32
        if not isinstance(tss, np.ndarray):
            tss = np.array(tss, dtype=np.float32)
        tss = tss.astype(np.float32)

        # shape guard: must be 2-D [nvar, L]
        if tss.ndim == 1:
            tss = tss[np.newaxis, :]
        if tss.ndim != 2:
            tss = tss.reshape(-1, tss.shape[-1])

        nvar, L = tss.shape

        # compute CWT
        cwt = cwt_transform(tss)      # [nvar, 3, L_cwt, 65]
        L_cwt = cwt.shape[2]

        # ── pad / trim time dimension ─────────────────────────────────
        T = min(L_cwt, self.max_L)
        if L_cwt < self.max_L:
            pad = torch.zeros(nvar, 3, self.max_L - L_cwt, 65)
            cwt = torch.cat([cwt, pad], dim=2)
        else:
            cwt = cwt[:, :, :self.max_L, :]

        # ── pad / trim nvar dimension ─────────────────────────────────
        if nvar < self.pad_nvar:
            pad = torch.zeros(self.pad_nvar - nvar, 3, self.max_L, 65)
            cwt = torch.cat([cwt, pad], dim=0)
        else:
            cwt = cwt[:self.pad_nvar, :, :, :]

        # cwt final shape: [pad_nvar, 3, max_L, 65]

        if self.transform is not None:
            cwt = self.transform(cwt)

        # ── label ──────────────────────────────────────────────────────
        if self.task_type == "classification":
            label = torch.tensor(meta["label"], dtype=torch.long)
        else:
            label = torch.tensor(meta["label"], dtype=torch.float32)

        return {
            "input":      cwt,                    # [pad_nvar, 3, max_L, 65]
            "label":      label,
            "subject_id": meta["subject_id"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader builders
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    ds_train: PersonalizedDownstreamDataset,
    ds_test:  PersonalizedDownstreamDataset,
    batch_size: int  = 32,
    num_workers: int = 4,
) -> tuple:
    """Build standard train/test DataLoaders."""
    train_loader = DataLoader(
        ds_train,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )
    return train_loader, test_loader


def build_subject_loaders(
    ds_train:    PersonalizedDownstreamDataset,
    ds_test:     PersonalizedDownstreamDataset,
    subject_id:  str,
    batch_size:  int = 16,
    num_workers: int = 2,
) -> tuple:
    """
    Build DataLoaders filtered to a single subject.
    Useful for per-subject LoRA fine-tuning experiments.
    """
    train_idx = ds_train.get_indices_for_subject(subject_id)
    test_idx  = ds_test.get_indices_for_subject(subject_id)

    train_loader = DataLoader(
        ds_train,
        batch_size  = batch_size,
        sampler     = SubsetRandomSampler(train_idx),
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size  = batch_size,
        sampler     = SubsetRandomSampler(test_idx),
        num_workers = num_workers,
        pin_memory  = True,
    )
    print(
        f"[Subject loader] {subject_id}: "
        f"train={len(train_idx)}, test={len(test_idx)}"
    )
    return train_loader, test_loader
