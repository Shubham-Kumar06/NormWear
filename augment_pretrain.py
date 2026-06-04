"""
augment_pretrain.py - NormWear paper Algorithm 1 (Time Series Mixup Augmentation).

Generates AUG_PER_SAMPLE augmented samples per original via chunk-swap between two
series that have the SAME channel count (nvar). Operates on raw `tss` only; CWT is
computed on-the-fly during training (caching CWT would need 144GB+).

Output: data/pretrain/wearable_pretrain_aug/aug_<baseIdx>_<j>.pkl  (~2.16M files, ~5GB)
"""
import os
import pickle
import random
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

INPUT_DIR = "/home/ug24/FoundationalModel/NormWear/data/pretrain/wearable_pretrain"
OUTPUT_DIR = "/home/ug24/FoundationalModel/NormWear/data/pretrain/wearable_pretrain_aug"
AUG_PER_SAMPLE = 9          # 9 augmented per original -> ~10x total dataset
NUM_WORKERS = 30
SEED = 42

# Globals populated in __main__ before the Pool forks (inherited copy-on-write on Linux).
TSS = []                      # list[np.ndarray float16], shape [nvar, L]
NVAR_GROUPS = defaultdict(list)  # nvar -> list[int] indices into TSS


def load_all():
    files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".pkl"))
    tss_list = []
    nvar_groups = defaultdict(list)
    for idx, fn in enumerate(files):
        with open(os.path.join(INPUT_DIR, fn), "rb") as f:
            d = pickle.load(f)
        tss = d["tss"]
        arr = tss if isinstance(tss, np.ndarray) else tss.numpy()
        arr = np.ascontiguousarray(arr.astype(np.float16))
        tss_list.append(arr)
        nvar_groups[arr.shape[0]].append(idx)
        if idx % 50000 == 0:
            print(f"  loaded {idx}/{len(files)}", flush=True)
    return tss_list, nvar_groups


def make_aug(base_idx):
    rng = random.Random(SEED + base_idx)
    x1 = TSS[base_idx]
    nvar, L1 = x1.shape
    group = NVAR_GROUPS[nvar]
    written = 0
    for j in range(AUG_PER_SAMPLE):
        out_path = os.path.join(OUTPUT_DIR, f"aug_{base_idx}_{j}.pkl")
        if os.path.exists(out_path):       # resume-safe
            written += 1
            continue
        p = rng.choice(group)
        x2 = TSS[p]
        L2 = x2.shape[1]
        Lc = min(L1, L2)
        lam = rng.randint(1, Lc)            # chunk size  ~ U(1, l)
        s1 = rng.randint(0, L1 - lam)       # start in x1
        s2 = rng.randint(0, L2 - lam)       # start in x2
        aug = x1.copy()
        aug[:, s1:s1 + lam] = x2[:, s2:s2 + lam]
        with open(out_path, "wb") as f:
            pickle.dump({"tss": aug, "fn": ("aug", f"aug_{base_idx}_{j}")}, f)
        written += 1
    return written


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading originals from {INPUT_DIR} ...", flush=True)
    TSS, NVAR_GROUPS = load_all()
    n = len(TSS)
    print(f"Loaded {n} originals. nvar groups: "
          f"{ {k: len(v) for k, v in NVAR_GROUPS.items()} }", flush=True)
    print(f"Generating {AUG_PER_SAMPLE}x = ~{n * AUG_PER_SAMPLE} augmented samples "
          f"with {NUM_WORKERS} workers ...", flush=True)

    total = 0
    with Pool(NUM_WORKERS) as pool:
        for i, w in enumerate(pool.imap_unordered(make_aug, range(n), chunksize=200)):
            total += w
            if i % 20000 == 0:
                print(f"  base {i}/{n} done, files written so far ~{total}", flush=True)

    n_files = sum(1 for f in os.listdir(OUTPUT_DIR) if f.endswith(".pkl"))
    print(f"Done. Augmented files on disk: {n_files}", flush=True)
