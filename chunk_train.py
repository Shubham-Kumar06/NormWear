"""
Chunk CWT Training Script for NormWear
- Processes 50K files at a time
- Precomputes CWT → trains → deletes → next chunk
- Runs automatically for all 400 epochs
"""

import os
import sys
import pickle
import torch
import numpy as np
import subprocess
import shutil

sys.path.insert(0, "/home/ug24/FoundationalModel")
from NormWear.pretrain_pipeline.dataset import cwt_wrap

# ─── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_DIR    = "/home/ug24/FoundationalModel/NormWear/data/pretrain/wearable_pretrain"
CHUNK_DIR    = "/home/ug24/FoundationalModel/NormWear/data/pretrain/chunk_cwt"
OUTPUT_DIR   = "/home/ug24/FoundationalModel/NormWear/data/results"
LOG_FILE     = "/home/ug24/FoundationalModel/NormWear/chunk_train_log.txt"

CHUNK_SIZE   = 50000
TOTAL_EPOCHS = 400
BATCH_SIZE   = 128
NUM_WORKERS  = 4
SAVE_EVERY   = 20   # save checkpoint every N epochs

# Resume settings (fill these if restarting after a crash)
START_EPOCH  = 0    # which epoch to resume from
START_CHUNK  = 0    # which chunk to resume from within that epoch
RESUME_CKPT  = ""   # path to checkpoint .pth file, or "" for fresh start
# ──────────────────────────────────────────────────────────────────────────────

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')

def get_all_files():
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".pkl")])
    log(f"Total files found: {len(files)}")
    return files

def split_chunks(files):
    chunks = [files[i:i+CHUNK_SIZE] for i in range(0, len(files), CHUNK_SIZE)]
    log(f"Total chunks per epoch: {len(chunks)} (chunk size: {CHUNK_SIZE})")
    return chunks

def precompute_chunk(chunk_files):
    """Precompute CWT for one chunk and save to CHUNK_DIR"""
    os.makedirs(CHUNK_DIR, exist_ok=True)
    log(f"    Precomputing CWT for {len(chunk_files)} files...")

    for i, fn in enumerate(chunk_files):
        out_path = os.path.join(CHUNK_DIR, fn)
        if os.path.exists(out_path):
            continue  # already computed (resume case)

        with open(os.path.join(INPUT_DIR, fn), "rb") as f:
            d = pickle.load(f)

        tss_raw = d['tss']
        if isinstance(tss_raw, torch.Tensor):
            tss = tss_raw.float()
        else:
            tss = torch.from_numpy(tss_raw).float()

        cwt = cwt_wrap(tss)                        # [nvar, 3, L, 65]
        L_cwt = cwt.size(2)
        tss_trimmed = tss[:, :L_cwt]

        out = {
            'tss': tss_trimmed.numpy(),
            'cwt': cwt.permute(0, 2, 3, 1).numpy()  # [nvar, L, 65, 3]
        }

        with open(out_path, "wb") as f:
            pickle.dump(out, f)

        if i % 5000 == 0:
            log(f"      [{i}/{len(chunk_files)}] precomputed...")

    log(f" CWT precomputed for chunk!")

def delete_chunk():
    """Delete CHUNK_DIR to free disk space"""
    if os.path.exists(CHUNK_DIR):
        shutil.rmtree(CHUNK_DIR)
        log(f" Chunk deleted — disk space freed!")

def get_latest_checkpoint():
    """Find most recent checkpoint in OUTPUT_DIR"""
    if not os.path.exists(OUTPUT_DIR):
        return ""
    ckpts = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".pth")]
    if not ckpts:
        return ""
    ckpts.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)))
    return os.path.join(OUTPUT_DIR, ckpts[-1])

def train_one_pass(epoch, chunk_idx, total_chunks, resume_ckpt=""):
    """Run pretrain_main for 1 epoch on current chunk"""
    log(f"    Training on chunk {chunk_idx+1}/{total_chunks}...")

    # Only save checkpoint at end of full epoch (last chunk)
    is_last_chunk = (chunk_idx == total_chunks - 1)
    should_save = is_last_chunk and (epoch % SAVE_EVERY == 0 or epoch == TOTAL_EPOCHS - 1)
    save_every = 1 if should_save else 999  # 999 = never save mid-epoch

    cmd = [
        "python3", "-m", "NormWear.pretrain_main",
        "--data_path", os.path.dirname(CHUNK_DIR),
        "--dataset_name", "chunk_cwt",
        "--output_dir", OUTPUT_DIR,
        "--batch_size", str(BATCH_SIZE),
        "--epochs", "1",
        "--warmup_epochs", "0",
        "--num_workers", str(NUM_WORKERS),
        "--save_every_epoch", str(save_every),
        "--remark", f"chunk_e{epoch:03d}_c{chunk_idx:02d}",
    ]

    if resume_ckpt and os.path.exists(resume_ckpt):
        cmd += ["--resume", resume_ckpt]

    log(f"    CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="/home/ug24/FoundationalModel")
    return result.returncode

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log("=" * 60)
    log("NormWear Chunk CWT Training Started!")
    log(f"Config: {TOTAL_EPOCHS} epochs, {CHUNK_SIZE} files/chunk, batch={BATCH_SIZE}")
    log("=" * 60)

    all_files = get_all_files()
    chunks = split_chunks(all_files)
    total_chunks = len(chunks)

    resume_ckpt = RESUME_CKPT

    for epoch in range(START_EPOCH, TOTAL_EPOCHS):
        log(f"\n{'='*60}")
        log(f"EPOCH {epoch+1}/{TOTAL_EPOCHS}")
        log(f"{'='*60}")

        start_c = START_CHUNK if epoch == START_EPOCH else 0

        for chunk_idx in range(start_c, total_chunks):
            chunk_files = chunks[chunk_idx]
            log(f"\n  --- Chunk {chunk_idx+1}/{total_chunks} "
                f"({len(chunk_files)} files) ---")

            # Step 1: Precompute CWT
            precompute_chunk(chunk_files)

            # Step 2: Check disk space before training
            statvfs = os.statvfs(CHUNK_DIR)
            free_gb = statvfs.f_frsize * statvfs.f_bavail / 1e9
            log(f"    Disk free: {free_gb:.1f} GB")
            if free_gb < 5:
                log("❌ Less than 5GB free! Stopping to avoid disk full.")
                delete_chunk()
                sys.exit(1)

            # Step 3: Train on chunk
            ret = train_one_pass(epoch, chunk_idx, total_chunks, resume_ckpt)

            if ret != 0:
                log(f"❌ Training failed at epoch {epoch}, chunk {chunk_idx}")
                delete_chunk()
                sys.exit(1)

            # Step 4: Delete chunk to free space
            delete_chunk()

            # Step 5: Update checkpoint for next pass
            resume_ckpt = get_latest_checkpoint()
            log(f"    Latest checkpoint: {resume_ckpt}")

        log(f" Epoch {epoch+1}/{TOTAL_EPOCHS} complete!")

    log("\Full training complete! 400 epochs done!")