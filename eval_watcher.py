"""
eval_watcher.py - Autonomously evaluate normwear_aug checkpoints on downstream as
they are produced by training, track the mean-AUC curve, and manage disk.

For each new checkpoint:
  1. run downstream_main (group 0, all 14 datasets) with remark aug_e<N>
  2. parse the results pkl, compute mean score, append to aug_eval_summary.txt
  3. delete that checkpoint's embedding folders (transient)
  4. keep only the best-so-far checkpoint and the latest one; delete the rest

Run from /home/ug24/FoundationalModel.  Stops once training has exited AND every
produced checkpoint has been evaluated.
"""
import os
import re
import glob
import time
import pickle
import subprocess

import numpy as np

ROOT = "/home/ug24/FoundationalModel"
CKPT_DIR = os.path.join(ROOT, "NormWear/data/results")
RESULTS_DIR = os.path.join(ROOT, "NormWear/data/results/downstream_results")
DOWNSTREAM_DIR = os.path.join(ROOT, "NormWear/data/wearable_downstream")
SUMMARY = os.path.join(ROOT, "NormWear/data/results/aug_eval_summary.txt")
CKPT_GLOB = os.path.join(CKPT_DIR, "normwear_aug_checkpoint-*.pth")
POLL_SECONDS = 180


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(SUMMARY, "a") as f:
        f.write(line + "\n")


def epoch_of(path):
    m = re.search(r"normwear_aug_checkpoint-(\d+)\.pth$", path)
    return int(m.group(1)) if m else None


def training_alive():
    r = subprocess.run(["pgrep", "-f", "pretrain_main"], capture_output=True)
    return r.returncode == 0


def run_eval(ckpt, epoch):
    remark = f"aug_e{epoch}"
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128")
    log(f"Evaluating epoch {epoch}: {os.path.basename(ckpt)}")
    with open(os.path.join(ROOT, f"NormWear/eval_e{epoch}.log"), "w") as lf:
        subprocess.run(
            ["python3", "-m", "NormWear.downstream_main",
             "--model_name", "normwear",
             "--model_weight_dir", ckpt,
             "--data_path", "NormWear/data",
             "--remark", remark,
             "--group", "0", "--prepare_embed", "1"],
            cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT,
        )
    return remark


def mean_score(remark):
    p = os.path.join(RESULTS_DIR, f"{remark}_results_all.pkl")
    if not os.path.isfile(p):
        return None, {}
    with open(p, "rb") as f:
        r = pickle.load(f)
    scores, per_task = [], {}
    for ds, tasks in r.items():
        for tname, runs in tasks.items():
            s = np.array(runs).mean(axis=0)[0] * 100
            per_task[tname] = round(float(s), 2)
            scores.append(s)
    return (float(np.mean(scores)) if scores else None), per_task


def cleanup_embeddings(remark):
    n = 0
    for d in glob.glob(os.path.join(DOWNSTREAM_DIR, f"*/{remark}_wav_embed")):
        subprocess.run(["rm", "-rf", d])
        n += 1
    log(f"  cleaned {n} embedding folders for {remark}")


def prune_checkpoints(best_epoch, latest_epoch, evaluated):
    # Only prune checkpoints that were ALREADY evaluated and are neither the
    # best-so-far nor the latest (latest is kept for a possible resume).
    for ckpt in glob.glob(CKPT_GLOB):
        e = epoch_of(ckpt)
        if e in evaluated and e != best_epoch and e != latest_epoch:
            os.remove(ckpt)
            log(f"  pruned checkpoint epoch {e} (keep best={best_epoch}, latest={latest_epoch})")


def main():
    log("=== eval_watcher started ===")
    evaluated = {}        # epoch -> mean score
    best_epoch, best_score = None, -1.0

    while True:
        ckpts = sorted(glob.glob(CKPT_GLOB), key=lambda p: epoch_of(p))
        produced = [epoch_of(c) for c in ckpts]
        latest_epoch = max(produced) if produced else None

        for ckpt in ckpts:
            e = epoch_of(ckpt)
            if e in evaluated or not os.path.isfile(ckpt):
                continue
            time.sleep(5)  # ensure checkpoint write finished
            remark = run_eval(ckpt, e)
            m, per_task = mean_score(remark)
            evaluated[e] = m
            cleanup_embeddings(remark)
            if m is not None:
                log(f"  epoch {e}: MEAN={m:.2f} | {per_task}")
                if m > best_score:
                    best_score, best_epoch = m, e
                    log(f"  >>> new best: epoch {e} = {m:.2f}")
            # disk management: keep best + latest only (among evaluated ones)
            prune_checkpoints(best_epoch, latest_epoch, evaluated)

        # termination: training done and all produced checkpoints evaluated
        if not training_alive():
            remaining = [e for e in produced if e not in evaluated]
            if not remaining:
                log(f"=== training finished, all checkpoints evaluated. "
                    f"BEST: epoch {best_epoch} = {best_score:.2f} ===")
                break
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
