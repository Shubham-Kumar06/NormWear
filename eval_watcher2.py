"""
eval_watcher2.py - Watch the corrected sub-epoch run (remark normwear_aug2) and
evaluate each step-checkpoint on the FAST subset (downstream group 2: wesad,
drive_fatigue, gameemo) to map the downstream peak quickly.

For each new normwear_aug2_checkpoint-stepNNNNNN.pth:
  1. run downstream_main --group 2 --remark aug2_s<step>
  2. parse results, compute subset-mean, append to aug2_eval_summary.txt
  3. delete that remark's embedding folders
  4. keep only best-so-far + latest checkpoints (disk control)

Run from /home/ug24/FoundationalModel. Stops when training has exited and all
produced step-checkpoints are evaluated.
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
SUMMARY = os.path.join(ROOT, "NormWear/data/results/aug2_eval_summary.txt")
CKPT_GLOB = os.path.join(CKPT_DIR, "normwear_aug2_checkpoint-step*.pth")
POLL_SECONDS = 180


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(SUMMARY, "a") as f:
        f.write(line + "\n")


def step_of(path):
    m = re.search(r"normwear_aug2_checkpoint-step(\d+)\.pth$", path)
    return int(m.group(1)) if m else None


def training_alive():
    return subprocess.run(["pgrep", "-f", "NormWear.pretrain_main"],
                          capture_output=True).returncode == 0


def run_eval(ckpt, step):
    remark = f"aug2_s{step}"
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128")
    log(f"Evaluating step {step}: {os.path.basename(ckpt)}")
    with open(os.path.join(ROOT, f"NormWear/eval_aug2_s{step}.log"), "w") as lf:
        subprocess.run(
            ["python3", "-m", "NormWear.downstream_main",
             "--model_name", "normwear", "--model_weight_dir", ckpt,
             "--data_path", "NormWear/data", "--remark", remark,
             "--group", "2", "--prepare_embed", "1"],
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
    for d in glob.glob(os.path.join(DOWNSTREAM_DIR, f"*/{remark}_wav_embed")):
        subprocess.run(["rm", "-rf", d])


def prune_checkpoints(best_step, latest_step, evaluated):
    for ckpt in glob.glob(CKPT_GLOB):
        s = step_of(ckpt)
        if s in evaluated and s != best_step and s != latest_step:
            os.remove(ckpt)
            log(f"  pruned step {s} (keep best={best_step}, latest={latest_step})")


def main():
    log("=== eval_watcher2 started (sub-epoch, group-2 subset) ===")
    evaluated = {}
    best_step, best_score = None, -1.0
    while True:
        ckpts = sorted(glob.glob(CKPT_GLOB), key=lambda p: step_of(p))
        produced = [step_of(c) for c in ckpts]
        latest_step = max(produced) if produced else None
        for ckpt in ckpts:
            s = step_of(ckpt)
            if s in evaluated or not os.path.isfile(ckpt):
                continue
            time.sleep(5)
            remark = run_eval(ckpt, s)
            m, per_task = mean_score(remark)
            evaluated[s] = m
            cleanup_embeddings(remark)
            if m is not None:
                log(f"  step {s}: SUBSET_MEAN={m:.2f} | {per_task}")
                if m > best_score:
                    best_score, best_step = m, s
                    log(f"  >>> new best: step {s} = {m:.2f}")
            prune_checkpoints(best_step, latest_step, evaluated)
        if not training_alive():
            remaining = [s for s in produced if s not in evaluated]
            if not remaining:
                log(f"=== training finished; best subset step {best_step} = {best_score:.2f} ===")
                break
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
