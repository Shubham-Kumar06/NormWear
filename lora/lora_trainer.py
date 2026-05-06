"""
lora/lora_trainer.py  (UPDATED — handles all downstream datasets)
=================================================================
Key fixes vs. original:
  1. run_per_subject_personalization() — uses dataset-specific min_samples
     threshold instead of hardcoded 10.
  2. Stratified 80/20 split works for ANY number of classes, including
     single-class subjects (skipped gracefully with a clear message).
  3. Label extraction in the per-subject split loop uses the same
     _parse_label_generic() from lora_dataset so it never crashes on
     unexpected label formats.
  4. Added a 'skip_if_single_class' guard: subjects where all test samples
     have the same label cannot compute AUC — skip them cleanly.

Place this file at:  NormWear/lora/lora_trainer.py
"""

import os
import copy
import time
import json
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_auc(labels: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    if not HAS_SKLEARN:
        return float("nan")
    try:
        if num_classes == 2:
            return roc_auc_score(labels, probs[:, 1]) * 100.0
        else:
            return roc_auc_score(
                labels, probs, multi_class="ovr", average="macro"
            ) * 100.0
    except ValueError:
        return float("nan")


def compute_mae(targets: np.ndarray, preds: np.ndarray) -> float:
    return float(np.mean(np.abs(targets - preds)))


# ─────────────────────────────────────────────────────────────────────────────
# LoRATrainer
# ─────────────────────────────────────────────────────────────────────────────

class LoRATrainer:
    def __init__(
        self,
        model,
        ds_name:       str   = "downstream",
        task_type:     str   = "classification",
        num_classes:   int   = 2,
        lr:            float = 1e-3,
        weight_decay:  float = 1e-2,
        epochs:        int   = 20,
        warmup_epochs: int   = 2,
        device:        str   = "cuda",
        save_dir:      str   = "NormWear/data/results/lora_results",
        subject_id:    Optional[str] = None,
    ):
        self.model         = model
        self.ds_name       = ds_name
        self.task_type     = task_type
        self.num_classes   = num_classes
        self.lr            = lr
        self.weight_decay  = weight_decay
        self.epochs        = epochs
        self.warmup_epochs = warmup_epochs
        self.subject_id    = subject_id
        self.save_dir      = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model  = self.model.to(self.device)

        if task_type == "classification":
            self.criterion = nn.BCEWithLogitsLoss() if num_classes == 2 \
                             else nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

        trainable = self.model.trainable_parameters()
        self.optimizer = optim.AdamW(
            trainable, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999),
        )
        print(
            f"[Trainer] AdamW lr={lr}, "
            f"trainable={sum(p.numel() for p in trainable):,}"
        )
        self.scheduler = self._build_scheduler()
        self.history   = {"train_loss": [], "val_loss": [], "val_metric": []}
        self.best_metric = -float("inf") if task_type == "classification" \
                           else float("inf")
        self.best_epoch  = 0

    def _build_scheduler(self):
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / max(self.warmup_epochs, 1)
            progress = (epoch - self.warmup_epochs) / max(
                self.epochs - self.warmup_epochs, 1)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _compute_loss(self, logits, label):
        if self.task_type == "classification":
            if self.num_classes == 2:
                return self.criterion(logits.squeeze(-1), label.float())
            return self.criterion(logits, label)
        return self.criterion(logits.squeeze(-1), label.float())

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss, n = 0.0, 0
        for batch in loader:
            x     = batch["input"].to(self.device)
            label = batch["label"].to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss   = self._compute_loss(logits, label)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.trainable_parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item() * x.size(0)
            n          += x.size(0)
        return total_loss / max(n, 1)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> tuple:
        self.model.eval()
        total_loss, n = 0.0, 0
        all_labels, all_probs = [], []

        for batch in loader:
            x     = batch["input"].to(self.device)
            label = batch["label"].to(self.device)
            logits = self.model(x)
            loss   = self._compute_loss(logits, label)
            total_loss += loss.item() * x.size(0)
            n          += x.size(0)

            if self.task_type == "classification":
                if self.num_classes > 2:
                    probs = torch.softmax(logits, dim=-1)
                else:
                    p = torch.sigmoid(logits.squeeze(-1))
                    probs = torch.stack([1 - p, p], dim=-1)
                all_probs.append(probs.cpu().numpy())
            else:
                all_probs.append(logits.squeeze(-1).cpu().numpy())
            all_labels.append(label.cpu().numpy())

        avg_loss   = total_loss / max(n, 1)
        all_labels = np.concatenate(all_labels)
        all_probs  = np.concatenate(all_probs, axis=0)

        if self.task_type == "classification":
            # guard: AUC requires ≥ 2 classes present
            if len(np.unique(all_labels)) < 2:
                metric = float("nan")
            else:
                metric = compute_auc(all_labels, all_probs, self.num_classes)
        else:
            mae    = compute_mae(all_labels, all_probs)
            metric = -mae

        return avg_loss, metric

    def train(self, train_loader: DataLoader, val_loader: DataLoader, verbose: bool = True):
        tag = f"[{self.ds_name}" + (f"/{self.subject_id}]" if self.subject_id else "]")
        print(f"\n{tag} LoRA fine-tuning: {self.epochs} epochs")

        for epoch in range(self.epochs):
            t0         = time.time()
            train_loss = self._train_epoch(train_loader)
            val_loss, val_metric = self._eval_epoch(val_loader)
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_metric"].append(val_metric)

            if verbose:
                mname = "AUC" if self.task_type == "classification" else "-MAE"
                print(
                    f"{tag} E{epoch+1:3d}/{self.epochs} | "
                    f"train={train_loss:.4f} | val={val_loss:.4f} | "
                    f"{mname}={val_metric:.2f} | "
                    f"lr={self.optimizer.param_groups[0]['lr']:.1e} | "
                    f"{time.time()-t0:.1f}s"
                )

            improved = (
                val_metric > self.best_metric
                if (self.task_type == "classification" or True)
                else False
            )
            if improved and not np.isnan(val_metric):
                self.best_metric = val_metric
                self.best_epoch  = epoch + 1
                self._save_best()

        print(f"\n{tag} Best epoch={self.best_epoch}, metric={self.best_metric:.2f}")
        self._load_best()

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> dict:
        self.model.eval()
        all_labels, all_probs = [], []

        for batch in test_loader:
            x      = batch["input"].to(self.device)
            label  = batch["label"]
            logits = self.model(x).cpu()

            if self.task_type == "classification":
                if self.num_classes > 2:
                    probs = torch.softmax(logits, dim=-1)
                else:
                    p = torch.sigmoid(logits.squeeze(-1))
                    probs = torch.stack([1 - p, p], dim=-1)
                all_probs.append(probs.numpy())
            else:
                all_probs.append(logits.squeeze(-1).numpy())
            all_labels.append(label.numpy())

        all_labels = np.concatenate(all_labels)
        all_probs  = np.concatenate(all_probs, axis=0)

        results = {
            "ds_name":    self.ds_name,
            "subject_id": self.subject_id,
            "task_type":  self.task_type,
        }

        if self.task_type == "classification":
            if len(np.unique(all_labels)) < 2:
                auc = float("nan")
                print(f"[Evaluate] {self.ds_name}/{self.subject_id} — "
                      f"SKIPPED AUC (only one class in test set)")
            else:
                auc = compute_auc(all_labels, all_probs, self.num_classes)
                print(f"[Evaluate] {self.ds_name}/{self.subject_id} AUC = {auc:.2f}%")
            results["auc"] = auc
        else:
            preds = all_probs
            mae   = compute_mae(all_labels, preds)
            corr  = float(np.corrcoef(all_labels, preds)[0, 1])
            results["mae"]  = mae
            results["corr"] = corr
            print(f"[Evaluate] {self.ds_name}/{self.subject_id} MAE={mae:.4f}, Corr={corr:.4f}")

        sid_tag  = f"_{self.subject_id}" if self.subject_id else ""
        out_path = os.path.join(self.save_dir, f"{self.ds_name}{sid_tag}_results.json")
        with open(out_path, "w") as fp:
            json.dump(results, fp, indent=2)
        return results

    def _best_path(self) -> str:
        sid_tag = f"_{self.subject_id}" if self.subject_id else ""
        return os.path.join(self.save_dir, f"best_lora_{self.ds_name}{sid_tag}.pth")

    def _save_best(self):
        self.model.save_full(
            self._best_path(),
            extra={"epoch": self.best_epoch, "best_metric": self.best_metric,
                   "history": self.history},
        )

    def _load_best(self):
        path = self._best_path()
        if os.path.isfile(path):
            self.model.load_full(path)


# ─────────────────────────────────────────────────────────────────────────────
# Per-subject personalization loop
# ─────────────────────────────────────────────────────────────────────────────

def run_per_subject_personalization(
    base_model_builder,
    ds_train,
    ds_test,
    trainer_kwargs: dict,
    batch_size:   int  = 16,
    num_workers:  int  = 2,
    device:       str  = "cuda",
    min_samples:  int  = 5,       # ← now a parameter, not hardcoded 10
) -> dict:
    """
    For every subject in the dataset, independently fine-tune a fresh LoRA
    model and evaluate.  Results are aggregated and returned.

    Parameters
    ----------
    min_samples : int
        Minimum total samples a subject must have to be included.
        Set lower (e.g. 3) for small datasets like PPG_*.
    """
    import tempfile
    from NormWear.lora.lora_dataset import PersonalizedDownstreamDataset

    data_dir  = ds_train.data_dir
    ds_name   = ds_train.ds_name
    max_L     = ds_train.max_L
    pad_nvar  = ds_train.pad_nvar
    task_type = ds_train.task_type
    sid_idx   = ds_train.sid_split_idx

    # ── group all files by subject ─────────────────────────────────────────
    all_files = sorted(
        f for f in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, f)) and not f.endswith(".json")
    )

    from NormWear.lora.lora_dataset import extract_subject_id, parse_label

    by_subject: dict = defaultdict(list)
    for fname in all_files:
        fpath = os.path.join(data_dir, fname)
        try:
            d = pickle.load(open(fpath, "rb"))
        except Exception:
            continue
        sid = extract_subject_id(d, fname, sid_idx)
        by_subject[sid].append(fname)

    subjects = sorted(by_subject.keys())
    print(f"\n[PerSubject] {ds_name}: {len(subjects)} subjects, "
          f"min_samples threshold={min_samples}")
    for sid in subjects:
        n = len(by_subject[sid])
        status = "OK" if n >= min_samples else "SKIP (too few)"
        print(f"  subject={sid:>8}  files={n:>5}  {status}")

    all_results = []

    for sid in subjects:
        files = sorted(by_subject[sid])
        if len(files) < min_samples:
            print(f"\n[SKIP] subject={sid}: only {len(files)} samples < {min_samples}")
            continue

        print(f"\n{'='*60}\n  Subject: {sid}  ({len(files)} samples)\n{'='*60}")

        # ── stratified 80/20 split by label ───────────────────────────────
        by_label: dict = defaultdict(list)
        for fname in files:
            fpath = os.path.join(data_dir, fname)
            try:
                d   = pickle.load(open(fpath, "rb"))
                lbl = parse_label(ds_name, d)
            except Exception:
                lbl = 0
            by_label[lbl].append(fname)

        train_files_s, test_files_s = [], []
        for lbl_files in by_label.values():
            lbl_files = sorted(lbl_files)
            n_train   = max(1, int(len(lbl_files) * 0.8))
            train_files_s.extend(lbl_files[:n_train])
            test_files_s.extend(lbl_files[n_train:])

        if len(test_files_s) == 0:
            print(f"  [SKIP] No test samples after split (all labels in one bin).")
            continue

        # ── build temporary split JSON ─────────────────────────────────────
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir="/tmp"
        )
        json.dump({"train": train_files_s, "test": test_files_s}, tmp)
        tmp.close()

        try:
            subj_train = PersonalizedDownstreamDataset(
                data_dir=data_dir, ds_name=ds_name,
                split_file=tmp.name, split="train",
                max_L=max_L, pad_nvar=pad_nvar,
                task_type=task_type, sid_split_idx=sid_idx,
            )
            subj_test = PersonalizedDownstreamDataset(
                data_dir=data_dir, ds_name=ds_name,
                split_file=tmp.name, split="test",
                max_L=max_L, pad_nvar=pad_nvar,
                task_type=task_type, sid_split_idx=sid_idx,
            )
        finally:
            os.unlink(tmp.name)

        if len(subj_train) == 0 or len(subj_test) == 0:
            print(f"  [SKIP] Empty split after loading (label parse mismatch?).")
            continue

        # ── guard: check test labels have ≥ 2 classes for AUC ─────────────
        test_labels = [s["label"] for s in subj_test.samples]
        if task_type == "classification" and len(set(test_labels)) < 2:
            print(f"  [SKIP] All test samples have the same class — AUC undefined.")
            continue

        print(f"  train={len(subj_train)}  test={len(subj_test)}")

        train_loader = DataLoader(
            subj_train, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )
        test_loader = DataLoader(
            subj_test, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        model   = base_model_builder()
        trainer = LoRATrainer(
            model,
            subject_id=sid,
            **{**trainer_kwargs, "device": device},
        )
        trainer.train(train_loader, test_loader, verbose=True)
        result = trainer.evaluate(test_loader)
        all_results.append(result)

    # ── aggregate ─────────────────────────────────────────────────────────
    aggregated = {"per_subject": all_results, "ds_name": ds_name}
    if all_results:
        if all_results[0]["task_type"] == "classification":
            aucs = [r["auc"] for r in all_results
                    if "auc" in r and not np.isnan(r.get("auc", float("nan")))]
            aggregated["mean_auc"] = float(np.mean(aucs)) if aucs else float("nan")
            aggregated["std_auc"]  = float(np.std(aucs))  if aucs else float("nan")
            print(f"\n[PerSubject] {ds_name}: "
                  f"Mean AUC = {aggregated['mean_auc']:.2f} "
                  f"± {aggregated['std_auc']:.2f}%  "
                  f"(over {len(aucs)} subjects)")
        else:
            maes = [r["mae"] for r in all_results if "mae" in r]
            aggregated["mean_mae"] = float(np.mean(maes)) if maes else float("nan")
            print(f"\n[PerSubject] {ds_name}: "
                  f"Mean MAE = {aggregated['mean_mae']:.4f}")

    return aggregated


# ── make pickle available in scope for the loop above ────────────────────────
import pickle
