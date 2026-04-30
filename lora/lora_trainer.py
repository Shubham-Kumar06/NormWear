"""
lora_trainer.py
===============
Training loop for LoRA personalization on NormWear downstream tasks.

Handles:
  • Classification  → optimizes BCE / CE loss, evaluates macro AUC
  • Regression      → optimizes MSE loss, evaluates MAE & correlation

Per-subject personalization mode:
  For each subject in the dataset, the trainer independently fine-tunes
  a fresh copy of LoRA weights (the base encoder is shared / frozen).

Usage (see lora_main.py for the CLI wrapper):
    trainer = LoRATrainer(model, args)
    trainer.train(train_loader, val_loader)
    trainer.evaluate(test_loader)
"""

import os
import copy
import time
import json
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
    """Macro-averaged ROC-AUC (handles binary and multi-class)."""
    if not HAS_SKLEARN:
        return float("nan")
    try:
        if num_classes == 2:
            return roc_auc_score(labels, probs[:, 1]) * 100.0
        else:
            return roc_auc_score(
                labels, probs,
                multi_class="ovr", average="macro"
            ) * 100.0
    except ValueError:
        return float("nan")


def compute_mae(targets: np.ndarray, preds: np.ndarray) -> float:
    return float(np.mean(np.abs(targets - preds)))


# ─────────────────────────────────────────────────────────────────────────────
# LoRATrainer
# ─────────────────────────────────────────────────────────────────────────────

class LoRATrainer:
    """
    Handles training and evaluation of a NormWearLoRA model.

    Args:
        model       : NormWearLoRA instance (encoder frozen, LoRA + head trainable)
        ds_name     : downstream dataset name (for logging / saving)
        task_type   : 'classification' or 'regression'
        num_classes : number of output classes
        lr          : learning rate for AdamW (LoRA + head)
        weight_decay: AdamW weight decay
        epochs      : number of fine-tuning epochs
        warmup_epochs: cosine LR warmup epochs
        device      : 'cuda' or 'cpu'
        save_dir    : directory to save LoRA checkpoints and results
        subject_id  : if fine-tuning per-subject, tag saves with this ID
    """

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

        # ── loss ──────────────────────────────────────────────────────
        if task_type == "classification":
            if num_classes == 2:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

        # ── optimizer: only trainable params ─────────────────────────
        trainable = self.model.trainable_parameters()
        self.optimizer = optim.AdamW(
            trainable,
            lr           = lr,
            weight_decay = weight_decay,
            betas        = (0.9, 0.999),
        )
        print(
            f"[Trainer] Optimizer: AdamW, lr={lr}, "
            f"trainable_params={sum(p.numel() for p in trainable):,}"
        )

        # ── scheduler: cosine with warmup ─────────────────────────────
        self.scheduler = self._build_scheduler()

        self.history = {
            "train_loss": [], "val_loss": [], "val_metric": []
        }
        self.best_metric = -float("inf") if task_type == "classification" \
                           else float("inf")
        self.best_epoch  = 0

    # ── LR scheduler ─────────────────────────────────────────────────

    def _build_scheduler(self):
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / max(self.warmup_epochs, 1)
            progress = (epoch - self.warmup_epochs) / max(
                self.epochs - self.warmup_epochs, 1
            )
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ── one epoch ────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0

        for batch in loader:
            x     = batch["input"].to(self.device)  # [B, nvar, 3, T, F]
            label = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x)                  # [B, C] or [B, 1]

            loss = self._compute_loss(logits, label)
            loss.backward()

            nn.utils.clip_grad_norm_(
                self.model.trainable_parameters(), max_norm=1.0
            )
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            n          += x.size(0)

        return total_loss / max(n, 1)

    def _compute_loss(self, logits, label):
        if self.task_type == "classification":
            if self.num_classes == 2:
                return self.criterion(
                    logits.squeeze(-1),
                    label.float()
                )
            else:
                return self.criterion(logits, label)
        else:
            return self.criterion(logits.squeeze(-1), label.float())

    # ── evaluation ────────────────────────────────────────────────────

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> tuple:
        """Returns (avg_loss, metric).
        metric = AUC*100 for classification, -MAE for regression.
        """
        self.model.eval()
        total_loss = 0.0
        n = 0
        all_labels = []
        all_probs  = []

        for batch in loader:
            x     = batch["input"].to(self.device)
            label = batch["label"].to(self.device)

            logits = self.model(x)
            loss   = self._compute_loss(logits, label)

            total_loss += loss.item() * x.size(0)
            n          += x.size(0)

            if self.task_type == "classification":
                probs = torch.softmax(logits, dim=-1) if self.num_classes > 2 \
                        else torch.sigmoid(logits).unsqueeze(-1)
                # for binary: make [B, 2] compatible with roc_auc_score
                if self.num_classes == 2 and probs.shape[-1] == 1:
                    probs = torch.cat([1 - probs, probs], dim=-1)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(label.cpu().numpy())
            else:
                all_probs.append(logits.squeeze(-1).cpu().numpy())
                all_labels.append(label.cpu().numpy())

        avg_loss   = total_loss / max(n, 1)
        all_labels = np.concatenate(all_labels)
        all_probs  = np.concatenate(all_probs, axis=0)

        if self.task_type == "classification":
            metric = compute_auc(all_labels, all_probs, self.num_classes)
        else:
            preds  = all_probs
            mae    = compute_mae(all_labels, preds)
            corr   = float(np.corrcoef(all_labels, preds)[0, 1])
            metric = -mae   # higher = better for scheduler/comparison
            print(f"  MAE={mae:.4f}  Corr={corr:.4f}", end="")

        return avg_loss, metric

    # ── main train loop ───────────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        verbose:      bool = True,
    ):
        """Fine-tune LoRA + head on train_loader, validate on val_loader."""
        tag = f"[{self.ds_name}" + (f"/{self.subject_id}]" if self.subject_id else "]")
        print(f"\n{tag} Starting LoRA fine-tuning: {self.epochs} epochs")

        for epoch in range(self.epochs):
            t0         = time.time()
            train_loss = self._train_epoch(train_loader)
            val_loss, val_metric = self._eval_epoch(val_loader)
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_metric"].append(val_metric)

            elapsed = time.time() - t0
            if verbose:
                metric_name = "AUC" if self.task_type == "classification" else "-MAE"
                print(
                    f"{tag} Epoch {epoch+1:3d}/{self.epochs} | "
                    f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                    f"val_{metric_name}={val_metric:.2f} | "
                    f"lr={self.optimizer.param_groups[0]['lr']:.2e} | "
                    f"{elapsed:.1f}s"
                )

            # save best checkpoint
            improved = (
                val_metric > self.best_metric if self.task_type == "classification"
                else val_metric > self.best_metric
            )
            if improved:
                self.best_metric = val_metric
                self.best_epoch  = epoch + 1
                self._save_best()

        print(
            f"\n{tag} Best epoch={self.best_epoch}, "
            f"best_metric={self.best_metric:.2f}"
        )
        self._load_best()   # restore best weights before returning

    # ── final evaluation ──────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> dict:
        """
        Run full evaluation on test_loader with the current (best) model.
        Returns a results dict and saves it to save_dir.
        """
        self.model.eval()
        all_labels = []
        all_probs  = []

        for batch in test_loader:
            x      = batch["input"].to(self.device)
            label  = batch["label"]
            logits = self.model(x).cpu()

            if self.task_type == "classification":
                probs = torch.softmax(logits, dim=-1) if self.num_classes > 2 \
                        else torch.sigmoid(logits)
                if self.num_classes == 2 and probs.shape[-1] == 1:
                    probs = torch.cat([1 - probs, probs], dim=-1)
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
            auc = compute_auc(all_labels, all_probs, self.num_classes)
            results["auc"] = auc
            print(f"[Evaluate] {self.ds_name} AUC = {auc:.2f}%")
        else:
            preds = all_probs
            mae   = compute_mae(all_labels, preds)
            corr  = float(np.corrcoef(all_labels, preds)[0, 1])
            results["mae"]  = mae
            results["corr"] = corr
            print(f"[Evaluate] {self.ds_name} MAE={mae:.4f}, Corr={corr:.4f}")

        # save results JSON
        sid_tag = f"_{self.subject_id}" if self.subject_id else ""
        out_path = os.path.join(
            self.save_dir, f"{self.ds_name}{sid_tag}_results.json"
        )
        with open(out_path, "w") as fp:
            json.dump(results, fp, indent=2)
        print(f"[Evaluate] Results saved → {out_path}")

        return results

    # ── checkpoint helpers ─────────────────────────────────────────────

    def _best_path(self) -> str:
        sid_tag = f"_{self.subject_id}" if self.subject_id else ""
        return os.path.join(
            self.save_dir, f"best_lora_{self.ds_name}{sid_tag}.pth"
        )

    def _save_best(self):
        self.model.save_full(
            self._best_path(),
            extra={
                "epoch":       self.best_epoch,
                "best_metric": self.best_metric,
                "history":     self.history,
            }
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
    batch_size:  int  = 16,
    num_workers: int  = 2,
    device:      str  = "cuda",
) -> dict:
    from .lora_dataset import PersonalizedDownstreamDataset
    from torch.utils.data import DataLoader
    from collections import defaultdict
    import os, json, tempfile

    data_dir  = ds_train.data_dir
    ds_name   = ds_train.ds_name
    max_L     = ds_train.max_L
    pad_nvar  = ds_train.pad_nvar
    task_type = ds_train.task_type

    # group all files by subject ID (first segment of filename)
    all_files = sorted(
        f for f in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, f)) and not f.endswith(".json")
    )
    by_subject = defaultdict(list)
    for f in all_files:
        sid = f.replace(".pkl", "").split("_")[0]
        by_subject[sid].append(f)

    subjects = sorted(by_subject.keys())
    print(f"\n[PerSubject] {len(subjects)} subjects, per-subject 80/20 split")
    for sid in subjects:
        n = len(by_subject[sid])
        print(f"  Subject {sid}: {int(n*0.8)} train | {n - int(n*0.8)} test")

    all_results = []
    for sid in subjects:
        print(f"\n{'='*60}\n  Subject: {sid}\n{'='*60}")
        files = sorted(by_subject[sid])
        if len(files) < 10:
            print(f"  [SKIP] Not enough data ({len(files)} samples)")
            continue

        # stratified 80/20 split — group by label to ensure all classes in test
        import pickle as _pkl
        by_label = defaultdict(list)
        for f in files:
            try:
                d_tmp = _pkl.load(open(os.path.join(data_dir, f), "rb"))
                lbl = d_tmp["label"][0]["class"] if isinstance(d_tmp.get("label"), list)                       else int(d_tmp.get("label", 0))
            except Exception:
                lbl = 0
            by_label[lbl].append(f)

        train_files_s, test_files_s = [], []
        for lbl_files in by_label.values():
            lbl_files = sorted(lbl_files)
            n = max(1, int(len(lbl_files) * 0.8))
            train_files_s.extend(lbl_files[:n])
            test_files_s.extend(lbl_files[n:])

        if len(test_files_s) == 0:
            print(f"  [SKIP] No test data after stratified split")
            continue

        n_train = len(train_files_s)
        split_data = {"train": train_files_s, "test": test_files_s}
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                          delete=False, dir="/tmp")
        json.dump(split_data, tmp)
        tmp.close()

        subj_train = PersonalizedDownstreamDataset(
            data_dir=data_dir, ds_name=ds_name, split_file=tmp.name,
            split="train", max_L=max_L, pad_nvar=pad_nvar, task_type=task_type)
        subj_test = PersonalizedDownstreamDataset(
            data_dir=data_dir, ds_name=ds_name, split_file=tmp.name,
            split="test", max_L=max_L, pad_nvar=pad_nvar, task_type=task_type)
        os.unlink(tmp.name)

        print(f"  Loaded: {len(subj_train)} train | {len(subj_test)} test")
        if len(subj_train) == 0 or len(subj_test) == 0:
            print(f"  [SKIP] Empty split")
            continue

        train_loader = DataLoader(subj_train, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  pin_memory=True, drop_last=False)
        test_loader  = DataLoader(subj_test, batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers,
                                  pin_memory=True)

        model   = base_model_builder()
        trainer = LoRATrainer(model, subject_id=sid,
                              **{**trainer_kwargs, "device": device})
        trainer.train(train_loader, test_loader, verbose=True)
        result = trainer.evaluate(test_loader)
        all_results.append(result)

    # aggregate
    aggregated = {"per_subject": all_results}
    if all_results:
        if all_results[0]["task_type"] == "classification":
            aucs = [r["auc"] for r in all_results
                    if "auc" in r and not np.isnan(r["auc"])]
            aggregated["mean_auc"] = float(np.mean(aucs)) if aucs else float("nan")
            aggregated["std_auc"]  = float(np.std(aucs))  if aucs else float("nan")
            print(f"\n[PerSubject] Mean AUC = {aggregated['mean_auc']:.2f}"
                  f" ± {aggregated['std_auc']:.2f}%")
        else:
            maes = [r["mae"] for r in all_results if "mae" in r]
            aggregated["mean_mae"] = float(np.mean(maes)) if maes else float("nan")
            print(f"\n[PerSubject] Mean MAE = {aggregated['mean_mae']:.4f}")
    return aggregated

