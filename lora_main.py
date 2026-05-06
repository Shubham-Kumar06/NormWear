"""
lora_main.py  (UPDATED — dataset-aware defaults)
================================================
Run from /home/ug24/FoundationalModel/:

  python3 -m NormWear.lora_main \\
      --model_weight NormWear/data/results/full_pretrain_checkpoint-399.pth \\
      --data_path    NormWear/data \\
      --ds_name      ecg_heart_cat \\
      --mode         per_subject

  All hyperparameters (max_L, num_classes, lr, epochs …) are automatically
  looked up from NormWear/lora/lora_config.py.  You can still override any
  of them explicitly on the command line.

Place this file at:  NormWear/lora_main.py
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="NormWear LoRA Personalization")

    # ── paths ─────────────────────────────────────────────────────────────
    p.add_argument("--model_weight", type=str,
                   default="NormWear/data/results/full_pretrain_checkpoint-399.pth")
    p.add_argument("--data_path",    type=str, default="NormWear/data")
    p.add_argument("--ds_name",      type=str, default="wesad")
    p.add_argument("--save_dir",     type=str,
                   default="NormWear/data/results/lora_results")

    # ── task (auto-filled from lora_config if not given) ──────────────────
    p.add_argument("--task_type",    type=str, default=None,
                   choices=["classification", "regression", None])
    p.add_argument("--num_classes",  type=int, default=None)
    p.add_argument("--mode",         type=str, default="per_subject",
                   choices=["global", "per_subject"])

    # ── LoRA (auto-filled from lora_config if not given) ──────────────────
    p.add_argument("--lora_rank",    type=int,   default=None)
    p.add_argument("--lora_alpha",   type=float, default=None)
    p.add_argument("--lora_dropout", type=float, default=None)

    # ── training (auto-filled from lora_config if not given) ──────────────
    p.add_argument("--epochs",        type=int,   default=None)
    p.add_argument("--warmup_epochs", type=int,   default=2)
    p.add_argument("--lr",            type=float, default=None)
    p.add_argument("--weight_decay",  type=float, default=1e-2)
    p.add_argument("--batch_size",    type=int,   default=None)
    p.add_argument("--num_workers",   type=int,   default=4)

    # ── model dims ────────────────────────────────────────────────────────
    p.add_argument("--embed_dim", type=int, default=768)
    p.add_argument("--nvar",      type=int, default=4)
    p.add_argument("--max_L",     type=int, default=None)

    # ── per-subject min samples ───────────────────────────────────────────
    p.add_argument("--min_samples", type=int, default=None,
                   help="Minimum files a subject needs to be included. "
                        "Defaults are set per-dataset in lora_config.py.")

    # ── diagnostic only: print subject info and exit ──────────────────────
    p.add_argument("--diagnose", action="store_true",
                   help="Print subject distribution and exit (no training).")

    return p.parse_args()


def _merge_config(args):
    """Fill None args from lora_config.py defaults for this dataset."""
    from NormWear.lora.lora_config import get_config
    cfg = get_config(args.ds_name)

    if args.task_type    is None: args.task_type    = cfg["task_type"]
    if args.num_classes  is None: args.num_classes  = cfg["num_classes"]
    if args.max_L        is None: args.max_L        = cfg["max_L"]
    if args.lora_rank    is None: args.lora_rank    = cfg["lora_rank"]
    if args.lora_alpha   is None: args.lora_alpha   = cfg["lora_alpha"]
    if args.lora_dropout is None: args.lora_dropout = cfg["lora_dropout"]
    if args.epochs       is None: args.epochs       = cfg["epochs"]
    if args.lr           is None: args.lr           = cfg["lr"]
    if args.batch_size   is None: args.batch_size   = cfg["batch_size"]
    if args.min_samples  is None: args.min_samples  = cfg["min_samples"]
    args._sid_split_idx = cfg["sid_split_idx"]
    return args


# ─────────────────────────────────────────────────────────────────────────────
# Load pretrained encoder
# ─────────────────────────────────────────────────────────────────────────────

def load_pretrained_encoder(ckpt_path: str):
    from NormWear.modules.normwear import NormWear

    model = NormWear(
        img_size=(387, 65), patch_size=(9, 5), in_chans=3, target_len=388,
        nvar=4, embed_dim=768, decoder_embed_dim=512, depth=12, num_heads=12,
        decoder_depth=2, mlp_ratio=4.0, fuse_freq=2, mask_t_prob=0.6,
        mask_f_prob=0.5, mask_prob=0.8, mask_scheme="random",
        use_cwt=True, is_pretrain=False,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd   = ckpt.get("model", ckpt.get("state_dict", ckpt))
    sd   = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[Encoder] Loaded — missing={len(missing)}, unexpected={len(unexpected)}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Build LoRA model
# ─────────────────────────────────────────────────────────────────────────────

def build_lora_model(encoder, args):
    import copy
    from NormWear.lora.lora_model import NormWearLoRA
    return NormWearLoRA(
        base_encoder   = copy.deepcopy(encoder),
        num_classes    = args.num_classes,
        task_type      = args.task_type,
        embed_dim      = args.embed_dim,
        lora_rank      = args.lora_rank,
        lora_alpha     = args.lora_alpha,
        lora_dropout   = args.lora_dropout,
        target_modules = ("qkv", "proj"),
        pooling        = "mean",
        nvar           = args.nvar,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Build datasets
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(args):
    from NormWear.lora.lora_dataset import PersonalizedDownstreamDataset

    ds_root    = os.path.join(args.data_path, "wearable_downstream",
                              args.ds_name, "sample_for_downstream")
    split_file = os.path.join(args.data_path, "wearable_downstream",
                              args.ds_name, "train_test_split.json")

    if not os.path.isdir(ds_root):
        raise FileNotFoundError(f"Dataset not found: {ds_root}")

    common = dict(
        data_dir       = ds_root,
        ds_name        = args.ds_name,
        split_file     = split_file if os.path.isfile(split_file) else None,
        max_L          = args.max_L,
        pad_nvar       = args.nvar,
        task_type      = args.task_type,
        sid_split_idx  = args._sid_split_idx,
    )
    ds_train = PersonalizedDownstreamDataset(**common, split="train")
    ds_test  = PersonalizedDownstreamDataset(**common, split="test")
    return ds_train, ds_test


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    args = _merge_config(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  Dataset : {args.ds_name}  |  Mode: {args.mode}")
    print(f"  task    : {args.task_type}  classes={args.num_classes}")
    print(f"  max_L   : {args.max_L}  nvar={args.nvar}")
    print(f"  LoRA    : rank={args.lora_rank}, alpha={args.lora_alpha}, "
          f"dropout={args.lora_dropout}")
    print(f"  Train   : epochs={args.epochs}, lr={args.lr}, "
          f"batch={args.batch_size}")
    if args.mode == "per_subject":
        print(f"  min_samples_per_subject = {args.min_samples}")
    print(f"{'='*60}\n")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── diagnose only ──────────────────────────────────────────────────────
    if args.diagnose:
        from NormWear.lora.lora_dataset import diagnose_subjects
        ds_root = os.path.join(args.data_path, "wearable_downstream",
                               args.ds_name, "sample_for_downstream")
        diagnose_subjects(ds_root, args.ds_name, args._sid_split_idx)
        return

    # ── load encoder ───────────────────────────────────────────────────────
    encoder = load_pretrained_encoder(args.model_weight)

    # ── datasets ───────────────────────────────────────────────────────────
    ds_train, ds_test = build_datasets(args)

    trainer_kwargs = dict(
        ds_name       = args.ds_name,
        task_type     = args.task_type,
        num_classes   = args.num_classes,
        lr            = args.lr,
        weight_decay  = args.weight_decay,
        epochs        = args.epochs,
        warmup_epochs = args.warmup_epochs,
        save_dir      = args.save_dir,
    )

    # ── GLOBAL mode ────────────────────────────────────────────────────────
    if args.mode == "global":
        from NormWear.lora.lora_dataset import build_dataloaders
        from NormWear.lora.lora_trainer import LoRATrainer

        model = build_lora_model(encoder, args)
        print(f"[Main] Params: {model.count_parameters()}")

        train_loader, test_loader = build_dataloaders(
            ds_train, ds_test,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        trainer = LoRATrainer(model, device=device, **trainer_kwargs)
        trainer.train(train_loader, test_loader, verbose=True)
        results = trainer.evaluate(test_loader)

        out = os.path.join(args.save_dir, f"{args.ds_name}_global_summary.json")
        with open(out, "w") as fp:
            json.dump(results, fp, indent=2)
        print(f"[Main] Saved → {out}")

    # ── PER-SUBJECT mode ───────────────────────────────────────────────────
    elif args.mode == "per_subject":
        from NormWear.lora.lora_trainer import run_per_subject_personalization

        def model_builder():
            return build_lora_model(encoder, args)

        aggregated = run_per_subject_personalization(
            base_model_builder = model_builder,
            ds_train           = ds_train,
            ds_test            = ds_test,
            trainer_kwargs     = trainer_kwargs,
            batch_size         = args.batch_size,
            num_workers        = args.num_workers,
            device             = device,
            min_samples        = args.min_samples,
        )

        out = os.path.join(args.save_dir,
                           f"{args.ds_name}_per_subject_summary.json")
        with open(out, "w") as fp:
            json.dump(aggregated, fp, indent=2, default=str)
        print(f"[Main] Per-subject summary → {out}")


if __name__ == "__main__":
    main()
