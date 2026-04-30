"""
lora_main.py

Command-line entry point for NormWear LoRA personalization.

Run from /home/ug24/FoundationalModel/ :

  # ── Global fine-tuning (all subjects together) ──────────────────────────
  python3 -m NormWear.lora_main \\
      --model_weight NormWear/data/results/full_pretrain_checkpoint-399.pth \\
      --data_path    NormWear/data \\
      --ds_name      wesad \\
      --task_type    classification \\
      --num_classes  3 \\
      --lora_rank    16 \\
      --lora_alpha   32 \\
      --epochs       20 \\
      --lr           1e-3 \\
      --mode         global

  # ── Per-subject personalization ─────────────────────────────────────────
  python3 -m NormWear.lora_main \\
      --model_weight NormWear/data/results/full_pretrain_checkpoint-399.pth \\
      --data_path    NormWear/data \\
      --ds_name      wesad \\
      --task_type    classification \\
      --num_classes  3 \\
      --lora_rank    16 \\
      --lora_alpha   32 \\
      --epochs       20 \\
      --lr           1e-3 \\
      --mode         per_subject
"""

import argparse
import json
import os
import sys

import torch

# ── make sure NormWear package is importable ──────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description="NormWear LoRA Personalization")

    # ── paths ──────────────────────────────────────────────────────────
    p.add_argument("--model_weight",  type=str,
                   default="NormWear/data/results/full_pretrain_checkpoint-399.pth",
                   help="Path to pretrained NormWear checkpoint")
    p.add_argument("--data_path",     type=str,
                   default="NormWear/data",
                   help="Root data directory")
    p.add_argument("--ds_name",       type=str,
                   default="wesad",
                   help="Downstream dataset name")
    p.add_argument("--save_dir",      type=str,
                   default="NormWear/data/results/lora_results",
                   help="Directory for saving LoRA checkpoints and results")

    # ── task ───────────────────────────────────────────────────────────
    p.add_argument("--task_type",     type=str,
                   default="classification",
                   choices=["classification", "regression"])
    p.add_argument("--num_classes",   type=int,   default=2)
    p.add_argument("--mode",          type=str,
                   default="global",
                   choices=["global", "per_subject"],
                   help="'global' = all subjects together; "
                        "'per_subject' = independent LoRA per subject")

    # ── LoRA hyper-params ──────────────────────────────────────────────
    p.add_argument("--lora_rank",     type=int,   default=16)
    p.add_argument("--lora_alpha",    type=float, default=32.0)
    p.add_argument("--lora_dropout",  type=float, default=0.1)

    # ── training hyper-params ──────────────────────────────────────────
    p.add_argument("--epochs",        type=int,   default=20)
    p.add_argument("--warmup_epochs", type=int,   default=2)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=1e-2)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--num_workers",   type=int,   default=4)

    # ── model dims (must match pretrain) ──────────────────────────────
    p.add_argument("--embed_dim",     type=int,   default=768)
    p.add_argument("--nvar",          type=int,   default=4)
    p.add_argument("--max_L",         type=int,   default=390)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Load pretrained NormWear encoder
# ─────────────────────────────────────────────────────────────────────────────

def load_pretrained_encoder(ckpt_path: str):
    """
    Load the NormWear model from the pretrain checkpoint and return it
    with is_pretrain=False so forward_encoder is accessible.
    """
    from NormWear.modules.normwear import NormWear

    # Build model (downstream mode — encoder only is used)
    model = NormWear(
        img_size          = (387, 65),
        patch_size        = (9, 5),
        in_chans          = 3,
        target_len        = 388,
        nvar              = 4,
        embed_dim         = 768,
        decoder_embed_dim = 512,
        depth             = 12,
        num_heads         = 12,
        decoder_depth     = 2,
        mlp_ratio         = 4.0,
        fuse_freq         = 2,
        mask_t_prob       = 0.6,
        mask_f_prob       = 0.5,
        mask_prob         = 0.8,
        mask_scheme       = "random",
        use_cwt           = True,
        is_pretrain       = False,   # ← downstream mode
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))

    # strip 'module.' prefix if saved with DataParallel
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[Encoder] Loaded from {ckpt_path}")
    print(f"  Missing  : {len(missing)}")
    print(f"  Unexpected: {len(unexpected)}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Build LoRA model
# ─────────────────────────────────────────────────────────────────────────────

def build_lora_model(encoder, args):
    from NormWear.lora.lora_model import NormWearLoRA
    import copy

    return NormWearLoRA(
        base_encoder   = copy.deepcopy(encoder),   # fresh copy per call
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

    ds_root    = os.path.join(args.data_path, "wearable_downstream", args.ds_name, "sample_for_downstream")
    split_file = os.path.join(args.data_path, "wearable_downstream", args.ds_name, "train_test_split.json")

    if not os.path.isdir(ds_root):
        raise FileNotFoundError(
            f"Downstream dataset not found: {ds_root}\n"
            f"Make sure --data_path and --ds_name are correct."
        )

    ds_train = PersonalizedDownstreamDataset(
        data_dir   = ds_root,
        ds_name    = args.ds_name,
        split_file = split_file if os.path.isfile(split_file) else None,
        split      = "train",
        max_L      = args.max_L,
        pad_nvar   = args.nvar,
        task_type  = args.task_type,
    )
    ds_test = PersonalizedDownstreamDataset(
        data_dir   = ds_root,
        ds_name    = args.ds_name,
        split_file = split_file if os.path.isfile(split_file) else None,
        split      = "test",
        max_L      = args.max_L,
        pad_nvar   = args.nvar,
        task_type  = args.task_type,
    )
    return ds_train, ds_test


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Main] Device: {device}")
    print(f"[Main] Dataset: {args.ds_name}  |  Mode: {args.mode}")
    print(f"[Main] LoRA rank={args.lora_rank}, alpha={args.lora_alpha}, "
          f"dropout={args.lora_dropout}")
    print(f"[Main] Epochs={args.epochs}, LR={args.lr}\n")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── load pretrained encoder ────────────────────────────────────────
    encoder = load_pretrained_encoder(args.model_weight)

    # ── datasets ───────────────────────────────────────────────────────
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
        # device passed separately to avoid duplicate kwarg in per_subject mode
    )

    # ── mode: global ───────────────────────────────────────────────────
    if args.mode == "global":
        from NormWear.lora.lora_dataset import build_dataloaders
        from NormWear.lora.lora_trainer import LoRATrainer

        model = build_lora_model(encoder, args)
        params = model.count_parameters()
        print(f"[Main] Parameters: {params}")

        train_loader, test_loader = build_dataloaders(
            ds_train, ds_test,
            batch_size  = args.batch_size,
            num_workers = args.num_workers,
        )

        trainer = LoRATrainer(model, device=device, **trainer_kwargs)
        trainer.train(train_loader, test_loader, verbose=True)
        results = trainer.evaluate(test_loader)

        # save summary
        out = os.path.join(args.save_dir, f"{args.ds_name}_global_summary.json")
        with open(out, "w") as fp:
            json.dump(results, fp, indent=2)
        print(f"[Main] Summary saved → {out}")

    # ── mode: per-subject ─────────────────────────────────────────────
    elif args.mode == "per_subject":
        from NormWear.lora.lora_trainer import run_per_subject_personalization
        import copy

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
        )

        # save aggregated results
        out = os.path.join(
            args.save_dir, f"{args.ds_name}_per_subject_summary.json"
        )
        with open(out, "w") as fp:
            json.dump(aggregated, fp, indent=2, default=str)
        print(f"[Main] Per-subject summary saved → {out}")


if __name__ == "__main__":
    main()
