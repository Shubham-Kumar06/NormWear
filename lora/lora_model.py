"""

NormWearLoRA: wraps the pretrained NormWear encoder with LoRA adapters
and a task-specific downstream head.

Architecture:
    Input [B, nvar, 3, T, F]
        ↓
    Frozen NormWear encoder  (+ LoRA delta-W in each attention block)
        ↓
    Mean pool over (nvar × patches)  →  [B, 768]
        ↓
    Trainable head  →  logits [B, C]  or  regression [B, 1]

Only LoRA matrices (lora_A, lora_B) and the head are trained.
"""

import torch
import torch.nn as nn

from .lora_layers import apply_lora_to_attention, save_lora_weights, load_lora_weights


class NormWearLoRA(nn.Module):
    """
    Args:
        base_encoder   : NormWear instance (is_pretrain=False recommended,
                         or full model — we only use the encoder half)
        num_classes    : output dimension for classification (≥2) or 1 for regression
        task_type      : 'classification' or 'regression'
        embed_dim      : transformer embedding dim (768 for NormWear)
        lora_rank      : LoRA rank r
        lora_alpha     : LoRA alpha scaling
        lora_dropout   : dropout before LoRA branch
        target_modules : which Linear attribute names to adapt
        pooling        : 'mean' (default) or 'cls'
        nvar           : max number of variables (sensors) per sample — used for
                         reshaping the encoder output
    """

    def __init__(
        self,
        base_encoder:    nn.Module,
        num_classes:     int   = 2,
        task_type:       str   = "classification",
        embed_dim:       int   = 768,
        lora_rank:       int   = 16,
        lora_alpha:      float = 32.0,
        lora_dropout:    float = 0.1,
        target_modules:  tuple = ("qkv", "proj"),
        pooling:         str   = "mean",
        nvar:            int   = 4,
    ):
        super().__init__()

        self.task_type   = task_type
        self.num_classes = num_classes
        self.pooling     = pooling
        self.embed_dim   = embed_dim
        self.nvar        = nvar

        # ── inject LoRA (freezes encoder, adds trainable A/B matrices) ──
        self.encoder, _ = apply_lora_to_attention(
            base_encoder,
            rank           = lora_rank,
            alpha          = lora_alpha,
            dropout        = lora_dropout,
            target_modules = target_modules,
        )

        # ── trainable downstream head ──────────────────────────────────
        if task_type == "classification":
            self.head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, num_classes),
            )
        else:  # regression
            self.head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, 1),
            )

        self._init_head()
        print(
            f"[NormWearLoRA] task={task_type}, classes={num_classes}, "
            f"pooling={pooling}, rank={lora_rank}, alpha={lora_alpha}"
        )

    # ── initialisation ────────────────────────────────────────────────

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── feature extraction ────────────────────────────────────────────

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the frozen encoder (+ LoRA) on *x* and return a pooled vector.

        Args:
            x : [B, nvar, 3, T, F]  (CWT scalogram, padded to nvar=4)

        Returns:
            features : [B, embed_dim]
        """
        B = x.shape[0]

        # ── call NormWear's encoder forward ───────────────────────────
        # NormWear.forward_encoder signature:
        #   forward_encoder(x, mask_ratio) → (latent, mask, ids_restore)
        # With mask_ratio=0.0 all patches are kept.
        if hasattr(self.encoder, "forward_encoder"):
            # NormWear expects [B*nvar, 3, L, F]
            B, nvar, C, L, F = x.shape
            x_in = x.view(B * nvar, C, L, F)           # [B*nvar, 3, L, F]
            # forward_encoder always masks + returns (latent, mask, ids_restore)
            # latent shape: [B*nvar, num_unmasked_patches+1, embed_dim]
            # cls token is at index 0; we use mean of patch tokens (index 1:)
            latent, _, _ = self.encoder.forward_encoder(x_in)
            latent = latent[:, 1:, :]                   # drop cls token → [B*nvar, P, D]
        else:
            raise AttributeError(
                "base_encoder must expose `forward_encoder(x)`. "
                "Make sure you pass the full NormWear model."
            )

        # ── reshape & pool ────────────────────────────────────────────
        latent = self._pool(latent, B)   # → [B, embed_dim]
        return latent

    def _pool(self, latent: torch.Tensor, B: int) -> torch.Tensor:
        """
        Handle different latent shapes and apply mean / cls pooling.

        Supported shapes:
          (a) [B * nvar, P, D]   — one row per (batch, variable) pair
          (b) [B, nvar * P, D]   — flattened (batch, var*patch)
          (c) [B, nvar, P, D]    — already separated
        """
        D = latent.shape[-1]

        # latent: [B*nvar, P, D] — reshape to [B, nvar, P, D] then mean pool
        if latent.dim() == 3:
            total, P, _ = latent.shape
            nvar_actual = total // B
            latent = latent.view(B, nvar_actual, P, D)  # [B, nvar, P, D]

        # latent: [B, nvar, P, D]
        return latent.mean(dim=(1, 2))   # mean over nvar and patches → [B, D]

    # ── forward ───────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, nvar, 3, T, F]

        Returns:
            logits : [B, num_classes]  (classification)
                  or [B, 1]            (regression)
        """
        features = self.get_features(x)   # [B, D]
        return self.head(features)        # [B, C] or [B, 1]

    # ── parameter helpers ─────────────────────────────────────────────

    def trainable_parameters(self):
        """All parameters with requires_grad=True (LoRA matrices + head)."""
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self):
        total  = sum(p.numel() for p in self.parameters())
        train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - train
        return {"total": total, "trainable": train, "frozen": frozen}

    # ── save / load ───────────────────────────────────────────────────

    def save_lora(self, path: str):
        """Save LoRA matrices only (small file, ~MB instead of ~GB)."""
        save_lora_weights(self.encoder, path)

    def load_lora(self, path: str):
        """Restore LoRA matrices from a previously saved file."""
        load_lora_weights(self.encoder, path)

    def save_full(self, path: str, extra: dict = None):
        """Save LoRA + head + metadata."""
        state = {
            "lora_and_head_state_dict": {
                k: v for k, v in self.state_dict().items()
                if "lora_A" in k or "lora_B" in k or k.startswith("head.")
            },
            "task_type":   self.task_type,
            "num_classes": self.num_classes,
            "embed_dim":   self.embed_dim,
        }
        if extra:
            state.update(extra)
        torch.save(state, path)
        print(f"[NormWearLoRA] Saved LoRA + head → {path}")

    def load_full(self, path: str):
        """Load LoRA + head from a save_full checkpoint."""
        ckpt = torch.load(path, map_location="cpu")
        missing, unexpected = self.load_state_dict(
            ckpt["lora_and_head_state_dict"], strict=False
        )
        lora_missing = [k for k in missing if "lora" in k or "head" in k]
        if lora_missing:
            print(f"[WARNING] Missing keys: {lora_missing}")
        print(f"[NormWearLoRA] Loaded LoRA + head ← {path}")
