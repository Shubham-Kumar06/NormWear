"""

Core LoRA implementation for NormWear.

LoRA (Low-Rank Adaptation) injects trainable low-rank matrices alongside frozen
pretrained Linear layers:

    W_new = W_frozen  +  (alpha / rank) * B @ A

where A is [rank, in_features] and B is [out_features, rank].
Only A and B are trained; W_frozen stays fixed.

Usage:
    model, lora_params = apply_lora_to_attention(encoder, rank=16, alpha=32)
"""

import math
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────
# 1.  LoRA linear layer
# ─────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that adds a frozen base weight
    plus trainable low-rank ΔW = (alpha/rank) * B @ A.

    Args:
        original_linear : the pretrained nn.Linear to wrap
        rank            : LoRA bottleneck dimension r
        alpha           : LoRA scaling factor (use 2×rank as a starting point)
        dropout         : dropout on input before the LoRA branch (0 = off)
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        in_features  = original_linear.in_features
        out_features = original_linear.out_features

        # ── freeze the original weight (and bias if any) ──
        self.original = original_linear
        for p in self.original.parameters():
            p.requires_grad = False

        self.rank    = rank
        self.alpha   = alpha
        self.scaling = alpha / rank          # applied to ΔW at each forward pass

        # ── LoRA matrices ──
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Initialisation: A ~ Kaiming uniform, B = 0
        # → ΔW = 0 at the start, so the model behaves identically to the
        #   pretrained checkpoint at epoch 0 of fine-tuning.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    # ── forward ──────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)                                   # frozen path
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))    # trainable ΔW
        return base_out + lora_out * self.scaling

    def extra_repr(self) -> str:
        return (
            f"in={self.original.in_features}, out={self.original.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}"
        )


# ─────────────────────────────────────────────────────────
# 2.  Inject LoRA into a NormWear encoder
# ─────────────────────────────────────────────────────────

def apply_lora_to_attention(
    model: nn.Module,
    rank: int   = 16,
    alpha: float = 32.0,
    dropout: float = 0.1,
    target_modules: tuple = ("qkv", "proj"),
) -> tuple:
    """
    Walk *model* and replace every nn.Linear whose attribute name is in
    *target_modules* with a LoRALinear.

    Steps
    -----
    1. Freeze ALL parameters in the model.
    2. Replace target Linear layers with LoRALinear (which re-introduces
       trainable lora_A / lora_B parameters).
    3. Return (model, lora_params_list).

    Args:
        model          : NormWear encoder (is_pretrain=False)
        rank           : LoRA rank r
        alpha          : LoRA alpha scaling
        dropout        : dropout before the LoRA branch
        target_modules : attribute names of Linear layers to adapt.
                         For a timm ViT attention block the defaults
                         ('qkv', 'proj') cover the QKV projection and the
                         output projection.

    Returns:
        model      : modified in-place
        lora_params: list[nn.Parameter] — only LoRA + head params (for optimizer)
    """

    # ── Step 1: freeze everything ──────────────────────────
    for param in model.parameters():
        param.requires_grad = False

    # ── Step 2: inject LoRA ────────────────────────────────
    replaced = 0
    for _module_name, module in model.named_modules():
        for attr in target_modules:
            linear = getattr(module, attr, None)
            if isinstance(linear, nn.Linear):
                setattr(
                    module, attr,
                    LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout)
                )
                replaced += 1

    # ── Step 3: collect trainable params ──────────────────
    lora_params = [p for p in model.parameters() if p.requires_grad]

    total = sum(p.numel() for p in model.parameters())
    lora  = sum(p.numel() for p in lora_params)
    pct   = 100 * lora / max(total, 1)

    print(f"[LoRA] Replaced {replaced} Linear layers "
          f"(rank={rank}, alpha={alpha}, dropout={dropout})")
    print(f"[LoRA] Total params  : {total:,}")
    print(f"[LoRA] LoRA trainable: {lora:,}  ({pct:.2f}%)")

    return model, lora_params


# ─────────────────────────────────────────────────────────
# 3.  Save / load LoRA weights only
# ─────────────────────────────────────────────────────────

def save_lora_weights(model: nn.Module, path: str) -> None:
    """
    Save only the LoRA weight tensors (lora_A, lora_B) — much smaller than
    saving the full model.
    """
    lora_state = {
        k: v for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }
    torch.save({"lora_state_dict": lora_state}, path)
    print(f"[LoRA] Saved {len(lora_state)} LoRA tensors → {path}")


def load_lora_weights(model: nn.Module, path: str) -> nn.Module:
    """
    Load LoRA weights into a model that already has LoRALinear layers injected.
    Non-LoRA keys are ignored (strict=False).
    """
    ckpt = torch.load(path, map_location="cpu")
    lora_state = ckpt["lora_state_dict"]
    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    lora_missing = [k for k in missing if "lora" in k]
    if lora_missing:
        raise RuntimeError(f"[LoRA] Missing LoRA keys in checkpoint: {lora_missing}")
    print(f"[LoRA] Loaded {len(lora_state)} LoRA tensors ← {path}")
    return model
