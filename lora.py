import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


#  Core LoRA linear layer
class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear with a low-rank update:
        W' = W + (B @ A) * (alpha / r)

    Only A and B are trained; W is frozen.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.original = original_linear          # frozen base weight
        self.r = r
        self.scaling = alpha / r

        in_features  = original_linear.in_features
        out_features = original_linear.out_features

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_dropout = nn.Dropout(p=dropout)

        # Kaiming init for A, zeros for B (so delta-W = 0 at start)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Freeze the original weight
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + lora_out * self.scaling

    def extra_repr(self):
        return (
            f"in={self.original.in_features}, "
            f"out={self.original.out_features}, "
            f"r={self.r}, scaling={self.scaling:.3f}"
        )


#  Injection helper
def inject_lora(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_modules: tuple = ("q_proj", "v_proj", "query", "value",
                             "to_q", "to_v", "q", "v"),
) -> nn.Module:
    """
    Walk the model tree and replace target Linear layers with LoRALinear.

    Args:
        model:          Pre-trained NormWear model (weights already loaded).
        r:              LoRA rank.
        alpha:          LoRA scaling factor.
        dropout:        Dropout on the LoRA path.
        target_modules: Sub-string patterns to match module names.
                        Adjust to match NormWear's actual layer names.

    Returns:
        The same model with LoRA layers injected (in-place modification).
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        # Walk to the parent so we can do setattr
        parts   = name.split(".")
        parent  = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr_name = parts[-1]

        if isinstance(module, nn.Linear):
            # Check if this layer name matches any target pattern
            if any(t in attr_name for t in target_modules):
                lora_layer = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
                setattr(parent, attr_name, lora_layer)
                replaced += 1

    print(f"[LoRA] Injected LoRA into {replaced} Linear layers "
          f"(r={r}, alpha={alpha})")
    return model


#  Parameter utilities
def freeze_base_model(model: nn.Module) -> None:
    """Freeze everything that is NOT a LoRA parameter."""
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad_(False)


def get_lora_params(model: nn.Module):
    """Return only LoRA parameters (for the optimizer)."""
    return [p for n, p in model.named_parameters()
            if "lora_A" in n or "lora_B" in n]


def count_params(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total":     total,
        "trainable": trainable,
        "frozen":    total - trainable,
        "ratio":     f"{100 * trainable / total:.4f}%",
    }


#  Save / Load LoRA weights only
def save_lora_weights(model: nn.Module, path: str) -> None:
    """Save only the LoRA delta weights (tiny file per subject"""
    lora_state = {
        k: v for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }
    torch.save(lora_state, path)
    print(f"[LoRA] Saved {len(lora_state)} LoRA tensors → {path}")


def load_lora_weights(model: nn.Module, path: str,
                      strict: bool = True) -> nn.Module:
    """Load LoRA delta weights back into a model that already has LoRA injected."""
    lora_state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"LoRA load mismatch!\n  missing: {missing}\n  unexpected: {unexpected}"
        )
    print(f"[LoRA] Loaded LoRA weights from {path}")
    return model