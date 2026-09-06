"""
models/lora.py
==============
Minimal LoRA implementation applied to linear projections.

LoRA replaces W·x with (W + BA)·x where:
    A ∈ ℝ^(r × d_in)   — down projection, init ~ N(0, 0.02)
    B ∈ ℝ^(d_out × r)  — up projection, init = 0
    scaling = alpha / r

Initialising B=0 means LoRA starts as identity — no disruption to
pretrained weights at step 0. Only A and B are trainable.
W stays frozen.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear with trainable LoRA adapters.
    Output = frozen_linear(x) + (B @ A @ x) * scaling
    """

    def __init__(
        self,
        frozen_linear : nn.Linear,
        r             : int,
        alpha         : float,
        dropout       : float = 0.0,
    ):
        super().__init__()
        self.frozen = frozen_linear
        self.r      = r
        self.scaling = alpha / r

        d_in  = frozen_linear.in_features
        d_out = frozen_linear.out_features

        # Freeze original weights
        for p in self.frozen.parameters():
            p.requires_grad = False

        # Trainable LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))

        # A ~ N(0, 1/sqrt(d_in)), B = 0 (identity init)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Match device and dtype of frozen layer
        device = frozen_linear.weight.device
        dtype  = frozen_linear.weight.dtype
        self.lora_A = nn.Parameter(self.lora_A.to(device=device, dtype=dtype))
        self.lora_B = nn.Parameter(self.lora_B.to(device=device, dtype=dtype))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x      = x.to(self.frozen.weight.dtype)
        base   = self.frozen(x)
        lora   = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base + lora

    def param_count(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()


def apply_lora_to_model(model, cfg, target_modules=("q_proj","k_proj","v_proj","o_proj")):
    """
    Walk all named modules in model. Replace any nn.Linear whose name
    ends with a target_module string with a LoRALinear wrapper.
    Returns count of replaced modules.
    """
    replaced = 0
    for name, module in model.named_modules():
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                # Navigate to parent and replace
                parts  = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                lora_layer = LoRALinear(
                    module,
                    r       = cfg.lora.r,
                    alpha   = cfg.lora.alpha,
                    dropout = cfg.lora.dropout,
                )
                setattr(parent, parts[-1], lora_layer)
                replaced += 1
                break
    return replaced
