"""
models/router.py
=================
Per-layer router: decides whether to run or skip each transformer block.

Router_i:
    input  : h ∈ ℝ^(B, T, hidden)  — hidden state entering block i
    output : gᵢ ∈ {0, 1}^(B, T)    — hard gate per token per block

Mechanism:
    score  = Linear_i(h).squeeze(-1)   → ℝ^(B, T)
    soft_g = sigmoid(score)             → (0, 1)
    hard_g = (soft_g > threshold).float()

Straight-Through Estimator (STE):
    Forward : uses hard_g  (discrete, no gradient through threshold)
    Backward: gradient flows through soft_g  (continuous)
    Implementation: gate = hard_g + soft_g - soft_g.detach()
    This is equivalent to: gate = soft_g + (hard_g - soft_g).detach()

At inference:
    Same hard gate — no train/inference mismatch.
    If gᵢ=0 for all tokens in sequence → skip block entirely (zero compute).
    If gᵢ=1 for any token → run block for those tokens.

Note: We use sequence-level gate (mean over T) for simplicity:
    gate = (soft_g.mean(dim=1) > threshold)  — one decision per sequence
    This is cleaner for agentic use: entire step takes cheap or expensive path.
    Token-level gating is possible but more complex (requires padding masks).
"""

import torch
import torch.nn as nn


class LayerRouter(nn.Module):
    """
    Lightweight per-layer router: Linear(hidden, 1) → STE gate.
    ~897 trainable params per layer.
    """

    def __init__(self, hidden: int, init_bias: float = 1.0, threshold: float = 0.5):
        super().__init__()
        self.linear    = nn.Linear(hidden, 1)
        self.threshold = threshold

        # Init bias so sigmoid(bias) ≈ 0.73 — start mostly running layers
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, init_bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, T, hidden]
        Returns:
            gate: [B, 1, 1]  — sequence-level hard gate, broadcastable
        """
        # Pool over sequence → single score per sample
        h_mean  = h.mean(dim=1)                          # [B, hidden]
        score   = self.linear(h_mean).squeeze(-1)        # [B]
        soft_g  = torch.sigmoid(score)                   # [B]  ∈ (0,1)

        # Hard gate (forward)
        hard_g  = (soft_g > self.threshold).float()      # [B]  ∈ {0,1}

        # STE: forward=hard, backward through soft
        gate    = hard_g - soft_g.detach() + soft_g      # [B]

        # Reshape for broadcasting with [B, T, hidden]
        return gate.unsqueeze(-1).unsqueeze(-1)           # [B, 1, 1]

    def gate_value(self) -> float:
        """Expected gate value from bias alone (for monitoring)."""
        return torch.sigmoid(self.linear.bias).item()


class RouterCollection(nn.Module):
    """One router per transformer layer.
    Middle layers initialised with lower bias — starts below threshold.
    This breaks symmetry so middle layers begin skipping immediately,
    allowing the LM loss gradient to teach layer importance from step 1.
    """

    def __init__(self, n_layers: int, hidden: int,
                 init_bias_early: float, init_bias_middle: float,
                 threshold: float, middle_start: int = 8, middle_end: int = 17):
        super().__init__()
        self.routers = nn.ModuleList([
            LayerRouter(
                hidden,
                init_bias = init_bias_middle if middle_start <= i < middle_end
                            else init_bias_early,
                threshold = threshold
            )
            for i in range(n_layers)
        ])

    def gate_values(self) -> list:
        """Current soft gate values for all layers — for logging."""
        return [r.gate_value() for r in self.routers]

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
