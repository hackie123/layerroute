"""
models/gated_qwen.py
=====================
Qwen2.5-0.5B with:
    1. LoRA adapters on Q,K,V,O projections
    2. Per-layer hard-gated skip connections (STE)
    3. Single LM loss training — no separate classification head

Forward pass (training):
    For each block i:
        gate_i = router_i(h)          ← STE: hard forward, soft backward
        h_out  = Block_i(h)           ← Qwen + LoRA
        h      = gate_i * h_out + (1 - gate_i) * h    ← gated output

    lm_logits = lm_head(norm(h))
    loss      = CrossEntropy(lm_logits, labels)

Forward pass (inference):
    Identical — same hard gate, no mismatch.
    gate_i = 0 → skip block (h unchanged, zero compute)
    gate_i = 1 → run block normally

What each component learns:
    LoRA adapters: improve Qwen's quality on agentic data
    Routers: learn when each block is useful vs redundant
        Tool call sequences → routers learn to skip middle blocks
        Planning sequences  → routers learn to keep all blocks

Trainable params:
    Routers     : 24 × 897  =   21,528
    LoRA (r=8)  : 24 × 4 projections × 2 × (896×8 + 8×896) = ~3.6M
    Total       : ~3.65M  (0.74% of 494M backbone)
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from typing import Optional, Tuple

from models.lora import apply_lora_to_model
from models.router import RouterCollection
from utils.config import SystemConfig, QWEN_SPEC


class GatedQwenLoRA(nn.Module):

    def __init__(self, hf_model, cfg: SystemConfig):
        super().__init__()
        self.cfg      = cfg
        self.hf_model = hf_model
        self.n_layers = QWEN_SPEC["n_layers"]
        self.hidden   = QWEN_SPEC["hidden"]

        # Step 1: Freeze ALL backbone parameters
        for p in self.hf_model.parameters():
            p.requires_grad = False
        n_frozen = sum(p.numel() for p in self.hf_model.parameters())
        print(f"  ✓ Frozen {n_frozen:,} backbone parameters")

        # Step 2: Apply LoRA to Q,K,V,O projections
        n_replaced = apply_lora_to_model(
            self.hf_model, cfg,
            target_modules=cfg.lora.target_modules
        )
        n_lora = sum(
            p.numel() for n, p in self.hf_model.named_parameters()
            if p.requires_grad
        )
        print(f"  ✓ LoRA applied to {n_replaced} projections — {n_lora:,} trainable params")

        # Step 3: Per-layer routers (STE hard gating)
        # Middle layers start below threshold — breaks symmetry immediately
        self.routers = RouterCollection(
            n_layers         = self.n_layers,
            hidden           = self.hidden,
            init_bias_early  = cfg.router.init_bias_early,
            init_bias_middle = cfg.router.init_bias_middle,
            threshold        = cfg.router.threshold,
        )
        print(f"  ✓ RouterCollection: {self.routers.param_count():,} params "
              f"(24 × Linear({self.hidden},1))")
        # Move routers to same device/dtype as backbone
        device = next(self.hf_model.parameters()).device
        self.routers = self.routers.to(device)

    @classmethod
    def from_pretrained(cls, cfg: SystemConfig) -> "GatedQwenLoRA":
        hf_name   = QWEN_SPEC["hf_name"]
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                     "float32": torch.float32}
        dtype = dtype_map[cfg.training.dtype]
        print(f"  Loading '{hf_name}'...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            hf_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )
        print(f"  ✓ Pretrained weights loaded")
        return cls(hf_model, cfg)

    # ── Manual layer loop ──────────────────────────────────────────

    def _forward_layers(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Manual forward through all 24 blocks with hard-gated skip connections.
        Returns: logits [B, T, vocab], gate_stats dict
        """
        transformer = self.hf_model.model
        device      = input_ids.device
        T           = input_ids.shape[1]

        hidden              = transformer.embed_tokens(input_ids)
        model_dtype         = hidden.dtype   # bfloat16
        position_ids        = torch.arange(T, device=device).unsqueeze(0)
        cos, sin            = transformer.rotary_emb(hidden, position_ids)
        position_embeddings = (cos, sin)

        layer_kwargs = dict(
            attention_mask      = None,
            position_ids        = position_ids,
            past_key_values     = None,
            use_cache           = False,
            position_embeddings = position_embeddings,
        )

        gate_values         = []
        soft_gates          = []   # kept for gate regularisation (has grad), batch-mean per layer
        soft_gates_per_sample = []  # [L] list of [B] tensors -- ADDED for the BCE ablation,
                                     # which needs per-SAMPLE gate scores against per-sample
                                     # provenance labels. Not used by the main LM+gate-reg path;
                                     # purely additive, existing behavior unchanged.
        layers_run          = 0

        for i, layer in enumerate(transformer.layers):
            # Router: soft gate (for reg grad) + hard STE gate (for forward)
            router   = self.routers.routers[i]
            h_mean   = hidden.mean(dim=1).to(router.linear.weight.dtype)
            score    = router.linear(h_mean).squeeze(-1)      # [B]
            soft_g   = torch.sigmoid(score)                   # [B] — has grad
            hard_g   = (soft_g > router.threshold).float()
            gate     = (hard_g - soft_g.detach() + soft_g)    # STE [B]
            gate     = gate.unsqueeze(-1).unsqueeze(-1)        # [B,1,1]

            soft_gates.append(soft_g.mean())   # scalar, differentiable
            soft_gates_per_sample.append(soft_g)   # [B], differentiable, per-sample

            # Stats (no grad) -- recorded BEFORE the skip decision below, so
            # gate_values always has exactly n_layers entries regardless of
            # whether this layer's compute is skipped.
            with torch.no_grad():
                g_val = gate.mean().item()
            gate_values.append(round(g_val, 4))
            layers_run += (gate.mean() > 0.5).float().item()

            # Inference-time compute skip: at training time the block must
            # still be run unconditionally so the STE backward pass has a
            # real forward value to connect gradient through (multiplying
            # by gate=0 discards the OUTPUT, but the layer's own forward
            # computation still has to exist for backprop). At inference,
            # no gradient is needed, so a closed gate can skip the block's
            # computation entirely rather than computing it and discarding
            # the result -- this is the actual FLOP saving the paper's
            # skip-differential and FLOPs-reduction numbers describe.
            if (not self.training) and gate.mean().item() < 0.5:
                continue

            # Run the block (LoRA adapters active inside)
            out    = layer(hidden, **layer_kwargs)
            h_out  = out[0] if isinstance(out, tuple) else out

            # Gated skip: h = gate * block_output + (1-gate) * h
            hidden = (gate * h_out + (1.0 - gate) * hidden).to(model_dtype)

        hidden  = transformer.norm(hidden)
        logits  = self.hf_model.lm_head(hidden)

        return logits, soft_gates, {
            "gate_values" : gate_values,
            "layers_run"  : layers_run,
            "skip_pct"    : round(100 * (self.n_layers - layers_run) / self.n_layers, 1),
            "soft_gates_per_sample": soft_gates_per_sample,  # ADDED: for BCE ablation
        }

    # ── Forward ────────────────────────────────────────────────────

    def forward(
        self,
        input_ids : torch.Tensor,
        labels    : Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """
        Returns: logits, loss (if labels provided), gate_stats
        """
        logits, soft_gates, gate_stats = self._forward_layers(input_ids)

        loss = None
        if labels is not None:
            # Standard LM loss — shift by 1
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            lm_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, QWEN_SPEC["vocab"]),
                shift_labels.view(-1),
                ignore_index = -100,
            )
            # Gate regularisation: penalise uniformly high gates
            # Encourages router to close gates on skippable layers
            # loss += lambda * mean(soft_gate_values)
            gate_reg = torch.stack(soft_gates).mean()
            loss = lm_loss + self.cfg.router.gate_reg_weight * gate_reg

        return logits, loss, gate_stats

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save_adapters(self, path: str):
        """Save only trainable params (LoRA + routers) — not full model."""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {k: v for k, v in self.state_dict().items()
                 if any(x in k for x in ["lora_A", "lora_B", "routers"])}
        torch.save(state, path)
        print(f"  ✓ Adapters saved → {path}  ({len(state)} tensors)")

    def load_adapters(self, path: str):
        state = torch.load(path, map_location="cpu", weights_only=True)
        missing, unexpected = self.load_state_dict(state, strict=False)
        print(f"  ✓ Adapters loaded ← {path}")
        if unexpected:
            print(f"  ⚠ Unexpected keys: {unexpected[:3]}")
