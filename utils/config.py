"""
utils/config.py
================
ConfGate v6 — LoRA + Per-Layer Hard-Gated Skip Connections.

Architecture:
    For each of 24 transformer blocks:
        router_i   : Linear(hidden, 1) → sigmoid → STE → hard gate gᵢ ∈ {0,1}
        if gᵢ = 1  : h = TransformerBlock_i(h)   with LoRA adapters
        if gᵢ = 0  : h = h                        skip entire block

    Trainable:
        - 24 router linears  (~21K params)
        - LoRA adapters on Q,K,V,O projections  (~3.6M params for r=8)

    Frozen:
        - All original Qwen weights (494M)

Training:
    Single pass, single loss:
        loss = CrossEntropy(lm_logits, next_token_labels)
    Data:
        Hermes / Glaive  → tool_call sequences (contain <tool_call> token)
        GSM8K / Turing   → planning / reasoning sequences
    The model learns which blocks to skip via the natural LM signal.
    No binary classification head. No two-stage training.

Straight-Through Estimator (STE):
    Forward : hard {0,1} gate
    Backward: gradient flows through sigmoid as if it were continuous
    Standard trick for discrete gating — used in MoD, VQ-VAE, etc.

Qwen2.5-0.5B specs:
    num_hidden_layers  : 24
    hidden_size        : 896
    num_q_heads        : 14
    num_kv_heads       : 2
    intermediate_size  : 4864
    vocab_size         : 151936
"""

from dataclasses import dataclass, field


QWEN_SPEC = {
    "hf_name"     : "Qwen/Qwen2.5-0.5B-Instruct",
    "n_layers"    : 24,
    "n_q_heads"   : 14,
    "n_kv_heads"  : 2,
    "hidden"      : 896,
    "head_dim"    : 64,
    "intermediate": 4864,
    "vocab"       : 151936,
}


@dataclass
class LoRAConfig:
    r           : int   = 8         # LoRA rank
    alpha       : float = 16.0      # LoRA scaling = alpha / r
    dropout     : float = 0.05
    # Which projections to apply LoRA to
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass
class RouterConfig:
    # Per-layer router: Linear(hidden, 1) → sigmoid → STE
    # init_bias is per-layer: early/late layers start open (1.0),
    # middle layers start closed (-1.0) to break symmetry immediately
    init_bias_early : float = 1.0    # layers 0-7, 17-23: sigmoid(1.0)=0.73
    init_bias_middle: float = -1.0   # layers 8-16: sigmoid(-1.0)=0.27 — below threshold
    threshold       : float = 0.5    # hard gate threshold
    # Gate regularisation: loss += gate_reg_weight * mean(soft_gates)
    # Penalises uniformly high gates — forces router to find skippable layers
    gate_reg_weight : float = 1.0    # increased from 0.05 — more aggressive skipping


@dataclass
class TrainingConfig:
    lr              : float = 2e-4   # standard LoRA fine-tuning LR
    batch_size      : int   = 4
    grad_accum      : int   = 4      # effective batch = 16
    max_steps       : int   = 1000
    warmup_steps    : int   = 100
    grad_clip       : float = 1.0
    log_every       : int   = 50
    eval_every      : int   = 200
    save_every      : int   = 500
    output_dir      : str   = "./checkpoints"
    dtype           : str   = "bfloat16"
    max_seq_len     : int   = 512
    ablation        : str   = "none"  # "none" | "bce_naive" | "bce_corrected"
                                       # -- see trainer.py for the label-direction
                                       # difference between the two BCE variants


@dataclass
class DataConfig:
    max_train_samples : int  = 5000   # per dataset
    val_split         : float = 0.1
    local_jsonl       : str  = None


@dataclass
class SystemConfig:
    lora    : LoRAConfig    = field(default_factory=LoRAConfig)
    router  : RouterConfig  = field(default_factory=RouterConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data    : DataConfig    = field(default_factory=DataConfig)
