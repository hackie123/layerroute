"""
main.py — ConfGate v6
======================
LoRA fine-tuning + per-layer hard-gated skip connections on Qwen2.5-0.5B.

Architecture:
    For each of 24 transformer blocks:
        router_i(h) → STE gate gᵢ ∈ {0,1}
        h = gᵢ * Block_i(h) + (1-gᵢ) * h

    Block_i contains Qwen weights + LoRA adapters.
    Single LM loss — no separate classification head.

Trainable:
    LoRA adapters  : ~3.6M params  (r=8, Q/K/V/O projections)
    Routers        : ~21K params   (24 × Linear(896,1))
    Total          : ~3.65M  (0.74% of 494M backbone)

Workflow:
    python main.py --mode train
    python main.py --mode demo --ckpt checkpoints/best_adapters.pt

CSV outputs (checkpoints/):
    training_log.csv  — loss, ppl, layers_run, skip_pct per step
    eval_log.csv      — val_loss, val_ppl, val_layers_run
    gate_log.csv      — per-layer gate values every log_every steps
    run_summary.csv   — one row per run
"""

import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F

from utils.config import (SystemConfig, LoRAConfig, RouterConfig,
                           TrainingConfig, DataConfig, QWEN_SPEC)
from models.gated_qwen import GatedQwenLoRA


def set_seed(seed: int):
    """Seeds everything that affects training reproducibility: LoRA/router
    weight init, dropout, and DataLoader batch shuffling order (which uses
    torch's global RNG, since no explicit generator= is passed -- confirmed
    by inspecting data/loader.py). Only the train/val SPLIT was previously
    seeded (fixed at 42, unrelated to this); nothing else was, meaning every
    prior training run was genuinely non-reproducible.
    cudnn.deterministic=True trades some training speed for exact
    reproducibility -- worth it here since the whole point is comparing
    seeds precisely."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"  \u2713 Seed set: {seed} (torch, numpy, random, cudnn.deterministic=True)")


def build_config(args) -> SystemConfig:
    return SystemConfig(
        lora     = LoRAConfig(
            r              = args.lora_r,
            alpha          = args.lora_alpha,
            dropout        = args.lora_dropout,
        ),
        router   = RouterConfig(
            init_bias_early  = args.router_init_early,
            init_bias_middle = args.router_init_middle,
            threshold        = args.router_threshold,
            gate_reg_weight  = args.gate_reg_weight,
        ),
        training = TrainingConfig(
            lr          = args.lr,
            batch_size  = args.batch_size,
            grad_accum  = args.grad_accum,
            max_steps   = args.max_steps,
            output_dir  = args.output_dir,
            max_seq_len = args.max_seq_len,
            ablation    = args.ablation,
        ),
        data     = DataConfig(
            max_train_samples = args.max_train_samples,
        ),
    )


def run_train(args):
    set_seed(args.seed)
    cfg   = build_config(args)

    print("\n[Model] Loading Qwen2.5-0.5B-Instruct + LoRA + Routers...")
    model = GatedQwenLoRA.from_pretrained(cfg)
    print(f"  Trainable : {model.trainable_params():,}")
    print(f"  Total     : {model.total_params():,}")

    from data.loader import build_dataloaders
    from utils.trainer import Trainer

    print("\n[Data] Building dataloaders...")
    train_l, val_l = build_dataloaders(cfg)

    trainer = Trainer(model, cfg)
    trainer.train(train_l, val_l)

    # Auto-run demo after training
    args.ckpt = f"{args.output_dir}/best_adapters.pt"
    run_demo(args)


def run_demo(args):
    from transformers import AutoTokenizer
    cfg   = build_config(args)
    model = GatedQwenLoRA.from_pretrained(cfg)

    if args.ckpt:
        model.load_adapters(args.ckpt)

    device = next(p for p in model.hf_model.parameters()).device
    model.routers = model.routers.to(device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(QWEN_SPEC["hf_name"], trust_remote_code=True)

    samples = [
        ("<tool_call> search_database(table='AUFK', filter='AUFNR=100023')",
         "TOOL CALL"),
        ("<tool_call> get_work_order_status(id=78234, priority=True)",
         "TOOL CALL"),
        ("Given the current backlog of 450 open work orders across 3 plants, "
         "develop a prioritization strategy that balances equipment criticality "
         "and crew capacity over the next 2 weeks.",
         "PLANNING"),
        ("Analyze the root cause of recurring pump failures in Plant B and "
         "propose a preventive maintenance schedule.",
         "PLANNING"),
    ]

    print("\n" + "═"*55)
    print("  CONFGATE v6 DEMO")
    print("  LoRA + Hard-Gated Skip Connections")
    print("═"*55)

    for text, label in samples:
        prompt = tok.apply_chat_template(
            [{"role":"user","content":text}],
            tokenize=False, add_generation_prompt=True
        )
        device = next(model.parameters()).device
        ids = tok.encode(prompt, return_tensors="pt",
                         max_length=cfg.training.max_seq_len, truncation=True).to(device)

        print(f"\n[{label}] {repr(text[:65])}")
        with torch.no_grad():
            logits, _, gate_stats = model(ids)

        print(f"  Layers run : {gate_stats['layers_run']:.0f}/24  "
              f"(skip {gate_stats['skip_pct']:.1f}%)")
        print(f"  Gate values: {gate_stats['gate_values']}")

        probs = F.softmax(logits[0, -1, :], dim=-1)
        top5  = torch.topk(probs, 5)
        print("  Top-5 next tokens:")
        for p, idx in zip(top5.values, top5.indices):
            print(f"    {repr(tok.decode([idx.item()])):15s} p={p.item():.4f}")

    # Gate analysis
    print("\n" + "═"*55)
    print("  GATE VALUES (lower = layer more often skipped)")
    gvals = model.routers.gate_values()
    for i, g in enumerate(gvals):
        bar = "█" * int(g * 20)
        print(f"  Layer {i:2d}: {g:.4f}  {bar}")


def parse_args():
    p = argparse.ArgumentParser(description="ConfGate v6 — LoRA + Skip Gates")
    p.add_argument("--mode",           default="demo",
                   choices=["train","demo"])

    # LoRA
    p.add_argument("--lora_r",         type=int,   default=8)
    p.add_argument("--lora_alpha",     type=float, default=16.0)
    p.add_argument("--lora_dropout",   type=float, default=0.05)

    # Router
    p.add_argument("--router_threshold",     type=float, default=0.5)
    p.add_argument("--router_init_early",    type=float, default=1.0,
                   help="Init bias for early/late layers (sigmoid(1.0)=0.73)")
    p.add_argument("--router_init_middle",   type=float, default=-1.0,
                   help="Init bias for middle layers 8-16 (sigmoid(-1.0)=0.27)")
    p.add_argument("--gate_reg_weight",      type=float, default=1.0,
                   help="Gate regularisation weight. Higher=more aggressive skipping.")

    # Training
    p.add_argument("--lr",             type=float, default=2e-4)
    p.add_argument("--batch_size",     type=int,   default=4)
    p.add_argument("--grad_accum",     type=int,   default=4)
    p.add_argument("--max_steps",      type=int,   default=1000)
    p.add_argument("--max_seq_len",    type=int,   default=512)
    p.add_argument("--output_dir",     default="./checkpoints")
    p.add_argument("--seed",           type=int,   default=2024,
                   help="Seeds torch/numpy/random + cudnn.deterministic for "
                        "reproducible training (weight init, dropout, batch "
                        "shuffling order). Independent of the fixed 42 used "
                        "for the train/val split.")
    p.add_argument("--ablation",       default="none",
                   choices=["none", "bce_naive", "bce_corrected"],
                   help="none: standard LM+gate-reg objective (default). "
                        "bce_naive: replaces gate-reg with BCE against "
                        "provenance labels directly (tool_call=1 trains gate "
                        "toward OPEN -- the literal reading of the paper's "
                        "Table 2 ablation description). bce_corrected: same "
                        "but with the label flipped (tool_call=1 trains gate "
                        "toward CLOSED/skip -- the direction actually "
                        "consistent with the paper's stated goal). See "
                        "utils/trainer.py for the full rationale.")
    p.add_argument("--max_train_samples", type=int, default=5000)

    # Inference
    p.add_argument("--ckpt",           default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 55)
    print("  CONFGATE v6 — LoRA + Hard-Gated Skip Connections")
    print("=" * 55)
    print(f"  mode         : {args.mode}")
    print(f"  LoRA r       : {args.lora_r}  alpha={args.lora_alpha}")
    print(f"  Router thresh: {args.router_threshold}")
    print(f"  Max steps    : {args.max_steps}")

    if args.mode == "train":
        run_train(args)
    else:
        run_demo(args)
