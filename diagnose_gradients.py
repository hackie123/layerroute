"""
diagnose_gradients.py
=======================
Tests whether the router's weight vector w_i receives systematically
weaker gradient signal than its bias term, which would explain why bias
converges cleanly and identically across every training variant we've
tried (implicit gate-reg, bce_naive, bce_corrected -- all converge to the
same ~0.27/~0.73 bias structure) while the weight direction (the only
part of the router that COULD encode step-type, since bias is uniform
across all inputs) stays effectively noise-dominated regardless of the
training signal.

Runs REAL training (same optimizer/scheduler setup as Trainer, same
--ablation choices) for a short number of steps, logging
||grad(router.linear.weight)|| and ||grad(router.linear.bias)|| per layer
at every step, before optimizer.step(). Reports the mean ratio per layer
across the run.

Usage:
    python diagnose_gradients.py --steps 200 --ablation none
    python diagnose_gradients.py --steps 200 --ablation bce_corrected
"""
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.config import SystemConfig, LoRAConfig, RouterConfig, TrainingConfig, DataConfig
from models.gated_qwen import GatedQwenLoRA
from data.loader import build_dataloaders


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--ablation", default="none", choices=["none", "bce_naive", "bce_corrected"])
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--max_train_samples", type=int, default=2000,
                   help="Smaller than the full 5000 -- this is a short "
                        "diagnostic run, not a full training reproduction.")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    cfg = SystemConfig(
        lora=LoRAConfig(r=8, alpha=16.0),
        router=RouterConfig(init_bias_early=1.0, init_bias_middle=-1.0,
                            threshold=0.5, gate_reg_weight=1.0),
        training=TrainingConfig(lr=2e-4, batch_size=4, grad_accum=4,
                                max_steps=args.steps, ablation=args.ablation),
        data=DataConfig(max_train_samples=args.max_train_samples),
    )

    print(f"[Model] Loading Qwen2.5-0.5B-Instruct + LoRA + Routers (ablation={args.ablation})...")
    model = GatedQwenLoRA.from_pretrained(cfg)
    device = next(model.parameters()).device

    print("[Data] Building dataloaders...")
    train_l, _ = build_dataloaders(cfg)
    train_iter = iter(train_l)

    trainable = [p_ for p_ in model.parameters() if p_.requires_grad]
    optimizer = optim.AdamW(trainable, lr=cfg.training.lr, weight_decay=0.01, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.max_steps, eta_min=1e-6)

    n_layers = model.n_layers
    weight_norms = [[] for _ in range(n_layers)]
    bias_norms   = [[] for _ in range(n_layers)]

    n_accum = cfg.training.grad_accum
    model.train()

    for step in range(args.steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_l)
            batch = next(train_iter)

        ids        = batch["input_ids"].to(device)
        labels     = batch["labels"].to(device)
        provenance = batch["provenance"].to(device)

        if args.ablation in ("bce_naive", "bce_corrected"):
            logits, soft_gates, gate_stats = model._forward_layers(ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            lm_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, logits.shape[-1]), shift_labels.view(-1), ignore_index=-100)
            bce_target = provenance if args.ablation == "bce_naive" else (1.0 - provenance)
            per_layer_soft = gate_stats["soft_gates_per_sample"]
            bce_loss = torch.stack([
                torch.nn.functional.binary_cross_entropy(sg, bce_target) for sg in per_layer_soft
            ]).mean()
            loss = lm_loss + cfg.router.gate_reg_weight * bce_loss
        else:
            _, loss, gate_stats = model(ids, labels=labels)

        (loss / n_accum).backward()

        # Log gradient norms BEFORE optimizer.step() clears/updates them
        for i, router in enumerate(model.routers.routers):
            w_grad = router.linear.weight.grad
            b_grad = router.linear.bias.grad
            weight_norms[i].append(w_grad.norm().item() if w_grad is not None else 0.0)
            bias_norms[i].append(b_grad.norm().item() if b_grad is not None else 0.0)

        if (step + 1) % n_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % 50 == 0:
            print(f"  step {step+1}/{args.steps}  loss={loss.item():.4f}  skip%={gate_stats['skip_pct']}")

    print(f"\n{'layer':<8}{'mean||grad_w||':<18}{'mean||grad_bias||':<20}{'ratio (w/bias)':<16}")
    print("-" * 62)
    ratios = []
    for i in range(n_layers):
        mw = np.mean(weight_norms[i])
        mb = np.mean(bias_norms[i])
        ratio = mw / mb if mb > 1e-12 else float("inf")
        ratios.append(ratio)
        print(f"{i:<8}{mw:<18.6f}{mb:<20.6f}{ratio:<16.4f}")

    print(f"\nMean ratio across all layers: {np.mean(ratios):.4f}")
    print("(ratio << 1 means weight gets much WEAKER gradient signal than bias --")
    print(" consistent with bias reliably converging while weight direction stays noisy)")


if __name__ == "__main__":
    main()
