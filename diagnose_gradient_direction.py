"""
diagnose_gradient_direction.py
================================
Tests gradient/weight DIRECTION stability, the one hypothesis not yet
ruled out. Magnitude is confirmed large (diagnose_gradients.py); this
checks whether that gradient reliably points somewhere USEFUL, or is
directionally noisy/arbitrary despite its size.

Two measurements, per layer, over a short real training run:

(1) Alignment with the "ideal" separating direction. probe_signal.py
    already showed tool_call vs planning is perfectly linearly separable
    in the FROZEN backbone's hidden states. Here we fit that same probe
    once (giving each layer's ideal separating direction, as a logistic-
    regression coefficient vector) and then track cosine similarity
    between the router's CURRENT weight vector and that ideal direction
    across training. Rising similarity = training is converging toward
    the direction that would actually solve step-type separation. Flat/
    near-zero = the weight vector is moving, but not toward that.
    Caveat: the probe's "ideal direction" is computed on frozen-backbone
    features; the router's actual input during training is partially
    LoRA-adapted and partially gated, so this is an approximation, not
    an exact target -- stated explicitly rather than overclaimed.

(2) Step-to-step gradient direction noise. Cosine similarity between
    consecutive steps' weight-gradient vectors, per layer. Low/unstable
    similarity means the gradient's DIRECTION changes a lot step to
    step even though its magnitude is large (matches the STE-near-
    threshold noise hypothesis); high similarity means gradient
    direction is at least locally consistent.

Usage:
    python diagnose_gradient_direction.py --steps 200 --ablation none --seed 2024
    python diagnose_gradient_direction.py --steps 200 --ablation none --seed 42
"""
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.config import SystemConfig, LoRAConfig, RouterConfig, TrainingConfig, DataConfig, QWEN_SPEC
from models.gated_qwen import GatedQwenLoRA
from data.loader import build_dataloaders


def compute_ideal_directions(n_eval=100, seed=2024, max_seq_len=512):
    """Fits one logistic-regression probe per layer on the FROZEN backbone's
    hidden states (same method as probe_signal.py), returns each layer's
    normalized coefficient vector as the 'ideal' separating direction."""
    print("[Ideal direction] Loading raw backbone for probe fitting...")
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_SPEC["hf_name"], torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(QWEN_SPEC["hf_name"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    import evaluate as ev
    samples = ev.load_eval_samples(n=n_eval, seed=seed)

    n_layers = QWEN_SPEC["n_layers"]
    hidden_by_layer = [[] for _ in range(n_layers + 1)]
    labels = []
    with torch.no_grad():
        for turns, step_type in samples:
            prompt = tok.apply_chat_template(turns, tokenize=False, add_generation_prompt=False)
            ids = tok(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len).input_ids.to(model.device)
            out = model(ids, output_hidden_states=True)
            for i, h in enumerate(out.hidden_states):
                hidden_by_layer[i].append(h.mean(dim=1).squeeze(0).float().cpu().numpy())
            labels.append(1 if step_type == "tool_call" else 0)  # match router.py convention below

    labels = np.array(labels)
    ideal_dirs = []
    for i in range(n_layers):
        # FIX: router index i's actual input is hidden_states[i] (the hidden
        # state ENTERING layer i, i.e. embedding output for i=0, output of
        # layer i-1 for i>0) -- confirmed by tracing gated_qwen.py's loop
        # precisely (h_mean is computed on `hidden` BEFORE layer i is
        # called). An earlier version of this script used [i+1], which
        # would have silently misaligned every layer's "ideal direction"
        # with the wrong router by one position.
        X = np.stack(hidden_by_layer[i])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X, labels)
        w = clf.coef_[0]
        w = w / (np.linalg.norm(w) + 1e-12)
        ideal_dirs.append(w)

    del model
    torch.cuda.empty_cache()
    return ideal_dirs


def cosine(a, b):
    a = a.flatten(); b = b.flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--ablation", default="none", choices=["none", "bce_naive", "bce_corrected"])
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--max_train_samples", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=10)
    args = p.parse_args()

    ideal_dirs = compute_ideal_directions(seed=args.seed)

    torch.manual_seed(args.seed)
    cfg = SystemConfig(
        lora=LoRAConfig(r=8, alpha=16.0),
        router=RouterConfig(init_bias_early=1.0, init_bias_middle=-1.0, threshold=0.5, gate_reg_weight=1.0),
        training=TrainingConfig(lr=2e-4, batch_size=4, grad_accum=4, max_steps=args.steps, ablation=args.ablation),
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
    n_accum = cfg.training.grad_accum

    align_history = [[] for _ in range(n_layers)]   # cosine(weight, ideal_dir) over time
    prev_grad = [None] * n_layers
    dirnoise_history = [[] for _ in range(n_layers)]  # cosine(grad_t, grad_{t-1})

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
            bce_loss = torch.stack([
                torch.nn.functional.binary_cross_entropy(sg, bce_target)
                for sg in gate_stats["soft_gates_per_sample"]
            ]).mean()
            loss = lm_loss + cfg.router.gate_reg_weight * bce_loss
        else:
            _, loss, gate_stats = model(ids, labels=labels)

        (loss / n_accum).backward()

        if (step + 1) % args.log_every == 0:
            with torch.no_grad():
                for i, router in enumerate(model.routers.routers):
                    w_vec = router.linear.weight.detach().flatten().float().cpu().numpy()
                    align = cosine(w_vec, ideal_dirs[i])
                    align_history[i].append(align)

                    g = router.linear.weight.grad
                    if g is not None:
                        g_np = g.detach().flatten().float().cpu().numpy()
                        if prev_grad[i] is not None:
                            dirnoise_history[i].append(cosine(g_np, prev_grad[i]))
                        prev_grad[i] = g_np.copy()

        if (step + 1) % n_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % 50 == 0:
            print(f"  step {step+1}/{args.steps}  loss={loss.item():.4f}  skip%={gate_stats['skip_pct']}")

    print(f"\n{'layer':<8}{'mean_align_w_ideal':<22}{'final_align':<14}{'mean_grad_dir_stability':<26}")
    print("-" * 70)
    for i in range(n_layers):
        mean_align = np.mean(align_history[i]) if align_history[i] else float("nan")
        final_align = align_history[i][-1] if align_history[i] else float("nan")
        mean_stability = np.mean(dirnoise_history[i]) if dirnoise_history[i] else float("nan")
        print(f"{i:<8}{mean_align:<22.4f}{final_align:<14.4f}{mean_stability:<26.4f}")

    all_align = [v for lst in align_history for v in lst]
    all_stab = [v for lst in dirnoise_history for v in lst]
    print(f"\nOverall mean alignment with ideal direction: {np.mean(all_align):.4f} "
         f"(near 0 = uncorrelated with useful signal; near +/-1 = strongly aligned/anti-aligned)")
    print(f"Overall mean step-to-step gradient direction stability: {np.mean(all_stab):.4f} "
         f"(near 0 = direction changes randomly step-to-step; near 1 = consistent direction)")


if __name__ == "__main__":
    main()
