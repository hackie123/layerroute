"""
check_router_triviality.py
=============================
Tests whether the router's per-sample adaptivity is REAL or TRIVIAL.

gate_values.csv (evaluate.py) only ever reports sigmoid(bias) -- the
BIAS-ONLY component, by its own docstring ("Expected gate value from bias
alone"). It has never told us whether the WEIGHT term (w_i . h_mean, the
only part that can differ per input) actually moves the gate meaningfully,
or whether it's negligible relative to bias -- i.e. whether the router is
architecturally adaptive (it is, by construction) but FUNCTIONALLY closer
to a fixed pattern (if weight's contribution is small) or genuinely
input-varying (if weight's contribution is substantial).

For each layer, over many held-out samples, reports:
  - bias contribution (constant per layer, same for every sample)
  - weight contribution (w_i . h_mean, varies per sample)
  - the REAL per-sample soft gate sigmoid(bias + weight_contribution)
  - std of the real gate across samples (near 0 = functionally fixed
    regardless of input; larger = genuine per-sample variation)
  - fraction of samples whose HARD decision (gate>0.5) would flip if
    weight's contribution were zeroed out (bias alone) -- directly answers
    "does the weight term ever change the actual skip/run decision, or is
    bias alone already deciding everything?"

Usage:
    python check_router_triviality.py --ckpt checkpoints_seed2024/best_adapters.pt
"""
import argparse
import numpy as np
import torch

from utils.config import SystemConfig, LoRAConfig, RouterConfig, TrainingConfig, QWEN_SPEC
from models.gated_qwen import GatedQwenLoRA
from transformers import AutoTokenizer


def build_model(ckpt, lora_r=8, lora_alpha=16.0, router_threshold=0.5):
    cfg = SystemConfig(
        lora=LoRAConfig(r=lora_r, alpha=lora_alpha),
        router=RouterConfig(threshold=router_threshold),
        training=TrainingConfig(dtype="bfloat16"),
    )
    model = GatedQwenLoRA.from_pretrained(cfg)
    model.load_adapters(ckpt)
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n_eval", type=int, default=100)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--max_seq_len", type=int, default=512)
    args = p.parse_args()

    model = build_model(args.ckpt)
    device = next(model.hf_model.parameters()).device
    tok = AutoTokenizer.from_pretrained(QWEN_SPEC["hf_name"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    import evaluate as ev
    samples = ev.load_eval_samples(n=args.n_eval, seed=args.seed)

    n_layers = model.n_layers
    # Per layer: bias contribution (scalar), and per-sample weight
    # contribution + real gate value.
    bias_contrib   = [None] * n_layers
    weight_contrib = [[] for _ in range(n_layers)]
    real_gate      = [[] for _ in range(n_layers)]
    flips          = [0] * n_layers

    transformer = model.hf_model.model

    with torch.no_grad():
        for turns, step_type in samples:
            prompt = tok.apply_chat_template(turns, tokenize=False, add_generation_prompt=False)
            ids = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_seq_len).input_ids.to(device)

            hidden = transformer.embed_tokens(ids)
            T = hidden.shape[1]
            position_ids = torch.arange(T, device=device).unsqueeze(0)
            cos, sin = transformer.rotary_emb(hidden, position_ids)
            layer_kwargs = dict(attention_mask=None, position_ids=position_ids,
                                past_key_values=None, use_cache=False,
                                position_embeddings=(cos, sin))

            for i, layer in enumerate(transformer.layers):
                router = model.routers.routers[i]
                h_mean = hidden.mean(dim=1).to(router.linear.weight.dtype)

                bias_val = router.linear.bias.item()
                w_contrib = (router.linear.weight @ h_mean.T).item()  # scalar: w . h_mean
                real_score = bias_val + w_contrib
                real_g = torch.sigmoid(torch.tensor(real_score)).item()
                bias_only_g = torch.sigmoid(torch.tensor(bias_val)).item()

                bias_contrib[i] = bias_val
                weight_contrib[i].append(w_contrib)
                real_gate[i].append(real_g)

                if (real_g > 0.5) != (bias_only_g > 0.5):
                    flips[i] += 1

                # Advance hidden using the REAL (hard) decision, matching
                # actual inference behavior, so later layers see realistic input.
                hard_g = 1.0 if real_g > router.threshold else 0.0
                if hard_g > 0.5:
                    out = layer(hidden, **layer_kwargs)
                    hidden = (out[0] if isinstance(out, tuple) else out).to(hidden.dtype)
                # else: hidden unchanged (skip)

    n_samples = len(samples)
    print(f"\n{'layer':<8}{'bias':<10}{'mean|w.h|':<12}{'std(real_gate)':<16}{'flips_vs_bias_only':<20}")
    print("-" * 66)
    for i in range(n_layers):
        wc = np.array(weight_contrib[i])
        rg = np.array(real_gate[i])
        print(f"{i:<8}{bias_contrib[i]:<10.3f}{np.mean(np.abs(wc)):<12.4f}"
             f"{np.std(rg):<16.4f}{flips[i]:<6d}/{n_samples:<14d}")

    all_std = [np.std(real_gate[i]) for i in range(n_layers)]
    all_flip_frac = [flips[i] / n_samples for i in range(n_layers)]
    print(f"\nMean std(real_gate) across layers: {np.mean(all_std):.4f} "
         f"(near 0 = gate functionally CONSTANT regardless of input at every layer)")
    print(f"Mean flip fraction (weight changes the hard decision): {np.mean(all_flip_frac):.4f} "
         f"(0 = weight NEVER changes skip/run decision anywhere -- router is functionally fixed, "
         f"despite being architecturally adaptive)")


if __name__ == "__main__":
    main()
