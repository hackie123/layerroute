"""
time_wall_clock.py
====================
Genuine wall-clock measurement, added because the original paper (and this
codebase's evaluate.py) reports only theoretical FLOPs reduction, never
measured time. This script measures real generation latency for:
  (1) Routed  -- trained gates active, genuine inference-time skip (the fix
      in gated_qwen.py -- verified to produce fewer real layer() calls, not
      just a smaller theoretical FLOPs number).
  (2) Baseline -- every gate forced open via direct bias manipulation on
      each router's linear.bias (same technique verified elsewhere: saves
      the exact original bias, overwrites so sigmoid saturates above
      threshold for every input, restores after timing). This model has no
      force_open flag at all, so this is the only way to get a genuine
      full-model baseline pass.

Generation is manual, token-by-token, through model(ids) directly -- NOT
.generate() (the model class has no such method; this matches how
evaluate.py and main.py's run_demo already call it, the only path that
exercises real gating).

Usage:
    python time_wall_clock.py --ckpt checkpoints_seed2024/best_adapters.pt \
        --seed 2024 --n_eval 30 --out_dir checkpoints_seed2024/paper_results
"""
import argparse
import csv
import json
import os
import random
import time

import torch
from transformers import AutoTokenizer

from utils.config import SystemConfig, LoRAConfig, RouterConfig, TrainingConfig, QWEN_SPEC
from models.gated_qwen import GatedQwenLoRA


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_model(ckpt, lora_r, lora_alpha, router_threshold):
    cfg = SystemConfig(
        lora=LoRAConfig(r=lora_r, alpha=lora_alpha),
        router=RouterConfig(threshold=router_threshold),
        training=TrainingConfig(dtype="bfloat16"),
    )
    model = GatedQwenLoRA.from_pretrained(cfg)
    model.load_adapters(ckpt)
    model.eval()
    return model


def load_eval_samples(n, seed):
    """Same sources/mix as evaluate.py's load_eval_samples, reusing its
    seeded shuffle fix so the sample SET matches what accuracy/PPL numbers
    were computed on for this seed."""
    import evaluate as ev
    return ev.load_eval_samples(n=n, seed=seed)


def time_query(model, tok, prompt, max_new_tokens, max_seq_len):
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len)
    ids = enc["input_ids"].to(next(model.hf_model.parameters()).device)
    prompt_len = ids.shape[1]
    eos_id = tok.eos_token_id

    _sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _, _ = model(ids)
            next_id = logits[0, -1, :].argmax(dim=-1, keepdim=True).unsqueeze(0)
            ids = torch.cat([ids, next_id], dim=1)
            if eos_id is not None and next_id.item() == eos_id:
                break
            if ids.shape[1] - prompt_len >= max_new_tokens:
                break
    _sync()
    ms = (time.perf_counter() - t0) * 1000.0
    new_tokens = ids.shape[1] - prompt_len
    return ms, new_tokens


def force_all_gates_open(model):
    saved = [r.linear.bias.data.clone() for r in model.routers.routers]
    for r in model.routers.routers:
        r.linear.bias.data.fill_(10.0)  # sigmoid(10.0) ~= 1.0, always > threshold
    return saved


def restore_gates(model, saved):
    for r, s in zip(model.routers.routers, saved):
        r.linear.bias.data.copy_(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--n_eval", type=int, default=30,
                   help="Fewer than the 100 used for accuracy/PPL -- ms/token "
                        "stabilizes quickly, and this runs once per seed on "
                        "top of an already-expensive 10-seed sweep.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--router_threshold", type=float, default=0.5)
    p.add_argument("--out_dir", default="checkpoints/paper_results")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = build_model(args.ckpt, args.lora_r, args.lora_alpha, args.router_threshold)
    tok = AutoTokenizer.from_pretrained(QWEN_SPEC["hf_name"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    samples = load_eval_samples(n=args.n_eval, seed=args.seed)

    records = []
    identical_count = 0
    for i, (turns, step_type) in enumerate(samples):
        prompt = tok.apply_chat_template(turns[:1], tokenize=False, add_generation_prompt=True)

        routed_ms, routed_tok = time_query(model, tok, prompt, args.max_new_tokens, args.max_seq_len)

        saved = force_all_gates_open(model)
        base_ms, base_tok = time_query(model, tok, prompt, args.max_new_tokens, args.max_seq_len)
        restore_gates(model, saved)

        records.append({
            "query_id": i, "step_type": step_type,
            "routed_ms": routed_ms, "routed_tokens": routed_tok,
            "baseline_ms": base_ms, "baseline_tokens": base_tok,
            "routed_ms_per_tok": routed_ms / max(routed_tok, 1),
            "baseline_ms_per_tok": base_ms / max(base_tok, 1),
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] routed={routed_ms:.1f}ms  baseline={base_ms:.1f}ms")

    ms_r = sum(r["routed_ms_per_tok"] for r in records) / len(records)
    ms_b = sum(r["baseline_ms_per_tok"] for r in records) / len(records)
    speedup = ms_b / ms_r if ms_r > 0 else float("nan")

    out_csv = os.path.join(args.out_dir, "wall_clock.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "seed": args.seed, "n_eval": args.n_eval,
        "routed_ms_per_tok": round(ms_r, 3), "baseline_ms_per_tok": round(ms_b, 3),
        "speedup": round(speedup, 3),
    }
    with open(os.path.join(args.out_dir, "wall_clock_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved -> {out_csv}")
    print(f"routed:   {ms_r:.2f} ms/token")
    print(f"baseline: {ms_b:.2f} ms/token")
    print(f"speedup:  {speedup:.3f}x")


if __name__ == "__main__":
    main()
