"""
evaluate.py
============
Post-training evaluation for ConfGate v6.
Run after training completes.

Generates all paper metrics in checkpoints/paper_results/:
    1. gate_values.csv          — per-layer final gate values
    2. routing_by_type.csv      — skip_pct per step type (tool_call vs planning)
    3. quality_degradation.csv  — perplexity full model vs gated model
    4. flops_table.csv          — FLOPs full vs average skipped
    5. lora_impact.csv          — loss before vs after LoRA (base vs fine-tuned)
    6. paper_summary.csv        — all key metrics in one row
    7. paper_results.txt        — human-readable report

Usage:
    python evaluate.py --ckpt checkpoints/best_adapters.pt --n_eval 100
"""

import os, csv, json, argparse, random, time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from utils.config import SystemConfig, LoRAConfig, RouterConfig, TrainingConfig, DataConfig, QWEN_SPEC
from models.gated_qwen import GatedQwenLoRA


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print(f"  Saved → {path}")


def build_model(args):
    cfg = SystemConfig(
        lora     = LoRAConfig(r=args.lora_r, alpha=args.lora_alpha),
        router   = RouterConfig(threshold=args.router_threshold),
        training = TrainingConfig(dtype="bfloat16", max_seq_len=args.max_seq_len),
    )
    cfg.validate() if hasattr(cfg, "validate") else None
    model = GatedQwenLoRA.from_pretrained(cfg)
    device = next(p for p in model.hf_model.parameters()).device
    model.routers = model.routers.to(device)
    if args.ckpt:
        model.load_adapters(args.ckpt)
    model.eval()
    return model, cfg


def load_eval_samples(n=100, seed=2024):
    """Load balanced tool_call and planning samples."""
    from datasets import load_dataset
    samples = []
    try:
        ds = load_dataset("NousResearch/hermes-function-calling-v1", split="train")
        ds = ds.select(range(min(n//2, len(ds))))
        for row in ds:
            turns = [{"role": "user" if t.get("from")=="human" else "assistant",
                      "content": t.get("value","")} for t in row.get("conversations",[])]
            if turns:
                samples.append((turns, "tool_call"))
        print(f"  tool_call: {sum(1 for _,t in samples if t=='tool_call')} samples")
    except Exception as e:
        print(f"  Hermes skipped: {e}")

    try:
        ds = load_dataset("openai/gsm8k", "main", split="train")
        ds = ds.select(range(min(n//2, len(ds))))
        for row in ds:
            turns = [{"role":"user","content":row["question"]},
                     {"role":"assistant","content":row["answer"]}]
            samples.append((turns, "planning"))
        print(f"  planning: {sum(1 for _,t in samples if t=='planning')} samples")
    except Exception as e:
        print(f"  GSM8K skipped: {e}")

    # FIX: previously unseeded -- eval sample selection was non-reproducible
    # even for an identical, seeded training checkpoint. Uses a local Random
    # instance (not random.shuffle on the global module) so this doesn't
    # perturb any other random state elsewhere in the process.
    random.Random(seed).shuffle(samples)
    return samples


# ── 1. Gate values ─────────────────────────────────────────────────

def eval_gate_values(model, out_dir):
    print("\n[1/6] Gate values...")
    gvals = model.routers.gate_values()
    rows  = [{"layer":i, "gate_value":round(g,6),
              "status":"skip_candidate" if g < 0.5 else "active"}
             for i, g in enumerate(gvals)]
    write_csv(os.path.join(out_dir,"gate_values.csv"),
              ["layer","gate_value","status"], rows)
    skip_n = sum(1 for g in gvals if g < 0.5)
    print(f"  Range: [{min(gvals):.4f}, {max(gvals):.4f}]")
    print(f"  Layers with gate < 0.5 (skip candidates): {skip_n}")
    return {
        "gate_min"     : round(min(gvals),4),
        "gate_max"     : round(max(gvals),4),
        "gate_variance": round(torch.tensor(gvals).var().item(),6),
        "n_skip_candidates": skip_n,
    }


# ── 2. Routing by step type ────────────────────────────────────────

def eval_routing_by_type(model, samples, tok, cfg, out_dir):
    print(f"\n[2/6] Routing by step type ({len(samples)} samples)...")
    rows = []
    stats = {"tool_call":{"layers_run":[],"skip_pct":[]},
             "planning" :{"layers_run":[],"skip_pct":[]}}

    for turns, step_type in samples:
        prompt = tok.apply_chat_template(turns, tokenize=False, add_generation_prompt=False)
        device = next(p for p in model.hf_model.parameters()).device
        ids    = tok.encode(prompt, return_tensors="pt",
                            max_length=cfg.training.max_seq_len, truncation=True).to(device)
        with torch.no_grad():
            _, _, gs = model(ids)
        rows.append({
            "step_type"  : step_type,
            "layers_run" : gs["layers_run"],
            "skip_pct"   : gs["skip_pct"],
            "gate_values": str(gs["gate_values"]),
        })
        stats[step_type]["layers_run"].append(gs["layers_run"])
        stats[step_type]["skip_pct"].append(gs["skip_pct"])

    write_csv(os.path.join(out_dir,"routing_by_type.csv"),
              ["step_type","layers_run","skip_pct","gate_values"], rows)

    def avg(lst): return round(sum(lst)/max(len(lst),1), 2)
    tc = stats["tool_call"]; pl = stats["planning"]
    print(f"  tool_call: avg_layers={avg(tc['layers_run'])}/24  skip={avg(tc['skip_pct'])}%")
    print(f"  planning : avg_layers={avg(pl['layers_run'])}/24  skip={avg(pl['skip_pct'])}%")
    return {
        "tool_avg_layers_run" : avg(tc["layers_run"]),
        "tool_avg_skip_pct"   : avg(tc["skip_pct"]),
        "plan_avg_layers_run" : avg(pl["layers_run"]),
        "plan_avg_skip_pct"   : avg(pl["skip_pct"]),
        "skip_diff_pct"       : round(avg(tc["skip_pct"]) - avg(pl["skip_pct"]), 2),
    }


# ── 3. Perplexity — gated vs full ─────────────────────────────────

def eval_perplexity(model, samples, tok, cfg, out_dir):
    print(f"\n[3/6] Perplexity — gated vs baseline ({len(samples)} samples)...")
    rows = []

    for turns, step_type in samples:
        prompt = tok.apply_chat_template(turns, tokenize=False, add_generation_prompt=False)
        device = next(p for p in model.hf_model.parameters()).device
        ids    = tok.encode(prompt, return_tensors="pt",
                            max_length=cfg.training.max_seq_len, truncation=True).to(device)
        labels = ids.clone()

        with torch.no_grad():
            # Gated model (routers active)
            _, loss_gated, gs = model(ids, labels=labels)
            ppl_gated = torch.exp(loss_gated).item()

            # Disable routers temporarily (all gates = 1, run all layers)
            for r in model.routers.routers:
                r.linear.bias.data.fill_(10.0)   # sigmoid(10) ≈ 1.0
            _, loss_full, _ = model(ids, labels=labels)
            ppl_full = torch.exp(loss_full).item()
            # Restore
            for r in model.routers.routers:
                r.linear.bias.data.fill_(cfg.router.init_bias_early)

        rows.append({
            "step_type"  : step_type,
            "ppl_full"   : round(min(ppl_full, 9999), 2),
            "ppl_gated"  : round(min(ppl_gated, 9999), 2),
            "ppl_delta"  : round(min(ppl_gated - ppl_full, 9999), 2),
            "layers_run" : gs["layers_run"],
            "skip_pct"   : gs["skip_pct"],
        })

    write_csv(os.path.join(out_dir,"quality_degradation.csv"),
              ["step_type","ppl_full","ppl_gated","ppl_delta","layers_run","skip_pct"], rows)

    tc_rows = [r for r in rows if r["step_type"]=="tool_call"]
    pl_rows = [r for r in rows if r["step_type"]=="planning"]
    def avg_col(lst, k): return round(sum(r[k] for r in lst)/max(len(lst),1), 3)

    print(f"  tool_call: ppl_full={avg_col(tc_rows,'ppl_full')}  "
          f"ppl_gated={avg_col(tc_rows,'ppl_gated')}  "
          f"delta={avg_col(tc_rows,'ppl_delta')}")
    print(f"  planning : ppl_full={avg_col(pl_rows,'ppl_full')}  "
          f"ppl_gated={avg_col(pl_rows,'ppl_gated')}  "
          f"delta={avg_col(pl_rows,'ppl_delta')}")

    return {
        "ppl_full_tool"    : avg_col(tc_rows,"ppl_full"),
        "ppl_gated_tool"   : avg_col(tc_rows,"ppl_gated"),
        "ppl_delta_tool"   : avg_col(tc_rows,"ppl_delta"),
        "ppl_full_plan"    : avg_col(pl_rows,"ppl_full"),
        "ppl_gated_plan"   : avg_col(pl_rows,"ppl_gated"),
        "ppl_delta_plan"   : avg_col(pl_rows,"ppl_delta"),
    }


# ── 4. FLOPs table ─────────────────────────────────────────────────

def eval_flops(model, routing_metrics, cfg, out_dir):
    print("\n[4/6] FLOPs table...")
    spec     = QWEN_SPEC
    hidden   = spec["hidden"]; n_q = spec["n_q_heads"]
    n_kv     = spec["n_kv_heads"]; head_dim = spec["head_dim"]
    ffn      = spec["intermediate"]; seq_len = cfg.training.max_seq_len
    n_layers = spec["n_layers"]

    attn_flops = (2*seq_len*hidden*n_q*head_dim + 2*seq_len*hidden*n_kv*head_dim*2 +
                  2*seq_len*seq_len*n_q*head_dim + 2*seq_len*n_q*head_dim*hidden)
    ffn_flops  = 3 * 2 * seq_len * hidden * ffn
    per_layer  = attn_flops + ffn_flops
    full_flops = per_layer * n_layers

    tc_layers  = routing_metrics["tool_avg_layers_run"]
    pl_layers  = routing_metrics["plan_avg_layers_run"]

    rows = [
        {"config":"Full model (no skip)","avg_layers":n_layers,
         "flops_per_seq":full_flops,"reduction_pct":0.0},
        {"config":f"tool_call (avg {tc_layers:.1f} layers)",
         "avg_layers":tc_layers,
         "flops_per_seq":int(per_layer*tc_layers),
         "reduction_pct":round(100*(n_layers-tc_layers)/n_layers,1)},
        {"config":f"planning (avg {pl_layers:.1f} layers)",
         "avg_layers":pl_layers,
         "flops_per_seq":int(per_layer*pl_layers),
         "reduction_pct":round(100*(n_layers-pl_layers)/n_layers,1)},
    ]
    write_csv(os.path.join(out_dir,"flops_table.csv"),
              ["config","avg_layers","flops_per_seq","reduction_pct"], rows)
    print(f"  Full model  : {full_flops:,} FLOPs/seq")
    print(f"  tool_call   : {rows[1]['flops_per_seq']:,}  (-{rows[1]['reduction_pct']}%)")
    print(f"  planning    : {rows[2]['flops_per_seq']:,}  (-{rows[2]['reduction_pct']}%)")
    return {
        "flops_full"          : full_flops,
        "flops_tool"          : rows[1]["flops_per_seq"],
        "flops_plan"          : rows[2]["flops_per_seq"],
        "flops_reduction_tool": rows[1]["reduction_pct"],
        "flops_reduction_plan": rows[2]["reduction_pct"],
    }


# ── 5. LoRA impact ─────────────────────────────────────────────────

def eval_lora_impact(model, samples, tok, cfg, out_dir):
    print(f"\n[5/6] LoRA impact — base vs fine-tuned ({min(len(samples),20)} samples)...")
    rows = []
    subset = samples[:20]

    for turns, step_type in subset:
        prompt = tok.apply_chat_template(turns, tokenize=False, add_generation_prompt=False)
        device = next(p for p in model.hf_model.parameters()).device
        ids    = tok.encode(prompt, return_tensors="pt",
                            max_length=cfg.training.max_seq_len, truncation=True).to(device)
        labels = ids.clone()

        with torch.no_grad():
            # With LoRA (fine-tuned)
            _, loss_lora, _ = model(ids, labels=labels)

            # Disable LoRA temporarily (scale to 0)
            for name, mod in model.named_modules():
                if hasattr(mod, "scaling"):
                    mod._saved_scaling = mod.scaling
                    mod.scaling = 0.0
            _, loss_base, _ = model(ids, labels=labels)
            # Restore
            for name, mod in model.named_modules():
                if hasattr(mod, "_saved_scaling"):
                    mod.scaling = mod._saved_scaling

        rows.append({
            "step_type"  : step_type,
            "loss_base"  : round(loss_base.item(), 4),
            "loss_lora"  : round(loss_lora.item(), 4),
            "loss_delta" : round(loss_lora.item() - loss_base.item(), 4),
        })

    write_csv(os.path.join(out_dir,"lora_impact.csv"),
              ["step_type","loss_base","loss_lora","loss_delta"], rows)

    def avg(lst, k): return round(sum(r[k] for r in lst)/max(len(lst),1),4)
    tc = [r for r in rows if r["step_type"]=="tool_call"]
    pl = [r for r in rows if r["step_type"]=="planning"]
    print(f"  tool_call: base={avg(tc,'loss_base')}  lora={avg(tc,'loss_lora')}  delta={avg(tc,'loss_delta')}")
    print(f"  planning : base={avg(pl,'loss_base')}  lora={avg(pl,'loss_lora')}  delta={avg(pl,'loss_delta')}")
    return {
        "lora_loss_delta_tool": avg(tc,"loss_delta"),
        "lora_loss_delta_plan": avg(pl,"loss_delta"),
    }


# ── 6. Paper summary ───────────────────────────────────────────────

def write_summary(all_metrics, out_dir, args):
    flat = {}
    for d in all_metrics.values(): flat.update(d)
    flat["checkpoint"] = args.ckpt or "no_ckpt"
    flat["n_eval"]     = args.n_eval
    flat["seed"]       = args.seed
    flat["lora_r"]     = args.lora_r

    write_csv(os.path.join(out_dir,"paper_summary.csv"), list(flat.keys()), [flat])

    rpt = os.path.join(out_dir,"paper_results.txt")
    gm  = all_metrics["gates"]
    rm  = all_metrics["routing"]
    qm  = all_metrics["quality"]
    fm  = all_metrics["flops"]
    lm  = all_metrics["lora"]

    with open(rpt,"w") as f:
        f.write("="*60+"\n")
        f.write("  CONFGATE v6 — PAPER RESULTS\n")
        f.write("  LoRA + Hard-Gated Skip Connections on Qwen2.5-0.5B\n")
        f.write("="*60+"\n\n")
        f.write(f"  Checkpoint   : {args.ckpt}\n")
        f.write(f"  Eval samples : {args.n_eval}\n")
        f.write(f"  LoRA rank    : {args.lora_r}\n\n")

        f.write("── Gate Learning ────────────────────────────────────────\n")
        f.write(f"  Gate range         : [{gm['gate_min']}, {gm['gate_max']}]\n")
        f.write(f"  Gate variance      : {gm['gate_variance']}\n")
        f.write(f"  Skip candidates    : {gm['n_skip_candidates']}/24\n\n")

        f.write("── Adaptive Routing ─────────────────────────────────────\n")
        f.write(f"  tool_call avg layers : {rm['tool_avg_layers_run']}/24  (skip {rm['tool_avg_skip_pct']}%)\n")
        f.write(f"  planning  avg layers : {rm['plan_avg_layers_run']}/24  (skip {rm['plan_avg_skip_pct']}%)\n")
        f.write(f"  Skip differential    : {rm['skip_diff_pct']}% more skipping on tool calls\n\n")

        f.write("── FLOPs ────────────────────────────────────────────────\n")
        f.write(f"  Full model    : {fm['flops_full']:,}\n")
        f.write(f"  tool_call     : {fm['flops_tool']:,}  (-{fm['flops_reduction_tool']}%)\n")
        f.write(f"  planning      : {fm['flops_plan']:,}  (-{fm['flops_reduction_plan']}%)\n\n")

        f.write("── Perplexity (quality) ─────────────────────────────────\n")
        f.write(f"  tool_call: ppl_full={qm['ppl_full_tool']}  ppl_gated={qm['ppl_gated_tool']}  delta={qm['ppl_delta_tool']}\n")
        f.write(f"  planning : ppl_full={qm['ppl_full_plan']}  ppl_gated={qm['ppl_gated_plan']}  delta={qm['ppl_delta_plan']}\n\n")

        f.write("── LoRA Impact ──────────────────────────────────────────\n")
        f.write(f"  tool_call loss delta : {lm['lora_loss_delta_tool']}\n")
        f.write(f"  planning  loss delta : {lm['lora_loss_delta_plan']}\n")
        f.write("="*60+"\n")

    print(f"  Saved → {rpt}")
    return rpt


# ── Main ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",             default="checkpoints/best_adapters.pt")
    p.add_argument("--n_eval",           type=int,   default=100)
    p.add_argument("--lora_r",           type=int,   default=8)
    p.add_argument("--lora_alpha",       type=float, default=16.0)
    p.add_argument("--router_threshold", type=float, default=0.5)
    p.add_argument("--max_seq_len",      type=int,   default=512)
    p.add_argument("--out_dir",          default="checkpoints/paper_results")
    p.add_argument("--seed",             type=int,   default=2024,
                   help="Seeds eval-sample selection (random.shuffle), "
                        "independent of the training seed.")
    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    t0      = time.perf_counter()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print("="*60)
    print("  CONFGATE v6 — PAPER EVALUATION")
    print("="*60)

    model, cfg = build_model(args)
    tok = AutoTokenizer.from_pretrained(QWEN_SPEC["hf_name"], trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    print("\n[Data] Loading eval samples...")
    samples = load_eval_samples(n=args.n_eval, seed=args.seed)

    all_metrics = {}
    all_metrics["gates"]   = eval_gate_values(model, out_dir)
    all_metrics["routing"] = eval_routing_by_type(model, samples, tok, cfg, out_dir)
    all_metrics["quality"] = eval_perplexity(model, samples, tok, cfg, out_dir)
    all_metrics["flops"]   = eval_flops(model, all_metrics["routing"], cfg, out_dir)
    all_metrics["lora"]    = eval_lora_impact(model, samples, tok, cfg, out_dir)

    print("\n[6/6] Writing paper summary...")
    write_summary(all_metrics, out_dir, args)

    print(f"\n  Done in {time.perf_counter()-t0:.0f}s")
    print(f"  All results → {out_dir}/")
    print(f"\n  Key metrics:")
    rm = all_metrics["routing"]
    fm = all_metrics["flops"]
    print(f"    tool_call skip : {rm['tool_avg_skip_pct']}%  planning skip: {rm['plan_avg_skip_pct']}%")
    print(f"    skip diff      : {rm['skip_diff_pct']}%")
    print(f"    FLOPs reduction: tool={fm['flops_reduction_tool']}%  plan={fm['flops_reduction_plan']}%")
