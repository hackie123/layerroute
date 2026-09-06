"""
check_length_confound_gate.py
================================
identify_gate_driver.py found sequence length (r=-0.51) correlates more
strongly with middle-layer gate score than the tool_call label (r=-0.40).
This checks whether tool_call has ANY independent explanatory power once
length is controlled for, via partial correlation -- the direct test of
"is step-type doing anything beyond riding on a length confound?"
r_{xy.z} = (r_xy - r_xz*r_yz) / sqrt((1-r_xz^2)(1-r_yz^2))
Verified against synthetic ground truth (confounded raw correlation
correctly collapses toward 0 once the true confound is partialed out)
before use here.

Usage:
    python check_length_confound_gate.py --ckpt checkpoints_seed2024/best_adapters.pt
"""
import argparse
import numpy as np
import torch
from scipy.stats import pearsonr

from utils.config import SystemConfig, LoRAConfig, RouterConfig, TrainingConfig, QWEN_SPEC
from models.gated_qwen import GatedQwenLoRA
from transformers import AutoTokenizer


def partial_corr(x, y, z):
    r_xy, _ = pearsonr(x, y)
    r_xz, _ = pearsonr(x, z)
    r_yz, _ = pearsonr(y, z)
    num = r_xy - r_xz * r_yz
    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    return num / denom if denom > 1e-12 else float("nan")


def build_model(ckpt):
    cfg = SystemConfig(lora=LoRAConfig(r=8, alpha=16.0), router=RouterConfig(threshold=0.5),
                       training=TrainingConfig(dtype="bfloat16"))
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
    p.add_argument("--middle_start", type=int, default=8)
    p.add_argument("--middle_end", type=int, default=17)
    args = p.parse_args()

    model = build_model(args.ckpt)
    device = next(model.hf_model.parameters()).device
    tok = AutoTokenizer.from_pretrained(QWEN_SPEC["hf_name"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    import evaluate as ev
    samples = ev.load_eval_samples(n=args.n_eval, seed=args.seed)
    transformer = model.hf_model.model

    seq_lengths, tool_labels, gate_scores = [], [], []
    with torch.no_grad():
        for turns, step_type in samples:
            prompt = tok.apply_chat_template(turns, tokenize=False, add_generation_prompt=False)
            ids = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_seq_len).input_ids.to(device)
            T = ids.shape[1]
            seq_lengths.append(T)
            tool_labels.append(1 if step_type == "tool_call" else 0)

            hidden = transformer.embed_tokens(ids)
            pos_ids = torch.arange(T, device=device).unsqueeze(0)
            cos, sin = transformer.rotary_emb(hidden, pos_ids)
            layer_kwargs = dict(attention_mask=None, position_ids=pos_ids,
                                past_key_values=None, use_cache=False,
                                position_embeddings=(cos, sin))
            h = hidden
            middle_scores = []
            for i, layer in enumerate(transformer.layers):
                router = model.routers.routers[i]
                h_mean = h.mean(dim=1).to(router.linear.weight.dtype)
                score = (router.linear.weight @ h_mean.T).item() + router.linear.bias.item()
                if args.middle_start <= i < args.middle_end:
                    middle_scores.append(score)
                hard_g = 1.0 if score > 0 else 0.0
                if hard_g > 0.5:
                    out = layer(h, **layer_kwargs)
                    h = (out[0] if isinstance(out, tuple) else out).to(hidden.dtype)
            gate_scores.append(np.mean(middle_scores))

    seq_lengths = np.array(seq_lengths, dtype=float)
    tool_labels = np.array(tool_labels, dtype=float)
    gate_scores = np.array(gate_scores, dtype=float)

    r_len_tool, _ = pearsonr(seq_lengths, tool_labels)
    r_len_gate, _ = pearsonr(seq_lengths, gate_scores)
    r_tool_gate, _ = pearsonr(tool_labels, gate_scores)
    partial_tool = partial_corr(tool_labels, gate_scores, seq_lengths)
    partial_len = partial_corr(seq_lengths, gate_scores, tool_labels)

    print(f"\nConfound check: corr(seq_length, tool_call_label) = {r_len_tool:.4f}")
    print(f"  (if large, length and step-type are entangled in this eval set)\n")
    print(f"{'comparison':<45}{'value'}")
    print("-" * 60)
    print(f"{'raw corr(tool_label, gate_score)':<45}{r_tool_gate:.4f}")
    print(f"{'partial corr(tool_label, gate_score | length)':<45}{partial_tool:.4f}")
    print(f"{'raw corr(length, gate_score)':<45}{r_len_gate:.4f}")
    print(f"{'partial corr(length, gate_score | tool_label)':<45}{partial_len:.4f}")

    print("\nInterpretation:")
    print("  If partial_tool collapses toward 0 while partial_len stays large:")
    print("    -> tool_call's apparent effect was entirely a length confound.")
    print("  If partial_tool stays meaningfully non-zero:")
    print("    -> tool_call retains SOME independent explanatory power beyond length.")


if __name__ == "__main__":
    main()
