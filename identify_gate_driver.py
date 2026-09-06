"""
identify_gate_driver.py
=========================
The router performs real, substantial per-sample computation in the
middle layers (87-100% flip rate vs. bias-only, check_router_triviality.py)
but does NOT track the claimed tool_call/planning distinction (alignment
~0.03, diagnose_gradient_direction.py). This tests two alternative
candidate drivers:

  (1) Sequence length -- h_mean is a MEAN over the sequence dimension;
      systematically different length distributions between sample types
      could produce length-correlated (not semantically meaningful) gate
      variation.
  (2) Prediction confidence -- the paper's own introduction motivates
      skipping via "the model's high confidence early in the residual
      stream suggests deep layers contribute marginally." This may be a
      REAL signal the router tracks, just not one that happens to align
      with tool_call/planning specifically (they only correlate with
      confidence on average, imperfectly).

For each candidate feature, reports both a linear correlation with the
per-sample middle-layer (8-16) real gate score, AND single-feature
logistic regression accuracy predicting the hard skip/run decision --
whichever feature predicts best is the more likely actual driver.

Caveat, stated explicitly: evaluate.py's eval-time samples are Hermes
(tool_call) vs GSM8K (planning) ONLY -- not the full 4-dataset training
mix -- so "tool_call vs planning" here is inseparable from "Hermes vs
GSM8K" as a dataset-source confound.

Usage:
    python identify_gate_driver.py --ckpt checkpoints_seed2024/best_adapters.pt
"""
import argparse
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.stats import pointbiserialr, pearsonr

from utils.config import SystemConfig, LoRAConfig, RouterConfig, TrainingConfig, QWEN_SPEC
from models.gated_qwen import GatedQwenLoRA
from transformers import AutoTokenizer


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

    seq_lengths   = []
    confidences   = []
    tool_labels   = []
    middle_gate_scores = []   # mean real_score (pre-sigmoid) across middle layers
    middle_gate_hard   = []   # mean hard decision across middle layers (fraction open)

    with torch.no_grad():
        for turns, step_type in samples:
            prompt = tok.apply_chat_template(turns, tokenize=False, add_generation_prompt=False)
            ids = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_seq_len).input_ids.to(device)
            T = ids.shape[1]
            seq_lengths.append(T)
            tool_labels.append(1 if step_type == "tool_call" else 0)

            # Confidence: mean max-softmax-probability over the sequence,
            # from the RAW backbone forward pass (matches the paper's own
            # "high confidence early" intuition) -- reuses the model's own
            # lm_head via a full (all-gates-open equivalent) pass for a
            # clean, gate-independent confidence measure.
            hidden = transformer.embed_tokens(ids)
            pos_ids = torch.arange(T, device=device).unsqueeze(0)
            cos, sin = transformer.rotary_emb(hidden, pos_ids)
            layer_kwargs = dict(attention_mask=None, position_ids=pos_ids,
                                past_key_values=None, use_cache=False,
                                position_embeddings=(cos, sin))
            h = hidden
            middle_scores = []
            middle_hard   = []
            for i, layer in enumerate(transformer.layers):
                router = model.routers.routers[i]
                h_mean = h.mean(dim=1).to(router.linear.weight.dtype)
                score = (router.linear.weight @ h_mean.T).item() + router.linear.bias.item()
                if args.middle_start <= i < args.middle_end:
                    middle_scores.append(score)
                    middle_hard.append(1.0 if score > 0 else 0.0)  # score>0 <=> sigmoid(score)>0.5
                hard_g = 1.0 if score > 0 else 0.0
                if hard_g > 0.5:
                    out = layer(h, **layer_kwargs)
                    h = (out[0] if isinstance(out, tuple) else out).to(hidden.dtype)

            middle_gate_scores.append(np.mean(middle_scores))
            middle_gate_hard.append(np.mean(middle_hard))

            # Confidence measure: full (ungated-equivalent) pass's own logits
            h_full = transformer.norm(h)
            logits = model.hf_model.lm_head(h_full)
            probs = torch.softmax(logits[0], dim=-1)
            max_probs = probs.max(dim=-1).values
            confidences.append(max_probs.mean().item())

    seq_lengths = np.array(seq_lengths, dtype=float)
    confidences = np.array(confidences, dtype=float)
    tool_labels = np.array(tool_labels, dtype=float)
    gate_scores = np.array(middle_gate_scores, dtype=float)
    gate_hard_frac = np.array(middle_gate_hard, dtype=float)
    gate_hard_decision = (gate_hard_frac > 0.5).astype(int)  # majority-open across middle layers

    print(f"\n{'feature':<20}{'corr_w_gate_score':<20}{'logreg_acc_predict_hard'}")
    print("-" * 60)
    for name, feat in [("seq_length", seq_lengths), ("confidence", confidences), ("tool_call_label", tool_labels)]:
        r, _ = pearsonr(feat, gate_scores)
        X = feat.reshape(-1, 1)
        # Standardize for fair logistic-regression comparison across features
        X = (X - X.mean()) / (X.std() + 1e-8)
        try:
            acc = cross_val_score(LogisticRegression(max_iter=1000), X, gate_hard_decision, cv=5).mean()
        except Exception:
            acc = float("nan")
        print(f"{name:<20}{r:<20.4f}{acc:<10.4f}")

    print(f"\nMiddle-layer (layers {args.middle_start}-{args.middle_end-1}) mean fraction gates open: "
         f"{gate_hard_frac.mean():.3f}")
    print("Interpretation: higher |correlation| and higher logreg accuracy = stronger candidate driver.")
    print("Baseline random-guess accuracy for a balanced binary target ~= 0.50.")


if __name__ == "__main__":
    main()
