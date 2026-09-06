"""
probe_signal.py
=================
Tests whether tool-call vs. planning is reliably linearly separable in the
FROZEN backbone's hidden states, before any LoRA/router training -- directly
testing the hypothesis that seed variance comes from the router being asked
to discover a signal that isn't robustly there to find, rather than from
training instability alone.

Method: extract h_mean at every layer (mean-pooled hidden state, EXACTLY
what the real router computes -- see models/router.py's h_i.mean(dim=1))
from the RAW pretrained model (no LoRA, no gates), fit a linear probe
(logistic regression) per layer to classify tool_call vs planning, and
report probe accuracy across MANY random train/test splits -- not just
one -- to see whether the *signal itself* is stable or noisy, mirroring
exactly the question we're asking about the router.

If probe accuracy is high AND stable across splits: the signal is robustly
present, and training-dynamics fragility (LoRA init, gate-reg interaction)
is a more likely explanation for seed variance than signal absence.
If probe accuracy is weak OR unstable across splits: supports the
hypothesis that the router is chasing a genuinely marginal, sample-
dependent signal, which would directly explain why different LoRA inits
(seeds) land on different, inconsistent separating directions.

Usage: python probe_signal.py --n_eval 100 --n_splits 20
"""
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utils.config import QWEN_SPEC


def extract_hidden_states(model, tok, samples, max_seq_len=512):
    """Returns hidden_by_layer: list of (n_samples, hidden_dim) arrays, one
    per layer (0..n_layers), and labels: list of 0/1 (tool_call=0, planning=1).
    Uses output_hidden_states=True on a plain forward pass -- the raw
    backbone, no LoRA, no gating -- exactly matching what a router at each
    layer WOULD see if inserted into an untrained model."""
    device = next(model.parameters()).device
    n_layers = QWEN_SPEC["n_layers"]
    hidden_by_layer = [[] for _ in range(n_layers + 1)]  # +1 for embedding output
    labels = []

    with torch.no_grad():
        for turns, step_type in samples:
            prompt = tok.apply_chat_template(turns, tokenize=False, add_generation_prompt=False)
            ids = tok(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len).input_ids.to(device)
            out = model(ids, output_hidden_states=True)
            # out.hidden_states: tuple of (1, T, H), length n_layers+1
            for i, h in enumerate(out.hidden_states):
                h_mean = h.mean(dim=1).squeeze(0).float().cpu().numpy()  # (H,)
                hidden_by_layer[i].append(h_mean)
            labels.append(0 if step_type == "tool_call" else 1)

    hidden_by_layer = [np.stack(layer_feats) for layer_feats in hidden_by_layer]
    labels = np.array(labels)
    return hidden_by_layer, labels


def probe_layer(X, y, n_splits, seed_base=0):
    """Fits a logistic-regression probe on MANY random train/test splits,
    reporting mean and STD of held-out accuracy -- the std across splits is
    the thing that matters here: a high mean with high std means the probe
    itself is unstable, i.e. the signal is real but fragile/sample-dependent,
    which would mirror the router's own instability."""
    accs = []
    for s in range(n_splits):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=seed_base + s, stratify=y
        )
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X_tr, y_tr)
        accs.append(clf.score(X_te, y_te))
    return np.mean(accs), np.std(accs), accs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_eval", type=int, default=100)
    p.add_argument("--n_splits", type=int, default=20,
                   help="Number of random train/test splits per layer -- "
                        "this is what reveals whether the SIGNAL is stable, "
                        "not just whether one split happens to look good.")
    p.add_argument("--max_seq_len", type=int, default=512)
    args = p.parse_args()

    print(f"Loading raw {QWEN_SPEC['hf_name']} (no LoRA, no router)...")
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_SPEC["hf_name"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(QWEN_SPEC["hf_name"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    import evaluate as ev
    samples = ev.load_eval_samples(n=args.n_eval, seed=2024)
    print(f"Loaded {len(samples)} samples "
         f"({sum(1 for _,t in samples if t=='tool_call')} tool_call, "
         f"{sum(1 for _,t in samples if t=='planning')} planning)")

    print("Extracting hidden states across all layers...")
    hidden_by_layer, labels = extract_hidden_states(model, tok, samples, args.max_seq_len)

    print(f"\n{'layer':<8}{'probe_acc_mean':<18}{'probe_acc_std':<16}{'note'}")
    print("-" * 60)
    results = []
    for i, X in enumerate(hidden_by_layer):
        mean_acc, std_acc, _ = probe_layer(X, labels, args.n_splits)
        note = ""
        if std_acc > 0.10:
            note = "UNSTABLE across splits"
        elif mean_acc < 0.65:
            note = "weak signal"
        results.append((i, mean_acc, std_acc))
        print(f"{i:<8}{mean_acc:<18.3f}{std_acc:<16.3f}{note}")

    accs_only = [r[1] for r in results]
    stds_only = [r[2] for r in results]
    print(f"\nAcross all {len(results)} layers: "
         f"mean(probe_acc)={np.mean(accs_only):.3f}  "
         f"mean(probe_std)={np.mean(stds_only):.3f}  "
         f"max(probe_std)={np.max(stds_only):.3f}")


if __name__ == "__main__":
    main()
