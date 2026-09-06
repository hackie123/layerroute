"""
compare_seeds.py
==================
Reads paper_summary.csv from every checkpoints_seed<N>/ directory produced
by the 10-seed sweep and ranks them by closeness to the paper's reported
numbers, using a normalized distance across the metrics that matter most
(skip differential is weighted highest, since it's the paper's headline
claim; the other metrics are secondary corroborating signals).

Usage: run from conf_gate6/ after all 10 seeds have completed:
    python compare_seeds.py
"""
import csv
import glob
import os

# Paper's reported values (Table 2 / Table 4, single unseeded run)
PAPER_TARGETS = {
    "tool_avg_skip_pct":   15.25,
    "plan_avg_skip_pct":    2.34,
    "skip_diff_pct":       12.91,
    "ppl_delta_tool":      -1.293,
    "ppl_delta_plan":      -1.296,
    "flops_reduction_tool": 15.2,
    "flops_reduction_plan":  2.3,
}

# Relative weight per metric in the overall closeness score. skip_diff_pct
# is the paper's headline claim (Section 5.1's "key result"), weighted
# highest; the rest are corroborating signals, weighted lower.
WEIGHTS = {
    "skip_diff_pct":        3.0,
    "tool_avg_skip_pct":    1.5,
    "plan_avg_skip_pct":    1.5,
    "ppl_delta_tool":       1.0,
    "ppl_delta_plan":       1.0,
    "flops_reduction_tool": 1.0,
    "flops_reduction_plan": 1.0,
}


def load_summary(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        row = next(reader)
    return {k: float(v) for k, v in row.items() if k in PAPER_TARGETS or k == "seed"}


def score(row):
    """Weighted relative-error distance from the paper's numbers. Lower is
    closer. Uses relative error (not absolute) so metrics on very different
    scales (percentages vs. FLOPs-reduction vs. PPL deltas) contribute
    comparably."""
    total = 0.0
    for metric, target in PAPER_TARGETS.items():
        if metric not in row:
            continue
        observed = row[metric]
        denom = abs(target) if abs(target) > 1e-6 else 1.0
        rel_err = abs(observed - target) / denom
        total += WEIGHTS[metric] * rel_err
    return total


def load_wall_clock(seed_dir):
    """Wall-clock timing is separate from paper_summary.csv (a genuinely
    new measurement the original paper never reported); load it alongside
    if present, tolerate its absence for older runs."""
    import json
    path = os.path.join(seed_dir, "paper_results", "wall_clock_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    paths = sorted(glob.glob("checkpoints_seed*/paper_results/paper_summary.csv"))
    if not paths:
        print("No checkpoints_seed*/paper_results/paper_summary.csv files found. "
             "Run the 10-seed sweep first.")
        return

    results = []
    for p in paths:
        seed_dir = p.split("/")[0]
        row = load_summary(p)
        row["_dist"] = score(row)
        row["_dir"] = seed_dir
        wc = load_wall_clock(seed_dir)
        row["_speedup"] = wc["speedup"] if wc else None
        results.append(row)

    results.sort(key=lambda r: r["_dist"])

    print(f"\n{'seed':<8}{'skip_diff':<12}{'tool_skip':<12}{'plan_skip':<12}"
         f"{'ppl_d_tool':<12}{'ppl_d_plan':<12}{'flops_r_tool':<14}"
         f"{'speedup':<10}{'distance':<10}")
    print("-" * 102)
    for r in results:
        seed = int(r.get("seed", -1))
        spd = f"{r['_speedup']:.2f}x" if r["_speedup"] is not None else "n/a"
        print(f"{seed:<8}{r['skip_diff_pct']:<12.2f}{r['tool_avg_skip_pct']:<12.2f}"
             f"{r['plan_avg_skip_pct']:<12.2f}{r['ppl_delta_tool']:<12.3f}"
             f"{r['ppl_delta_plan']:<12.3f}{r['flops_reduction_tool']:<14.2f}"
             f"{spd:<10}{r['_dist']:<10.3f}")

    best = results[0]
    print(f"\n>>> Closest to paper: seed={int(best.get('seed',-1))} "
         f"({best['_dir']}), distance={best['_dist']:.3f}")
    print(f">>> Paper targets:    skip_diff=12.91  tool_skip=15.25  plan_skip=2.34  "
         f"ppl_d_tool=-1.293  ppl_d_plan=-1.296  flops_r_tool=15.2")

    print(f"\n{'='*40}\nSummary statistics across all {len(results)} seeds\n{'='*40}")
    for metric in ["skip_diff_pct", "tool_avg_skip_pct", "plan_avg_skip_pct"]:
        vals = [r[metric] for r in results]
        print(f"{metric}: mean={sum(vals)/len(vals):.2f}  "
             f"min={min(vals):.2f}  max={max(vals):.2f}  "
             f"(paper: {PAPER_TARGETS[metric]:.2f})")


if __name__ == "__main__":
    main()
