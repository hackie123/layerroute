"""
analyze_gate_stability.py
============================
Reads gate_values.csv from all 10 checkpoints_seed<N>/ directories and
checks, per layer, whether the CONVERGED gate value is stable across seeds
or a "swing" layer that flips between skip-candidate and active depending
on random init. Directly tests whether skip-differential instability is
concentrated in a few layers or diffuse across the middle band.

Usage: python analyze_gate_stability.py
"""
import csv
import glob
import numpy as np


def load_gate_values(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return {int(row["layer"]): float(row["gate_value"]) for row in reader}


def main():
    paths = sorted(glob.glob("checkpoints_seed*/paper_results/gate_values.csv"))
    if not paths:
        print("No gate_values.csv files found.")
        return

    per_seed = {}
    for p in paths:
        seed_dir = p.split("/")[0]
        per_seed[seed_dir] = load_gate_values(p)

    n_layers = len(next(iter(per_seed.values())))
    seeds = sorted(per_seed.keys())

    print(f"Loaded {len(seeds)} seeds x {n_layers} layers\n")
    print(f"{'layer':<8}{'mean':<10}{'std':<10}{'min':<10}{'max':<10}{'flips':<8}{'note'}")
    print("-" * 70)

    swing_layers = []
    for layer in range(n_layers):
        vals = [per_seed[s][layer] for s in seeds]
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        # "Flips" = how many seeds disagree with the MAJORITY skip/active decision
        decisions = [v < 0.5 for v in vals]
        majority = sum(decisions) > len(decisions) / 2
        flips = sum(1 for d in decisions if d != majority)

        note = ""
        if flips >= 3:
            note = "SWING LAYER (unstable decision across seeds)"
            swing_layers.append(layer)
        elif std_v > 0.15:
            note = "high variance"

        print(f"{layer:<8}{mean_v:<10.3f}{std_v:<10.3f}{min(vals):<10.3f}"
             f"{max(vals):<10.3f}{flips:<8}{note}")

    print(f"\n{'='*50}")
    print(f"Swing layers (>=3/{len(seeds)} seeds disagree with majority): {swing_layers}")
    print(f"{'='*50}")

    if swing_layers:
        print(f"\nInstability is {'CONCENTRATED in ' + str(len(swing_layers)) + ' of ' + str(n_layers) + ' layers' if len(swing_layers) < n_layers // 2 else 'DIFFUSE across most layers'}.")
    else:
        print("\nNo swing layers found -- gate CONVERGENCE is stable across seeds. "
             "(If skip_diff_pct still varies wildly, the instability may be in "
             "PER-SAMPLE/step-type routing behavior rather than the converged "
             "gate structure itself -- worth checking routing_by_type.csv next.)")


if __name__ == "__main__":
    main()
