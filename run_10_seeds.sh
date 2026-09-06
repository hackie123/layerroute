#!/bin/bash
# run_10_seeds.sh
# Trains + evaluates LayerRoute at 10 different seeds, saving each to its
# own checkpoints_seed<N>/ directory. Run compare_seeds.py after this
# completes to rank all 10 against the paper's reported numbers.
# NOTE: deliberately NOT using `set -e` -- one seed's failure should not
# abort the remaining 9 in a 10x-training-run sweep.

SEEDS="2024 42 123 7 100 256 1234 31415 8675309 999"

for SEED in $SEEDS; do
  echo "=================================================="
  echo "=== SEED $SEED: training ==="
  echo "=================================================="
  python main.py --mode train --max_steps 3000 --seed "$SEED" \
    --output_dir "./checkpoints_seed${SEED}" \
    2>&1 | tee "train_seed${SEED}.log"

  echo "=== SEED $SEED: evaluating (accuracy/PPL/FLOPs) ==="
  python evaluate.py --ckpt "checkpoints_seed${SEED}/best_adapters.pt" \
    --seed "$SEED" \
    --out_dir "checkpoints_seed${SEED}/paper_results" \
    2>&1 | tee "eval_seed${SEED}.log"

  echo "=== SEED $SEED: timing (wall-clock, real skip) ==="
  python time_wall_clock.py --ckpt "checkpoints_seed${SEED}/best_adapters.pt" \
    --seed "$SEED" --n_eval 30 \
    --out_dir "checkpoints_seed${SEED}/paper_results" \
    2>&1 | tee "time_seed${SEED}.log"

  echo "=== SEED $SEED: done ==="
done

echo "=================================================="
echo "=== ALL 10 SEEDS COMPLETE -- run compare_seeds.py ==="
echo "=================================================="
