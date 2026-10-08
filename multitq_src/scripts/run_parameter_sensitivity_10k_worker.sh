#!/usr/bin/env bash
set -Eeuo pipefail

GPU_ID="${1:?usage: run_parameter_sensitivity_10k_worker.sh GPU_ID FACTOR}"
FACTOR="${2:?usage: run_parameter_sensitivity_10k_worker.sh GPU_ID FACTOR}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
source "$(dirname "$0")/parameter_sensitivity_10k_common.sh"

case "$FACTOR" in
  prefix)
    EXPERIMENTS=(
      "prefix_0.00|0.00|0.30|0.30"
      "prefix_0.25|0.25|0.30|0.30"
      "prefix_0.75|0.75|0.30|0.30"
    )
    ;;
  stop)
    EXPERIMENTS=(
      "stop_0.00|0.50|0.00|0.30"
      "stop_0.15|0.50|0.15|0.30"
      "stop_0.45|0.50|0.45|0.30"
    )
    ;;
  counterfactual)
    EXPERIMENTS=(
      "counterfactual_0.00|0.50|0.30|0.00"
      "counterfactual_0.15|0.50|0.30|0.15"
      "counterfactual_0.45|0.50|0.30|0.45"
    )
    ;;
  *)
    echo "FACTOR must be prefix, stop, or counterfactual" >&2
    exit 2
    ;;
esac

trap 'mark_worker_failed "$FACTOR" "$?"' ERR
wait_for_balanced_test_cache

for specification in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r label prefix stop counterfactual <<<"$specification"
  overrides=(
    --set "loss.prefix=$prefix"
    --set "loss.stop=$stop"
    --set "loss.counterfactual=$counterfactual"
  )
  run_stage "${label}_train10k" "${DECGC[@]}" train-reasoner \
    --config "$CONFIG" --retriever-checkpoint best \
    --reranker-checkpoint best --limit "$TRAIN_QUESTIONS" \
    --seed "$SAMPLE_SEED" --resume auto "${overrides[@]}"
  run_stage "${label}_evaluate_test10k" "${DECGC[@]}" evaluate \
    --config "$CONFIG" --split test --checkpoint best \
    --limit "$TEST_QUESTIONS" --seed "$SAMPLE_SEED" \
    --resume auto "${overrides[@]}"
done

printf 'done\n' >"$STATUS_DIR/$FACTOR.done"
echo "GPU $GPU_ID completed factor: $FACTOR"
