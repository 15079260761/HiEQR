#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/three_loss_common.sh"

# Use SENSITIVITY_EVAL_QUESTIONS=full for the final paper run. The default 500
# is a quick matched check before committing to all 54,584 MultiTQ test items.
EVAL_QUESTIONS="${SENSITIVITY_EVAL_QUESTIONS:-500}"
if [[ "$EVAL_QUESTIONS" == "full" ]]; then
  EVAL_LIMIT=()
else
  [[ "$EVAL_QUESTIONS" =~ ^[1-9][0-9]*$ ]] || {
    echo "SENSITIVITY_EVAL_QUESTIONS must be a positive integer or 'full'" >&2
    exit 2
  }
  EVAL_LIMIT=(--limit "$EVAL_QUESTIONS")
fi

# The default point is shared by all three curves, so 12 plotted points require
# only 10 unique trainings.
EXPERIMENTS=(
  "prefix_0.00|0.00|0.30|0.30"
  "prefix_0.25|0.25|0.30|0.30"
  "default|0.50|0.30|0.30"
  "prefix_0.75|0.75|0.30|0.30"
  "stop_0.00|0.50|0.00|0.30"
  "stop_0.15|0.50|0.15|0.30"
  "stop_0.45|0.50|0.45|0.30"
  "counterfactual_0.00|0.50|0.30|0.00"
  "counterfactual_0.15|0.50|0.30|0.15"
  "counterfactual_0.45|0.50|0.30|0.45"
)

# Parsing and candidate chains are independent of reasoner loss weights.
run_stage sensitivity_parse_test "${DECGC[@]}" parse-questions \
  --config "$CONFIG" --split test "${EVAL_LIMIT[@]}" --resume auto
run_stage sensitivity_build_test_chains "${DECGC[@]}" build-chain-cache \
  --config "$CONFIG" --split test "${EVAL_LIMIT[@]}" \
  --retriever-checkpoint best --reranker-checkpoint best --resume auto

for specification in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r label prefix stop counterfactual <<<"$specification"
  overrides=(
    --set "loss.prefix=$prefix"
    --set "loss.stop=$stop"
    --set "loss.counterfactual=$counterfactual"
  )
  run_stage "sensitivity_${label}_train" "${DECGC[@]}" train-reasoner \
    --config "$CONFIG" \
    --retriever-checkpoint best --reranker-checkpoint best \
    --limit 10000 --resume auto "${overrides[@]}"
  run_stage "sensitivity_${label}_evaluate_${EVAL_QUESTIONS}" \
    "${DECGC[@]}" evaluate --config "$CONFIG" --split test \
    --checkpoint best --resume auto "${EVAL_LIMIT[@]}" "${overrides[@]}"
done

OUTPUT_PREFIX="outputs/reports/parameter_sensitivity_multitq_${EVAL_QUESTIONS}"
python -B multitq_3loss_src/scripts/collect_parameter_sensitivity.py \
  --config "$CONFIG" \
  --questions "$EVAL_QUESTIONS" \
  --output-prefix "$OUTPUT_PREFIX"
