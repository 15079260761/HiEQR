#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/parameter_sensitivity_10k_common.sh"

factors=(prefix stop counterfactual)
while true; do
  all_done=true
  for factor in "${factors[@]}"; do
    if [[ -f "$STATUS_DIR/$factor.failed" ]]; then
      echo "Worker failed: $factor; see $LOG_DIR" >&2
      exit 1
    fi
    if [[ ! -f "$STATUS_DIR/$factor.done" ]]; then
      all_done=false
    fi
  done
  [[ "$all_done" == true ]] && break
  sleep 20
done

python -B multitq_3loss_src/scripts/collect_parameter_sensitivity.py \
  --config "$CONFIG" --questions "$TEST_QUESTIONS" --exclude-default \
  --output-prefix \
  "outputs/reports/parameter_sensitivity_multitq_test10k_nondefault"
