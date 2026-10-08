#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/parameter_sensitivity_10k_common.sh"

on_failure() {
  local status=$?
  printf '%s\n' "$status" >"$prepare_failed_marker"
  exit "$status"
}
trap on_failure ERR

run_stage sensitivity_10k_doctor "${DECGC[@]}" doctor --config "$CONFIG"
run_stage sensitivity_10k_parse_test "${DECGC[@]}" parse-questions \
  --config "$CONFIG" --split test --limit "$TEST_QUESTIONS" \
  --seed "$SAMPLE_SEED" --resume auto

run_stage sensitivity_10k_verify_balanced_sample python -B \
  multitq_3loss_src/scripts/verify_balanced_sample.py \
  --config "$CONFIG" --split test --limit "$TEST_QUESTIONS" \
  --seed "$SAMPLE_SEED" \
  --require-parse-manifest \
  --output-prefix \
  "outputs/reports/parameter_sensitivity_multitq_test10k_sample"

run_stage sensitivity_10k_build_test_chains "${DECGC[@]}" build-chain-cache \
  --config "$CONFIG" --split test --limit "$TEST_QUESTIONS" \
  --seed "$SAMPLE_SEED" --retriever-checkpoint best \
  --reranker-checkpoint best --resume auto

temporary_marker="$ready_marker.tmp"
printf 'ready\n' >"$temporary_marker"
mv -f "$temporary_marker" "$ready_marker"
echo "Balanced MultiTQ test-10k cache is ready."
