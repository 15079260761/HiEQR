#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/three_loss_common.sh"

if [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
fi

TRAIN_QUESTIONS=10000
TEST_QUESTIONS=10000
SAMPLE_SEED=42
RUN_TOKEN="${PARAM_RUN_TOKEN:?PARAM_RUN_TOKEN must identify this tmux launch}"
LOG_DIR="$PROJECT_ROOT/outputs/logs/parameter_sensitivity_10k/$RUN_TOKEN"
STATUS_DIR="$PROJECT_ROOT/outputs/manifests/parameter_sensitivity_10k/$RUN_TOKEN"
mkdir -p "$LOG_DIR" "$STATUS_DIR"

ready_marker="$STATUS_DIR/test10k.ready"
prepare_failed_marker="$STATUS_DIR/test10k.failed"

wait_for_balanced_test_cache() {
  while [[ ! -f "$ready_marker" ]]; do
    if [[ -f "$prepare_failed_marker" ]]; then
      echo "Balanced test-10k preparation failed; see $LOG_DIR" >&2
      return 1
    fi
    sleep 20
  done
}

mark_worker_failed() {
  local factor="$1"
  local status="$2"
  printf '%s\n' "$status" >"$STATUS_DIR/$factor.failed"
}
