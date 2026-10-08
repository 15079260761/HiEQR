#!/usr/bin/env bash

# Shared environment for the paper-aligned MultiTQ four-loss experiment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${DECGC_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_ROOT"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT/multitq_src${PYTHONPATH:+:$PYTHONPATH}"

DECGC=(python -B -m decgc.cli)

CONFIG="multitq_src/configs/experiments/multitq_4loss_main.yaml"
LOG_DIR="outputs/logs/main-4loss-multitq-10k"
mkdir -p "$LOG_DIR"

run_stage() {
  local name="$1"
  shift
  local log_file="$LOG_DIR/${name}.log"
  local started=$SECONDS
  printf '\n[%s] START %s\n' "$(date '+%F %T')" "$name" | tee -a "$log_file"
  set +e
  "$@" 2>&1 | tee -a "$log_file"
  local status=${PIPESTATUS[0]}
  set -e
  if (( status != 0 )); then
    printf '[%s] FAIL %s (%ds, exit=%d)\n' \
      "$(date '+%F %T')" "$name" "$((SECONDS-started))" "$status" | tee -a "$log_file"
    return "$status"
  fi
  printf '[%s] DONE %s (%ds)\n' \
    "$(date '+%F %T')" "$name" "$((SECONDS-started))" | tee -a "$log_file"
}
