#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${DECGC_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT/cronquestions_src${PYTHONPATH:+:$PYTHONPATH}"

CONFIG=cronquestions_src/configs/experiments/cronquestions_full.yaml
LOG_DIR=cronquestions_outputs/logs
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/full_test_$(date +%Y%m%d_%H%M%S).log"

run_stage() {
  local name="$1"
  shift
  local started=$SECONDS
  printf '\n[%s] START %s\n' "$(date '+%F %T')" "$name" | tee -a "$LOG_FILE"
  set +e
  "$@" 2>&1 | tee -a "$LOG_FILE"
  local status=${PIPESTATUS[0]}
  set -e
  if (( status != 0 )); then
    printf '[%s] FAIL %s (%ds, exit=%d)\n' "$(date '+%F %T')" "$name" "$((SECONDS-started))" "$status" | tee -a "$LOG_FILE"
    return "$status"
  fi
  printf '[%s] DONE %s (%ds)\n' "$(date '+%F %T')" "$name" "$((SECONDS-started))" | tee -a "$LOG_FILE"
}

run_stage cleanup-start python -m decgc.cli cleanup-obsolete-artifacts --config "$CONFIG" --execute
run_stage parse-test python -m decgc.cli parse-questions --config "$CONFIG" --split test --no-online --resume auto
run_stage cache-test-chains python -m decgc.cli build-chain-cache --config "$CONFIG" --split test \
  --retriever-checkpoint best --reranker-checkpoint best --resume auto
run_stage evaluate-full-test python -m decgc.cli evaluate --config "$CONFIG" --split test --checkpoint best --resume auto

RUN_ID=$(sed -n 's/^Frozen evaluation output: //p' "$LOG_FILE" | tail -n 1 | tr -d '\r')
if [[ ! "$RUN_ID" =~ ^frozen-[0-9a-f]{12}$ ]]; then
  echo "Unable to resolve the exact frozen run id from $LOG_FILE: $RUN_ID" >&2
  exit 3
fi
DIAGNOSTICS="cronquestions_outputs/reports/$RUN_ID.diagnostics.json"
if [[ ! -f "$DIAGNOSTICS" ]]; then
  echo "Expected diagnostics file is missing: $DIAGNOSTICS" >&2
  exit 4
fi
run_stage export-paper-metrics python -B cronquestions_src/scripts/export_paper_metrics.py \
  --diagnostics "$DIAGNOSTICS" --model DECGC --decimals 1
run_stage export-results python -m decgc.cli export-results --config "$CONFIG"
run_stage cleanup-finish python -m decgc.cli cleanup-obsolete-artifacts --config "$CONFIG" --execute
printf '\nFull-test log: %s\nDiagnostics: %s\n' "$LOG_FILE" "$DIAGNOSTICS"
