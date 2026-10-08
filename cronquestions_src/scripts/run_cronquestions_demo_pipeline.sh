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

CONFIG=cronquestions_src/configs/experiments/cronquestions_demo.yaml
LOG_DIR=cronquestions_outputs/logs
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/demo_pipeline_$(date +%Y%m%d_%H%M%S).log"

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
run_stage doctor python -m decgc.cli doctor --config "$CONFIG"
run_stage inspect-data python -m decgc.cli inspect-data --config "$CONFIG" --resume auto
run_stage build-kg-index python -m decgc.cli build-kg-index --config "$CONFIG" --resume auto
run_stage parse-train python -m decgc.cli parse-questions --config "$CONFIG" --split train --no-online --resume auto
run_stage parse-valid python -m decgc.cli parse-questions --config "$CONFIG" --split valid --no-online --resume auto
run_stage parse-test-demo python -m decgc.cli parse-questions --config "$CONFIG" --split test --limit 500 --no-online --resume auto
run_stage verify-parse-quality python -m decgc.cli verify-parse-quality --config "$CONFIG"
run_stage build-retriever-data python -m decgc.cli build-retriever-data --config "$CONFIG" --resume auto
run_stage cleanup-before-retriever python -m decgc.cli cleanup-obsolete-artifacts --config "$CONFIG" --execute
run_stage train-retriever python -m decgc.cli train-retriever --config "$CONFIG" --resume auto
run_stage encode-facts python -m decgc.cli encode-facts --config "$CONFIG" --retriever-checkpoint best --resume auto
run_stage build-fact-index python -m decgc.cli build-fact-index --config "$CONFIG" --resume auto
run_stage build-reranker-data python -m decgc.cli build-reranker-data --config "$CONFIG" --retriever-checkpoint best --resume auto
run_stage cleanup-before-reranker python -m decgc.cli cleanup-obsolete-artifacts --config "$CONFIG" --execute
run_stage train-reranker python -m decgc.cli train-reranker --config "$CONFIG" --resume auto
run_stage cache-train-chains python -m decgc.cli build-chain-cache --config "$CONFIG" --split train --resume auto
run_stage cache-valid-chains python -m decgc.cli build-chain-cache --config "$CONFIG" --split valid --resume auto
run_stage build-counterfactuals python -m decgc.cli build-counterfactuals --config "$CONFIG" --split train --resume auto
run_stage cleanup-before-reasoner python -m decgc.cli cleanup-obsolete-artifacts --config "$CONFIG" --execute
run_stage train-reasoner python -m decgc.cli train-reasoner --config "$CONFIG" --resume auto
run_stage demo-evaluate python -m decgc.cli demo-evaluate --config "$CONFIG" --split test --checkpoint best --limit 500 --resume auto
run_stage demo-predict python -m decgc.cli demo-predict --config "$CONFIG" --split test --checkpoint best --num-examples 20 \
  --export-evidence --export-graph-json --export-human-readable --resume auto
run_stage generate-report python -m decgc.cli generate-demo-report --config "$CONFIG" --checkpoint best --resume auto
run_stage verify-targets python -m decgc.cli verify-demo-targets --config "$CONFIG"
run_stage cleanup-finish python -m decgc.cli cleanup-obsolete-artifacts --config "$CONFIG" --execute
printf '\nPipeline log: %s\n' "$LOG_FILE"
