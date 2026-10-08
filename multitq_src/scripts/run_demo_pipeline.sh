#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${DECGC_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
CONFIG=multitq_src/configs/experiments/multitq_demo.yaml
LOG_DIR=outputs/logs
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
    printf '[%s] FAIL  %s (%ds, exit=%d)\n' \
      "$(date '+%F %T')" "$name" "$((SECONDS-started))" "$status" | tee -a "$LOG_FILE"
    return "$status"
  fi
  printf '[%s] DONE  %s (%ds)\n' "$(date '+%F %T')" "$name" "$((SECONDS-started))" | tee -a "$LOG_FILE"
}

run_stage cleanup-obsolete decgc cleanup-obsolete-artifacts --config "$CONFIG" --execute
run_stage doctor decgc doctor --config "$CONFIG"
run_stage inspect-data decgc inspect-data --config "$CONFIG" --resume auto
run_stage build-kg-index decgc build-kg-index --config "$CONFIG" --resume auto
# Demo development never touches the held-out test set.
run_stage parse-train decgc parse-questions --config "$CONFIG" --split train --resume auto
run_stage parse-valid decgc parse-questions --config "$CONFIG" --split valid --resume auto
run_stage verify-parse-quality decgc verify-parse-quality --config "$CONFIG"
run_stage build-retriever-data decgc build-retriever-data --config "$CONFIG" --resume auto
run_stage cleanup-before-retriever decgc cleanup-obsolete-artifacts --config "$CONFIG" --execute
run_stage train-retriever decgc train-retriever --config "$CONFIG" --resume auto
run_stage encode-facts decgc encode-facts --config "$CONFIG" --retriever-checkpoint best --resume auto
run_stage build-fact-index decgc build-fact-index --config "$CONFIG" --resume auto
run_stage build-reranker-data decgc build-reranker-data --config "$CONFIG" --retriever-checkpoint best --resume auto
run_stage cleanup-before-reranker decgc cleanup-obsolete-artifacts --config "$CONFIG" --execute
run_stage train-reranker decgc train-reranker --config "$CONFIG" --resume auto
run_stage cache-train-chains decgc build-chain-cache --config "$CONFIG" --split train --resume auto
run_stage cache-valid-chains decgc build-chain-cache --config "$CONFIG" --split valid --resume auto
run_stage build-counterfactuals decgc build-counterfactuals --config "$CONFIG" --split train --resume auto
run_stage cleanup-before-reasoner decgc cleanup-obsolete-artifacts --config "$CONFIG" --execute
run_stage train-reasoner decgc train-reasoner --config "$CONFIG" --resume auto
run_stage demo-evaluate decgc demo-evaluate --config "$CONFIG" --split valid --checkpoint best --limit 500 --resume auto
run_stage demo-predict decgc demo-predict --config "$CONFIG" --split valid --checkpoint best --num-examples 20 \
  --export-evidence --export-graph-json --export-human-readable --resume auto
run_stage generate-report decgc generate-demo-report --config "$CONFIG" --checkpoint best --resume auto
run_stage verify-targets decgc verify-demo-targets --config "$CONFIG"
printf '\nPipeline log: %s\n' "$LOG_FILE"
