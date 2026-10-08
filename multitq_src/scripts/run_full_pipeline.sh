#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${DECGC_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
CONFIG=multitq_src/configs/experiments/multitq_full.yaml
echo "Run only after freezing the full configuration; official test is evaluated once."
decgc evaluate --config "$CONFIG" --split test --checkpoint best --resume auto
decgc export-results --config "$CONFIG"
