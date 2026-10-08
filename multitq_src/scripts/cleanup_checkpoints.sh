#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${DECGC_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_ROOT"
CONFIG=multitq_src/configs/experiments/multitq_demo.yaml
decgc cleanup-checkpoints --config "$CONFIG" --dry-run
read -r -p "Execute this checkpoint cleanup? [y/N] " answer
if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
  decgc cleanup-checkpoints --config "$CONFIG" --execute
fi

