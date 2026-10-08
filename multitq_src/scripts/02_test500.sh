#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/four_loss_common.sh"

run_stage parse_test500 "${DECGC[@]}" parse-questions \
  --config "$CONFIG" --split test --limit 500 --resume auto

run_stage build_test500_chains "${DECGC[@]}" build-chain-cache \
  --config "$CONFIG" --split test --limit 500 \
  --retriever-checkpoint best --reranker-checkpoint best --resume auto

run_stage evaluate_test500 "${DECGC[@]}" evaluate \
  --config "$CONFIG" --split test --checkpoint best --limit 500 --resume auto

RUN_ID=$(python -B - "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path

from decgc.config import load_config

config, _ = load_config(Path(sys.argv[1]))
candidates = []
for path in config.paths.predictions_root.glob("frozen-*/evaluation_manifest.json"):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("completed") and data.get("questions") == 500:
        candidates.append(path)
if not candidates:
    raise SystemExit("No completed 500-question frozen evaluation found")
print(max(candidates, key=lambda p: p.stat().st_mtime_ns).parent.name)
PY
)

python -B multitq_src/scripts/export_paper_metrics.py \
  --diagnostics "outputs/reports/$RUN_ID.diagnostics.json" \
  --output-prefix "outputs/reports/$RUN_ID.paper_metrics" \
  --model "DECGC-4Loss" --decimals 1

echo "Test-500 run: $RUN_ID"
echo "Metrics: outputs/reports/$RUN_ID.json"
echo "Evidence: outputs/evidence/$RUN_ID/supporting_evidence_chains.jsonl"
