#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/cronquestions_4loss_common.sh"

run_stage parse_test500 "${DECGC[@]}" parse-questions \
  --config "$CONFIG" --split test --limit 500 --no-online --resume auto
run_stage reuse_test500_chains "${DECGC[@]}" build-chain-cache \
  --config "$CONFIG" --split test --limit 500 \
  --retriever-checkpoint best --reranker-checkpoint best --resume auto
run_stage evaluate_test500 "${DECGC[@]}" evaluate \
  --config "$CONFIG" --split test --checkpoint best --limit 500 --resume auto

RUN_ID=$(python -B - "$CONFIG" <<'PY'
import json, sys
from pathlib import Path
from decgc.config import load_config
config, _ = load_config(Path(sys.argv[1]))
candidates=[]
for path in config.paths.predictions_root.glob("frozen-*/evaluation_manifest.json"):
    data=json.loads(path.read_text(encoding="utf-8"))
    signature=data.get("signature", {})
    if (data.get("completed") and data.get("questions")==500
            and signature.get("config_hash")==config.hash):
        candidates.append(path)
if not candidates: raise SystemExit("No completed 500-question evaluation found")
print(max(candidates,key=lambda path:path.stat().st_mtime_ns).parent.name)
PY
)

run_stage export_test500_paper_metrics python -B \
  cronquestions_src/scripts/export_paper_metrics.py \
  --diagnostics "cronquestions_outputs/reports/$RUN_ID.diagnostics.json" \
  --output-prefix "cronquestions_outputs/reports/$RUN_ID.four_loss.paper_metrics" \
  --model "DECGC-4Loss" --decimals 1

echo "Test-500 run: $RUN_ID"
echo "Metrics: cronquestions_outputs/reports/$RUN_ID.json"
echo "Evidence: cronquestions_outputs/evidence/$RUN_ID/supporting_evidence_chains.jsonl"
