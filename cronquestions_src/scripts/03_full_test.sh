#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/cronquestions_3loss_common.sh"

run_stage parse_full_test "${DECGC[@]}" parse-questions \
  --config "$FULL_CONFIG" --split test --no-online --resume auto
run_stage reuse_full_test_chains "${DECGC[@]}" build-chain-cache \
  --config "$FULL_CONFIG" --split test \
  --retriever-checkpoint best --reranker-checkpoint best --resume auto
run_stage evaluate_full_test "${DECGC[@]}" evaluate \
  --config "$FULL_CONFIG" --split test --checkpoint best --resume auto

RUN_ID=$(python -B - "$FULL_CONFIG" <<'PY'
import json, sys
from pathlib import Path
from decgc.config import load_config
config, _ = load_config(Path(sys.argv[1]))
candidates=[]
for path in config.paths.predictions_root.glob("frozen-*/evaluation_manifest.json"):
    data=json.loads(path.read_text(encoding="utf-8"))
    signature=data.get("signature", {})
    if (data.get("completed") and data.get("questions")==30000
            and signature.get("config_hash")==config.hash):
        candidates.append(path)
if not candidates: raise SystemExit("No completed 30,000-question evaluation found")
print(max(candidates,key=lambda path:path.stat().st_mtime_ns).parent.name)
PY
)

run_stage export_full_paper_metrics python -B \
  cronquestions_src/scripts/export_paper_metrics.py \
  --diagnostics "cronquestions_outputs/reports/$RUN_ID.diagnostics.json" \
  --output-prefix "cronquestions_outputs/reports/$RUN_ID.three_loss.paper_metrics" \
  --model "HiEQR" --decimals 1

python -B cronquestions_src/scripts/verify_three_loss_full_result.py \
  --run-id "$RUN_ID"
run_stage export_result_manifest "${DECGC[@]}" export-results \
  --config "$FULL_CONFIG"

echo "Official full-test run: $RUN_ID"
echo "Metrics: cronquestions_outputs/reports/$RUN_ID.json"
echo "Diagnostics: cronquestions_outputs/reports/$RUN_ID.diagnostics.json"
echo "Paper table: cronquestions_outputs/reports/$RUN_ID.three_loss.paper_metrics.csv"
echo "Evidence: cronquestions_outputs/evidence/$RUN_ID/"
