#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/four_loss_common.sh"

FULL_OVERRIDES=(
  --set demo.evaluation_max_samples=null
  --set evaluation.split=test
  --set evaluation.max_samples=null
)

run_stage parse_full_test "${DECGC[@]}" parse-questions \
  --config "$CONFIG" --split test --resume auto "${FULL_OVERRIDES[@]}"

run_stage build_full_test_chains "${DECGC[@]}" build-chain-cache \
  --config "$CONFIG" --split test \
  --retriever-checkpoint best --reranker-checkpoint best --resume auto \
  "${FULL_OVERRIDES[@]}"

run_stage evaluate_full_test "${DECGC[@]}" evaluate \
  --config "$CONFIG" --split test --checkpoint best --resume auto \
  "${FULL_OVERRIDES[@]}"

RUN_ID=$(python -B - "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path

from decgc.config import load_config

config, _ = load_config(Path(sys.argv[1]), [
    "demo.evaluation_max_samples=null",
    "evaluation.split=test",
    "evaluation.max_samples=null",
])
expected = 54584
candidates = []
for path in config.paths.predictions_root.glob("frozen-*/evaluation_manifest.json"):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("completed") and data.get("questions") == expected:
        candidates.append(path)
if not candidates:
    raise SystemExit(f"No completed {expected}-question frozen evaluation found")
print(max(candidates, key=lambda p: p.stat().st_mtime_ns).parent.name)
PY
)

python -B multitq_src/scripts/export_paper_metrics.py \
  --diagnostics "outputs/reports/$RUN_ID.diagnostics.json" \
  --output-prefix "outputs/reports/$RUN_ID.paper_metrics" \
  --model "DECGC-4Loss" --decimals 1

python -B - "$RUN_ID" <<'PY'
import json
import sys
from pathlib import Path

run_id = sys.argv[1]
prediction = Path("outputs/predictions") / run_id / "predictions.jsonl"
evidence = Path("outputs/evidence") / run_id
count = sum(1 for line in prediction.open(encoding="utf-8") if line.strip())
assert count == 54584, f"expected 54584 predictions, got {count}"
required = (
    evidence / "supporting_evidence_chains.jsonl",
    evidence / "evidence_validation.json",
    evidence / "human_readable_evidence.md",
    evidence / "all_chain_audit.manifest.json",
)
assert all(path.exists() for path in required), "one or more evidence outputs are missing"
validation = json.loads((evidence / "evidence_validation.json").read_text(encoding="utf-8"))
print("Full test predictions:", count)
print("Evidence outputs verified:", evidence)
print("Evidence validation:", json.dumps(validation, ensure_ascii=False))
PY

run_stage export_result_manifest "${DECGC[@]}" export-results \
  --config "$CONFIG" "${FULL_OVERRIDES[@]}"

echo "Official full-test run: $RUN_ID"
echo "Metrics: outputs/reports/$RUN_ID.json"
echo "Diagnostics: outputs/reports/$RUN_ID.diagnostics.json"
echo "Paper table: outputs/reports/$RUN_ID.paper_metrics.csv"
echo "Evidence: outputs/evidence/$RUN_ID/"
