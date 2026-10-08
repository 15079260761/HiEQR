"""Audit the persisted 30k result and report the target without rewriting it."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--run-id", required=True)
arguments = parser.parse_args()
root = Path("cronquestions_outputs")
prediction = root / "predictions" / arguments.run_id / "predictions.jsonl"
evidence = root / "evidence" / arguments.run_id
diagnostics = root / "reports" / f"{arguments.run_id}.diagnostics.json"
count = sum(1 for line in prediction.open(encoding="utf-8") if line.strip())
assert count == 30000, f"expected 30000 predictions, got {count}"
required = (
    evidence / "supporting_evidence_chains.jsonl",
    evidence / "evidence_validation.json",
    evidence / "human_readable_evidence.md",
    evidence / "all_chain_audit.manifest.json",
)
assert all(path.exists() for path in required), "one or more evidence outputs are missing"
overall = json.loads(diagnostics.read_text(encoding="utf-8"))["overall"]
result = {
    "samples": count,
    "hits1": overall["hits1"],
    "hits10": overall["hits10"],
    "declared_hits1_target": 0.985,
    "target_met": overall["hits1"] >= 0.985,
}
print(json.dumps(result, ensure_ascii=False, indent=2))
if not result["target_met"]:
    print("WARNING: Hits@1 target not met; persisted experimental values are unchanged.")
