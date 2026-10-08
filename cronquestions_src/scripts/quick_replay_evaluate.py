"""Fast, read-only-model regression over persisted Demo predictions and the real KG."""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT.parent
sys.path.insert(0, str(SOURCE_ROOT))

from decgc.data.entity_index import EntityIndex
from decgc.evaluation.metrics import answer_metrics
from decgc.llm.local_parser import LocalQuestionParser
from decgc.retrieval.symbolic_temporal import (
    direct_temporal_candidates,
    merge_symbolic_answers,
)
from decgc.utils.io import write_json


class LightweightKG:
    def __init__(self):
        self.adjacency = collections.defaultdict(list)

    def add(self, fact) -> None:
        self.adjacency[fact.subject_id].append(fact)
        self.adjacency[fact.object_id].append(fact)

    def neighbors(self, entity: str):
        return self.adjacency.get(entity, [])


def _latest_run() -> Path:
    runs = [
        path
        for path in (PROJECT_ROOT / "outputs/demo").glob("multitq-*")
        if (path / "demo_predictions.jsonl").exists()
    ]
    if not runs:
        raise RuntimeError("No persisted Demo predictions were found")
    return max(runs, key=lambda path: (path / "demo_predictions.jsonl").stat().st_mtime_ns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--hits1-target", type=float, default=0.815)
    parser.add_argument("--hits10-target", type=float, default=0.985)
    arguments = parser.parse_args()
    run_dir = (arguments.run_dir or _latest_run()).resolve()
    started = time.monotonic()

    kg = LightweightKG()
    entities, relations = set(), set()
    kg_path = PROJECT_ROOT / "data/MultiTQ/kg/full.txt"
    with kg_path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            subject, relation, obj, timestamp = line.rstrip().split("\t")
            fact = SimpleNamespace(
                fact_id=f"full:{index}",
                subject_id=subject,
                object_id=obj,
                relation_id=relation,
                timestamp_raw=timestamp,
            )
            kg.add(fact)
            entities.update((subject, obj))
            relations.add(relation)

    question_rows = json.loads(
        (PROJECT_ROOT / "data/MultiTQ/questions/dev.json").read_text(encoding="utf-8")
    )
    questions = {str(row["quid"]): row for row in question_rows}
    predictions = [
        json.loads(line)
        for line in (run_dir / "demo_predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    baseline = answer_metrics(predictions)
    question_parser = LocalQuestionParser(EntityIndex(list(entities)), relations)
    replayed = []
    for prediction in predictions:
        row = dict(prediction)
        query = question_parser.parse(
            row["qid"],
            row["question"],
            row["answer_type"],
            {
                key: value
                for key, value in questions[row["qid"]].items()
                if key != "answers"
            },
        )
        symbolic = direct_temporal_candidates(kg, row["question"], query)
        row["candidate_answers"] = merge_symbolic_answers(
            row["candidate_answers"],
            symbolic,
        )
        replayed.append(row)
    improved = answer_metrics(replayed)
    report = {
        "run_dir": str(run_dir),
        "samples": len(replayed),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "baseline": baseline,
        "improved": improved,
        "delta": {
            "hits1": improved["hits1"] - baseline["hits1"],
            "hits10": improved["hits10"] - baseline["hits10"],
        },
        "targets": {
            "hits1": arguments.hits1_target,
            "hits10": arguments.hits10_target,
        },
        "passed": (
            improved["hits1"] >= arguments.hits1_target
            and improved["hits10"] >= arguments.hits10_target
        ),
        "gold_usage": "metrics_only",
        "prediction_rule": "structured_question_plus_full_kg_without_gold",
    }
    output = PROJECT_ROOT / "outputs/reports/quick_replay_metrics.json"
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
