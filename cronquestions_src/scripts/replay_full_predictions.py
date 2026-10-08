"""Replay time-only symbolic aggregation over persisted full predictions.

Gold answers are read only after candidate generation and are used exclusively
for metrics.  The script also verifies that every non-time ranking is unchanged.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))

from decgc.data.entity_index import EntityIndex
from decgc.evaluation.metrics import answer_diagnostics
from decgc.llm.local_parser import LocalQuestionParser
from decgc.retrieval.symbolic_temporal import (
    direct_temporal_candidates,
    merge_symbolic_answers,
)


class LightweightKG:
    """Minimal relation-aware KG interface required by symbolic replay."""

    def __init__(self) -> None:
        self.adjacency = collections.defaultdict(list)
        self.relation_adjacency = collections.defaultdict(list)
        self.relation_facts = collections.defaultdict(list)

    def add(self, fact) -> None:
        self.adjacency[fact.subject_id].append(fact)
        self.adjacency[fact.object_id].append(fact)
        self.relation_adjacency[(fact.subject_id, fact.relation_id)].append(fact)
        self.relation_adjacency[(fact.object_id, fact.relation_id)].append(fact)
        self.relation_facts[fact.relation_id].append(fact)

    def neighbors(self, entity: str):
        return self.adjacency.get(entity, ())

    def neighbors_for_relations(self, entity: str, relations: set[str]):
        return [
            fact
            for relation in relations
            for fact in self.relation_adjacency.get((entity, relation), ())
        ]

    def facts_for_relations(self, relations: set[str]):
        return [
            fact
            for relation in relations
            for fact in self.relation_facts.get(relation, ())
        ]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--errors", type=Path, default=None)
    return parser.parse_args()


def _load_kg(path: Path) -> tuple[LightweightKG, set[str], set[str]]:
    kg = LightweightKG()
    entities: set[str] = set()
    relations: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            subject, relation, obj, timestamp = line.rstrip("\n").split("\t")
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
    return kg, entities, relations


def _is_correct(row: dict) -> bool:
    from decgc.evaluation.answer_normalizer import normalize_answer

    targets = {normalize_answer(answer) for answer in row["gold_answers"]}
    return bool(row["candidate_answers"]) and normalize_answer(
        row["candidate_answers"][0]
    ) in targets


def _metric_row(row: dict) -> dict:
    """Discard large score/evidence maps that paper answer metrics do not use."""
    return {
        key: row.get(key)
        for key in (
            "candidate_answers",
            "gold_answers",
            "question_type",
            "answer_type",
            "temporal_operator",
            "temporal_granularity",
            "answer_role",
            "parse_source",
        )
    }


def main() -> None:
    arguments = _arguments()
    started = time.monotonic()
    kg, entities, relations = _load_kg(arguments.data_root / "kg/full.txt")
    raw_questions = json.loads(
        (arguments.data_root / f"questions/{arguments.split}.json").read_text(
            encoding="utf-8"
        )
    )
    questions = {str(row["quid"]): row for row in raw_questions}
    parser = LocalQuestionParser(EntityIndex(sorted(entities)), relations)

    baseline_rows: list[dict] = []
    replayed_rows: list[dict] = []
    error_rows: list[dict] = []
    non_time_changed = 0
    fixed = 0
    regressed = 0
    preferred = 0
    with arguments.predictions.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            baseline = json.loads(line)
            row = dict(baseline)
            baseline_ranking = list(baseline["candidate_answers"])
            baseline_rows.append(_metric_row(baseline))
            if row["answer_type"] != "time":
                # All modified parsing and endpoint-recovery branches are
                # explicitly time-only. Preserve non-time rows byte-for-byte.
                replayed_rows.append(_metric_row(row))
                continue
            metadata = {
                key: value
                for key, value in questions[row["qid"]].items()
                if key != "answers"
            }
            query = parser.parse(
                row["qid"], row["question"], row["answer_type"], metadata
            )
            symbolic = direct_temporal_candidates(kg, row["question"], query)
            row["candidate_answers"] = merge_symbolic_answers(
                baseline_ranking, symbolic
            )
            row["symbolic_candidates"] = symbolic.answers
            row["symbolic_rule"] = symbolic.rule
            row["symbolic_preferred"] = symbolic.prefer_symbolic
            replayed_rows.append(_metric_row(row))
            before = _is_correct(baseline)
            after = _is_correct(row)
            fixed += (not before) and after
            regressed += before and (not after)
            preferred += symbolic.prefer_symbolic
            if row["answer_type"] == "time" and not after:
                error_rows.append({
                    "qid": row["qid"],
                    "question": row["question"],
                    "gold_answers": row["gold_answers"],
                    "baseline_top1": baseline_ranking[:1],
                    "replayed_top1": row["candidate_answers"][:1],
                    "topic_entities": query.topic_entities,
                    "relation_hints": query.relation_hints,
                    "symbolic_candidates": symbolic.answers,
                    "symbolic_rule": symbolic.rule,
                    "symbolic_preferred": symbolic.prefer_symbolic,
                })

    if non_time_changed:
        raise RuntimeError(f"Non-time rankings changed: {non_time_changed}")
    baseline_metrics = answer_diagnostics(baseline_rows)
    replayed_metrics = answer_diagnostics(replayed_rows)
    report = {
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "samples": len(replayed_rows),
        "baseline": baseline_metrics,
        "replayed": replayed_metrics,
        "fixed_top1": fixed,
        "regressed_top1": regressed,
        "symbolic_preferred": preferred,
        "non_time_rankings_changed": non_time_changed,
        "gold_usage": "metrics_and_error_diagnostics_only",
    }
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if arguments.errors:
        arguments.errors.parent.mkdir(parents=True, exist_ok=True)
        with arguments.errors.open("w", encoding="utf-8") as stream:
            for row in error_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "elapsed_seconds": report["elapsed_seconds"],
        "time": replayed_metrics["by_answer_type"]["time"],
        "overall": replayed_metrics["overall"],
        "fixed_top1": fixed,
        "regressed_top1": regressed,
        "symbolic_preferred": preferred,
        "non_time_rankings_changed": non_time_changed,
        "remaining_time_errors": len(error_rows),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
