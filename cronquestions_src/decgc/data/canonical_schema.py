"""Canonical field conversion functions."""

from __future__ import annotations

from .temporal_parser import parse_time
from ..schemas import QAExample, TemporalFact
from ..utils.hashing import object_hash


def canonical_fact(parts: list[str], split: str, row_index: int) -> TemporalFact:
    if len(parts) != 4:
        raise ValueError(f"Expected four tab-separated KG fields, found {len(parts)}")
    subject, relation, obj, timestamp = (part.strip() for part in parts)
    parsed = parse_time(timestamp)
    record_hash = object_hash(parts)
    return TemporalFact(
        fact_id=f"{split}:{row_index}:{record_hash[:12]}",
        subject_id=subject,
        subject_text=subject.replace("_", " "),
        relation_id=relation,
        relation_text=relation.replace("_", " "),
        object_id=obj,
        object_text=obj.replace("_", " "),
        timestamp_raw=timestamp,
        timestamp_normalized=parsed["normalized"],
        time_start=parsed["start"],
        time_end=parsed["end"],
        time_granularity=str(parsed["granularity"]),
        source_relative_path=f"kg/{split}.txt",
        source_row_index=row_index,
        source_record_hash=record_hash,
    )


def canonical_question(record: dict, split: str, row_index: int, source_name: str) -> QAExample:
    required = {"quid", "question", "answers"}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"Question record is missing: {sorted(missing)}")
    return QAExample(
        qid=str(record["quid"]),
        question=str(record["question"]),
        answers=[str(item) for item in record["answers"]],
        split=split,
        question_type=record.get("qlabel"),
        answer_type=record.get("answer_type"),
        raw_record=record,
        source_relative_path=f"questions/{source_name}",
        source_row_index=row_index,
    )

