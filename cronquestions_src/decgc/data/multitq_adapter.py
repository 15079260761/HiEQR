"""CronQuestions adapter.

The historical module name is retained so the unchanged DECGC pipeline can use
the new dataset without altering the retrieval/reranking/reasoning algorithm.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..config import Config
from ..schemas import QAExample, TemporalFact
from ..utils.hashing import object_hash
from ..utils.io import read_json
from .canonical_schema import canonical_fact, canonical_question


SIMPLE_TYPES = {"simple_entity", "simple_time"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, set):
        return sorted((_json_safe(item) for item in value), key=str)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class CronQuestionsAdapter:
    def __init__(self, config: Config):
        self.config = config
        self.dataset = config.dataset
        self._entities: dict[str, str] | None = None
        self._relations: dict[str, str] | None = None

    def kg_path(self, split: str) -> Path:
        return self.config.paths.kg_root / self.dataset["kg_files"][split]

    def question_path(self, split: str) -> Path:
        return self.config.paths.questions_root / self.dataset["question_files"][split]

    @staticmethod
    def _load_text_map(path: Path) -> dict[str, str]:
        result: dict[str, str] = {}
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                key, separator, text = line.rstrip("\r\n").partition("\t")
                if separator:
                    result[key] = text.strip() or key
        return result

    def _ensure_text_maps(self) -> None:
        if self._entities is None:
            self._entities = self._load_text_map(
                self.config.paths.kg_root / self.dataset["entity_text_file"]
            )
        if self._relations is None:
            self._relations = self._load_text_map(
                self.config.paths.kg_root / self.dataset["relation_text_file"]
            )

    def iter_facts(self, split: str = "full", limit: int | None = None) -> Iterator[TemporalFact]:
        self._ensure_text_maps()
        assert self._entities is not None and self._relations is not None
        with self.kg_path(split).open("r", encoding=self.dataset.get("encoding", "utf-8")) as stream:
            for index, line in enumerate(stream):
                if limit is not None and index >= limit:
                    return
                if not line.strip():
                    continue
                parts = [part.strip() for part in line.rstrip("\r\n").split("\t")]
                if len(parts) != 5:
                    raise ValueError(
                        f"CronQuestions KG row {index} in {split!r} has {len(parts)} fields; expected 5"
                    )
                subject, relation, obj, start_year, end_year = parts
                start = f"{int(start_year):04d}-01-01"
                end = f"{int(end_year):04d}-12-31"
                raw_time = start_year if start_year == end_year else f"{start_year}/{end_year}"
                record_hash = object_hash(parts)
                yield TemporalFact(
                    fact_id=f"{split}:{index}:{record_hash[:12]}",
                    subject_id=subject,
                    subject_text=self._entities.get(subject, subject),
                    relation_id=relation,
                    relation_text=self._relations.get(relation, relation),
                    object_id=obj,
                    object_text=self._entities.get(obj, obj),
                    timestamp_raw=raw_time,
                    timestamp_normalized=raw_time,
                    time_start=start,
                    time_end=end,
                    time_granularity="year",
                    source_relative_path=f"kg/{self.dataset['kg_files'][split]}",
                    source_row_index=index,
                    source_record_hash=record_hash,
                )

    def load_questions(self, split: str, limit: int | None = None) -> list[QAExample]:
        filename = self.dataset["question_files"][split]
        with self.question_path(split).open("rb") as stream:
            records = pickle.load(stream)
        selected = records if limit is None else records[:limit]
        result: list[QAExample] = []
        for index, source in enumerate(selected):
            record = _json_safe(source)
            cron_type = str(record["type"])
            annotation = dict(record.get("annotation", {}))
            raw_id = str(record["uniq_id"])
            # uniq_id repeats within and across official files; row index makes
            # cache keys and predictions stable and collision free.
            qid = f"{split}:{index}:{raw_id}"
            paraphrases = record.get("paraphrases") or []
            question = str(paraphrases[0] if paraphrases else record["question"])
            raw_question = str(record["question"])
            official_entities = [str(value) for value in record.get("entities", [])]
            official_entities.sort(
                key=lambda value: (
                    raw_question.find(value) < 0,
                    raw_question.find(value),
                    value,
                )
            )
            operator = annotation.get("adj") or annotation.get("type")
            if not operator and annotation.get("time") is not None:
                operator = "equal"
            record.update({
                "_dataset": "CronQuestions",
                "cron_type": cron_type,
                "qtype": operator,
                "time_level": "year",
                "official_entities": official_entities,
                "official_relations": sorted(record.get("relations", [])),
                "official_times": [str(value) for value in sorted(record.get("times", []))],
            })
            result.append(QAExample(
                qid=qid,
                question=question,
                answers=[str(item) for item in sorted(record["answers"], key=str)],
                split=split,
                question_type="Simple" if cron_type in SIMPLE_TYPES else "Complex",
                answer_type=str(record["answer_type"]).casefold(),
                raw_record=record,
                source_relative_path=f"questions/{filename}",
                source_row_index=index,
            ))
        return result


class LegacyMultiTQAdapter:
    """Retain the original adapter for core regression fixtures."""

    def __init__(self, config: Config):
        self.config = config
        self.dataset = config.dataset

    def kg_path(self, split: str) -> Path:
        return self.config.paths.kg_root / self.dataset["kg_files"][split]

    def question_path(self, split: str) -> Path:
        return self.config.paths.questions_root / self.dataset["question_files"][split]

    def iter_facts(self, split: str = "full", limit: int | None = None) -> Iterator[TemporalFact]:
        with self.kg_path(split).open("r", encoding=self.dataset.get("encoding", "utf-8")) as stream:
            for index, line in enumerate(stream):
                if limit is not None and index >= limit:
                    return
                if line.strip():
                    yield canonical_fact(
                        line.rstrip("\r\n").split(self.dataset.get("delimiter", "\t")),
                        split,
                        index,
                    )

    def load_questions(self, split: str, limit: int | None = None) -> list[QAExample]:
        filename = self.dataset["question_files"][split]
        records = read_json(self.question_path(split))
        selected = records if limit is None else records[:limit]
        return [
            canonical_question(record, split, index, filename)
            for index, record in enumerate(selected)
        ]


class MultiTQAdapter:
    """Backward-compatible dataset adapter factory used by the core pipeline."""

    def __new__(cls, config: Config):
        if config.dataset.get("name") == "CronQuestions":
            return CronQuestionsAdapter(config)
        return LegacyMultiTQAdapter(config)
