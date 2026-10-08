"""Adapter for the inspected MultiTQ distribution."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..config import Config
from ..schemas import QAExample, TemporalFact
from ..utils.io import read_json
from .canonical_schema import canonical_fact, canonical_question


class MultiTQAdapter:
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
                    yield canonical_fact(line.rstrip("\r\n").split(self.dataset.get("delimiter", "\t")), split, index)

    def load_questions(self, split: str, limit: int | None = None) -> list[QAExample]:
        filename = self.dataset["question_files"][split]
        records = read_json(self.question_path(split))
        selected = records if limit is None else records[:limit]
        return [canonical_question(record, split, index, filename) for index, record in enumerate(selected)]

