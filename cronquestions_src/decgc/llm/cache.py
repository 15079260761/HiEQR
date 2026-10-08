"""Validated parse cache keyed by question and prompt version."""

from __future__ import annotations

from pathlib import Path

from ..schemas import StructuredQuery
from ..utils.file_lock import cache_lock
from ..utils.hashing import object_hash
from ..utils.io import read_json, write_json


class ParseCache:
    def __init__(self, root: Path):
        self.root = root

    def path_for(self, question: str, version: str) -> Path:
        return self.root / f"{object_hash({'question': question, 'version': version})}.json"

    def get(self, question: str, version: str) -> StructuredQuery | None:
        path = self.path_for(question, version)
        return StructuredQuery.model_validate(read_json(path)) if path.exists() else None

    def put(self, question: str, version: str, value: StructuredQuery) -> None:
        path = self.path_for(question, version)
        with cache_lock(path):
            write_json(path, value.model_dump(mode="json"))

