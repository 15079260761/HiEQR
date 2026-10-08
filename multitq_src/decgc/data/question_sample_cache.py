"""Persistent deterministic samples that avoid reparsing the full question JSON."""

from __future__ import annotations

import json
from pathlib import Path

from ..schemas import QAExample
from ..utils.hashing import object_hash
from ..utils.io import read_json, write_json, write_jsonl
from .multitq_adapter import MultiTQAdapter
from .sampler import stratified_sample
from .source_signature import file_signature

CACHE_VERSION = 2


def load_or_build_sample(
    adapter: MultiTQAdapter,
    cache_root: Path,
    split: str,
    limit: int,
    seed: int,
) -> tuple[list[QAExample], bool]:
    """Return a finite sample and whether it was loaded from persistent cache."""
    directory = cache_root / "question_samples"
    stem = f"{split}-n{limit}-seed{seed}"
    rows_path = directory / f"{stem}.jsonl"
    manifest_path = directory / f"{stem}.manifest.json"
    source = file_signature(adapter.question_path(split))
    expected = {
        "cache_version": CACHE_VERSION,
        "split": split,
        "limit": limit,
        "seed": seed,
        "source": source,
    }
    if rows_path.exists() and manifest_path.exists():
        manifest = read_json(manifest_path)
        if all(manifest.get(key) == value for key, value in expected.items()):
            rows = [
                QAExample.model_validate(json.loads(line))
                for line in rows_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(rows) == manifest.get("questions"):
                return rows, True

    examples = adapter.load_questions(split)
    rows = stratified_sample(examples, limit, seed)
    write_jsonl(rows_path, (row.model_dump(mode="json") for row in rows))
    write_json(manifest_path, {
        **expected,
        "questions": len(rows),
        "selection_hash": object_hash([row.qid for row in rows]),
    })
    return rows, False
