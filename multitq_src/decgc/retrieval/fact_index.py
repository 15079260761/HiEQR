"""Dependency-light exact cosine index with manifest freshness checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..utils.io import write_json


class FactIndex:
    def __init__(self, ids: list[str], embeddings: np.ndarray, manifest: dict):
        self.ids = ids
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.manifest = manifest
        self.id_to_index = {fact_id: index for index, fact_id in enumerate(ids)}

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        scores = self.embeddings @ query.astype(np.float32)
        indices = np.argsort(-scores)[:top_k]
        return [(self.ids[index], float(scores[index])) for index in indices]

    def score_ids(self, query: np.ndarray, fact_ids: list[str]) -> list[float]:
        query = query.astype(np.float32)
        return [
            float(self.embeddings[self.id_to_index[fact_id]] @ query)
            if fact_id in self.id_to_index else float("-inf")
            for fact_id in fact_ids
        ]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        write_json(directory / "ids.json", self.ids)
        write_json(directory / "manifest.json", self.manifest)

    @classmethod
    def load(cls, directory: Path, expected: dict | None = None) -> "FactIndex":
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if expected and any(manifest.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Fact index is stale; model/data/config hash mismatch")
        ids = json.loads((directory / "ids.json").read_text(encoding="utf-8"))
        return cls(
            ids,
            np.load(directory / "embeddings.npy", mmap_mode="r"),
            manifest,
        )
