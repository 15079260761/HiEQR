"""Read-only recursive data inventory and schema inference."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..config import Config
from ..utils.hashing import file_hash
from ..utils.io import write_json


def _encoding(path: Path) -> str:
    prefix = path.read_bytes()[:4096]
    prefix.decode("utf-8")
    return "utf-8"


def _preview(path: Path) -> list[Any]:
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if isinstance(data, list):
            return data[:3]
        return list(data.items())[:3]
    if path.suffix in {".txt", ""}:
        rows = []
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    rows.append(line.rstrip("\r\n").split("\t"))
                if len(rows) == 3:
                    break
        return rows
    return []


def inspect_data(config: Config, resume: bool = False) -> dict[str, Any]:
    config.prepare_outputs()
    inventory_path = config.paths.manifests_root / "multitq_file_inventory.json"
    if resume and inventory_path.exists():
        return json.loads(inventory_path.read_text(encoding="utf-8"))
    files = []
    for root in (config.paths.kg_root, config.paths.questions_root):
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(config.paths.dataset_root).as_posix()
            item: dict[str, Any] = {
                "relative_path": relative, "suffix": path.suffix, "bytes": path.stat().st_size,
                "sha256": file_hash(path), "encoding": None, "preview": [],
            }
            if path.suffix in {".json", ".txt", ".yaml", ""}:
                try:
                    item["encoding"] = _encoding(path)
                    item["preview"] = _preview(path)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    item["encoding"] = "binary-or-unknown"
            files.append(item)
    schema: dict[str, Any] = {"kg": {}, "questions": {}}
    stats: dict[str, Any] = {"files": len(files), "kg_rows": {}, "question_rows": {}, "categories": {}}
    for split, filename in config.dataset["kg_files"].items():
        path = config.paths.kg_root / filename
        with path.open("r", encoding="utf-8") as stream:
            count = sum(1 for line in stream if line.strip())
        stats["kg_rows"][split] = count
        schema["kg"][split] = {"file": filename, "delimiter": "tab", "columns": 4,
                               "fields": ["subject", "relation", "object", "timestamp"]}
    for split, filename in config.dataset["question_files"].items():
        records = json.loads((config.paths.questions_root / filename).read_text(encoding="utf-8"))
        stats["question_rows"][split] = len(records)
        keys = sorted({key for row in records[:100] for key in row})
        schema["questions"][split] = {"file": filename, "container": "json-array", "fields": keys}
        stats["categories"][split] = {
            "question_type": dict(Counter(row.get("qlabel") for row in records)),
            "answer_type": dict(Counter(row.get("answer_type") for row in records)),
            "operator": dict(Counter(row.get("qtype") for row in records)),
        }
    inventory = {"dataset_root": str(config.paths.dataset_root), "files": files}
    write_json(inventory_path, inventory)
    write_json(config.paths.manifests_root / "multitq_schema.json", schema)
    write_json(config.paths.manifests_root / "multitq_statistics.json", stats)
    return inventory

