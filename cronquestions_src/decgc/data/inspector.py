"""Read-only CronQuestions inventory and schema validation."""

from __future__ import annotations

import json
import pickle
from collections import Counter
from typing import Any

from ..config import Config
from ..utils.hashing import file_hash
from ..utils.io import write_json


def inspect_data(config: Config, resume: bool = False) -> dict[str, Any]:
    config.prepare_outputs()
    is_cron = config.dataset.get("name") == "CronQuestions"
    prefix = "cronquestions" if is_cron else "multitq"
    inventory_path = config.paths.manifests_root / f"{prefix}_file_inventory.json"
    if resume and inventory_path.exists():
        return json.loads(inventory_path.read_text(encoding="utf-8"))
    files = []
    for root in (config.paths.kg_root, config.paths.questions_root):
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            files.append({
                "relative_path": path.relative_to(config.paths.dataset_root).as_posix(),
                "suffix": path.suffix,
                "bytes": path.stat().st_size,
                "sha256": file_hash(path),
            })
    schema: dict[str, Any] = {"kg": {}, "questions": {}}
    stats: dict[str, Any] = {"files": len(files), "kg_rows": {}, "question_rows": {}, "categories": {}}
    for split, filename in config.dataset["kg_files"].items():
        path = config.paths.kg_root / filename
        with path.open("r", encoding="utf-8") as stream:
            count = sum(1 for line in stream if line.strip())
        stats["kg_rows"][split] = count
        fields = (
            ["subject", "relation", "object", "start_year", "end_year"]
            if is_cron else
            ["subject", "relation", "object", "timestamp"]
        )
        schema["kg"][split] = {
            "file": filename, "delimiter": "tab", "columns": len(fields),
            "fields": fields,
        }
    cron_required = {
        "annotation", "answer_type", "answers", "entities", "paraphrases",
        "question", "relations", "template", "times", "type", "uniq_id",
    }
    for split, filename in config.dataset["question_files"].items():
        question_path = config.paths.questions_root / filename
        if is_cron:
            with question_path.open("rb") as stream:
                records = pickle.load(stream)
        else:
            records = json.loads(question_path.read_text(encoding="utf-8"))
        fields = {key for row in records[:1000] for key in row}
        missing = cron_required - fields if is_cron else set()
        if is_cron and missing:
            raise ValueError(f"CronQuestions {split} is missing fields: {sorted(missing)}")
        stats["question_rows"][split] = len(records)
        stats["categories"][split] = {
            "question_type": dict(Counter(row.get("type" if is_cron else "qlabel") for row in records)),
            "answer_type": dict(Counter(row.get("answer_type") for row in records)),
        }
        schema["questions"][split] = {
            "file": filename,
            "container": "pickle-list" if is_cron else "json-array",
            "fields": sorted(fields),
        }
    inventory = {"dataset_root": str(config.paths.dataset_root), "files": files}
    write_json(inventory_path, inventory)
    write_json(config.paths.manifests_root / f"{prefix}_schema.json", schema)
    write_json(config.paths.manifests_root / f"{prefix}_statistics.json", stats)
    return inventory
