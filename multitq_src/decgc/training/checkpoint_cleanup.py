"""Conservative checkpoint cleanup retaining best/latest and newest files."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..utils.io import write_jsonl


def cleanup_run(root: Path, keep_last: int = 2, execute: bool = False) -> dict:
    protected = set()
    for pointer_name in ("best.json", "latest.json"):
        pointer = root / pointer_name
        if pointer.exists():
            protected.add(json.loads(pointer.read_text(encoding="utf-8"))["path"])
    checkpoints = sorted(root.glob("*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    protected.update(path.name for path in checkpoints[:keep_last])
    candidates = [path for path in checkpoints if path.name not in protected]
    temporary = list(root.glob("*.tmp"))
    bytes_to_release = sum(path.stat().st_size for path in candidates + temporary)
    removed = []
    if execute:
        for path in candidates + temporary:
            removed.append(path.name)
            path.unlink()
        log = root / "cleanup_log.jsonl"
        previous = []
        if log.exists():
            previous = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line]
        previous.append({"time": datetime.now(timezone.utc).isoformat(), "removed": removed, "bytes": bytes_to_release})
        write_jsonl(log, previous)
    return {"execute": execute, "candidates": [path.name for path in candidates + temporary],
            "bytes_to_release": bytes_to_release, "protected": sorted(protected), "removed": removed}

