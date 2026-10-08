"""Atomic UTF-8 serialization helpers."""

from __future__ import annotations

import json
import gzip
import os
import uuid
from pathlib import Path
from typing import Any, Iterable

from filelock import FileLock


def _unique_temporary(path: Path) -> Path:
    """Return a same-directory temporary path safe across processes/threads."""
    return path.parent / (
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )


def _publish_temporary(temporary: Path, path: Path) -> None:
    """Serialize final replacement and discard the pre-v0.3 fixed temp."""
    with FileLock(str(path) + ".write.lock", timeout=120):
        os.replace(temporary, path)
        legacy = path.with_suffix(path.suffix + ".tmp")
        if legacy != temporary:
            legacy.unlink(missing_ok=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _unique_temporary(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        _publish_temporary(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def read_json_gzip(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write_json_gzip(path: Path, value: Any) -> None:
    """Atomically write one compressed JSON value."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _unique_temporary(path)
    try:
        with gzip.open(
            temporary,
            "wt",
            encoding="utf-8",
            newline="\n",
            compresslevel=6,
        ) as stream:
            json.dump(value, stream, ensure_ascii=False, default=str)
            stream.write("\n")
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        _publish_temporary(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    """Atomically stream JSONL without materializing the whole file in RAM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _unique_temporary(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        _publish_temporary(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl_gzip(path: Path, rows: Iterable[Any]) -> None:
    """Atomically stream compressed JSONL for large resumable shard sets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _unique_temporary(path)
    try:
        with gzip.open(
            temporary,
            "wt",
            encoding="utf-8",
            newline="\n",
            compresslevel=6,
        ) as stream:
            for row in rows:
                stream.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        _publish_temporary(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_text_lines(path: Path, lines: Iterable[str]) -> None:
    """Atomically stream text fragments without assembling them in memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _unique_temporary(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for line in lines:
                stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        _publish_temporary(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: Path, row: Any) -> None:
    """Durably append one record so long-running preprocessing can resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
