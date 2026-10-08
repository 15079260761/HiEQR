"""Cheap source fingerprints used to validate large persistent caches."""

from __future__ import annotations

from pathlib import Path

from ..utils.hashing import object_hash


def file_signature(path: Path) -> dict[str, int | str]:
    """Return a fast fingerprint without rereading a multi-GB/large source file."""
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def directory_signature(path: Path) -> str:
    """Fingerprint model artifacts by relative name, size, and modification time."""
    rows = []
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        stat = item.stat()
        rows.append((item.relative_to(path).as_posix(), stat.st_size, stat.st_mtime_ns))
    return object_hash(rows)
