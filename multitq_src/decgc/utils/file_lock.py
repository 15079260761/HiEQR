"""Cross-process cache locking."""

from pathlib import Path

from filelock import FileLock


def cache_lock(target: Path) -> FileLock:
    target.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(target) + ".lock")

