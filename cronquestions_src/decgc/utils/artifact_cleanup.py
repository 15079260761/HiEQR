"""Conservative cleanup of artifacts that cannot participate in a resume."""

from __future__ import annotations

import shutil
from pathlib import Path


def path_size(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        return path.stat().st_size
    return sum(
        child.stat().st_size
        for child in path.rglob("*")
        if child.is_file()
    )


def remove_scoped_paths(
    paths: list[Path],
    scope: Path,
    execute: bool,
) -> dict:
    """Remove explicit targets only when every resolved path is under scope."""
    scope = scope.resolve()
    unique: list[Path] = []
    seen = set()
    for path in paths:
        if not path.exists() or path in seen:
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(scope) or resolved == scope:
            raise RuntimeError(f"Refusing cleanup outside artifact scope: {path}")
        seen.add(path)
        unique.append(path)
    # If a directory is selected, do not separately process children.
    selected = set(unique)
    unique = [
        path
        for path in unique
        if not any(parent in selected for parent in path.parents)
    ]
    bytes_to_release = sum(path_size(path) for path in unique)
    removed = []
    if execute:
        for path in unique:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
            removed.append(str(path.relative_to(scope)))
    return {
        "execute": execute,
        "candidates": [str(path.relative_to(scope)) for path in unique],
        "bytes_to_release": bytes_to_release,
        "removed": removed,
    }


def obsolete_artifact_candidates(
    outputs_root: Path,
    current_runs: dict[str, Path],
    keep_old_runs: int,
) -> list[Path]:
    """Find only replaceable temporaries, completed shards, and old run IDs."""
    candidates: list[Path] = []
    if not outputs_root.exists():
        return candidates

    # Interrupted atomic writes are never valid resume points.
    candidates.extend(outputs_root.rglob("*.tmp"))

    # Counterfactuals are generated on demand by reasoner training since data
    # version 3; legacy materialized files are redundant.
    counterfactual_root = outputs_root / "cache" / "counterfactuals"
    if counterfactual_root.exists():
        candidates.extend(counterfactual_root.glob("*.jsonl"))

    # Once the consolidated retrieval file and its manifest are durable, the
    # per-question construction shards are no longer consulted.
    retrieval_root = outputs_root / "cache" / "retrieval_pairs"
    if (
        (retrieval_root / "train.jsonl").exists()
        and (retrieval_root / "train.manifest.json").exists()
    ):
        candidates.extend(retrieval_root.glob("train-shards-*"))

    # Keep the active stage run plus a small configured number of recent
    # historical runs. Older stage hashes cannot be selected by --resume auto.
    for stage, current in current_runs.items():
        stage_root = outputs_root / "runs" / stage
        if not stage_root.exists():
            continue
        # Runs may be legacy ``stage/multitq-<hash>`` directories or named
        # ``stage/<experiment>/dataset-<hash>`` directories.  Select only
        # immutable hash directories, never their human-readable parent.
        run_directories = [
            path
            for path in stage_root.rglob("*")
            if path.is_dir()
            and path.name.startswith(("multitq-", "cronquestions-"))
        ]
        historical = sorted(
            (
                path
                for path in run_directories
                if path.resolve() != current.resolve()
            ),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        candidates.extend(historical[max(0, keep_old_runs):])
    return candidates
