"""Sharded candidate-chain cache and manifest."""

from pathlib import Path

from ..utils.io import read_json, write_json, write_jsonl_gzip


def chain_shard_path(directory: Path, qid: str) -> Path:
    return directory / f"{qid}.jsonl.gz"


def chain_shard_files(directory: Path) -> list[Path]:
    """Return one shard per qid, preferring compressed current storage."""
    by_qid = {
        path.name.removesuffix(".jsonl"): path
        for path in directory.glob("*.jsonl")
    }
    by_qid.update({
        path.name.removesuffix(".jsonl.gz"): path
        for path in directory.glob("*.jsonl.gz")
    })
    return [by_qid[qid] for qid in sorted(by_qid)]


def chain_shard_is_current(root: Path, split: str, qid: str, expected: dict) -> bool:
    directory = root / split
    shard = chain_shard_path(directory, qid)
    sidecar = directory / f"{qid}.manifest.json"
    if not shard.exists() or not sidecar.exists():
        return False
    manifest = read_json(sidecar)
    return all(manifest.get(key) == value for key, value in expected.items())


def write_chain_cache(root: Path, split: str, qid: str, chains: list, manifest: dict) -> None:
    directory = root / split
    shard = chain_shard_path(directory, qid)
    write_jsonl_gzip(
        shard,
        (chain.model_dump(mode="json") for chain in chains),
    )
    write_json(directory / f"{qid}.manifest.json", manifest)
    write_json(directory / "manifest.json", manifest)
    # The compressed shard is now durable and has a matching sidecar, so the
    # legacy uncompressed copy can no longer participate in resume.
    (directory / f"{qid}.jsonl").unlink(missing_ok=True)
