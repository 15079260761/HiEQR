import os

import pytest

from decgc.utils.artifact_cleanup import (
    obsolete_artifact_candidates,
    remove_scoped_paths,
)
from decgc.utils.io import write_jsonl


def test_write_jsonl_streams_generator_and_replaces_failed_temporary(tmp_path):
    target = tmp_path / "rows.jsonl"
    temporary = target.with_suffix(".jsonl.tmp")
    temporary.write_text("incomplete", encoding="utf-8")
    consumed = []

    def rows():
        for index in range(3):
            consumed.append(index)
            yield {"index": index}

    write_jsonl(target, rows())

    assert consumed == [0, 1, 2]
    assert target.read_text(encoding="utf-8").count("\n") == 3
    assert not temporary.exists()


def test_cleanup_removes_only_obsolete_resume_artifacts(tmp_path):
    outputs = tmp_path / "outputs"
    counterfactuals = outputs / "cache/counterfactuals"
    retrieval = outputs / "cache/retrieval_pairs"
    label = outputs / "runs/retriever/main-3loss-multitq-10k"
    current = label / "multitq-current"
    old_new = outputs / "runs/retriever/multitq-old-new"
    old_stale = outputs / "runs/retriever/older-experiment/multitq-old-stale"
    for directory in (
        counterfactuals,
        retrieval / "train-shards-old",
        current,
        old_new,
        old_stale,
    ):
        directory.mkdir(parents=True)
    (counterfactuals / "train.jsonl").write_bytes(b"legacy")
    (counterfactuals / "train.jsonl.tmp").write_bytes(b"failed")
    (retrieval / "train.jsonl").write_bytes(b"complete")
    (retrieval / "train.manifest.json").write_text("{}", encoding="utf-8")
    (retrieval / "train-shards-old/000.jsonl").write_bytes(b"duplicate")
    (current / "keep.pt").write_bytes(b"current")
    (old_new / "keep.pt").write_bytes(b"fallback")
    (old_stale / "remove.pt").write_bytes(b"obsolete")
    os.utime(old_stale, (1, 1))
    os.utime(old_new, (2, 2))

    candidates = obsolete_artifact_candidates(
        outputs,
        {
            "retriever": current,
            "reranker": outputs / "runs/reranker/current",
            "reasoner": outputs / "runs/reasoner/current",
        },
        keep_old_runs=1,
    )
    result = remove_scoped_paths(candidates, outputs, execute=True)

    assert result["bytes_to_release"] > 0
    assert not (counterfactuals / "train.jsonl").exists()
    assert not (retrieval / "train-shards-old").exists()
    assert not old_stale.exists()
    assert current.exists()
    assert label.exists()
    assert old_new.exists()
    assert (retrieval / "train.jsonl").exists()


def test_cleanup_refuses_recursive_target_outside_outputs(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(RuntimeError, match="outside artifact scope"):
        remove_scoped_paths([outside], outputs, execute=True)
