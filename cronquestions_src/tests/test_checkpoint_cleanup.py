import json

from decgc.training.checkpoint_cleanup import cleanup_run


def test_cleanup_protects_best_and_latest(tmp_path):
    for index in range(4):
        (tmp_path / f"{index}.pt").write_bytes(bytes(index + 1))
    (tmp_path / "best.json").write_text(json.dumps({"path": "0.pt"}), encoding="utf-8")
    (tmp_path / "latest.json").write_text(json.dumps({"path": "3.pt"}), encoding="utf-8")
    result = cleanup_run(tmp_path, keep_last=1, execute=True)
    assert (tmp_path / "0.pt").exists() and (tmp_path / "3.pt").exists()
    assert "0.pt" not in result["removed"] and "3.pt" not in result["removed"]

