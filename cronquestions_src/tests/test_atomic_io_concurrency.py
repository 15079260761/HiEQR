from concurrent.futures import ThreadPoolExecutor
import json

from decgc.utils.io import write_json


def test_atomic_json_writes_use_unique_concurrent_temporaries(tmp_path):
    target = tmp_path / "manifest.json"

    def write(index: int) -> None:
        write_json(target, {"writer": index, "payload": "x" * 1024})

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(write, range(200)))

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["writer"] in range(200)
    assert payload["payload"] == "x" * 1024
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))
