from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from decgc.config import load_config


@pytest.fixture()
def tiny_root(tmp_path: Path):
    root = tmp_path / "project"
    (root / "data/MultiTQ/kg").mkdir(parents=True)
    (root / "data/MultiTQ/questions").mkdir(parents=True)
    (root / "models/bge-base-en-v1.5").mkdir(parents=True)
    (root / "models/bge-reranker-base").mkdir(parents=True)
    (root / "src/prompts").mkdir(parents=True)
    facts = [
        "Alice\tvisit\tIraq\t2007-02-27",
        "Iraq\tpraise\tJapan\t2007-03-01",
        "Bob\tvisit\tIraq\t2008-01-01",
    ]
    for name in ("train", "valid", "test", "full"):
        (root / f"data/MultiTQ/kg/{name}.txt").write_text("\n".join(facts) + "\n", encoding="utf-8")
    questions = [{"quid": 1, "question": "Who visited Iraq on 27 February 2007?", "answers": ["Alice"],
                  "answer_type": "entity", "time_level": "day", "qtype": "equal", "qlabel": "Single"}]
    for name in ("train", "dev", "test"):
        (root / f"data/MultiTQ/questions/{name}.json").write_text(json.dumps(questions), encoding="utf-8")
    return root


@pytest.fixture()
def config(tiny_root: Path):
    config_path = Path(__file__).parents[1] / "configs/experiments/multitq_smoke.yaml"
    return load_config(config_path, [f"project.root={tiny_root.as_posix()}"])[0]
