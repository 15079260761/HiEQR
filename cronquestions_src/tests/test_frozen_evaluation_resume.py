from types import SimpleNamespace
from types import ModuleType
import sys

import pytest

pytest.importorskip("torch")

try:
    import typer  # noqa: F401
except ModuleNotFoundError:
    typer_stub = ModuleType("typer")

    class _Typer:
        def __init__(self, *args, **kwargs):
            pass

        def command(self, *args, **kwargs):
            return lambda function: function

    typer_stub.Typer = _Typer
    typer_stub.Option = lambda default=None, *args, **kwargs: default
    typer_stub.BadParameter = ValueError
    sys.modules["typer"] = typer_stub

import decgc.cli as cli
from decgc.utils.io import read_json, write_json, write_jsonl_gzip


def _prediction(qid: str) -> dict:
    return {
        "qid": qid,
        "question": "Who visited Iraq?",
        "predicted_answer": "Alice",
        "candidate_answers": ["Alice"],
        "candidate_scores": {"Alice": 1.0},
        "gold_answers": ["Alice"],
        "question_type": "Single",
        "answer_type": "entity",
        "temporal_operator": "equal",
        "temporal_granularity": "day",
        "answer_role": "subject",
        "parse_source": "local",
        "candidate_chain_counts": {"alice": 1},
        "candidate_best_hops": {"alice": 1},
        "fallback": False,
    }


def test_frozen_evaluation_writes_per_question_resume_items(config, monkeypatch):
    config.prepare_outputs()
    example = SimpleNamespace(qid="1")
    monkeypatch.setattr(cli, "_examples", lambda *_args, **_kwargs: [example])
    monkeypatch.setattr(cli, "_parsed", lambda *_args, **_kwargs: {"1": object()})
    monkeypatch.setattr(
        cli,
        "_kg_index",
        lambda *_args, **_kwargs: SimpleNamespace(adjacency={}, facts={}),
    )
    monkeypatch.setattr(cli, "_load_reasoner", lambda *_args, **_kwargs: None)
    calls = []

    def evaluate_one(*_args, **_kwargs):
        calls.append("1")
        return _prediction("1"), [], [], []

    monkeypatch.setattr(cli, "_evaluate_question", evaluate_one)
    chain_root = config.paths.cache_root / "candidate_chains/test"
    write_json(chain_root / "manifest.json", {"version": 1})
    write_jsonl_gzip(chain_root / "1.jsonl.gz", [])

    run_id, metrics = cli._run_frozen_evaluation(
        config,
        "test",
        None,
        "auto",
    )
    prediction_dir = config.paths.predictions_root / run_id
    assert metrics["hits1"] == 1.0
    assert len(calls) == 1
    assert read_json(prediction_dir / "evaluation_manifest.json")["questions"] == 1
    assert (prediction_dir / "resume_items/1.json.gz").exists()

    # Simulate interruption during final output assembly. The durable
    # question result must be reused instead of running inference again.
    (prediction_dir / "evaluation_manifest.json").unlink()
    (prediction_dir / "predictions.jsonl").unlink()
    resumed_run_id, resumed_metrics = cli._run_frozen_evaluation(
        config,
        "test",
        None,
        "auto",
    )
    assert resumed_run_id == run_id
    assert resumed_metrics["hits1"] == 1.0
    assert len(calls) == 1
    assert len((prediction_dir / "predictions.jsonl").read_text().splitlines()) == 1
