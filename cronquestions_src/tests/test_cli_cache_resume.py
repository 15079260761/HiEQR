from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
import sys

import pytest
import numpy as np

torch = pytest.importorskip("torch")

try:
    import typer  # noqa: F401
except ModuleNotFoundError:
    typer_stub = ModuleType("typer")

    class _Typer:
        def __init__(self, *args, **kwargs):
            pass

        def command(self, *args, **kwargs):
            return lambda function: function

    class _Exit(Exception):
        pass

    typer_stub.Typer = _Typer
    typer_stub.Option = lambda default=None, *args, **kwargs: default
    typer_stub.Exit = _Exit
    sys.modules["typer"] = typer_stub

import decgc.cli as cli
from decgc.data.canonical_schema import canonical_fact
from decgc.data.kg_index import KGIndex
from decgc.data.multitq_adapter import MultiTQAdapter
from decgc.utils.io import read_json, write_json


def test_retriever_data_removes_shards_after_consolidated_cache_is_durable(tiny_root: Path):
    config_path = Path(__file__).parents[1] / "configs/experiments/multitq_smoke.yaml"
    overrides = [f"project.root={tiny_root.as_posix()}"]
    cli.build_kg_index(
        config_path, resume="none", limit=None, seed=None, log_level="INFO", set_values=overrides,
    )
    cli.parse_questions(
        config_path,
        split="train",
        resume="auto",
        limit=None,
        seed=None,
        log_level="INFO",
        no_online=True,
        set_values=overrides,
    )
    cli.build_retriever_data(
        config_path, resume="auto", limit=None, seed=None, log_level="INFO", set_values=overrides,
    )

    cache = tiny_root / "outputs/cache/retrieval_pairs"
    consolidated = cache / "train.jsonl"
    consolidated_mtime = consolidated.stat().st_mtime_ns
    assert not list(cache.glob("train-shards-*"))

    cli.build_retriever_data(
        config_path, resume="auto", limit=None, seed=None, log_level="INFO", set_values=overrides,
    )

    assert consolidated.stat().st_mtime_ns == consolidated_mtime


def test_missing_parse_is_repaired_without_invalidating_primary_cache(tiny_root: Path):
    config_path = Path(__file__).parents[1] / "configs/experiments/multitq_smoke.yaml"
    overrides = [f"project.root={tiny_root.as_posix()}"]
    config, _ = cli.load_config(config_path, overrides)
    cli.build_kg_index(
        config_path, resume="auto", limit=None, seed=None,
        log_level="INFO", set_values=overrides,
    )
    primary = cli._parse_path(config, "test")
    primary.parent.mkdir(parents=True, exist_ok=True)
    primary.write_text("", encoding="utf-8")
    original_bytes = primary.read_bytes()
    original_mtime = primary.stat().st_mtime_ns
    examples = MultiTQAdapter(config).load_questions("test")

    cli._ensure_local_parse_cache(config, "test", examples)

    assert primary.read_bytes() == original_bytes
    assert primary.stat().st_mtime_ns == original_mtime
    repair = cli._parse_repair_path(config, "test")
    assert repair.exists()
    assert str(examples[0].qid) in cli._parsed(config, "test")
    repair_mtime = repair.stat().st_mtime_ns

    cli._ensure_local_parse_cache(config, "test", examples)
    assert repair.stat().st_mtime_ns == repair_mtime


def test_fact_encoding_resumes_from_last_durable_batch(tiny_root: Path, monkeypatch):
    config_path = Path(__file__).parents[1] / "configs/experiments/multitq_smoke.yaml"
    overrides = [f"project.root={tiny_root.as_posix()}"]
    facts = [
        canonical_fact([f"Subject_{index}", "visit", "Iraq", "2007-02-27"], "full", index)
        for index in range(260)
    ]
    index = KGIndex(facts)
    kg_dir = tiny_root / "outputs/cache/kg_indices"
    index.save(kg_dir / "multitq_full.pkl")
    write_json(kg_dir / "multitq_full.manifest.json", {"data_hash": index.data_hash})

    instances = []

    class FakeEncoder:
        def __init__(self, fail_on_call=None):
            self.model = SimpleNamespace(config=SimpleNamespace(hidden_size=3))
            self.fail_on_call = fail_on_call
            self.batch_lengths = []

        def to(self, device):
            return self

        def eval(self):
            return self

        def encode(self, texts):
            self.batch_lengths.append(len(texts))
            if self.fail_on_call == len(self.batch_lengths):
                raise RuntimeError("simulated interruption")
            return torch.ones((len(texts), 3), dtype=torch.float32)

        def state_dict(self):
            return {}

    def encoder_factory(path):
        encoder = FakeEncoder(fail_on_call=2 if not instances else None)
        instances.append(encoder)
        return encoder

    monkeypatch.setattr(cli, "BGEEncoder", encoder_factory)
    monkeypatch.setattr(cli, "select_device", lambda: torch.device("cpu"))

    with pytest.raises(RuntimeError, match="simulated interruption"):
        cli.encode_facts(
            config_path,
            retriever_checkpoint="best",
            resume="auto",
            limit=None,
            seed=None,
            log_level="INFO",
            set_values=overrides,
        )

    partial_manifest = (
        tiny_root / "outputs/cache/fact_embeddings/embeddings.partial.manifest.json"
    )
    assert read_json(partial_manifest)["completed_facts"] == 128

    cli.encode_facts(
        config_path,
        retriever_checkpoint="best",
        resume="auto",
        limit=None,
        seed=None,
        log_level="INFO",
        set_values=overrides,
    )

    assert instances[1].batch_lengths == [128, 4]
    assert not partial_manifest.exists()
    assert (tiny_root / "outputs/cache/fact_embeddings/embeddings.npy").exists()


def test_counterfactual_stage_removes_legacy_materialization(tiny_root: Path):
    config_path = Path(__file__).parents[1] / "configs/experiments/multitq_smoke.yaml"
    overrides = [f"project.root={tiny_root.as_posix()}"]
    chain_root = tiny_root / "outputs/cache/candidate_chains/train"
    chain_root.mkdir(parents=True)
    (chain_root / "1.jsonl").write_text(
        '{"qid":"1","state":"COMPLETED"}\n',
        encoding="utf-8",
    )
    counterfactual_root = tiny_root / "outputs/cache/counterfactuals"
    counterfactual_root.mkdir(parents=True)
    (counterfactual_root / "train.jsonl").write_bytes(b"legacy")
    (counterfactual_root / "train.jsonl.tmp").write_bytes(b"incomplete")

    cli.build_counterfactuals(
        config_path,
        split="train",
        resume="auto",
        limit=None,
        seed=None,
        log_level="INFO",
        set_values=overrides,
    )

    manifest = read_json(counterfactual_root / "train.manifest.json")
    assert manifest["counterfactual_data_version"] == 3
    assert manifest["materialized"] is False
    assert manifest["records"] == 4
    assert not (counterfactual_root / "train.jsonl").exists()
    assert not (counterfactual_root / "train.jsonl.tmp").exists()


def test_fact_index_reuses_embedding_file_with_hardlink(tiny_root: Path):
    config_path = Path(__file__).parents[1] / "configs/experiments/multitq_smoke.yaml"
    overrides = [f"project.root={tiny_root.as_posix()}"]
    source = tiny_root / "outputs/cache/fact_embeddings"
    target = tiny_root / "outputs/cache/fact_index"
    source.mkdir(parents=True)
    target.mkdir(parents=True)
    manifest = {"facts": 2, "retriever_hash": "current"}
    np.save(source / "embeddings.npy", np.ones((2, 3), dtype=np.float32))
    write_json(source / "ids.json", ["a", "b"])
    write_json(source / "manifest.json", manifest)
    np.save(target / "embeddings.npy", np.zeros((2, 3), dtype=np.float32))
    write_json(target / "ids.json", ["a", "b"])
    write_json(target / "manifest.json", manifest)

    cli.build_fact_index(
        config_path,
        resume="auto",
        limit=None,
        seed=None,
        log_level="INFO",
        set_values=overrides,
    )

    assert (source / "embeddings.npy").samefile(target / "embeddings.npy")
