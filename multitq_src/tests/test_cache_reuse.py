from pathlib import Path

from decgc.retrieval.candidate_cache import chain_shard_is_current, write_chain_cache
from decgc.utils.stage_cache import (
    completed_transformer_training,
    stage_config_hash,
)


def test_stage_hash_ignores_unrelated_later_experiment_settings(config):
    baseline_retriever = stage_config_hash(config, "retriever")
    baseline_reranker = stage_config_hash(config, "reranker")
    baseline_reasoner = stage_config_hash(config, "reasoner")

    changed = config.model_copy(deep=True)
    changed.model_extra["reasoner_training"]["learning_rate"] = 9.0e-4

    assert stage_config_hash(changed, "retriever") == baseline_retriever
    assert stage_config_hash(changed, "reranker") == baseline_reranker
    assert stage_config_hash(changed, "reasoner") != baseline_reasoner


def test_retriever_hash_changes_only_for_retriever_training_settings(config):
    baseline = stage_config_hash(config, "retriever")
    changed = config.model_copy(deep=True)
    changed.model_extra["retriever_training"]["learning_rate"] = 9.0e-6

    assert stage_config_hash(changed, "retriever") != baseline


def test_reporting_and_validation_sample_changes_reuse_all_training_runs(config):
    baseline = {
        stage: stage_config_hash(config, stage)
        for stage in ("retriever", "reranker", "reasoner")
    }
    changed = config.model_copy(deep=True)
    changed.model_extra["demo"]["valid_max_samples"] = 99
    changed.model_extra["demo"]["evaluation_max_samples"] = 77
    changed.model_extra["demo"]["qualitative_examples"] = 3
    changed.model_extra["evaluation"]["max_samples"] = 12

    assert {
        stage: stage_config_hash(changed, stage)
        for stage in ("retriever", "reranker", "reasoner")
    } == baseline


def test_metric_target_changes_reuse_all_training_runs(config):
    baseline = {
        stage: stage_config_hash(config, stage)
        for stage in ("retriever", "reranker", "reasoner")
    }
    changed = config.model_copy(deep=True)
    changed.model_extra.setdefault("quality_gate", {})["hits1_min"] = 0.999
    changed.model_extra["quality_gate"]["hits10_min"] = 0.999

    assert {
        stage: stage_config_hash(changed, stage)
        for stage in ("retriever", "reranker", "reasoner")
    } == baseline


def test_candidate_chain_shards_have_independent_manifests(tmp_path: Path):
    root = tmp_path / "candidate_chains"
    manifest = {"chain_cache_version": 3, "search_signature": "current"}
    write_chain_cache(root, "train", "question-1", [], manifest)

    assert chain_shard_is_current(root, "train", "question-1", manifest)
    assert not chain_shard_is_current(
        root,
        "train",
        "question-1",
        {"chain_cache_version": 3, "search_signature": "changed"},
    )
    assert (root / "train/question-1.manifest.json").exists()
    shard = root / "train/question-1.jsonl.gz"
    assert shard.exists()
    assert shard.read_bytes()[:2] == b"\x1f\x8b"
    assert not (root / "train/question-1.jsonl").exists()


def test_completed_transformer_run_is_detected_without_loading_model(tmp_path: Path):
    run = tmp_path / "run"
    (run / "best_model").mkdir(parents=True)
    (run / "checkpoints").mkdir()
    (run / "best_model/config.json").write_text("{}", encoding="utf-8")
    (run / "best_model/model.safetensors").write_bytes(b"weights")
    (run / "checkpoints/latest.json").write_text(
        '{"path":"epoch-001.pt"}',
        encoding="utf-8",
    )
    (run / "training_summary.json").write_text(
        '{"global_step":12}',
        encoding="utf-8",
    )

    assert completed_transformer_training(run) == {"global_step": 12}
    (run / "best_model/model.safetensors").unlink()
    assert completed_transformer_training(run) is None
