"""Stage-scoped signatures that keep unrelated experiment changes reusable."""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from .hashing import object_hash
from .io import read_json

REASONER_TRAINING_DATA_VERSION = 12
RETRIEVER_MODELING_VERSION = 4


def completed_transformer_training(run_dir: Path) -> dict | None:
    """Return an atomic completed run before loading its dataset or model."""
    summary_path = run_dir / "training_summary.json"
    best_model = run_dir / "best_model"
    latest = run_dir / "checkpoints/latest.json"
    weights_present = any(
        (best_model / name).exists()
        for name in ("model.safetensors", "pytorch_model.bin")
    )
    if not (
        summary_path.exists()
        and latest.exists()
        and (best_model / "config.json").exists()
        and weights_present
    ):
        return None
    summary = read_json(summary_path)
    return summary if isinstance(summary, dict) else None


def manifest_signature(path: Path) -> str:
    if not path.exists():
        return "missing"
    payload = read_json(path)
    if isinstance(payload, dict):
        payload = {
            key: value
            for key, value in payload.items()
            if key not in {"resolved_config_hash", "config_hash"}
        }
    return object_hash(payload)


def stage_config_hash(config: Config, stage: str) -> str:
    """Hash only inputs that can affect a training stage, not the whole experiment."""
    demo = config.model_extra.get("demo", {})
    training_sample = {
        key: demo.get(key)
        for key in (
            "train_max_samples",
            "sampling_seed",
            "stratified_sampling",
        )
    }
    common = {
        "seed": config.project.seed,
        "dataset": config.dataset,
        "training_sample": training_sample,
    }
    if stage == "retriever":
        payload = {
            **common,
            "modeling_version": RETRIEVER_MODELING_VERSION,
            "pretrained_model": str(config.paths.pretrained_retriever),
            "data": manifest_signature(
                config.paths.cache_root / "retrieval_pairs" / "train.manifest.json"
            ),
            "settings": config.model_extra["retriever_training"],
        }
    elif stage == "reranker":
        payload = {
            **common,
            "pretrained_model": str(config.paths.pretrained_reranker),
            "data": manifest_signature(
                config.paths.cache_root / "reranker_pairs" / "train.manifest.json"
            ),
            "settings": config.model_extra["reranker_training"],
        }
    elif stage == "reasoner":
        payload = {
            **common,
            "training_data_version": REASONER_TRAINING_DATA_VERSION,
            "retriever_run": stage_config_hash(config, "retriever"),
            "reranker_run": stage_config_hash(config, "reranker"),
            "chain_data": manifest_signature(
                config.paths.cache_root / "candidate_chains" / "train" / "manifest.json"
            ),
            "validation_chain_data": manifest_signature(
                config.paths.cache_root / "candidate_chains" / "valid" / "manifest.json"
            ),
            "model": config.model_extra["model"],
            "loss": config.model_extra["loss"],
            "settings": config.model_extra["reasoner_training"],
            "minimum_positive_coverage": config.model_extra.get(
                "quality_gate",
                {},
            ).get("reasoner_positive_question_coverage_min", 0.0),
        }
    else:
        return config.hash
    return object_hash(payload)
