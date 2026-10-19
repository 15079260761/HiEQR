from pathlib import Path

import pytest

from decgc.config import load_config

pytest.importorskip("torch")

from decgc.training.reasoner_trainer import loss_ablation_metadata, reasoner_loss_weights
from decgc.utils.stage_cache import stage_config_hash


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    return load_config(ROOT / "configs" / "experiments" / name)[0]


def test_three_loss_configuration_reuses_only_upstream_stage_hashes():
    legacy = _load("cronquestions_demo.yaml")
    three = _load("cronquestions_3loss_main.yaml")
    assert stage_config_hash(legacy, "retriever") == stage_config_hash(three, "retriever")
    assert stage_config_hash(legacy, "reranker") == stage_config_hash(three, "reranker")
    assert stage_config_hash(legacy, "reasoner") != stage_config_hash(three, "reasoner")


def test_exact_unnormalized_three_loss_objective_is_auditable():
    config = _load("cronquestions_3loss_main.yaml")
    weights = reasoner_loss_weights(config)
    assert weights == {
        "prefix": 0.5,
        "stop": 0.3,
        "counterfactual": 0.3,
    }
    metadata = loss_ablation_metadata(weights)
    assert metadata["active_losses"] == ["prefix", "stop", "counterfactual"]
    assert metadata["removed_losses"] == []
    assert config.model_extra["ablation"]["weight_normalization"] is False


def test_full_evaluation_uses_same_three_loss_reasoner_stage():
    demo = _load("cronquestions_3loss_main.yaml")
    full = _load("cronquestions_3loss_full.yaml")
    assert stage_config_hash(demo, "reasoner") == stage_config_hash(full, "reasoner")
    assert full.model_extra["evaluation"]["max_samples"] is None
