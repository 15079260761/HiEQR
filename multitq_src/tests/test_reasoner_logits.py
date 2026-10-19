import pytest
from types import SimpleNamespace

torch = pytest.importorskip("torch")

from decgc.models.decgc_model import DECGCModel
from decgc.models.losses import grouped_candidate_answer_ranking_loss
from decgc.training.reasoner_trainer import (
    _forward_prefix_sequences,
    reasoner_runtime_preflight,
    train_reasoner,
    validation_ranking_metrics,
)


def test_prefix_step_exposes_logits_and_matching_probabilities():
    model = DECGCModel(fact_size=4, hidden_size=8)
    output = model.prefix_step(
        previous=torch.zeros(2, 8),
        fact=torch.ones(2, 4),
        hop=torch.tensor([1, 2]),
        temporal=torch.tensor([1.0, 0.0]),
        type_score=torch.tensor([1.0, 0.0]),
    )

    assert torch.allclose(
        output["absolute_reliability"],
        torch.sigmoid(output["reliability_logits"]),
    )
    assert torch.allclose(
        output["stop_probability"],
        torch.sigmoid(output["stop_logits"]),
    )


def test_reasoner_trainer_completes_an_epoch_with_logit_losses(config, tmp_path):
    model = DECGCModel(fact_size=4, hidden_size=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    batch = {
        "previous": torch.zeros(2, 8),
        "fact": torch.ones(2, 4),
        "hop": torch.tensor([1, 2]),
        "temporal": torch.tensor([1.0, 0.0]),
        "type_score": torch.tensor([1.0, 0.0]),
        "prefix_label": torch.tensor([1.0, 0.0]),
        "stop_label": torch.tensor([1.0, 0.0]),
        "questions": ["q1", "q2"],
        "group_ids": ["q1", "q2"],
        "fact_texts": ["f1", "f2"],
        "cf_fact_texts": [],
    }

    result = train_reasoner(
        model,
        [batch],
        optimizer,
        config,
        tmp_path,
        resume="none",
        checkpoint_config_hash="test-stage",
    )

    assert result["training_complete"]
    assert result["active_losses"] == ["prefix", "stop", "counterfactual"]
    assert (tmp_path / "checkpoints/latest.json").exists()


def test_loss_ablation_computes_only_prefix(
    config,
    tmp_path,
    monkeypatch,
):
    model = DECGCModel(fact_size=4, hidden_size=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    batch = {
        "previous": torch.zeros(2, 8),
        "fact": torch.ones(2, 4),
        "hop": torch.tensor([1, 2]),
        "temporal": torch.tensor([1.0, 0.0]),
        "type_score": torch.tensor([1.0, 0.0]),
        "prefix_label": torch.tensor([1.0, 0.0]),
        "stop_label": torch.tensor([1.0, 0.0]),
        "questions": ["q1", "q1"],
        "group_ids": ["q1", "q1"],
        "fact_texts": ["f1", "f2"],
        "answer_texts": ["Alice", "Bob"],
        "cf_fact_texts": [],
    }
    ablation = config.model_copy(deep=True)
    ablation.model_extra["loss"]["stop"] = 0.0
    ablation.model_extra["loss"]["counterfactual"] = 0.0
    calls = []
    original = torch.nn.functional.binary_cross_entropy_with_logits

    def tracked_bce(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "decgc.training.reasoner_trainer.functional.binary_cross_entropy_with_logits",
        tracked_bce,
    )
    result = train_reasoner(
        model,
        [batch],
        optimizer,
        ablation,
        tmp_path,
        resume="none",
        checkpoint_config_hash="three-loss-test",
    )

    assert result["active_losses"] == ["prefix"]
    # One BCE call is the prefix loss. A stop-loss BCE would make this two.
    assert len(calls) == 1


def test_training_reconstructs_multihop_previous_state():
    class DummyRetriever(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones(4))
            self.model = SimpleNamespace(config=SimpleNamespace(hidden_size=4))

        def encode(self, texts):
            scales = torch.tensor(
                [float(len(text)) for text in texts],
                device=self.anchor.device,
            )[:, None]
            return self.anchor[None, :] * scales

    class DummyReranker(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.1))

        def score(self, questions, texts):
            return self.anchor.expand(len(texts))

    model = DECGCModel(
        fact_size=4,
        hidden_size=8,
        retriever=DummyRetriever(),
        reranker=DummyReranker(),
    )
    batch = {
        "questions": ["q1", "q2"],
        "fact_sequences": [["f1"], ["f1", "f2"]],
    }
    tensors = {
        "previous": torch.zeros(2, 8),
        "fact": torch.zeros(2, 4),
        "hop": torch.tensor([1, 2]),
        "temporal": torch.ones(2),
        "type_score": torch.ones(2),
    }

    output, pre_final = _forward_prefix_sequences(
        model,
        batch,
        tensors,
        torch.device("cpu"),
    )

    assert output["reliability_logits"].shape == (2,)
    assert torch.count_nonzero(pre_final[0]) == 0
    assert torch.count_nonzero(pre_final[1]) > 0


def test_multihop_state_buffer_tracks_amp_output_dtype():
    class HalfPrecisionSequenceModel:
        retriever = object()
        chain_encoder = SimpleNamespace(
            gru=SimpleNamespace(hidden_size=4),
        )

        def encode_live(self, questions, fact_texts):
            return torch.ones(
                len(questions),
                3,
                dtype=torch.float16,
            )

        def prefix_step(self, previous, fact, hop, temporal, type_score):
            assert previous.dtype == fact.dtype == torch.float16
            chain = previous + fact[:, :1]
            logits = chain[:, 0]
            return {
                "chain": chain,
                "reliability_logits": logits,
                "stop_logits": logits,
            }

    batch = {
        "questions": ["q1", "q2"],
        "fact_sequences": [["f1"], ["f1", "f2"]],
    }
    tensors = {
        "temporal": torch.ones(2),
        "type_score": torch.ones(2),
    }

    output, pre_final = _forward_prefix_sequences(
        HalfPrecisionSequenceModel(),
        batch,
        tensors,
        torch.device("cpu"),
    )

    assert output["chain"].dtype == torch.float16
    assert pre_final.dtype == torch.float16
    assert torch.count_nonzero(pre_final[1]) > 0


def test_reasoner_preflight_exercises_half_precision_sequence_path():
    class HalfPrecisionSequenceModel:
        retriever = object()
        chain_encoder = SimpleNamespace(
            gru=SimpleNamespace(hidden_size=4),
        )
        training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True):
            self.training = mode
            return self

        def encode_live(self, questions, fact_texts):
            return torch.ones(len(questions), 4, dtype=torch.float16)

        def prefix_step(self, previous, fact, hop, temporal, type_score):
            chain = previous.to(fact.dtype) + fact
            logits = chain[:, 0]
            return {
                "chain": chain,
                "reliability_logits": logits,
                "stop_logits": logits,
            }

        def score_answers(self, chains, answers):
            return chains[:, 0]

        def score_interactions(self, chains, answers, groups):
            return chains[:, 0]

    batch = {
        "questions": ["q1"],
        "group_ids": ["q1"],
        "answer_texts": ["Alice"],
        "fact_sequences": [["fact"]],
        "cf_fact_texts": ["counterfactual"],
        "cf_mask": torch.ones(1),
        "hop": torch.ones(1, dtype=torch.long),
        "temporal": torch.ones(1),
        "type_score": torch.ones(1),
    }

    result = reasoner_runtime_preflight(
        HalfPrecisionSequenceModel(),
        [batch],
        torch.device("cpu"),
        use_amp=False,
    )

    assert result["state_dtype"] == "torch.float16"
    assert result["batch_items"] == 1


def test_answer_scorer_is_trainable_from_candidate_text():
    class DummyRetriever(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones(4))
            self.model = SimpleNamespace(config=SimpleNamespace(hidden_size=4))

        def encode(self, texts):
            values = torch.tensor(
                [[float(len(text)), 1.0, 0.5, -0.5] for text in texts],
                device=self.anchor.device,
            )
            return values * self.anchor

    class DummyReranker(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.1))

        def score(self, questions, texts):
            return self.anchor.expand(len(texts))

    model = DECGCModel(
        fact_size=4,
        hidden_size=8,
        retriever=DummyRetriever(),
        reranker=DummyReranker(),
    )
    chains = torch.randn(2, 8)

    logits = model.score_answers(chains, ["Alice", "Bob"])
    logits.sum().backward()

    assert logits.shape == (2,)
    assert model.answer_projection.weight.grad is not None
    assert next(model.answer_scorer.parameters()).grad is not None


def test_candidate_answer_loss_is_invariant_to_redundant_weaker_path():
    base_logits = torch.tensor([2.0, 1.0], requires_grad=True)
    base_loss = grouped_candidate_answer_ranking_loss(
        base_logits,
        torch.tensor([1.0, 0.0]),
        ["q1", "q1"],
        ["Gold Answer", "wrong"],
    )
    duplicated_logits = torch.tensor([2.0, -100.0, 1.0], requires_grad=True)
    duplicated_loss = grouped_candidate_answer_ranking_loss(
        duplicated_logits,
        torch.tensor([1.0, 1.0, 0.0]),
        ["q1", "q1", "q1"],
        ["Gold Answer", "gold_answer", "wrong"],
    )

    assert torch.allclose(base_loss, duplicated_loss)
    duplicated_loss.backward()
    assert duplicated_logits.grad is not None


def test_validation_checkpoint_metric_uses_complete_question_group():
    model = DECGCModel(fact_size=4, hidden_size=8)
    batch = {
        "previous": torch.zeros(2, 8),
        "fact": torch.stack([torch.ones(4), -torch.ones(4)]),
        "hop": torch.tensor([1, 1]),
        "temporal": torch.tensor([1.0, 0.0]),
        "type_score": torch.tensor([1.0, 0.0]),
        "prefix_label": torch.tensor([1.0, 0.0]),
        "stop_label": torch.tensor([1.0, 0.0]),
        "questions": ["same question", "same question"],
        "group_ids": ["q1", "q1"],
        "fact_texts": ["positive", "negative"],
        "answer_texts": ["Alice", "Bob"],
        "cf_fact_texts": [],
    }

    metrics = validation_ranking_metrics(
        model,
        [batch],
        torch.device("cpu"),
    )

    assert metrics["groups"] == 1
    assert metrics["hits1"] in {0.0, 1.0}
    assert metrics["loss"] >= 0.0


def test_full_reasoner_epoch_uses_validation_and_real_counterfactual_mask(
    config,
    tmp_path,
):
    class DummyRetriever(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones(4))
            self.model = SimpleNamespace(config=SimpleNamespace(hidden_size=4))

        def encode(self, texts):
            scales = torch.tensor(
                [float(len(text)) for text in texts],
                device=self.anchor.device,
            )[:, None]
            return self.anchor[None, :] * scales

    class DummyReranker(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.1))

        def score(self, questions, texts):
            return self.anchor.expand(len(texts))

    model = DECGCModel(
        fact_size=4,
        hidden_size=8,
        retriever=DummyRetriever(),
        reranker=DummyReranker(),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    batch = {
        "previous": torch.zeros(2, 8),
        "fact": torch.zeros(2, 4),
        "hop": torch.tensor([1, 1]),
        "temporal": torch.tensor([1.0, 0.0]),
        "type_score": torch.tensor([1.0, 1.0]),
        "prefix_label": torch.tensor([1.0, 0.0]),
        "stop_label": torch.tensor([1.0, 0.0]),
        "cf_mask": torch.tensor([1.0, 0.0]),
        "questions": ["Who visited?", "Who visited?"],
        "group_ids": ["q1", "q1"],
        "fact_texts": ["gold fact", "wrong fact"],
        "answer_texts": ["Alice", "Bob"],
        "fact_sequences": [["gold fact"], ["wrong fact"]],
        "cf_fact_texts": ["shifted-time fact", "wrong fact"],
    }

    result = train_reasoner(
        model,
        [batch],
        optimizer,
        config,
        tmp_path,
        resume="none",
        validation_batches=[batch],
        checkpoint_config_hash="validation-stage",
    )

    assert result["training_complete"]
    assert result["latest_validation"]["groups"] == 1
    assert (tmp_path / "checkpoints/best.json").exists()


def test_chain_cooperation_and_competition_are_trainable():
    model = DECGCModel(fact_size=4, hidden_size=8)
    chains = torch.randn(3, 8, requires_grad=True)

    logits = model.score_interactions(
        chains,
        ["Alice", "Alice", "Bob"],
        ["q1", "q1", "q1"],
    )
    logits.sum().backward()

    assert logits.shape == (3,)
    assert chains.grad is not None
    assert next(model.interaction_head.parameters()).grad is not None
