import pytest

torch = pytest.importorskip("torch")

from decgc.models.losses import (
    combined_loss,
    grouped_answer_ranking_loss,
    multi_positive_candidate_loss,
    multi_positive_inbatch_loss,
    prefix_bce,
)


def test_combined_loss_matches_equation_7_three_term_objective():
    result = combined_loss(
        torch.tensor([0.3, -0.2]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.4, -0.1]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.8]),
        torch.tensor([0.1]),
        {"prefix": 0.5, "stop": 0.3, "counterfactual": 0.3},
    )

    assert set(result) == {"total", "prefix", "stop", "counterfactual"}
    expected = 0.5 * result["prefix"] + 0.3 * result["stop"] + 0.3 * result["counterfactual"]
    assert torch.allclose(result["total"], expected)


def test_prefix_bce_uses_absolute_reliability_logits():
    logits = torch.tensor([1.4, -1.4], requires_grad=True)
    loss = prefix_bce(logits, torch.tensor([1.0, 0.0]))
    loss.backward()
    assert logits.grad is not None


def test_prefix_bce_is_safe_inside_autocast():
    logits = torch.tensor([1.4, -1.4], requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = prefix_bce(logits, torch.tensor([1.0, 0.0]))
    loss.backward()
    assert torch.isfinite(loss)


def test_grouped_ranking_loss_promotes_gold_prefix_within_question():
    logits = torch.tensor([0.0, 2.0, -1.0, 1.0], requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = grouped_answer_ranking_loss(logits, labels, ["q1", "q1", "q2", "q2"])

    loss.backward()

    assert logits.grad[0] < 0
    assert logits.grad[1] > 0
    assert logits.grad[2] < 0
    assert logits.grad[3] > 0


def test_multi_positive_inbatch_loss_does_not_penalize_same_question_positive():
    good = torch.tensor([
        [4.0, 3.0, -2.0],
        [3.0, 4.0, -2.0],
        [-2.0, -2.0, 4.0],
    ])
    bad = torch.tensor([
        [4.0, -2.0, 3.0],
        [-2.0, 4.0, 3.0],
        [3.0, 3.0, 4.0],
    ])
    groups = ["q1", "q1", "q2"]

    assert multi_positive_inbatch_loss(good, groups) < multi_positive_inbatch_loss(
        bad,
        groups,
    )


def test_explicit_hard_negatives_participate_in_temperature_scaled_infonce():
    easy = torch.tensor([
        [5.0, -2.0, -4.0, -3.0],
        [-2.0, 5.0, -3.0, -4.0],
    ])
    hard = easy.clone()
    hard[0, 2] = 6.0

    assert multi_positive_candidate_loss(
        easy,
        ["q1", "q2"],
        positive_columns=2,
    ) < multi_positive_candidate_loss(
        hard,
        ["q1", "q2"],
        positive_columns=2,
    )
