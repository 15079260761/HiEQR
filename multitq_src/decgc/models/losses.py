"""Numerically stable joint supervision on prefix and stop logits."""

import torch
import torch.nn.functional as functional

from ..evaluation.answer_normalizer import normalize_answer


def prefix_bce(reliability_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Supervise absolute reliability logits, never relative softmax weights."""
    return functional.binary_cross_entropy_with_logits(reliability_logits, labels.float())


def counterfactual_margin(positive: torch.Tensor, negative: torch.Tensor, margin: float) -> torch.Tensor:
    return functional.relu(margin - positive + negative).mean()


def grouped_answer_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_ids: list[str],
) -> torch.Tensor:
    """Put probability mass on gold-supporting prefixes within each question."""
    losses = []
    for group_id in dict.fromkeys(group_ids):
        indices = [
            index
            for index, candidate_group in enumerate(group_ids)
            if candidate_group == group_id
        ]
        group_index = torch.tensor(indices, dtype=torch.long, device=logits.device)
        group_logits = logits.index_select(0, group_index)
        group_labels = labels.index_select(0, group_index).bool()
        if group_labels.any() and (~group_labels).any():
            losses.append(
                torch.logsumexp(group_logits, dim=0)
                - torch.logsumexp(group_logits[group_labels], dim=0)
            )
    if not losses:
        return logits.sum() * 0
    return torch.stack(losses).mean()


def grouped_candidate_answer_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_ids: list[str],
    answer_texts: list[str],
) -> torch.Tensor:
    """Rank unique answers, so duplicate evidence paths cannot change the target.

    Several prefixes can end in the same candidate answer. Treating those prefixes
    as separate classes makes the loss depend on how many paths happened to be
    generated for an answer. We instead pool each answer with a max operation and
    perform the listwise loss over unique normalized answers per question.
    """
    if not (
        logits.shape[0] == labels.shape[0] == len(group_ids) == len(answer_texts)
    ):
        raise ValueError("candidate-answer ranking metadata must match logits")
    losses = []
    for group_id in dict.fromkeys(group_ids):
        question_indices = [
            index
            for index, candidate_group in enumerate(group_ids)
            if candidate_group == group_id
        ]
        answer_groups: dict[str, list[int]] = {}
        for index in question_indices:
            answer_groups.setdefault(
                normalize_answer(answer_texts[index]),
                [],
            ).append(index)
        answer_logits = []
        answer_labels = []
        for indices in answer_groups.values():
            index_tensor = torch.tensor(
                indices,
                dtype=torch.long,
                device=logits.device,
            )
            answer_logits.append(logits.index_select(0, index_tensor).max())
            answer_labels.append(
                labels.index_select(0, index_tensor).bool().any()
            )
        pooled_logits = torch.stack(answer_logits)
        pooled_labels = torch.stack(answer_labels)
        if pooled_labels.any() and (~pooled_labels).any():
            losses.append(
                torch.logsumexp(pooled_logits, dim=0)
                - torch.logsumexp(pooled_logits[pooled_labels], dim=0)
            )
    if not losses:
        return logits.sum() * 0
    return torch.stack(losses).mean()


def multi_positive_inbatch_loss(
    logits: torch.Tensor,
    group_ids: list[str],
) -> torch.Tensor:
    """Multi-positive InfoNCE without treating same-question positives as negatives."""
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("in-batch logits must be a square matrix")
    if logits.shape[0] != len(group_ids):
        raise ValueError("group_ids must match the logits batch dimension")
    return multi_positive_candidate_loss(
        logits,
        group_ids,
        positive_columns=len(group_ids),
    )


def multi_positive_candidate_loss(
    logits: torch.Tensor,
    group_ids: list[str],
    positive_columns: int,
) -> torch.Tensor:
    """InfoNCE over positive columns followed by arbitrary hard negatives."""
    if logits.ndim != 2 or logits.shape[0] != len(group_ids):
        raise ValueError("candidate logits rows must match group_ids")
    if positive_columns != len(group_ids) or logits.shape[1] < positive_columns:
        raise ValueError(
            "the leading positive columns must correspond one-to-one with group_ids"
        )
    losses = []
    for row, group_id in enumerate(group_ids):
        positives = torch.tensor(
            [
                column < positive_columns
                and group_ids[column] == group_id
                for column in range(logits.shape[1])
            ],
            dtype=torch.bool,
            device=logits.device,
        )
        losses.append(
            torch.logsumexp(logits[row], dim=0)
            - torch.logsumexp(logits[row][positives], dim=0)
        )
    return torch.stack(losses).mean()


def combined_loss(answer_logits, answer_labels, reliability_logits, prefix_labels,
                  stop_logits, stop_labels, positive_scores, counterfactual_scores,
                  weights: dict) -> dict[str, torch.Tensor]:
    components = {
        "answer": functional.cross_entropy(answer_logits, answer_labels),
        "prefix": prefix_bce(reliability_logits, prefix_labels),
        "stop": functional.binary_cross_entropy_with_logits(stop_logits, stop_labels.float()),
        "counterfactual": counterfactual_margin(
            positive_scores, counterfactual_scores, float(weights.get("counterfactual_margin", 0.2))
        ),
    }
    total = (weights.get("answer", 1.0) * components["answer"]
             + weights.get("prefix", 0.5) * components["prefix"]
             + weights.get("stop", 0.3) * components["stop"]
             + weights.get("counterfactual", 0.3) * components["counterfactual"])
    return {"total": total, **components}
