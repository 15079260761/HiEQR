"""Balanced weak supervision for chain-prefix reliability and stopping."""

from __future__ import annotations

from ..evaluation.answer_normalizer import normalize_answer
from ..retrieval.type_consistency import answer_type_score
from ..schemas import ChainPrefix

REASONER_DATA_VERSION = 4


def is_weak_positive(
    chain: ChainPrefix,
    gold_answers: list[str],
    answer_type: str | None = None,
) -> bool:
    """A prefix is positive when its current candidate already answers the question."""
    gold = {normalize_answer(answer) for answer in gold_answers}
    return (
        chain.state != "PRUNED"
        and bool(chain.candidate_answer)
        and normalize_answer(chain.candidate_answer or "") in gold
        and chain.temporal_score > 0
        and answer_type_score(chain.candidate_answer, answer_type) > 0
    )


def is_optimal_stop_prefix(
    chain: ChainPrefix,
    candidate_rows: list[dict],
    gold_answers: list[str],
    answer_type: str | None = None,
) -> bool:
    """Label the earliest sufficient prefix on a concrete evidence route.

    A later gold-supporting prefix remains a positive reliability example, but
    it is not a positive stopping example when one of its proper prefixes was
    already sufficient.  This separates evidence correctness from the learned
    stopping boundary.
    """
    if not is_weak_positive(chain, gold_answers, answer_type):
        return False
    fact_ids = tuple(fact.fact_id for fact in chain.facts)
    for row in candidate_rows:
        shorter = ChainPrefix.model_validate(row)
        shorter_ids = tuple(fact.fact_id for fact in shorter.facts)
        if (
            0 < len(shorter_ids) < len(fact_ids)
            and fact_ids[: len(shorter_ids)] == shorter_ids
            and is_weak_positive(shorter, gold_answers, answer_type)
        ):
            return False
    return True


def _hardness(chain: ChainPrefix) -> tuple[float, float, int, str]:
    reranker = chain.reranker_scores[-1] if chain.reranker_scores else 0.0
    retrieval = chain.retrieval_scores[-1] if chain.retrieval_scores else chain.semantic_score
    return float(reranker), float(retrieval), -chain.hop, chain.chain_id


def select_balanced_prefixes(
    rows: list[dict],
    gold_answers: list[str],
    max_prefixes: int,
    answer_type: str | None = None,
) -> list[tuple[ChainPrefix, bool]]:
    """Select positives first and the strongest confusing negatives for each question."""
    if max_prefixes <= 0:
        raise ValueError("max_prefixes must be positive")
    unique: dict[tuple[tuple[str, ...], str], ChainPrefix] = {}
    for row in rows:
        chain = ChainPrefix.model_validate(row)
        if chain.state == "PRUNED" or not chain.facts or not chain.candidate_answer:
            continue
        key = (
            tuple(fact.fact_id for fact in chain.facts),
            normalize_answer(chain.candidate_answer),
        )
        existing = unique.get(key)
        if existing is None or _hardness(chain) > _hardness(existing):
            unique[key] = chain

    positives = sorted(
        (
            chain for chain in unique.values()
            if is_weak_positive(chain, gold_answers, answer_type)
        ),
        key=_hardness,
        reverse=True,
    )
    negatives = sorted(
        (
            chain for chain in unique.values()
            if not is_weak_positive(chain, gold_answers, answer_type)
        ),
        key=_hardness,
        reverse=True,
    )
    selected: list[tuple[ChainPrefix, bool]] = []
    if positives:
        selected.append((positives.pop(0), True))
    if negatives and len(selected) < max_prefixes:
        selected.append((negatives.pop(0), False))
    pools = (positives, negatives)
    while len(selected) < max_prefixes and any(pools):
        for pool, label in ((positives, True), (negatives, False)):
            if pool and len(selected) < max_prefixes:
                selected.append((pool.pop(0), label))
    return selected


def pack_question_groups(samples: list[dict], batch_size: int) -> list[list[dict]]:
    """Pack batches without ever splitting one question's ranking group."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    groups: dict[str, list[dict]] = {}
    for sample in samples:
        qid = str(sample["qid"])
        groups.setdefault(qid, []).append(sample)

    batches: list[list[dict]] = []
    current: list[dict] = []
    for group in groups.values():
        if current and len(current) + len(group) > batch_size:
            batches.append(current)
            current = []
        if len(group) > batch_size:
            batches.append(group)
        else:
            current.extend(group)
    if current:
        batches.append(current)
    return batches


def collate_question_batches(samples: list[dict], batch_size: int) -> list[dict]:
    """Tensor-collate complete question groups while preserving text metadata."""
    import torch

    batches = []
    for group in pack_question_groups(samples, batch_size):
        cf_fact_texts: list[str] = []
        cf_questions: list[str] = []
        cf_types: list[str] = []
        cf_owner_indices: list[int] = []
        for owner_index, item in enumerate(group):
            texts = item.get("cf_fact_texts", [])
            types = item.get("cf_types", [])
            if len(texts) != len(types):
                raise ValueError("counterfactual texts and types must align")
            for text, counterfactual_type in zip(texts, types):
                cf_fact_texts.append(text)
                cf_questions.append(item["questions"][0])
                cf_types.append(counterfactual_type)
                cf_owner_indices.append(owner_index)
        batches.append({
            "previous": torch.cat([item["previous"] for item in group]),
            "fact": torch.cat([item["fact"] for item in group]),
            "hop": torch.cat([item["hop"] for item in group]),
            "temporal": torch.cat([item["temporal"] for item in group]),
            "type_score": torch.cat([item["type_score"] for item in group]),
            "prefix_label": torch.cat([item["prefix_label"] for item in group]),
            "stop_label": torch.cat([item["stop_label"] for item in group]),
            "questions": [text for item in group for text in item["questions"]],
            "group_ids": [item["qid"] for item in group],
            "fact_texts": [text for item in group for text in item["fact_texts"]],
            "answer_texts": [text for item in group for text in item["answer_texts"]],
            "fact_sequences": [
                sequence for item in group for sequence in item["fact_sequences"]
            ],
            "cf_fact_texts": cf_fact_texts,
            "cf_questions": cf_questions,
            "cf_types": cf_types,
            "cf_owner_indices": torch.tensor(cf_owner_indices, dtype=torch.long),
            "cf_mask": torch.cat([item["cf_mask"] for item in group]),
        })
    return batches
