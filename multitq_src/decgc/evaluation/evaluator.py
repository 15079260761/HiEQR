"""Offline beam inference and answer aggregation."""

from .answer_normalizer import normalize_answer
from ..schemas import ChainPrefix


def stop_candidates(chains: list[ChainPrefix]) -> list[ChainPrefix]:
    """Treat every legal prefix as a possible learned stopping point."""
    unique: dict[tuple[tuple[str, ...], str], ChainPrefix] = {}
    for chain in chains:
        if chain.state == "PRUNED" or not chain.facts or not chain.candidate_answer:
            continue
        key = (
            tuple(fact.fact_id for fact in chain.facts),
            chain.candidate_answer,
        )
        candidate = chain.model_copy(deep=True)
        candidate.state = "COMPLETED"
        if chain.state != "COMPLETED":
            candidate.stop_reason = "learned_stop_candidate"
        existing = unique.get(key)
        if existing is None or candidate.reranker_scores[-1:] > existing.reranker_scores[-1:]:
            unique[key] = candidate
    return list(unique.values())


def aggregate_completed(chains: list) -> tuple[list[str], dict[str, float]]:
    """Rank unique answers by their strongest learned evidence chain.

    Summing every path lets high-degree entities win merely because beam search
    generated many redundant routes to them. Answer-level max pooling removes that
    degree bias. Hop is already represented by the chain encoder, so applying a
    second fixed length penalty here would make inference disagree with the
    answer-level validation objective and suppress genuine multi-hop evidence.
    """
    grouped: dict[str, tuple[str, float]] = {}
    for chain in chains:
        if chain.state == "COMPLETED" and chain.candidate_answer:
            temporal_factor = 0.1 + 0.9 * max(
                0.0,
                min(1.0, float(chain.temporal_score)),
            )
            # relative_weight is already the answer-level normalization of
            # reliability + stopping + answer + interaction logits after
            # reasoner rescoring. Multiplying those probabilities again would
            # double-count the same evidence and diverge from validation.
            support = chain.relative_weight * temporal_factor
            key = normalize_answer(chain.candidate_answer)
            existing = grouped.get(key)
            if existing is None or support > existing[1]:
                grouped[key] = (chain.candidate_answer, support)
    scores = {
        answer: support
        for answer, support in grouped.values()
    }
    ranking = sorted(scores, key=scores.get, reverse=True)
    return ranking, scores
