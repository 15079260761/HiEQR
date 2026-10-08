"""Bounded weak-positive and hard-negative retriever pair construction."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

from ..data.temporal_parser import parse_time, temporal_compatible
from ..retrieval.fact_formatter import format_fact

RETRIEVER_DATA_VERSION = 4


def _iter_unique_neighbor_ids(kg_index, entities: Iterable[str]) -> Iterator[str]:
    """Yield adjacency ids in stable KG order without materializing a large neighborhood."""
    seen: set[str] = set()
    for entity in entities:
        for fact_id in kg_index.adjacency.get(entity, ()):
            if fact_id not in seen:
                seen.add(fact_id)
                yield fact_id


def _answer_ids(answers: Iterable[str]) -> set[str]:
    values: set[str] = set()
    for answer in answers:
        text = str(answer).strip()
        values.update((text, text.replace(" ", "_")))
    return values


def _supports_answer(
    fact,
    answers: set[str],
    answer_type: str | None = None,
) -> bool:
    if answer_type == "time":
        valid_times = [
            answer
            for answer in answers
            if parse_time(answer)["normalized"] is not None
        ]
        return any(
            temporal_compatible(fact.timestamp_raw, [answer], "equal")
            for answer in valid_times
        )
    return bool(answers & {
        fact.subject_id,
        fact.subject_text,
        fact.object_id,
        fact.object_text,
        fact.timestamp_raw,
        fact.timestamp_normalized,
    })


def _relation_matches(fact, hints: list[str]) -> bool:
    if not hints:
        return True
    relation = f"{fact.relation_id} {fact.relation_text}".replace("_", " ").casefold()
    return any(
        hint.replace("_", " ").casefold() in relation
        or relation in hint.replace("_", " ").casefold()
        for hint in hints
    )


def _direction_matches(fact, query) -> bool:
    topics = set(query.topic_entities)
    if query.answer_role == "time" and len(topics) >= 2:
        return topics.issubset({fact.subject_id, fact.object_id})
    if query.answer_role == "subject":
        return fact.object_id in topics
    if query.answer_role == "object":
        return fact.subject_id in topics
    return bool(topics & {fact.subject_id, fact.object_id})


def _positive_matches(fact, query, require_direction: bool = True) -> bool:
    return (
        _relation_matches(fact, query.relation_hints)
        and (not require_direction or _direction_matches(fact, query))
        and temporal_compatible(
            fact.timestamp_raw,
            query.temporal_constraints,
            query.temporal_operator,
        )
    )


def build_retriever_records_for_example(
    example,
    kg_index,
    query,
    max_pairs_per_question: int = 4,
    max_neighbor_facts_scanned: int = 4096,
) -> list[dict]:
    """Build a small deterministic set of weakly supervised triples for one question.

    Positives contain an answer in the topic-entity neighborhood. Negatives come from
    that same neighborhood, making them harder than arbitrary KG facts. Both sides are
    capped so high-degree entities cannot dominate runtime or training.
    """
    if max_pairs_per_question <= 0:
        raise ValueError("max_pairs_per_question must be positive")
    if max_neighbor_facts_scanned <= 0:
        raise ValueError("max_neighbor_facts_scanned must be positive")

    answers = _answer_ids(example.answers)
    positive_ids: list[str] = []
    negative_ids: list[str] = []

    # Entity answers can be joined through the adjacency index directly. This finds
    # topic-to-answer facts without walking a high-degree topic neighborhood first.
    topic_entities = set(query.topic_entities)
    for scanned, fact_id in enumerate(_iter_unique_neighbor_ids(kg_index, answers), 1):
        fact = kg_index.facts[fact_id]
        if (
            (fact.subject_id in topic_entities or fact.object_id in topic_entities)
            and _supports_answer(fact, answers, example.answer_type)
            and _positive_matches(fact, query)
        ):
            positive_ids.append(fact_id)
            if len(positive_ids) == max_pairs_per_question:
                break
        if scanned == max_neighbor_facts_scanned:
            break

    # A single linear pass replaces the former list-membership test, whose worst-case
    # cost was O(neighborhood_size * positive_count) with full Pydantic equality.
    hard_negative_ids: list[str] = []
    fallback_negative_ids: list[str] = []
    for scanned, fact_id in enumerate(_iter_unique_neighbor_ids(kg_index, query.topic_entities), 1):
        fact = kg_index.facts[fact_id]
        if _supports_answer(fact, answers, example.answer_type):
            if (
                _positive_matches(fact, query)
                and len(positive_ids) < max_pairs_per_question
                and fact_id not in positive_ids
            ):
                positive_ids.append(fact_id)
        elif _relation_matches(fact, query.relation_hints) and _direction_matches(fact, query):
            if len(hard_negative_ids) < max_pairs_per_question:
                hard_negative_ids.append(fact_id)
        elif len(fallback_negative_ids) < max_pairs_per_question:
            fallback_negative_ids.append(fact_id)
        if (
            len(positive_ids) == max_pairs_per_question
            and len(hard_negative_ids) == max_pairs_per_question
        ):
            break
        if scanned == max_neighbor_facts_scanned:
            break
    negative_ids = (hard_negative_ids + fallback_negative_ids)[:max_pairs_per_question]

    # Preserve the previous weak-supervision fallback: an answer-adjacent fact can be
    # positive even when the parser failed to connect the answer to a topic entity.
    if not positive_ids:
        for fact_id in _iter_unique_neighbor_ids(kg_index, answers):
            fact = kg_index.facts[fact_id]
            if (
                _supports_answer(fact, answers, example.answer_type)
                and _positive_matches(fact, query, require_direction=False)
            ):
                positive_ids.append(fact_id)
                if len(positive_ids) == max_pairs_per_question:
                    break

    if not negative_ids:
        positive_set = set(positive_ids)
        for scanned, fact_id in enumerate(kg_index.facts, 1):
            fact = kg_index.facts[fact_id]
            if (
                fact_id not in positive_set
                and not _supports_answer(
                    fact,
                    answers,
                    example.answer_type,
                )
            ):
                negative_ids.append(fact_id)
                if len(negative_ids) == max_pairs_per_question:
                    break
            if scanned == max_neighbor_facts_scanned:
                break

    return [
        {
            "query": example.question,
            "positive": format_fact(kg_index.facts[positive_id]),
            "negative": format_fact(kg_index.facts[negative_id]),
        }
        for positive_id, negative_id in zip(positive_ids, negative_ids)
    ]


def build_retriever_records(
    examples,
    kg_index,
    parsed_queries,
    max_pairs_per_question: int = 4,
    max_neighbor_facts_scanned: int = 4096,
    progress: Callable[[int, int, int], None] | None = None,
) -> list[dict]:
    records = []
    total = len(examples)
    for processed, example in enumerate(examples, 1):
        query = parsed_queries[example.qid]
        records.extend(build_retriever_records_for_example(
            example,
            kg_index,
            query,
            max_pairs_per_question,
            max_neighbor_facts_scanned,
        ))
        if progress is not None:
            progress(processed, total, len(records))
    return records
