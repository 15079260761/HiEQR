import pytest

from decgc.data.kg_index import KGIndex
from decgc.data.canonical_schema import canonical_fact
from decgc.data.multitq_adapter import MultiTQAdapter
from decgc.retrieval.beam_search import TemporalBeamSearch
from decgc.schemas import StructuredQuery


def test_beam_paths_are_connected_and_terminal(config):
    index = KGIndex(list(MultiTQAdapter(config).iter_facts("full")))
    query = StructuredQuery(qid="q", topic_entities=["Alice"], relation_hints=["visit"],
                            temporal_constraints=["2007-02-27"], temporal_operator="equal",
                            answer_type="entity", answer_role="object", parse_confidence=1,
                            parse_source="test")
    completed, audit = TemporalBeamSearch(index, beam_size=2, max_hops=2).run("Who visited?", query)
    assert completed
    assert all(chain.state == "COMPLETED" for chain in completed)
    assert all(a.object_id in {b.subject_id, b.object_id} or a.subject_id in {b.subject_id, b.object_id}
               for chain in audit for a, b in zip(chain.facts, chain.facts[1:]))


def test_first_and_last_temporal_operators_preserve_direction(config):
    index = KGIndex(list(MultiTQAdapter(config).iter_facts("full")))
    base = {
        "qid": "q",
        "topic_entities": ["Iraq"],
        "relation_hints": ["visit"],
        "temporal_constraints": [],
        "answer_type": "entity",
        "answer_role": "subject",
        "parse_confidence": 1,
        "parse_source": "test",
    }

    first, _ = TemporalBeamSearch(index, beam_size=1, max_hops=1).run(
        "Who first visited Iraq?",
        StructuredQuery(**base, temporal_operator="first"),
    )
    last, _ = TemporalBeamSearch(index, beam_size=1, max_hops=1).run(
        "Who last visited Iraq?",
        StructuredQuery(**base, temporal_operator="last"),
    )

    assert first[0].candidate_answer == "Alice"
    assert last[0].candidate_answer == "Bob"


def test_relative_anchor_entity_is_resolved_to_event_time(config):
    index = KGIndex(list(MultiTQAdapter(config).iter_facts("full")))
    query = StructuredQuery(
        qid="q",
        topic_entities=["Iraq"],
        target_entities=["Alice"],
        relation_hints=["visit"],
        temporal_constraints=[],
        temporal_operator="after",
        answer_type="entity",
        answer_role="subject",
        parse_confidence=1,
        parse_source="test",
    )

    completed, _ = TemporalBeamSearch(index, beam_size=2, max_hops=1).run(
        "After Alice, who visited Iraq?",
        query,
    )

    assert [chain.candidate_answer for chain in completed] == ["Bob"]


def test_equal_multi_anchor_keeps_requested_granularity(config):
    index = KGIndex(list(MultiTQAdapter(config).iter_facts("full")))
    query = StructuredQuery(
        qid="q",
        topic_entities=["Iraq"],
        target_entities=["Alice"],
        relation_hints=["visit"],
        temporal_constraints=[],
        temporal_operator="equal_multi",
        temporal_granularity="year",
        answer_type="entity",
        answer_role="subject",
        parse_confidence=1,
        parse_source="test",
    )
    search = TemporalBeamSearch(index)

    resolved = search._resolve_anchor_constraints(
        query,
        "Who visited Iraq at the corresponding granularity as Alice?",
    )

    assert resolved.temporal_constraints == ["2007"]


def test_answer_role_filters_inverse_direction_noise(config):
    index = KGIndex(list(MultiTQAdapter(config).iter_facts("full")))
    query = StructuredQuery(
        qid="q",
        topic_entities=["Iraq"],
        relation_hints=[],
        temporal_constraints=[],
        temporal_operator=None,
        answer_type="entity",
        answer_role="subject",
        parse_confidence=1,
        parse_source="test",
    )

    completed, _ = TemporalBeamSearch(index, beam_size=8, max_hops=1).run(
        "Who acted on Iraq?",
        query,
    )

    assert {chain.candidate_answer for chain in completed} == {"Alice", "Bob"}


def test_time_answer_requires_both_known_entities_in_same_fact(config):
    index = KGIndex(list(MultiTQAdapter(config).iter_facts("full")))
    query = StructuredQuery(
        qid="q",
        topic_entities=["Alice", "Iraq"],
        relation_hints=["visit"],
        temporal_constraints=[],
        temporal_operator="first",
        answer_type="time",
        answer_role="time",
        parse_confidence=1,
        parse_source="test",
    )

    completed, _ = TemporalBeamSearch(index, beam_size=8, max_hops=1).run(
        "When did Alice first visit Iraq?",
        query,
    )

    assert {chain.candidate_answer for chain in completed} == {"2007-02-27"}


def test_beam_reserves_slots_for_distinct_candidate_answers():
    facts = [
        canonical_fact(
            ["Topic", "visit", "Duplicate", f"2007-01-{day:02d}"],
            "full",
            day,
        )
        for day in range(1, 7)
    ]
    facts.append(
        canonical_fact(
            ["Topic", "visit", "Distinct", "2007-02-01"],
            "full",
            20,
        )
    )
    query = StructuredQuery(
        qid="q",
        topic_entities=["Topic"],
        relation_hints=["visit"],
        temporal_constraints=[],
        temporal_operator=None,
        answer_type="entity",
        answer_role="object",
        parse_confidence=1,
        parse_source="test",
    )

    completed, _ = TemporalBeamSearch(
        KGIndex(facts),
        beam_size=2,
        max_hops=1,
        max_paths_per_answer=1,
    ).run("Who did Topic visit?", query)

    assert {chain.candidate_answer for chain in completed} == {
        "Duplicate",
        "Distinct",
    }


def test_ordinal_cutoff_cannot_drop_true_temporal_extreme():
    torch = pytest.importorskip("torch")

    facts = [
        canonical_fact(
            [f"Actor_{index}", "visit", "Topic", f"2007-01-{index + 1:02d}"],
            "full",
            index,
        )
        for index in range(20)
    ]

    class ReverseTemporalReranker:
        def score(self, questions, texts):
            # Prefer later facts semantically; symbolic first must still win.
            return torch.arange(len(texts), dtype=torch.float32)

    query = StructuredQuery(
        qid="q",
        topic_entities=["Topic"],
        relation_hints=["visit"],
        temporal_constraints=[],
        temporal_operator="first",
        answer_type="entity",
        answer_role="subject",
        parse_confidence=1,
        parse_source="test",
    )
    completed, _ = TemporalBeamSearch(
        KGIndex(facts),
        reranker=ReverseTemporalReranker(),
        beam_size=1,
        retrieve_top_k=20,
        rerank_top_k=4,
        max_hops=1,
    ).run("Who first visited Topic?", query)

    assert completed[0].candidate_answer == "Actor_0"
