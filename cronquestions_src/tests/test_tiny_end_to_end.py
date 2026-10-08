from decgc.data.kg_index import KGIndex
from decgc.data.multitq_adapter import MultiTQAdapter
from decgc.evaluation.evaluator import aggregate_completed, stop_candidates
from decgc.retrieval.beam_search import TemporalBeamSearch
from decgc.schemas import StructuredQuery


def test_tiny_pipeline_outputs_answer_and_chain(config):
    index = KGIndex(list(MultiTQAdapter(config).iter_facts("full")))
    query = StructuredQuery(qid="1", topic_entities=["Alice"], relation_hints=["visit"],
                            temporal_constraints=["2007-02-27"], temporal_operator="equal",
                            answer_type="entity", answer_role="object", parse_confidence=1, parse_source="test")
    completed, _ = TemporalBeamSearch(index, beam_size=2, max_hops=1).run("Who visited Iraq?", query)
    ranking, _ = aggregate_completed(completed)
    assert ranking and completed[0].facts[0].source_relative_path == "kg/full.txt"


def test_active_prefix_is_a_valid_learned_stop_candidate(config):
    index = KGIndex(list(MultiTQAdapter(config).iter_facts("full")))
    query = StructuredQuery(qid="1", topic_entities=["Iraq"], relation_hints=["visit"],
                            temporal_constraints=["2007-02-27"], temporal_operator="equal",
                            answer_type="entity", answer_role="subject", parse_confidence=1, parse_source="test")
    _, audit = TemporalBeamSearch(index, beam_size=3, max_hops=2).run(
        "Who visited Iraq on 27 February 2007?", query,
    )

    candidates = stop_candidates(audit)

    assert any(chain.hop == 1 and chain.candidate_answer == "Alice" for chain in candidates)


def test_answer_aggregation_respects_symbolic_temporal_preference(config):
    index = KGIndex(list(MultiTQAdapter(config).iter_facts("full")))
    query = StructuredQuery(
        qid="1",
        topic_entities=["Iraq"],
        relation_hints=["visit"],
        temporal_constraints=[],
        temporal_operator="first",
        answer_type="entity",
        answer_role="subject",
        parse_confidence=1,
        parse_source="test",
    )
    completed, _ = TemporalBeamSearch(index, beam_size=2, max_hops=1).run(
        "Who first visited Iraq?",
        query,
    )
    for chain in completed:
        chain.absolute_reliability = 1.0
        chain.relative_weight = 0.5
        chain.stop_probability = 1.0
        chain.answer_score = 1.0

    ranking, scores = aggregate_completed(completed)

    assert ranking[0] == "Alice"
    assert scores["Alice"] > scores["Bob"]


def test_answer_aggregation_is_not_biased_by_duplicate_paths(config):
    index = KGIndex(list(MultiTQAdapter(config).iter_facts("full")))
    query = StructuredQuery(
        qid="1",
        topic_entities=["Iraq"],
        relation_hints=["visit"],
        temporal_constraints=[],
        temporal_operator=None,
        answer_type="entity",
        answer_role="subject",
        parse_confidence=1,
        parse_source="test",
    )
    completed, _ = TemporalBeamSearch(index, beam_size=2, max_hops=1).run(
        "Who visited Iraq?",
        query,
    )
    by_answer = {chain.candidate_answer: chain for chain in completed}
    alice = by_answer["Alice"]
    bob = by_answer["Bob"]
    for chain, weight in ((alice, 0.4), (bob, 0.3)):
        chain.absolute_reliability = 1.0
        chain.relative_weight = weight
        chain.stop_probability = 1.0
        chain.answer_score = 1.0
        chain.temporal_score = 1.0
    duplicate_bob = []
    for index_value in range(12):
        duplicate = bob.model_copy(deep=True)
        duplicate.chain_id = f"bob-{index_value}"
        duplicate_bob.append(duplicate)

    ranking, scores = aggregate_completed([alice, bob, *duplicate_bob])

    assert ranking[0] == "Alice"
    assert scores["Alice"] > scores["Bob"]


def test_answer_aggregation_follows_learned_answer_weight_across_hops(config):
    index = KGIndex(list(MultiTQAdapter(config).iter_facts("full")))
    query = StructuredQuery(
        qid="1",
        topic_entities=["Iraq"],
        relation_hints=["visit"],
        temporal_constraints=[],
        temporal_operator=None,
        answer_type="entity",
        answer_role="subject",
        parse_confidence=1,
        parse_source="test",
    )
    completed, _ = TemporalBeamSearch(index, beam_size=2, max_hops=1).run(
        "Who visited Iraq?",
        query,
    )
    by_answer = {chain.candidate_answer: chain for chain in completed}
    alice = by_answer["Alice"]
    bob = by_answer["Bob"]
    for chain in (alice, bob):
        chain.absolute_reliability = 1.0
        chain.stop_probability = 1.0
        chain.answer_score = 1.0
        chain.temporal_score = 1.0
    alice.relative_weight = 0.4
    alice.hop = 1
    bob.relative_weight = 0.6
    bob.hop = 2

    ranking, _ = aggregate_completed([alice, bob])

    assert ranking[0] == "Bob"


def test_time_answers_are_projected_to_question_granularity(config):
    index = KGIndex(list(MultiTQAdapter(config).iter_facts("full")))
    query = StructuredQuery(
        qid="1",
        topic_entities=["Alice", "Iraq"],
        relation_hints=["visit"],
        temporal_constraints=[],
        temporal_operator="equal",
        temporal_granularity="year",
        answer_type="time",
        answer_role="time",
        parse_confidence=1,
        parse_source="test",
    )

    completed, _ = TemporalBeamSearch(index, beam_size=2, max_hops=1).run(
        "In which year did Alice visit Iraq?",
        query,
    )

    assert {chain.candidate_answer for chain in completed} == {"2007"}


def test_equal_multi_with_first_orders_inside_explicit_time_scope(config):
    index = KGIndex(list(MultiTQAdapter(config).iter_facts("full")))
    query = StructuredQuery(
        qid="1",
        topic_entities=["Iraq"],
        relation_hints=["visit"],
        temporal_constraints=["2007"],
        temporal_operator="equal_multi",
        temporal_granularity="year",
        answer_type="entity",
        answer_role="subject",
        parse_confidence=1,
        parse_source="test",
    )

    completed, _ = TemporalBeamSearch(index, beam_size=2, max_hops=1).run(
        "In 2007, who was the first to visit Iraq?",
        query,
    )

    assert completed[0].candidate_answer == "Alice"


def test_answer_aggregation_does_not_double_count_reasoner_components(config):
    index = KGIndex(list(MultiTQAdapter(config).iter_facts("full")))
    query = StructuredQuery(
        qid="1",
        topic_entities=["Iraq"],
        relation_hints=["visit"],
        temporal_constraints=[],
        temporal_operator=None,
        answer_type="entity",
        answer_role="subject",
        parse_confidence=1,
        parse_source="test",
    )
    completed, _ = TemporalBeamSearch(index, beam_size=2, max_hops=1).run(
        "Who visited Iraq?",
        query,
    )
    by_answer = {chain.candidate_answer: chain for chain in completed}
    alice = by_answer["Alice"]
    bob = by_answer["Bob"]
    alice.relative_weight = 0.6
    alice.absolute_reliability = 0.01
    alice.stop_probability = 0.01
    alice.answer_score = 0.01
    bob.relative_weight = 0.4
    bob.absolute_reliability = 1.0
    bob.stop_probability = 1.0
    bob.answer_score = 1.0
    for chain in (alice, bob):
        chain.temporal_score = 1.0

    ranking, _ = aggregate_completed([alice, bob])

    assert ranking[0] == "Alice"
