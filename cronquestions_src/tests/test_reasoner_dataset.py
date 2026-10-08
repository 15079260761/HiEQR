from decgc.data.kg_index import KGIndex
from decgc.data.multitq_adapter import MultiTQAdapter
from decgc.retrieval.beam_search import TemporalBeamSearch
from decgc.schemas import StructuredQuery
from decgc.training.reasoner_dataset import (
    is_optimal_stop_prefix,
    pack_question_groups,
    select_balanced_prefixes,
)


def test_balanced_prefix_selection_keeps_gold_and_hard_negative(config):
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
    _, audit = TemporalBeamSearch(index, beam_size=3, max_hops=2).run(
        "Who visited Iraq on 27 February 2007?",
        query,
    )

    selected = select_balanced_prefixes(
        [chain.model_dump(mode="json") for chain in audit],
        ["Alice"],
        max_prefixes=2,
    )

    assert len(selected) == 2
    assert {label for _, label in selected} == {True, False}
    assert any(chain.candidate_answer == "Alice" and label for chain, label in selected)


def test_pack_question_groups_never_splits_a_question():
    samples = [
        {"qid": "q1", "value": 1},
        {"qid": "q1", "value": 2},
        {"qid": "q1", "value": 3},
        {"qid": "q2", "value": 4},
        {"qid": "q2", "value": 5},
        {"qid": "q3", "value": 6},
        {"qid": "q3", "value": 7},
        {"qid": "q3", "value": 8},
        {"qid": "q3", "value": 9},
        {"qid": "q3", "value": 10},
    ]

    batches = pack_question_groups(samples, batch_size=4)

    for qid in {"q1", "q2", "q3"}:
        containing = [batch for batch in batches if any(row["qid"] == qid for row in batch)]
        assert len(containing) == 1
    assert any(len(batch) == 5 for batch in batches)


def test_stop_label_marks_earliest_sufficient_prefix(config):
    index = KGIndex(list(MultiTQAdapter(config).iter_facts("full")))
    query = StructuredQuery(
        qid="1",
        topic_entities=["Iraq"],
        relation_hints=["visit"],
        temporal_constraints=[],
        answer_type="entity",
        answer_role="subject",
        parse_confidence=1,
        parse_source="test",
    )
    _, audit = TemporalBeamSearch(index, beam_size=3, max_hops=1).run(
        "Who visited Iraq?", query
    )
    earliest = next(chain for chain in audit if chain.candidate_answer == "Alice")
    extra_fact = next(fact for fact in index.facts.values() if fact.fact_id != earliest.facts[0].fact_id)
    later = earliest.model_copy(update={
        "chain_id": "later-positive",
        "parent_chain_id": earliest.chain_id,
        "hop": 2,
        "facts": earliest.facts + [extra_fact],
    })
    rows = [earliest.model_dump(mode="json"), later.model_dump(mode="json")]

    assert is_optimal_stop_prefix(earliest, rows, ["Alice"], "entity")
    assert not is_optimal_stop_prefix(later, rows, ["Alice"], "entity")
