from decgc.data.kg_index import KGIndex
from decgc.data.multitq_adapter import MultiTQAdapter
from decgc.retrieval.retriever_dataset import (
    build_retriever_records,
    build_retriever_records_for_example,
)
from decgc.schemas import StructuredQuery, TemporalFact


def _query(qid: str = "1") -> StructuredQuery:
    return StructuredQuery(
        qid=qid,
        topic_entities=["Iraq"],
        relation_hints=["visit"],
        temporal_constraints=["2007-02-27"],
        temporal_operator="equal",
        answer_type="entity",
        answer_role="subject",
        parse_confidence=1,
        parse_source="test",
    )


def test_retriever_pairs_are_bounded_and_neighborhood_hard(config):
    example = MultiTQAdapter(config).load_questions("train")[0]
    index = KGIndex(list(MultiTQAdapter(config).iter_facts("full")))

    records = build_retriever_records_for_example(
        example, index, _query(), max_pairs_per_question=1,
    )

    assert len(records) == 1
    assert "[SUBJECT] Alice" in records[0]["positive"]
    assert "Iraq" in records[0]["negative"]
    assert "[SUBJECT] Alice" not in records[0]["negative"]


def test_retriever_builder_does_not_compare_full_fact_models(config, monkeypatch):
    example = MultiTQAdapter(config).load_questions("train")[0]
    index = KGIndex(list(MultiTQAdapter(config).iter_facts("full")))
    progress = []

    def reject_model_equality(self, other):
        raise AssertionError("full TemporalFact equality must not be used")

    monkeypatch.setattr(TemporalFact, "__eq__", reject_model_equality)
    records = build_retriever_records(
        [example],
        index,
        {example.qid: _query(example.qid)},
        max_pairs_per_question=2,
        progress=lambda processed, total, count: progress.append((processed, total, count)),
    )

    assert records
    assert progress == [(1, 1, len(records))]


def test_duplicate_topic_entities_do_not_duplicate_pairs(config):
    example = MultiTQAdapter(config).load_questions("train")[0]
    index = KGIndex(list(MultiTQAdapter(config).iter_facts("full")))
    query = _query()
    query.topic_entities = ["Iraq", "Iraq"]

    records = build_retriever_records_for_example(
        example, index, query, max_pairs_per_question=4,
    )

    assert len(records) == 1


def test_high_degree_neighborhood_has_a_hard_scan_limit(config):
    example = MultiTQAdapter(config).load_questions("train")[0]
    index = KGIndex(list(MultiTQAdapter(config).iter_facts("full")))

    records = build_retriever_records_for_example(
        example,
        index,
        _query(),
        max_pairs_per_question=1,
        max_neighbor_facts_scanned=1,
    )

    # The first neighbor/global fact is the positive. With a hard scan limit of one,
    # construction stops instead of walking farther to find a negative.
    assert records == []


def test_temporally_wrong_answer_fact_is_never_labeled_positive(config):
    example = MultiTQAdapter(config).load_questions("train")[0]
    example.answers = ["Bob"]
    index = KGIndex(list(MultiTQAdapter(config).iter_facts("full")))

    records = build_retriever_records_for_example(
        example,
        index,
        _query(),
        max_pairs_per_question=2,
    )

    assert records == []


def test_coarse_time_answer_matches_day_granularity_kg_fact(config):
    example = MultiTQAdapter(config).load_questions("train")[0]
    example.answer_type = "time"
    index = KGIndex(list(MultiTQAdapter(config).iter_facts("full")))
    query = StructuredQuery(
        qid=example.qid,
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

    example.answers = ["2007"]
    year_records = build_retriever_records_for_example(
        example,
        index,
        query,
        max_pairs_per_question=1,
    )
    example.answers = ["2007-02"]
    month_records = build_retriever_records_for_example(
        example,
        index,
        query,
        max_pairs_per_question=1,
    )

    assert year_records
    assert month_records
    assert "2007-02-27" in year_records[0]["positive"]
