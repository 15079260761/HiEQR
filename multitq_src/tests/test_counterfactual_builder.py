from decgc.data.kg_index import KGIndex
from decgc.data.multitq_adapter import MultiTQAdapter
from decgc.retrieval.beam_search import TemporalBeamSearch
from decgc.retrieval.counterfactual_builder import build_counterfactuals
from decgc.retrieval.fact_formatter import format_fact
from decgc.schemas import ChainPrefix, StructuredQuery


def test_counterfactuals_change_real_fact_fields_without_marker(config):
    index = KGIndex(list(MultiTQAdapter(config).iter_facts("full")))
    query = StructuredQuery(
        qid="q",
        topic_entities=["Iraq"],
        relation_hints=["visit"],
        temporal_constraints=["2007-02-27"],
        temporal_operator="equal",
        answer_type="entity",
        answer_role="subject",
        parse_confidence=1,
        parse_source="test",
    )
    completed, _ = TemporalBeamSearch(index, beam_size=1, max_hops=1).run(
        "Who visited Iraq?",
        query,
    )

    payloads = build_counterfactuals(
        completed[0],
        alternative_relation="praise",
        alternative_entity="Japan",
    )
    chains = [ChainPrefix.model_validate(payload) for payload in payloads]
    rendered = [format_fact(chain.facts[-1]) for chain in chains]

    assert chains[0].facts[-1].timestamp_raw.startswith("2047")
    assert chains[1].facts[-1].relation_id == "praise"
    # Entity corruption changes the non-answer/bridge endpoint while keeping
    # the target answer fixed; tail corruption changes the answer endpoint.
    assert chains[2].candidate_answer == completed[0].candidate_answer
    assert chains[3].candidate_answer == "Japan"
    assert all("[COUNTERFACTUAL]" not in text for text in rendered)
