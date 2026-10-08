from decgc.data.kg_index import KGIndex
from decgc.data.multitq_adapter import MultiTQAdapter
from decgc.evaluation.evidence_validator import validate_chain
from decgc.schemas import ChainPrefix, StructuredQuery


def test_evidence_fact_is_traceable(config):
    facts = list(MultiTQAdapter(config).iter_facts("full"))
    index = KGIndex(facts)
    chain = ChainPrefix(chain_id="c", qid="q", hop=1, facts=[facts[0]],
                        entity_path=["Alice", "Iraq"], relation_path=["visit"],
                        time_path=["2007-02-27"], current_entity="Iraq",
                        candidate_answer="Iraq", state="COMPLETED")
    query = StructuredQuery(qid="q", temporal_constraints=["2007-02-27"], temporal_operator="equal",
                            answer_type="entity", parse_confidence=1, parse_source="test")
    result = validate_chain(chain, "Iraq", index, query)
    assert result["eligible_supporting_chain"]


def test_fallback_is_not_completed_evidence(config):
    facts = list(MultiTQAdapter(config).iter_facts("full"))
    chain = ChainPrefix(chain_id="c", qid="q", hop=1, facts=[facts[0]], entity_path=["Alice", "Iraq"],
                        relation_path=["visit"], time_path=["2007-02-27"], current_entity="Iraq",
                        candidate_answer="Iraq", state="ACTIVE")
    query = StructuredQuery(qid="q", parse_confidence=1, parse_source="test")
    assert not validate_chain(chain, "Iraq", KGIndex(facts), query)["eligible_supporting_chain"]

