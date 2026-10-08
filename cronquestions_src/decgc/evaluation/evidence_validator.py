"""Strict source, connectivity, temporal, state, and answer validation."""

from __future__ import annotations

from ..data.temporal_parser import temporal_compatible
from ..schemas import ChainPrefix, StructuredQuery
from ..retrieval.type_consistency import answer_type_score
from .answer_normalizer import normalize_answer


def validate_chain(chain: ChainPrefix, predicted_answer: str, kg_index, query: StructuredQuery,
                   completed_only: bool = True) -> dict:
    existence = all(fact.fact_id in kg_index.facts for fact in chain.facts)
    exact = all(fact.fact_id in kg_index.facts and
                kg_index.facts[fact.fact_id].source_record_hash == fact.source_record_hash for fact in chain.facts)
    connected = all(bool({a.subject_id, a.object_id} & {b.subject_id, b.object_id})
                    for a, b in zip(chain.facts, chain.facts[1:]))
    temporal = all(temporal_compatible(fact.timestamp_raw, query.temporal_constraints, query.temporal_operator)
                   for fact in chain.facts)
    answer = normalize_answer(chain.candidate_answer or "") == normalize_answer(predicted_answer)
    answer_type = answer_type_score(chain.candidate_answer, query.answer_type) > 0
    state = chain.state == "COMPLETED" if completed_only else chain.state != "PRUNED"
    return {"all_facts_exist_in_kg": existence and exact, "path_connected": connected,
            "temporal_constraints_satisfied": temporal, "answer_supported_by_chain": answer,
            "answer_type_matched": answer_type,
            "eligible_supporting_chain": (
                existence and exact and connected and temporal and answer
                and answer_type and state
            )}
