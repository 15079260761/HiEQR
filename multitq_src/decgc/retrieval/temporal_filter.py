"""Hard temporal and path legality filters."""

from ..data.temporal_parser import temporal_compatible
from ..schemas import ChainPrefix, StructuredQuery, TemporalFact


def legal_extension(chain: ChainPrefix | None, fact: TemporalFact, query: StructuredQuery) -> bool:
    if chain is not None:
        if chain.state != "ACTIVE":
            return False
        if fact.fact_id in {item.fact_id for item in chain.facts}:
            return False
        if chain.current_entity not in {fact.subject_id, fact.object_id}:
            return False
    return temporal_compatible(fact.timestamp_raw, query.temporal_constraints, query.temporal_operator)

