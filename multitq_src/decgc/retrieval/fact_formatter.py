"""The single canonical textual representation of a temporal fact."""

from ..schemas import TemporalFact

FACT_TEMPLATE = "[SUBJECT] {subject}\n[RELATION] {relation}\n[OBJECT] {object}\n[TIME] {time}"


def format_fact(fact: TemporalFact) -> str:
    return FACT_TEMPLATE.format(
        subject=fact.subject_text, relation=fact.relation_text,
        object=fact.object_text, time=fact.timestamp_raw,
    )

