"""Answer-type compatibility used by weak supervision and neural rescoring."""

from __future__ import annotations

from ..data.temporal_parser import parse_time


def answer_type_score(candidate_answer: str | None, answer_type: str | None) -> float:
    """Return an auditable compatibility score without consulting gold answers.

    MultiTQ and CronQuestions expose the coarse answer types ``entity`` and
    ``time``.  Entity candidates are KG endpoint identifiers; time candidates
    must be parseable temporal values.  Unknown dataset labels remain neutral
    instead of being silently treated as a perfect match.
    """
    candidate = str(candidate_answer or "").strip()
    if not candidate:
        return 0.0
    expected = str(answer_type or "").strip().casefold()
    if expected == "time":
        parsed = parse_time(candidate)
        return 1.0 if parsed.get("normalized") is not None else 0.0
    if expected == "entity":
        return 1.0
    return 0.5
