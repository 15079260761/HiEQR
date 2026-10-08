"""Deterministic time/relation/entity counterfactual construction."""

from __future__ import annotations

import copy

from ..data.temporal_parser import parse_time
from ..schemas import ChainPrefix


def build_counterfactuals(
    chain: ChainPrefix,
    alternative_relation: str = "__counterfactual_relation__",
    alternative_entity: str = "__counterfactual_entity__",
) -> list[dict]:
    results = []
    if not chain.facts:
        return results
    for kind in ("time", "relation", "entity", "tail"):
        payload = copy.deepcopy(chain.model_dump(mode="json"))
        final = payload["facts"][-1]
        if kind == "time":
            parsed = parse_time(final["timestamp_raw"])
            normalized = str(parsed["normalized"] or "1900-01-01")
            if normalized[:4].isdigit():
                normalized = f"{int(normalized[:4]) + 40:04d}{normalized[4:]}"
            final["timestamp_raw"] = normalized
            final["timestamp_normalized"] = normalized
        elif kind == "relation":
            final["relation_id"] = alternative_relation
            final["relation_text"] = alternative_relation.replace("_", " ")
        elif kind == "entity":
            # Perturb the non-answer/bridge endpoint while preserving the
            # candidate answer.  For multi-hop paths this breaks the final
            # connection; for one-hop paths it replaces the anchor endpoint.
            candidate = payload.get("candidate_answer")
            if final["subject_id"] == candidate:
                final["object_id"] = alternative_entity
                final["object_text"] = alternative_entity.replace("_", " ")
            else:
                final["subject_id"] = alternative_entity
                final["subject_text"] = alternative_entity.replace("_", " ")
        else:  # tail
            # Perturb the answer-bearing terminal endpoint and therefore the
            # candidate answer.  This is intentionally distinct from entity.
            candidate = payload.get("candidate_answer")
            if final["subject_id"] == candidate:
                final["subject_id"] = alternative_entity
                final["subject_text"] = alternative_entity.replace("_", " ")
            else:
                final["object_id"] = alternative_entity
                final["object_text"] = alternative_entity.replace("_", " ")
            payload["candidate_answer"] = alternative_entity
        payload["counterfactual_type"] = kind
        results.append(payload)
    return results
