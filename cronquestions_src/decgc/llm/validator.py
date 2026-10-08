"""LLM output allow-list validation."""

from __future__ import annotations

from typing import Any


def validate_parse(payload: dict[str, Any], entities: set[str], relations: set[str], operators: set[str]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Structured parse must be a JSON object")

    def allowed_list(name: str, allowed: set[str]) -> list[str]:
        values = payload.get(name, [])
        if not isinstance(values, list):
            return []
        return [item for item in values if isinstance(item, str) and item in allowed]

    payload["topic_entities"] = allowed_list("topic_entities", entities)
    payload["target_entities"] = allowed_list("target_entities", entities)
    payload["relation_hints"] = allowed_list("relation_hints", relations)
    constraints = payload.get("temporal_constraints", [])
    payload["temporal_constraints"] = (
        [item for item in constraints if isinstance(item, str)] if isinstance(constraints, list) else []
    )
    if payload.get("temporal_operator") not in operators:
        payload["temporal_operator"] = None
    if payload.get("answer_type") not in {"entity", "time"}:
        payload["answer_type"] = None
    if payload.get("answer_role") not in {"subject", "object", "time"}:
        payload["answer_role"] = None
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    payload["confidence"] = min(1.0, max(0.0, confidence))
    return payload
