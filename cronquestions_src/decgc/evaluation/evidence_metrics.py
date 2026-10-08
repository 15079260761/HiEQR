"""Aggregate evidence quality."""


def evidence_metrics(validations: list[dict], predictions: list[dict]) -> dict[str, float]:
    denominator = max(1, len(validations))
    return {
        "evidence_fact_validity": sum(item["all_facts_exist_in_kg"] for item in validations) / denominator,
        "temporal_validity": sum(item["temporal_constraints_satisfied"] for item in validations) / denominator,
        "answer_chain_consistency": sum(item["answer_supported_by_chain"] for item in validations) / denominator,
        "completed_chain_precision": sum(item["eligible_supporting_chain"] for item in validations) / denominator,
        "fallback_rate": sum(bool(item.get("fallback")) for item in predictions) / max(1, len(predictions)),
    }

