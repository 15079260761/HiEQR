"""Explicit metric acceptance gates over persisted evaluation results."""

from __future__ import annotations


def evaluate_metric_gate(
    metrics: dict[str, float],
    required: dict[str, float],
) -> dict:
    checks = {
        name: {
            "actual": float(metrics.get(name, 0.0)),
            "required": float(threshold),
            "passed": float(metrics.get(name, 0.0)) >= float(threshold),
        }
        for name, threshold in required.items()
    }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
        "source": "persisted_demo_metrics",
    }
