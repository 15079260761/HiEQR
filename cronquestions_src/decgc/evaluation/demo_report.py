"""Deterministic Demo report generator."""

from pathlib import Path

from ..utils.io import atomic_text


def generate_report(path: Path, summary: dict, successes: list[dict], failures: list[dict]) -> None:
    lines = ["# DECGC MultiTQ Demo Report", "",
             "> This validation Demo is not a formal test-set result.", "",
             "## Scale and metrics", ""]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Successful examples", ""])
    lines.extend(f"- {row.get('qid')}: {row.get('predicted_answer')}" for row in successes[:5])
    lines.extend(["", "## Failure examples", ""])
    lines.extend(f"- {row.get('qid')}: {row.get('predicted_answer')}" for row in failures[:5])
    atomic_text(path, "\n".join(lines) + "\n")
