"""Paper-ready Hits@K table export from persisted evaluation diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PAPER_COLUMNS = (
    "Model",
    "Hits@1 Overall (%)",
    "Hits@1 Multiple (%)",
    "Hits@1 Single (%)",
    "Hits@1 Entity (%)",
    "Hits@1 Time (%)",
    "Hits@10 Overall (%)",
    "Hits@10 Multiple (%)",
    "Hits@10 Single (%)",
    "Hits@10 Entity (%)",
    "Hits@10 Time (%)",
)


def _metric(group: dict, metric: str, decimals: int) -> float:
    if metric not in group:
        raise KeyError(f"Diagnostic group is missing {metric!r}")
    return round(100.0 * float(group[metric]), decimals)


def paper_metric_row(
    diagnostics: dict,
    model: str = "DECGC",
    decimals: int = 1,
) -> dict[str, str | float]:
    """Return the exact Overall/Multiple/Single/Entity/Time paper columns."""
    question_types = diagnostics.get("by_question_type", {})
    answer_types = diagnostics.get("by_answer_type", {})
    groups = {
        "Overall": diagnostics.get("overall", {}),
        "Multiple": question_types.get("Multiple", {}),
        "Single": question_types.get("Single", {}),
        "Entity": answer_types.get("entity", {}),
        "Time": answer_types.get("time", {}),
    }
    missing = [name for name, group in groups.items() if not group]
    if missing:
        raise KeyError(
            "Diagnostics are missing required paper groups: "
            + ", ".join(missing)
        )

    row: dict[str, str | float] = {"Model": model}
    for metric, label in (("hits1", "Hits@1"), ("hits10", "Hits@10")):
        for group_name, group in groups.items():
            row[f"{label} {group_name} (%)"] = _metric(
                group,
                metric,
                decimals,
            )
    return row


def export_paper_metrics(
    diagnostics_path: Path,
    output_prefix: Path,
    model: str = "DECGC",
    decimals: int = 1,
) -> dict[str, Path]:
    """Write JSON, CSV, and Markdown views without running inference."""
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    row = paper_metric_row(diagnostics, model=model, decimals=decimals)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    json_path = Path(f"{output_prefix}.json")
    csv_path = Path(f"{output_prefix}.csv")
    markdown_path = Path(f"{output_prefix}.md")

    json_path.write_text(
        json.dumps(
            {
                "source_diagnostics": str(diagnostics_path),
                "units": "percent",
                "decimals": decimals,
                "columns": list(PAPER_COLUMNS),
                "row": row,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PAPER_COLUMNS)
        writer.writeheader()
        writer.writerow(row)

    header = "| " + " | ".join(PAPER_COLUMNS) + " |"
    separator = "| " + " | ".join("---" for _ in PAPER_COLUMNS) + " |"
    values = "| " + " | ".join(str(row[column]) for column in PAPER_COLUMNS) + " |"
    markdown_path.write_text(
        "\n".join((header, separator, values)) + "\n",
        encoding="utf-8",
    )
    return {
        "json": json_path,
        "csv": csv_path,
        "markdown": markdown_path,
    }
