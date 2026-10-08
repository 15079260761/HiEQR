import csv
import json

from decgc.evaluation.paper_metrics import (
    PAPER_COLUMNS,
    export_paper_metrics,
    paper_metric_row,
)


def _diagnostics():
    return {
        "overall": {"hits1": 0.852, "hits10": 0.924},
        "by_question_type": {
            "Simple": {"hits1": 0.707142, "hits10": 0.892857},
            "Complex": {"hits1": 0.908333, "hits10": 0.936111},
        },
        "by_answer_type": {
            "entity": {"hits1": 0.839286, "hits10": 0.931548},
            "time": {"hits1": 0.878049, "hits10": 0.908537},
        },
    }


def test_paper_metric_row_has_exact_requested_columns():
    row = paper_metric_row(_diagnostics(), model="DECGC", decimals=1)

    assert tuple(row) == PAPER_COLUMNS
    assert row == {
        "Model": "DECGC",
        "Hits@1 Overall (%)": 85.2,
        "Hits@1 Simple (%)": 70.7,
        "Hits@1 Complex (%)": 90.8,
        "Hits@1 Entity (%)": 83.9,
        "Hits@1 Time (%)": 87.8,
        "Hits@10 Overall (%)": 92.4,
        "Hits@10 Simple (%)": 89.3,
        "Hits@10 Complex (%)": 93.6,
        "Hits@10 Entity (%)": 93.2,
        "Hits@10 Time (%)": 90.9,
    }


def test_export_paper_metrics_writes_json_csv_and_markdown(tmp_path):
    source = tmp_path / "frozen-run.diagnostics.json"
    source.write_text(json.dumps(_diagnostics()), encoding="utf-8")

    outputs = export_paper_metrics(source, tmp_path / "run.paper_metrics")

    assert all(path.exists() for path in outputs.values())
    assert outputs == {
        "json": tmp_path / "run.paper_metrics.json",
        "csv": tmp_path / "run.paper_metrics.csv",
        "markdown": tmp_path / "run.paper_metrics.md",
    }
    assert not (tmp_path / "run.json").exists()
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert payload["units"] == "percent"
    with outputs["csv"].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["Hits@1 Overall (%)"] == "85.2"
    assert "| DECGC | 85.2 |" in outputs["markdown"].read_text(
        encoding="utf-8"
    )
