"""Export paper-table Hits@1/Hits@10 columns from a frozen test result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT.parent
sys.path.insert(0, str(SOURCE_ROOT))

from decgc.evaluation.paper_metrics import export_paper_metrics


def _latest_diagnostics() -> Path:
    candidates = list(
        (PROJECT_ROOT / "outputs" / "reports").glob(
            "frozen-*.diagnostics.json"
        )
    )
    if not candidates:
        raise RuntimeError(
            "No frozen test diagnostics found under outputs/reports"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, default=None)
    parser.add_argument("--model", default="DECGC")
    parser.add_argument("--decimals", type=int, default=1)
    parser.add_argument("--output-prefix", type=Path, default=None)
    arguments = parser.parse_args()
    diagnostics = (arguments.diagnostics or _latest_diagnostics()).resolve()
    if arguments.decimals < 0:
        parser.error("--decimals must be non-negative")
    prefix = arguments.output_prefix
    if prefix is None:
        run_id = diagnostics.name.removesuffix(".diagnostics.json")
        prefix = (
            PROJECT_ROOT
            / "outputs"
            / "reports"
            / f"{run_id}.paper_metrics"
        )
    paths = export_paper_metrics(
        diagnostics,
        prefix.resolve(),
        model=arguments.model,
        decimals=arguments.decimals,
    )
    payload = {
        "source": str(diagnostics),
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
