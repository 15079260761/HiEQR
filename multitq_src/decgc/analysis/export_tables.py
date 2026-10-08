"""Flat CSV metric export."""

from pathlib import Path

import pandas as pd


def export_metrics(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
