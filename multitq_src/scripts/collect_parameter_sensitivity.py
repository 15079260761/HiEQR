#!/usr/bin/env python3
"""Collect the matched one-factor loss-weight evaluations into plot-ready files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from decgc.config import load_config


DEFAULT = {"prefix": 0.50, "stop": 0.30, "counterfactual": 0.30}
LEVELS = {
    "prefix": (0.00, 0.25, 0.50, 0.75),
    "stop": (0.00, 0.15, 0.30, 0.45),
    "counterfactual": (0.00, 0.15, 0.30, 0.45),
}
METRICS = ("hits1", "hits10", "mrr", "candidate_answer_recall", "fallback_rate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--questions",
        default="500",
        help="Exact evaluation size, or 'full' to select the largest completed run.",
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument(
        "--exclude-default",
        action="store_true",
        help="Write only the nine non-default points (for staged experiments).",
    )
    return parser.parse_args()


def config_for(config_path: Path, weights: dict[str, float]):
    overrides = [f"loss.{name}={value}" for name, value in weights.items()]
    config, _ = load_config(config_path, overrides)
    return config


def completed_evaluation(config, questions: str) -> tuple[str, int, dict]:
    candidates: list[tuple[int, Path, dict]] = []
    for manifest_path in config.paths.predictions_root.glob(
        "frozen-*/evaluation_manifest.json"
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        signature = manifest.get("signature", {})
        count = int(manifest.get("questions", 0))
        if not manifest.get("completed") or signature.get("config_hash") != config.hash:
            continue
        if questions != "full" and count != int(questions):
            continue
        candidates.append((count, manifest_path, manifest))
    if not candidates:
        raise FileNotFoundError(
            f"No completed evaluation for config={config.hash[:12]}, questions={questions}"
        )
    _, manifest_path, manifest = max(
        candidates,
        key=lambda item: (item[0], item[1].stat().st_mtime_ns),
    )
    return manifest_path.parent.name, int(manifest["questions"]), manifest["metrics"]


def main() -> None:
    args = parse_args()
    rows = []
    for factor, levels in LEVELS.items():
        for level in levels:
            weights = dict(DEFAULT)
            weights[factor] = level
            if args.exclude_default and weights == DEFAULT:
                continue
            config = config_for(args.config, weights)
            run_id, question_count, metrics = completed_evaluation(
                config,
                args.questions,
            )
            row = {
                "factor": factor,
                "level": level,
                "prefix": weights["prefix"],
                "stop": weights["stop"],
                "counterfactual": weights["counterfactual"],
                "is_default": weights == DEFAULT,
                "run_id": run_id,
                "questions": question_count,
            }
            row.update({name: metrics.get(name) for name in METRICS})
            rows.append(row)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    fieldnames = list(rows[0])
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
