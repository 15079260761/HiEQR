#!/usr/bin/env python3
"""Verify and report the deterministic stratified question sample."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from decgc.config import load_config
from decgc.data.multitq_adapter import MultiTQAdapter
from decgc.data.question_sample_cache import load_or_build_sample
from decgc.data.sampler import stratified_sample
from decgc.utils.hashing import object_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument(
        "--require-parse-manifest",
        action="store_true",
        help="Also require the split parse manifest to use the same QID selection.",
    )
    return parser.parse_args()


def stratum(example) -> tuple[str, str, str, str]:
    values = (
        example.question_type,
        example.answer_type,
        example.raw_record.get("qtype"),
        example.raw_record.get("time_level"),
    )
    return tuple("<null>" if value is None else str(value) for value in values)


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    config, _ = load_config(args.config)
    config.prepare_outputs()
    adapter = MultiTQAdapter(config)
    population = adapter.load_questions(args.split)
    expected = stratified_sample(population, args.limit, args.seed)
    cached, reused = load_or_build_sample(
        adapter,
        config.paths.cache_root,
        args.split,
        args.limit,
        args.seed,
    )
    expected_qids = [example.qid for example in expected]
    cached_qids = [example.qid for example in cached]
    if cached_qids != expected_qids:
        raise RuntimeError(
            "Cached sample does not exactly match the current deterministic "
            "stratified sampler; do not start parameter experiments."
        )
    if len(cached_qids) != args.limit:
        raise RuntimeError(
            f"Expected {args.limit} sampled questions, found {len(cached_qids)}"
        )
    if len(set(cached_qids)) != len(cached_qids):
        raise RuntimeError("The balanced sample contains duplicate question IDs")
    selection_hash = object_hash(cached_qids)
    parse_manifest_matches = None
    parse_manifest_path = (
        config.paths.cache_root
        / "question_parses"
        / f"{args.split}.manifest.json"
    )
    if parse_manifest_path.exists():
        parse_manifest = json.loads(parse_manifest_path.read_text(encoding="utf-8"))
        parse_manifest_matches = (
            parse_manifest.get("selection_hash") == selection_hash
        )
    if args.require_parse_manifest and parse_manifest_matches is not True:
        raise RuntimeError(
            "Question-parse manifest is missing or does not match the verified "
            "balanced sample; do not build chains or start evaluation."
        )

    population_counts = Counter(stratum(example) for example in population)
    sample_counts = Counter(stratum(example) for example in cached)
    rows = []
    for key in sorted(population_counts):
        population_count = population_counts[key]
        sample_count = sample_counts.get(key, 0)
        rows.append({
            "question_type": key[0],
            "answer_type": key[1],
            "qtype": key[2],
            "time_level": key[3],
            "population_count": population_count,
            "sample_count": sample_count,
            "population_share": population_count / len(population),
            "sample_share": sample_count / len(cached),
        })

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".strata.csv")
    json_path = args.output_prefix.with_suffix(".audit.json")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps({
            "verified": True,
            "method": "deterministic_round_robin_stratified_sample",
            "stratification_keys": [
                "question_type",
                "answer_type",
                "raw_record.qtype",
                "raw_record.time_level",
            ],
            "split": args.split,
            "seed": args.seed,
            "requested_questions": args.limit,
            "sampled_questions": len(cached),
            "unique_qids": len(set(cached_qids)),
            "population_questions": len(population),
            "population_strata": len(population_counts),
            "sampled_strata": len(sample_counts),
            "selection_hash": selection_hash,
            "parse_manifest": str(parse_manifest_path),
            "parse_manifest_matches": parse_manifest_matches,
            "cache_reused": reused,
            "strata_csv": str(csv_path),
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Balanced sample verified: {len(cached)} unique questions")
    print(f"Selection hash: {selection_hash}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
