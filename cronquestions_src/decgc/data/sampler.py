"""Deterministic stratified sampling without changing official splits."""

from __future__ import annotations

import random
from collections import defaultdict

from ..schemas import QAExample


def stratified_sample(examples: list[QAExample], limit: int | None, seed: int) -> list[QAExample]:
    if limit is None or len(examples) <= limit:
        return examples
    groups: dict[
        tuple[str | None, str | None, str | None, str | None],
        list[QAExample],
    ] = defaultdict(list)
    for example in examples:
        if example.raw_record.get("_dataset") == "CronQuestions":
            key = (
                example.raw_record.get("cron_type"),
                example.answer_type,
                None,
                None,
            )
        else:
            key = (
                example.question_type,
                example.answer_type,
                example.raw_record.get("qtype"),
                example.raw_record.get("time_level"),
            )
        groups[key].append(example)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    # Preserve the official joint distribution using largest-remainder
    # allocation.  Round-robin balancing can severely over-sample rare
    # CronQuestions templates and no longer represents the source split.
    total = len(examples)
    exact = {key: limit * len(values) / total for key, values in groups.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = limit - sum(quotas.values())
    order = sorted(
        groups,
        key=lambda key: (-(exact[key] - quotas[key]), str(key)),
    )
    for key in order[:remaining]:
        quotas[key] += 1
    result = [
        example
        for key in sorted(groups, key=str)
        for example in groups[key][:quotas[key]]
    ]
    rng.shuffle(result)
    return result
