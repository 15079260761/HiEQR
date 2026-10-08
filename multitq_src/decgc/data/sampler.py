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
        groups[(
            example.question_type,
            example.answer_type,
            example.raw_record.get("qtype"),
            example.raw_record.get("time_level"),
        )].append(example)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    result: list[QAExample] = []
    keys = sorted(groups, key=str)
    while len(result) < limit and keys:
        remaining = []
        for key in keys:
            if groups[key] and len(result) < limit:
                result.append(groups[key].pop())
            if groups[key]:
                remaining.append(key)
        keys = remaining
    return result
