"""Deterministic query-grouped train/validation splitting."""

from __future__ import annotations

from ..utils.hashing import object_hash


def split_records_by_query(
    records: list[dict],
    validation_fraction: float,
) -> tuple[list[dict], list[dict]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(str(record["query"]), []).append(record)
    if len(groups) < 2:
        return list(records), list(records)

    threshold = round(validation_fraction * 10_000)
    training, validation = [], []
    for query, group in groups.items():
        bucket = int(object_hash(query)[:8], 16) % 10_000
        (validation if bucket < threshold else training).extend(group)

    if not validation:
        query = min(groups, key=lambda value: object_hash(value))
        validation = groups[query]
        training = [
            record
            for candidate, group in groups.items()
            if candidate != query
            for record in group
        ]
    if not training:
        query = max(groups, key=lambda value: object_hash(value))
        training = groups[query]
        validation = [
            record
            for candidate, group in groups.items()
            if candidate != query
            for record in group
        ]
    return training, validation
