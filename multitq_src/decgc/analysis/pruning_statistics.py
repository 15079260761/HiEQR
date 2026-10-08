"""Pruning reason frequencies."""

from collections import Counter


def pruning_statistics(chains: list) -> dict:
    return dict(Counter(chain.stop_reason for chain in chains if chain.state == "PRUNED"))

