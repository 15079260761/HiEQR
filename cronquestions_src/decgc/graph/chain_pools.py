"""Disjoint active/completed/pruned pools."""

from dataclasses import dataclass, field

from ..schemas import ChainPrefix


@dataclass
class ChainPools:
    active: list[ChainPrefix] = field(default_factory=list)
    completed: list[ChainPrefix] = field(default_factory=list)
    pruned: list[ChainPrefix] = field(default_factory=list)

    @classmethod
    def partition(cls, chains: list[ChainPrefix]) -> "ChainPools":
        return cls(
            active=[chain for chain in chains if chain.state == "ACTIVE"],
            completed=[chain for chain in chains if chain.state == "COMPLETED"],
            pruned=[chain for chain in chains if chain.state == "PRUNED"],
        )

