"""Typed graph records."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphNode:
    node_id: str
    node_type: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source: str
    target: str
    edge_type: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceGraph:
    qid: str
    hop: int
    nodes: list[GraphNode]
    edges: list[GraphEdge]

    def to_dict(self) -> dict:
        return {"qid": self.qid, "hop": self.hop,
                "nodes": [vars(node) for node in self.nodes],
                "edges": [vars(edge) for edge in self.edges]}

