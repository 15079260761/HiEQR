"""Graph structural validation."""

from .graph_types import EvidenceGraph
from ..constants import EDGE_TYPES, NODE_TYPES


def validate_graph(graph: EvidenceGraph) -> list[str]:
    errors = []
    node_ids = {node.node_id for node in graph.nodes}
    for node in graph.nodes:
        if node.node_type not in NODE_TYPES:
            errors.append(f"unknown node type {node.node_type}")
    for edge in graph.edges:
        if edge.edge_type not in EDGE_TYPES:
            errors.append(f"unknown edge type {edge.edge_type}")
        if edge.source not in node_ids or edge.target not in node_ids:
            errors.append(f"dangling edge {edge.source}->{edge.target}")
    return errors

