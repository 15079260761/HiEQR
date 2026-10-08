"""Heterogeneous dynamic graph builder; pruned chains are audit-only."""

from __future__ import annotations

from itertools import combinations

from .graph_types import EvidenceGraph, GraphEdge, GraphNode
from .interaction_rules import competition, cooperation


class DynamicGraphBuilder:
    def build(self, qid: str, hop: int, chains: list, exclusive: bool = True) -> EvidenceGraph:
        chains = [chain for chain in chains if chain.state != "PRUNED"]
        nodes: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []
        for chain in chains:
            chain_node = f"chain:{chain.chain_id}"
            nodes[chain_node] = GraphNode(chain_node, "CHAIN", {"weight": chain.relative_weight})
            if chain.candidate_answer:
                answer_node = f"answer:{chain.candidate_answer}"
                nodes[answer_node] = GraphNode(answer_node, "ANSWER", {"text": chain.candidate_answer})
                edges.append(GraphEdge(chain_node, answer_node, "SUPPORTS_ANSWER",
                                       {"reliability": chain.absolute_reliability}))
            for fact in chain.facts:
                subject, obj = f"entity:{fact.subject_id}", f"entity:{fact.object_id}"
                nodes.setdefault(subject, GraphNode(subject, "ENTITY", {"text": fact.subject_text}))
                nodes.setdefault(obj, GraphNode(obj, "ENTITY", {"text": fact.object_text}))
                edges.append(GraphEdge(subject, obj, "FACT_EDGE", {
                    "fact_id": fact.fact_id, "relation": fact.relation_id, "timestamp": fact.timestamp_raw,
                    "hop": chain.hop, "temporal_compatibility": chain.temporal_score,
                    "source_relative_path": fact.source_relative_path, "source_row_index": fact.source_row_index,
                }))
                edges.append(GraphEdge(chain_node, subject, "CHAIN_MEMBERSHIP", {"fact_id": fact.fact_id}))
                edges.append(GraphEdge(chain_node, obj, "CHAIN_MEMBERSHIP", {"fact_id": fact.fact_id}))
        for left, right in combinations(chains, 2):
            is_cooperative, reason = cooperation(left, right)
            if is_cooperative:
                edges.append(GraphEdge(f"chain:{left.chain_id}", f"chain:{right.chain_id}", "COOPERATION", {"reason": reason}))
            is_competing, reason = competition(left, right, exclusive)
            if is_competing:
                edges.append(GraphEdge(f"chain:{left.chain_id}", f"chain:{right.chain_id}", "COMPETITION", {"reason": reason}))
        return EvidenceGraph(qid, hop, list(nodes.values()), edges)

