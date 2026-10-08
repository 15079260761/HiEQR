"""Graph-to-tensor batching metadata."""

import torch


def edge_index(graph, node_to_index: dict[str, int]) -> tuple[torch.Tensor, list[str]]:
    pairs, types = [], []
    for edge in graph.edges:
        pairs.append([node_to_index[edge.source], node_to_index[edge.target]])
        types.append(edge.edge_type)
    return torch.tensor(pairs, dtype=torch.long).T if pairs else torch.empty((2, 0), dtype=torch.long), types
