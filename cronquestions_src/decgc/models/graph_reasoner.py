"""Stacked relational graph reasoning."""

import torch

from .relational_graph_layer import RelationalGraphLayer


class GraphReasoner(torch.nn.Module):
    def __init__(self, hidden_size: int, relation_count: int, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [RelationalGraphLayer(hidden_size, relation_count, dropout) for _ in range(layers)]
        )

    def forward(self, nodes: torch.Tensor, **graph_inputs) -> torch.Tensor:
        for layer in self.layers:
            nodes = layer(nodes, **graph_inputs)
        return nodes

