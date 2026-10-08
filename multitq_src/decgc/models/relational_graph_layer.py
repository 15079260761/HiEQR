"""Pure-PyTorch relation-aware, weight-modulated message passing."""

import torch


class RelationalGraphLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, relation_count: int, dropout: float = 0.1):
        super().__init__()
        self.transforms = torch.nn.Parameter(torch.empty(relation_count, hidden_size, hidden_size))
        torch.nn.init.xavier_uniform_(self.transforms)
        self.edge_gate = torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size), torch.nn.Sigmoid())
        self.cooperation_gate = torch.nn.Sequential(torch.nn.Linear(hidden_size * 2, hidden_size), torch.nn.Sigmoid())
        self.competition_gate = torch.nn.Sequential(torch.nn.Linear(hidden_size * 2, hidden_size), torch.nn.Sigmoid())
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, nodes: torch.Tensor, edge_index: torch.Tensor, relation_ids: torch.Tensor,
                chain_weights: torch.Tensor, edge_features: torch.Tensor,
                cooperation: torch.Tensor | None = None, competition: torch.Tensor | None = None) -> torch.Tensor:
        source, target = edge_index
        transformed = torch.bmm(nodes[source].unsqueeze(1), self.transforms[relation_ids]).squeeze(1)
        messages = transformed * chain_weights[:, None] * self.edge_gate(edge_features)
        aggregated = torch.zeros_like(nodes).index_add(0, target, messages)
        if cooperation is not None:
            aggregated = aggregated + self.cooperation_gate(torch.cat([nodes, cooperation], -1)) * cooperation
        if competition is not None:
            suppression = self.competition_gate(torch.cat([nodes, competition], -1)) * competition
            aggregated = aggregated * (1.0 - torch.sigmoid(suppression))
        return self.norm(nodes + self.dropout(aggregated))

