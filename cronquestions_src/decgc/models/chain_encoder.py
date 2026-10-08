"""GRU chain-prefix representation with hop embedding."""

import torch


class ChainEncoder(torch.nn.Module):
    def __init__(self, fact_size: int, hidden_size: int, max_hops: int = 8):
        super().__init__()
        self.hop_embedding = torch.nn.Embedding(max_hops + 1, 32)
        self.gru = torch.nn.GRUCell(fact_size + 32, hidden_size)

    def forward(self, previous: torch.Tensor, fact: torch.Tensor, hop: torch.Tensor) -> torch.Tensor:
        return self.gru(torch.cat([fact, self.hop_embedding(hop)], dim=-1), previous)

