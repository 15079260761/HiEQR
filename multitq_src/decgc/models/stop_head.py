"""Per-prefix stop probability."""

import torch


class StopHead(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.network = torch.nn.Sequential(torch.nn.Linear(hidden_size + 1, hidden_size // 2),
                                           torch.nn.GELU(), torch.nn.Linear(hidden_size // 2, 1))

    def forward(self, chain: torch.Tensor, reliability: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.network(torch.cat([chain, reliability[:, None]], dim=-1)).squeeze(-1)
        return logits, torch.sigmoid(logits)
