"""Absolute reliability r and masked relative weight alpha."""

import torch


class ReliabilityHead(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(hidden_size + 2, hidden_size), torch.nn.GELU(), torch.nn.Linear(hidden_size, 1)
        )

    def forward(self, chain: torch.Tensor, temporal: torch.Tensor, type_score: torch.Tensor,
                mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.network(torch.cat([chain, temporal[:, None], type_score[:, None]], dim=-1)).squeeze(-1)
        reliability = torch.sigmoid(logits)
        masked_logits = logits.masked_fill(~mask, -torch.inf) if mask is not None else logits
        relative = torch.softmax(masked_logits, dim=-1)
        return logits, reliability, relative

