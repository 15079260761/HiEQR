"""Completed-chain answer contribution and aggregation."""

import torch


class AnswerScorer(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = torch.nn.Sequential(torch.nn.Linear(hidden_size * 2, hidden_size),
                                         torch.nn.GELU(), torch.nn.Linear(hidden_size, 1))

    def forward(
        self,
        chains: torch.Tensor,
        answers: torch.Tensor,
        relative_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        contributions = self.score(torch.cat([chains, answers], dim=-1)).squeeze(-1)
        return contributions if relative_weights is None else relative_weights * contributions
