"""Query-aware fact encoder adapter."""

import torch


class QueryFactEncoder(torch.nn.Module):
    def __init__(self, bge_encoder, projection_size: int):
        super().__init__()
        self.bge = bge_encoder
        input_size = int(bge_encoder.model.config.hidden_size) * 3
        self.projection = torch.nn.Linear(input_size, projection_size)

    def forward(self, questions: list[str], facts: list[str]) -> torch.Tensor:
        encode_queries = getattr(
            self.bge,
            "encode_queries",
            self.bge.encode,
        )
        query = encode_queries(questions)
        fact = self.bge.encode(facts)
        return self.projection(torch.cat([query, fact, query * fact], dim=-1))
