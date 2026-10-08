"""Trainable local BGE bi-encoder."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer

from ..models.losses import (
    multi_positive_candidate_loss,
    multi_positive_inbatch_loss,
)


class BGEEncoder(torch.nn.Module):
    DEFAULT_QUERY_INSTRUCTION = (
        "Represent this sentence for searching relevant passages: "
    )

    def __init__(
        self,
        model_path,
        normalize: bool = True,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        temperature: float = 0.01,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.normalize = normalize
        self.query_instruction = query_instruction
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

    def encode(self, texts: list[str], max_length: int = 192) -> torch.Tensor:
        device = next(self.parameters()).device
        tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        tokens = {key: value.to(device) for key, value in tokens.items()}
        embeddings = self.model(**tokens).last_hidden_state[:, 0]
        return functional.normalize(embeddings, dim=-1) if self.normalize else embeddings

    def encode_queries(
        self,
        texts: list[str],
        max_length: int = 192,
    ) -> torch.Tensor:
        return self.encode(
            [f"{self.query_instruction}{text}" for text in texts],
            max_length=max_length,
        )

    def forward(self, queries: list[str], positives: list[str], negatives: list[str] | None = None) -> torch.Tensor:
        query_vectors = self.encode_queries(queries)
        passage_vectors = self.encode(positives)
        if negatives:
            negative_vectors = self.encode(negatives)
            candidate_vectors = torch.cat(
                [passage_vectors, negative_vectors],
                dim=0,
            )
            return multi_positive_candidate_loss(
                (query_vectors @ candidate_vectors.T) / self.temperature,
                queries,
                positive_columns=len(positives),
            )
        return multi_positive_inbatch_loss(
            (query_vectors @ passage_vectors.T) / self.temperature,
            queries,
        )

    def train_last_n_layers(self, count: int) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        layers = getattr(getattr(self.model, "encoder", None), "layer", [])
        for layer in list(layers)[-count:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        for parameter in self.model.pooler.parameters() if getattr(self.model, "pooler", None) else []:
            parameter.requires_grad = True
