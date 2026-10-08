"""Trainable local BGE cross-encoder reranker."""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BGEReranker(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

    def score(self, queries: list[str], texts: list[str], max_length: int = 256) -> torch.Tensor:
        device = next(self.parameters()).device
        tokens = self.tokenizer(queries, texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        output = self.model(**{key: value.to(device) for key, value in tokens.items()}).logits
        return output.squeeze(-1)

    def forward(self, queries: list[str], texts: list[str], labels: torch.Tensor) -> torch.Tensor:
        logits = self.score(queries, texts)
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.to(logits.device).float())

