"""Answer-blind candidate-generation component ablations.

The lexical encoder is deliberately parameter-free and corpus-independent: it
does not fit IDF statistics, inspect labels, or receive answer strings.  It is
used only as a deterministic replacement for dense semantic recall.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from .component_ablation_protocol import (
    BASELINE,
    NO_CROSS_ENCODER_RERANKER,
    NO_DENSE_RETRIEVER,
    SUPPORTED_VARIANTS,
    candidate_chain_root,
    candidate_generation_variant,
    hashed_lexical_counts,
)


class AnswerBlindLexicalEncoder(torch.nn.Module):
    """Stable feature hashing over word unigrams/bigrams, with no fitted state."""

    version = "answer-blind-hashed-lexical-v1"

    def __init__(self, dimensions: int = 8192):
        super().__init__()
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = int(dimensions)
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

    def encode(self, texts: list[str], max_length: int = 0) -> torch.Tensor:
        del max_length
        vectors = torch.zeros(
            len(texts),
            self.dimensions,
            dtype=torch.float32,
            device=self._device_anchor.device,
        )
        for row, text in enumerate(texts):
            for bucket, value in hashed_lexical_counts(
                text,
                self.dimensions,
            ).items():
                vectors[row, bucket] += value
        return functional.normalize(vectors, dim=-1)

    def encode_queries(
        self,
        texts: list[str],
        max_length: int = 0,
    ) -> torch.Tensor:
        return self.encode(texts, max_length=max_length)
