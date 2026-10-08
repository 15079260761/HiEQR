"""Structure-first candidate retrieval."""

from __future__ import annotations

import torch

from ..schemas import StructuredQuery
from .fact_formatter import format_fact


class Retriever:
    def __init__(self, encoder, kg_index, fact_index=None):
        self.encoder = encoder
        self.kg_index = kg_index
        self.fact_index = fact_index

    def retrieve(self, question: str, query: StructuredQuery, top_k: int) -> list[dict]:
        candidates = {}
        for entity in query.topic_entities:
            for fact in self.kg_index.neighbors(entity):
                candidates[fact.fact_id] = fact
        if not candidates:
            candidates = dict(self.kg_index.facts)
        texts = [format_fact(fact) for fact in candidates.values()]
        with torch.inference_mode():
            encode_queries = getattr(
                self.encoder,
                "encode_queries",
                self.encoder.encode,
            )
            query_vector = encode_queries([question])[0]
            fact_vectors = self.encoder.encode(texts)
            scores = fact_vectors @ query_vector
        ordered = scores.argsort(descending=True)[:top_k].tolist()
        facts = list(candidates.values())
        return [{"fact": facts[index], "dense_score": float(scores[index]), "entity_connection_score": 1.0,
                 "relation_score": 0.0, "temporal_score": 1.0} for index in ordered]
