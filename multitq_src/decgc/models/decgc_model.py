"""Integrated differentiable dynamic-chain model."""

import torch

from .answer_scorer import AnswerScorer
from .chain_encoder import ChainEncoder
from .reliability_head import ReliabilityHead
from .stop_head import StopHead


class DECGCModel(torch.nn.Module):
    def __init__(self, fact_size: int, hidden_size: int, retriever=None, reranker=None):
        super().__init__()
        self.retriever = retriever
        self.reranker = reranker
        if retriever is not None:
            embedding_size = int(retriever.model.config.hidden_size)
            self.live_projection = torch.nn.Linear(embedding_size * 3, fact_size)
            self.reranker_projection = torch.nn.Linear(1, fact_size)
            self.answer_projection = torch.nn.Linear(embedding_size, hidden_size)
        self.chain_encoder = ChainEncoder(fact_size, hidden_size)
        self.reliability_head = ReliabilityHead(hidden_size)
        self.stop_head = StopHead(hidden_size)
        self.answer_scorer = AnswerScorer(hidden_size)
        self.interaction_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 3, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, 1),
        )

    def encode_live(self, questions: list[str], fact_texts: list[str]) -> torch.Tensor:
        if self.retriever is None or self.reranker is None:
            raise RuntimeError("Live BGE components were not attached")
        encode_queries = getattr(
            self.retriever,
            "encode_queries",
            self.retriever.encode,
        )
        query = encode_queries(questions)
        fact = self.retriever.encode(fact_texts)
        pair = self.live_projection(torch.cat([query, fact, query * fact], dim=-1))
        reranker_score = self.reranker.score(questions, fact_texts)[:, None]
        return pair + self.reranker_projection(reranker_score)

    def prefix_step(self, previous, fact, hop, temporal, type_score, mask=None):
        chain = self.chain_encoder(previous, fact, hop)
        logits, reliability, relative = self.reliability_head(chain, temporal, type_score, mask)
        stop_logits, stop = self.stop_head(chain, reliability)
        return {"chain": chain, "reliability_logits": logits, "absolute_reliability": reliability,
                "relative_weight": relative, "stop_logits": stop_logits, "stop_probability": stop}

    def score_answers(self, chains: torch.Tensor, answer_texts: list[str]) -> torch.Tensor:
        """Score the actual candidate answer against its completed chain state."""
        if self.retriever is None:
            raise RuntimeError("Live BGE components were not attached")
        answers = self.answer_projection(self.retriever.encode(answer_texts))
        return self.answer_scorer(chains, answers)

    def score_interactions(
        self,
        chains: torch.Tensor,
        answer_texts: list[str],
        group_ids: list[str],
    ) -> torch.Tensor:
        """Learn cooperation between same-answer chains and competition otherwise."""
        if len(answer_texts) != chains.shape[0] or len(group_ids) != chains.shape[0]:
            raise ValueError("interaction metadata must match the chain batch")
        normalized = [
            " ".join(text.replace("_", " ").casefold().split())
            for text in answer_texts
        ]
        features = []
        for index, (answer, group_id) in enumerate(zip(normalized, group_ids)):
            cooperative = [
                candidate
                for candidate, (other_answer, other_group) in enumerate(
                    zip(normalized, group_ids)
                )
                if candidate != index
                and other_group == group_id
                and other_answer == answer
            ]
            competing = [
                candidate
                for candidate, (other_answer, other_group) in enumerate(
                    zip(normalized, group_ids)
                )
                if other_group == group_id and other_answer != answer
            ]
            cooperation_state = (
                chains[cooperative].mean(dim=0)
                if cooperative
                else torch.zeros_like(chains[index])
            )
            competition_state = (
                chains[competing].mean(dim=0)
                if competing
                else torch.zeros_like(chains[index])
            )
            features.append(torch.cat([
                chains[index],
                cooperation_state,
                competition_state,
            ]))
        return self.interaction_head(torch.stack(features)).squeeze(-1)
