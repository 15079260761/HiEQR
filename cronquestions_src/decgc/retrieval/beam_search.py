"""Question-aware temporal beam search that never accepts gold answers."""

from __future__ import annotations

import math

from ..data.temporal_parser import parse_time
from ..schemas import ChainPrefix, ChainSnapshot, StructuredQuery
from ..utils.hashing import object_hash
from .fact_formatter import format_fact
from .temporal_filter import legal_extension
from .type_consistency import answer_type_score


class TemporalBeamSearch:
    def __init__(self, kg_index, retriever=None, reranker=None, fact_index=None,
                 beam_size: int = 8, max_hops: int = 3, retrieve_top_k: int = 48,
                 rerank_top_k: int = 16, rerank_batch_size: int = 16,
                 stop_threshold: float = 0.65, prune_threshold: float = 0.10,
                 max_paths_per_answer: int = 2):
        self.kg = kg_index
        self.retriever = retriever
        self.reranker = reranker
        self.fact_index = fact_index
        self.beam_size = beam_size
        self.max_hops = max_hops
        self.retrieve_top_k = retrieve_top_k
        self.rerank_top_k = rerank_top_k
        self.rerank_batch_size = rerank_batch_size
        self.stop_threshold = stop_threshold
        self.prune_threshold = prune_threshold
        self.max_paths_per_answer = max(1, max_paths_per_answer)
        self._query_vector = None

    @staticmethod
    def _relation_matches(fact, hints: list[str]) -> bool:
        if not hints:
            return True
        relation = f"{fact.relation_id} {fact.relation_text}".replace("_", " ").casefold()
        return any(
            hint.replace("_", " ").casefold() in relation
            or relation in hint.replace("_", " ").casefold()
            for hint in hints
        )

    @staticmethod
    def _answer_role_matches(fact, query: StructuredQuery) -> bool:
        topics = set(query.topic_entities)
        if query.answer_role == "time" and len(topics) >= 2:
            return topics.issubset({fact.subject_id, fact.object_id})
        if query.answer_role == "subject":
            return fact.object_id in topics
        if query.answer_role == "object":
            return fact.subject_id in topics
        return True

    def _resolve_anchor_constraints(
        self,
        query: StructuredQuery,
        question: str = "",
    ) -> StructuredQuery:
        """Resolve relative-time anchor entities to their KG event timestamps."""
        if query.temporal_constraints or not query.target_entities or not query.topic_entities:
            return query
        if query.temporal_operator not in {
            "before",
            "after",
            "before_last",
            "after_first",
            "equal_multi",
        }:
            return query
        topics = set(query.topic_entities)
        targets = set(query.target_entities)
        anchors = []
        # CronQuestions time-join templates may name a queried role plus both
        # endpoints of a separate reference event. Resolve that target-target
        # event before falling back to the original topic-target join.
        if len(targets) >= 2:
            for target in targets:
                for fact in self.kg.neighbors(target):
                    endpoints = {fact.subject_id, fact.object_id}
                    if len(endpoints.intersection(targets)) < 2:
                        continue
                    if not self._relation_matches(fact, query.relation_hints):
                        continue
                    parsed = parse_time(fact.timestamp_raw)
                    if parsed["normalized"] is not None:
                        anchors.append(str(parsed["normalized"]))
        for topic in topics:
            for fact in self.kg.neighbors(topic):
                endpoints = {fact.subject_id, fact.object_id}
                if not endpoints.intersection(targets):
                    continue
                if not self._relation_matches(fact, query.relation_hints):
                    continue
                parsed = parse_time(fact.timestamp_raw)
                if parsed["normalized"] is None:
                    continue
                anchor = str(parsed["normalized"])
                lowered = question.casefold()
                if query.temporal_operator == "equal_multi":
                    if query.temporal_granularity == "year" or "same year" in lowered:
                        anchor = anchor[:4]
                    elif query.temporal_granularity == "month" or "same month" in lowered:
                        anchor = anchor[:7]
                anchors.append(anchor)
        return query.model_copy(update={"temporal_constraints": list(dict.fromkeys(anchors))}) if anchors else query

    @staticmethod
    def _temporal_order(item: tuple) -> str:
        parsed = parse_time(item[0].timestamp_raw)
        return str(parsed["start"] or parsed["normalized"] or "9999-12-31")

    @staticmethod
    def _ordering_operator(question: str, query: StructuredQuery) -> str | None:
        """Recover the ordinal half of compound equal+first/last questions."""
        operator = query.temporal_operator
        if operator != "equal_multi":
            return operator
        lowered = question.casefold()
        if "first" in lowered:
            return "first"
        if "last" in lowered:
            return "last"
        return operator

    @staticmethod
    def _candidate_answer(fact, current: str, query: StructuredQuery) -> str:
        if query.answer_type != "time":
            return current
        parsed = parse_time(fact.timestamp_raw)
        normalized = str(parsed["normalized"] or fact.timestamp_raw)
        if query.temporal_granularity == "year":
            return normalized[:4]
        if query.temporal_granularity == "month":
            return normalized[:7]
        return normalized

    def _extension_answer(
        self,
        fact,
        chain: ChainPrefix | None,
        query: StructuredQuery,
    ) -> str:
        if chain is None:
            current = (
                fact.object_id
                if fact.subject_id in query.topic_entities
                else fact.subject_id
            )
        else:
            current = (
                fact.object_id
                if fact.subject_id == chain.current_entity
                else fact.subject_id
            )
        return self._candidate_answer(fact, current, query)

    def _diverse_extensions(
        self,
        ranked: list[tuple],
        chain: ChainPrefix | None,
        query: StructuredQuery,
        limit: int,
    ) -> list[tuple]:
        """Reserve beam slots for distinct answers before redundant paths."""
        selected = []
        deferred = []
        counts: dict[str, int] = {}
        for item in ranked:
            answer = self._extension_answer(item[0], chain, query)
            key = " ".join(answer.replace("_", " ").casefold().split())
            if counts.get(key, 0) == 0:
                selected.append(item)
                counts[key] = 1
            else:
                deferred.append((key, item))
            if len(selected) == limit:
                return selected
        for key, item in deferred:
            if counts.get(key, 0) >= self.max_paths_per_answer:
                continue
            selected.append(item)
            counts[key] = counts.get(key, 0) + 1
            if len(selected) == limit:
                break
        return selected

    def _diverse_chains(
        self,
        ranked: list[ChainPrefix],
        limit: int,
    ) -> list[ChainPrefix]:
        selected = []
        deferred = []
        counts: dict[str, int] = {}
        for chain in ranked:
            key = " ".join(
                (chain.candidate_answer or "")
                .replace("_", " ")
                .casefold()
                .split()
            )
            if counts.get(key, 0) == 0:
                selected.append(chain)
                counts[key] = 1
            else:
                deferred.append((key, chain))
            if len(selected) == limit:
                return selected
        for key, chain in deferred:
            if counts.get(key, 0) >= self.max_paths_per_answer:
                continue
            selected.append(chain)
            counts[key] = counts.get(key, 0) + 1
            if len(selected) == limit:
                break
        return selected

    def _extensions(self, question: str, query: StructuredQuery, chain: ChainPrefix | None) -> list[tuple]:
        if chain is None:
            facts = []
            for entity in query.topic_entities:
                facts.extend(self.kg.neighbors(entity))
            if not facts:
                if self.fact_index is not None and self._query_vector is not None:
                    facts = [
                        self.kg.facts[fact_id]
                        for fact_id, _ in self.fact_index.search(self._query_vector, self.retrieve_top_k)
                        if fact_id in self.kg.facts
                    ]
                else:
                    facts = list(self.kg.facts.values())[: self.beam_size * 8]
        else:
            facts = self.kg.neighbors(chain.current_entity)
        legal = [fact for fact in facts if legal_extension(chain, fact, query)]
        if not legal:
            return []
        if chain is None:
            role_matched = [
                fact for fact in legal if self._answer_role_matches(fact, query)
            ]
            if role_matched:
                legal = role_matched
        relation_matched = [
            fact for fact in legal if self._relation_matches(fact, query.relation_hints)
        ]
        if relation_matched:
            legal = relation_matched
        ordering_operator = self._ordering_operator(question, query)
        if ordering_operator in {"first", "after_first"}:
            legal.sort(
                key=lambda fact: str(
                    parse_time(fact.timestamp_raw)["start"] or "9999-12-31"
                )
            )
            legal = legal[: self.retrieve_top_k]
        elif ordering_operator in {"last", "before_last"}:
            legal.sort(
                key=lambda fact: str(
                    parse_time(fact.timestamp_raw)["start"] or ""
                ),
                reverse=True,
            )
            legal = legal[: self.retrieve_top_k]
        hints = " ".join(query.relation_hints).casefold()
        if self.retriever is not None and self._query_vector is not None:
            import torch
            if self.fact_index is not None:
                semantic = self.fact_index.score_ids(
                    self._query_vector, [fact.fact_id for fact in legal]
                )
            else:
                vectors = []
                with torch.inference_mode():
                    for start in range(0, len(legal), 128):
                        vectors.append(self.retriever.encode(
                            [format_fact(fact) for fact in legal[start:start + 128]]
                        ).cpu())
                fact_vectors = torch.cat(vectors)
                semantic = (fact_vectors @ torch.from_numpy(self._query_vector)).tolist()
        else:
            semantic = [
                1.0 if hints and any(word in fact.relation_text.casefold() for word in hints.split()) else 0.25
                for fact in legal
            ]
        dense_order = sorted(range(len(legal)), key=lambda index: semantic[index], reverse=True)
        dense_order = dense_order[: min(self.retrieve_top_k, len(dense_order))]
        recalled = [legal[index] for index in dense_order]
        recalled_semantic = [semantic[index] for index in dense_order]
        rerank_scores = list(recalled_semantic)
        if self.reranker is not None:
            import torch
            scores = []
            with torch.inference_mode():
                for start in range(0, len(recalled), self.rerank_batch_size):
                    batch = recalled[start:start + self.rerank_batch_size]
                    scores.extend(torch.sigmoid(self.reranker.score(
                        [question] * len(batch), [format_fact(fact) for fact in batch]
                    )).cpu().tolist())
            rerank_scores = scores
        scored = list(zip(recalled, recalled_semantic, rerank_scores))
        if ordering_operator in {"first", "after_first"}:
            # Ordinal semantics is a hard part of the question. Do not let a
            # dense/reranker cutoff discard the true temporal extreme.
            ranked = sorted(
                scored,
                key=lambda item: (self._temporal_order(item), -item[2]),
            )[: self.rerank_top_k]
        elif ordering_operator in {"last", "before_last"}:
            ranked = sorted(
                scored,
                key=lambda item: (self._temporal_order(item), item[2]),
                reverse=True,
            )[: self.rerank_top_k]
        else:
            ranked = sorted(
                scored,
                key=lambda item: item[2],
                reverse=True,
            )[: self.rerank_top_k]
        selected = self._diverse_extensions(
            ranked,
            chain,
            query,
            self.beam_size,
        )
        denominator = max(len(selected) - 1, 1)
        return [
            (
                fact,
                semantic,
                rerank,
                1.0 - temporal_rank / denominator
                if ordering_operator in {"first", "last", "before_last", "after_first"}
                else 1.0,
            )
            for temporal_rank, (fact, semantic, rerank) in enumerate(selected)
        ]

    def run(self, question: str, query: StructuredQuery) -> tuple[list[ChainPrefix], list[ChainPrefix]]:
        query = self._resolve_anchor_constraints(query, question)
        if self.retriever is not None:
            import torch
            with torch.inference_mode():
                encode_queries = getattr(
                    self.retriever,
                    "encode_queries",
                    self.retriever.encode,
                )
                self._query_vector = (
                    encode_queries([question])[0].detach().cpu().numpy()
                )
        else:
            self._query_vector = None
        active: list[ChainPrefix | None] = [None]
        completed: list[ChainPrefix] = []
        audit: list[ChainPrefix] = []
        for hop in range(1, self.max_hops + 1):
            expanded: list[ChainPrefix] = []
            for parent in active:
                extensions = self._extensions(question, query, parent)
                if parent is not None and not extensions:
                    parent.state = "COMPLETED"
                    parent.stop_reason = "no_legal_extension"
                    parent.stop_probability = 1.0
                    parent.state_history.append(ChainSnapshot(
                        hop=parent.hop, absolute_reliability=parent.absolute_reliability,
                        relative_weight=parent.relative_weight, stop_probability=1.0, state="COMPLETED",
                    ))
                    completed.append(parent)
                for fact, semantic, rerank, temporal_score in extensions:
                    if parent is None:
                        current = fact.object_id if fact.subject_id in query.topic_entities else fact.subject_id
                    else:
                        current = fact.object_id if fact.subject_id == parent.current_entity else fact.subject_id
                    facts = ([] if parent is None else parent.facts) + [fact]
                    reliability = 1 / (
                        1 + math.exp(-(0.45 * semantic + 0.35 * rerank + 0.20 * temporal_score))
                    )
                    stop = min(0.99, 0.15 + 0.25 * hop + (0.25 if query.answer_type == "time" else 0.0))
                    state = "COMPLETED" if stop >= self.stop_threshold or hop == self.max_hops else "ACTIVE"
                    prefix = ChainPrefix(
                        chain_id=object_hash([query.qid, [item.fact_id for item in facts]])[:20],
                        parent_chain_id=None if parent is None else parent.chain_id, qid=query.qid, hop=hop,
                        facts=facts,
                        entity_path=([fact.subject_id] if parent is None else parent.entity_path) + [current],
                        relation_path=([ ] if parent is None else parent.relation_path) + [fact.relation_id],
                        time_path=([ ] if parent is None else parent.time_path) + [fact.timestamp_raw],
                        current_entity=current,
                        candidate_answer=self._candidate_answer(fact, current, query),
                        retrieval_scores=([ ] if parent is None else parent.retrieval_scores) + [float(semantic)],
                        reranker_scores=([ ] if parent is None else parent.reranker_scores) + [float(rerank)],
                        semantic_score=float(semantic), temporal_score=float(temporal_score),
                        type_score=answer_type_score(
                            self._candidate_answer(fact, current, query),
                            query.answer_type,
                        ),
                        absolute_reliability=reliability, stop_probability=stop, state=state,
                        state_history=[] if parent is None else list(parent.state_history),
                        stop_reason="max_hops" if hop == self.max_hops else ("sufficient_evidence" if state == "COMPLETED" else None),
                    )
                    expanded.append(prefix)
            if not expanded:
                break
            logits = [math.log(max(item.absolute_reliability, 1e-8)) for item in expanded]
            denominator = sum(math.exp(value) for value in logits)
            for chain, value in zip(expanded, logits):
                chain.relative_weight = math.exp(value) / denominator
                if chain.absolute_reliability < self.prune_threshold:
                    chain.state, chain.stop_reason = "PRUNED", "low_reliability"
                chain.state_history.append(ChainSnapshot(
                    hop=hop, absolute_reliability=chain.absolute_reliability,
                    relative_weight=chain.relative_weight, stop_probability=chain.stop_probability, state=chain.state,
                ))
            ranked = sorted(expanded, key=lambda item: item.relative_weight, reverse=True)
            active = self._diverse_chains(
                [item for item in ranked if item.state == "ACTIVE"],
                self.beam_size,
            )
            completed.extend(item for item in ranked if item.state == "COMPLETED")
            audit.extend(ranked)
        return sorted(completed, key=lambda item: item.relative_weight, reverse=True), audit
