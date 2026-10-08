"""Confidence-gated parser with mandatory offline cache."""

from __future__ import annotations

import logging

from ..constants import TEMPORAL_OPERATORS
from ..schemas import StructuredQuery
from .cache import ParseCache
from .validator import validate_parse

LOGGER = logging.getLogger(__name__)
DEFAULT_PROMPT_VERSION = "2.7.0"


class SelectiveParser:
    def __init__(self, local_parser, cache: ParseCache, client=None, threshold: float = 0.8,
                 version: str = DEFAULT_PROMPT_VERSION):
        self.local_parser = local_parser
        self.cache = cache
        self.client = client
        self.threshold = threshold
        self.version = version

    def parse(self, qid: str, question: str, answer_type: str | None, entities: set[str], relations: set[str],
              allow_online: bool = False, metadata: dict | None = None) -> StructuredQuery:
        cached = self.cache.get(question, self.version)
        if cached:
            return cached.model_copy(update={"qid": qid})
        local = self.local_parser.parse(qid, question, answer_type, metadata)
        if local.parse_confidence >= self.threshold or not allow_online or self.client is None:
            self.cache.put(question, self.version, local)
            return local
        try:
            payload = validate_parse(
                self.client.parse(
                    question,
                    sorted(set(local.topic_entities + local.target_entities)),
                    sorted(relations),
                    sorted(TEMPORAL_OPERATORS),
                ),
                entities, relations, TEMPORAL_OPERATORS,
            )
        except Exception as error:
            LOGGER.warning(
                "DeepSeek parse failed for qid=%s; continuing with local fallback: %s: %s",
                qid, type(error).__name__, error,
            )
            fallback = local.model_copy(update={"parse_source": "local_fallback", "prompt_version": self.version})
            self.cache.put(question, self.version, fallback)
            return fallback
        parsed = StructuredQuery(
            qid=qid,
            topic_entities=(
                local.topic_entities
                if metadata and metadata.get("entity_positions")
                else payload["topic_entities"] or local.topic_entities
            ),
            target_entities=(
                local.target_entities
                if metadata and metadata.get("entity_positions")
                else payload["target_entities"] or local.target_entities
            ),
            relation_hints=payload["relation_hints"] or local.relation_hints,
            temporal_constraints=local.temporal_constraints or payload["temporal_constraints"],
            temporal_operator=local.temporal_operator or payload["temporal_operator"],
            temporal_granularity=local.temporal_granularity,
            answer_type=local.answer_type or payload["answer_type"],
            answer_role=local.answer_role or payload["answer_role"],
            parse_confidence=payload["confidence"],
            parse_source="deepseek", prompt_version=self.version,
        )
        self.cache.put(question, self.version, parsed)
        return parsed
