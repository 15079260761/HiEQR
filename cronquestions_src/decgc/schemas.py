"""Canonical data, chain, interaction, and evidence schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .constants import CHAIN_STATES


class TemporalFact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fact_id: str
    subject_id: str
    subject_text: str
    relation_id: str
    relation_text: str
    object_id: str
    object_text: str
    timestamp_raw: str
    timestamp_normalized: str | None = None
    time_start: str | None = None
    time_end: str | None = None
    time_granularity: str
    source_relative_path: str
    source_row_index: int
    source_record_hash: str


class QAExample(BaseModel):
    qid: str
    question: str
    answers: list[str]
    split: Literal["train", "valid", "test"]
    question_type: str | None = None
    answer_type: str | None = None
    raw_record: dict[str, Any]
    source_relative_path: str
    source_row_index: int


class StructuredQuery(BaseModel):
    qid: str
    topic_entities: list[str] = Field(default_factory=list)
    target_entities: list[str] = Field(default_factory=list)
    relation_hints: list[str] = Field(default_factory=list)
    temporal_constraints: list[str] = Field(default_factory=list)
    temporal_operator: str | None = None
    temporal_granularity: str | None = None
    answer_type: str | None = None
    answer_role: str | None = None
    parse_confidence: float = Field(ge=0, le=1)
    parse_source: str
    prompt_version: str | None = None


class ChainSnapshot(BaseModel):
    hop: int
    absolute_reliability: float
    relative_weight: float
    stop_probability: float
    state: str
    evidence_gain: float = 0.0


class ChainPrefix(BaseModel):
    chain_id: str
    parent_chain_id: str | None = None
    qid: str
    hop: int
    facts: list[TemporalFact]
    entity_path: list[str]
    relation_path: list[str]
    time_path: list[str]
    current_entity: str
    candidate_answer: str | None = None
    retrieval_scores: list[float] = Field(default_factory=list)
    reranker_scores: list[float] = Field(default_factory=list)
    semantic_score: float = 0.0
    temporal_score: float = 0.0
    type_score: float = 0.0
    legality_score: float = 1.0
    absolute_reliability: float = 0.0
    relative_weight: float = 0.0
    stop_probability: float = 0.0
    answer_score: float = 1.0
    state: str = "ACTIVE"
    state_history: list[ChainSnapshot] = Field(default_factory=list)
    stop_reason: str | None = None
    graph_interactions: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("state")
    @classmethod
    def state_is_valid(cls, value: str) -> str:
        if value not in CHAIN_STATES:
            raise ValueError(f"Invalid chain state: {value}")
        return value


class GraphInteraction(BaseModel):
    other_chain_id: str
    interaction_type: Literal["cooperation", "competition"]
    interaction_reason: str
    interaction_confidence: float = Field(ge=0, le=1)
