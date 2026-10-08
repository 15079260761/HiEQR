"""Serializable bidirectional KG adjacency index with provenance."""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

from ..schemas import TemporalFact
from ..utils.hashing import object_hash


class KGIndex:
    def __init__(self, facts: list[TemporalFact]):
        self.facts = {fact.fact_id: fact for fact in facts}
        self.adjacency: dict[str, list[str]] = defaultdict(list)
        self.relation_adjacency: dict[tuple[str, str], list[str]] | None = None
        self.facts_by_relation: dict[str, list[str]] | None = None
        for fact in facts:
            self.adjacency[fact.subject_id].append(fact.fact_id)
            self.adjacency[fact.object_id].append(fact.fact_id)
        self.data_hash = object_hash([fact.source_record_hash for fact in facts])

    def neighbors(self, entity: str) -> list[TemporalFact]:
        return [self.facts[fact_id] for fact_id in self.adjacency.get(entity, [])]

    def neighbors_for_relations(
        self,
        entity: str,
        relations: set[str],
    ) -> list[TemporalFact]:
        """Return relation-filtered neighbors with a backward-compatible index.

        Old serialized KG indexes do not contain ``relation_adjacency``.  Build
        it lazily on first temporal evaluation so existing cloud caches remain
        reusable and produce exactly the same fact set.
        """
        relation_adjacency = getattr(self, "relation_adjacency", None)
        if relation_adjacency is None:
            relation_adjacency = defaultdict(list)
            for fact in self.facts.values():
                relation_adjacency[(fact.subject_id, fact.relation_id)].append(
                    fact.fact_id
                )
                relation_adjacency[(fact.object_id, fact.relation_id)].append(
                    fact.fact_id
                )
            self.relation_adjacency = relation_adjacency
        seen = set()
        result = []
        for relation in relations:
            for fact_id in relation_adjacency.get((entity, relation), []):
                if fact_id in seen:
                    continue
                seen.add(fact_id)
                result.append(self.facts[fact_id])
        return result

    def facts_for_relations(self, relations: set[str]) -> list[TemporalFact]:
        facts_by_relation = getattr(self, "facts_by_relation", None)
        if facts_by_relation is None:
            facts_by_relation = defaultdict(list)
            for fact in self.facts.values():
                facts_by_relation[fact.relation_id].append(fact.fact_id)
            self.facts_by_relation = facts_by_relation
        return [
            self.facts[fact_id]
            for relation in relations
            for fact_id in facts_by_relation.get(relation, [])
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "KGIndex":
        with path.open("rb") as stream:
            value = pickle.load(stream)
        if not isinstance(value, cls):
            raise TypeError("Invalid KG index")
        return value
