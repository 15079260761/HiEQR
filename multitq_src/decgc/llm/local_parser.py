"""Deterministic local parser for MultiTQ-style English questions."""

from __future__ import annotations

import re

from ..data.temporal_parser import parse_time
from ..schemas import StructuredQuery

_MONTHS = (
    "January|February|March|April|May|June|July|August|"
    "September|October|November|December|"
    "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
_DATE = re.compile(
    rf"\b(?:"
    rf"\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})\s+\d{{4}}|"
    rf"(?:{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}|"
    rf"(?:{_MONTHS})\s+\d{{4}}|"
    rf"(?:{_MONTHS}),?\s+\d{{4}}|"
    rf"\d{{1,2}}\.\d{{1,2}}\.\d{{4}}|"
    rf"\d{{4}}-\d{{1,2}}-\d{{1,2}}|\d{{4}}-\d{{1,2}}|\d{{4}}"
    rf")\b",
    re.IGNORECASE,
)
_TIME_RELATION_PATTERNS = (
    (r"\b(?:ask|asks|asked)\b|\b(?:make|made|making)\b.*\b(?:appeal|request)\b",
     ("make an appeal or request",)),
    (r"\b(?:appeal|appealed|request|requested)\b",
     ("make an appeal or request",)),
    (r"\b(?:offer|offers|offered|offering)\b.*\bcooperat",
     ("express intent to cooperate",)),
    (r"\b(?:denunciation|denunciations)\b", ("criticize or denounce",)),
    (r"\b(?:formally\s+)?(?:sign|signed|signing)\b.*\bagreements?\b",
     ("sign formal agreement",)),
    (r"\b(?:start|started|begin|began)\b.*\bnegotiat",
     ("engage in negotiation",)),
    (r"\b(?:want|wants|wanted|wish|wishes|wished|intend|intends|intended|"
     r"intention|intent|would like)\b.*\bdiplomatic cooperation\b",
     ("express intent to engage in diplomatic cooperation (such as policy support)",)),
    (r"\b(?:express|expressed|announce|announced|intend|intended|wish|wished)\b"
     r".*\b(?:intention|intent|engage|establish|participate)\b.*\bdiplomatic cooperation\b",
     ("express intent to engage in diplomatic cooperation (such as policy support)",)),
    (r"\b(?:seek|seeks|sought)\b.*\bdiplomatic cooperation\b",
     ("express intent to engage in diplomatic cooperation (such as policy support)",)),
    (r"\b(?:express|expressed)\b.*\bcooperation\b",
     ("express intent to cooperate",)),
    (r"\bunconventional (?:mass )?(?:violence|force)\b",
     ("use unconventional violence",)),
    (r"\bsmall arms and light weapons\b",
     ("fight with small arms and light weapons",)),
    (r"\b(?:repatriat\w*|return\w*|releas\w*)\b.*\b(?:people|persons|citizens|dissidents|protesters|staff|criminals?)\b",
     ("return, release person(s)",)),
)


# High-precision mappings for the ICEWS event families used to generate
# MultiTQ templates. Specific patterns must precede generic ones.
_RELATION_PATTERNS = (
    (r"\b(?:received|receive|hosted|host)\b.*\bvisit\b", ("host a visit",)),
    (r"\b(?:visited|visit|visiting)\b", ("make a visit",)),
    (r"\b(?:praised|praise|commended|commend|endorsed|endorse|supported|support)\b",
     ("praise or endorse",)),
    (r"\b(?:condemned|condemn|condemnation|criticism|criticised|criticized|criticise|criticize|"
     r"denounced|denounce|blamed|blame)\b",
     ("criticize or denounce",)),
    (r"\baccus(?:e|ed|ing)\b", ("accuse",)),
    (r"\bthreaten(?:ed|ing|s)?\b", ("threaten",)),
    (r"\binvestigat(?:e|ed|ing|es)\b", ("investigate",)),
    (r"\breject(?:ed|ing|s)?\b", ("reject",)),
    (r"\bunconventional (?:mass )?violence\b", ("use unconventional mass violence",)),
    (r"\bconventional military (?:force|forces)\b", ("use conventional military force",)),
    (r"\bsmall arms and light weapons\b", ("use small arms and light weapons",)),
    (r"\b(?:pessimistic(?:ally)?|negative)\s+(?:comment|comments|remark|remarks)\b"
     r"|\bpessimism\b",
     ("make pessimistic comment",)),
    (r"\b(?:optimistic(?:ally)?|optimism)\b", ("make optimistic comment",)),
    (r"\b(?:formally )?(?:sign|signed)\b.*\bagreement\b", ("sign formal agreement",)),
    (r"\b(?:telephone|telephoned|phone|conversation)\b", ("discuss by telephone",)),
    (r"\b(?:returned|return|released|release)\b", ("return release",)),
    (r"\b(?:provided|provide|offered|offer)\b.*\bhumanitarian\b.*\b(?:aid|assistance)\b",
     ("provide humanitarian aid",)),
    (r"\b(?:received|receive|receives)\b.*\bhumanitarian\b.*\b(?:aid|assistance)\b",
     ("provide humanitarian aid",)),
    (r"\b(?:provided|provide|offered|offer)\b.*\b(?:aid|assistance)\b", ("provide aid",)),
    (r"\b(?:appeal|appealed)\b", ("appeal",)),
    (r"\b(?:request|requested|ask for|asks for|asked for)\b",
     ("make an appeal or request",)),
    (r"\b(?:wanted|wished|intended|intention|intent|would like)\b.*"
     r"\b(?:engage|establish|participate)\b.*\bdiplomatic cooperation\b",
     ("express intent to engage in diplomatic cooperation",)),
    (r"\b(?:want|wants|wanted|wish|wishes|wished|intended|intention|intent|would like)\b.*\b(?:meet|negotiat)",
     ("express intent to meet or negotiate",)),
    (r"\b(?:negotiated|negotiating|negotiate|negotiation)\b", ("engage in negotiation",)),
    (r"\b(?:want|wants|wanted|wish|wishes|wished|intended|intention|intent|interest|would like)\b.*\bcooperat",
     ("express intent to cooperate",)),
    (r"\b(?:decline|declined|declines)\b", ("reject",)),
    (r"\b(?:recommend|recommended|recommends)\b", ("praise or endorse",)),
    (r"\b(?:say|said|speak|spoke)\b.*\b(?:negative|negatively)\b",
     ("make pessimistic comment",)),
    (r"\b(?:diplomatic cooperation|offered cooperation|establish cooperation)\b",
     ("engage in diplomatic cooperation",)),
)


class LocalQuestionParser:
    def __init__(self, entity_index=None, known_relations: set[str] | None = None):
        self.entity_index = entity_index
        self.known_relations = sorted(known_relations or ())

    @staticmethod
    def _stem(token: str) -> str:
        if token.endswith("ied") and len(token) > 4:
            return token[:-3] + "y"
        for suffix in ("ing", "ed", "es", "s"):
            if token.endswith(suffix) and len(token) > len(suffix) + 3:
                return token[: -len(suffix)]
        return token

    def _relations(self, lowered: str, precise: bool = False) -> list[str]:
        inferred = []
        normalized_known = {
            relation: " ".join(
                relation.replace("_", " ").casefold().split()
            )
            for relation in self.known_relations
        }
        patterns = (
            _TIME_RELATION_PATTERNS + _RELATION_PATTERNS
            if precise
            else _RELATION_PATTERNS
        )
        for pattern, markers in patterns:
            if not re.search(pattern, lowered):
                continue
            exact = [
                relation
                for relation, normalized in normalized_known.items()
                if any(
                    normalized == marker
                    for marker in markers
                )
            ]
            matched = exact or [
                relation
                for relation, normalized in normalized_known.items()
                if any(
                    marker in normalized or normalized in marker
                    for marker in markers
                )
            ]
            if matched:
                for relation in matched:
                    if relation not in inferred:
                        inferred.append(relation)
            elif not self.known_relations:
                fallback = markers[0].replace(" ", "_")
                if fallback not in inferred:
                    inferred.append(fallback)
            if inferred:
                break
        question_tokens = {
            self._stem(token)
            for token in re.findall(r"[a-z]+", lowered)
            if len(token) >= 4
        }
        if not inferred:
            for relation in self.known_relations:
                relation_tokens = {
                    self._stem(token)
                    for token in re.findall(
                        r"[a-z]+",
                        relation.replace("_", " ").casefold(),
                    )
                    if len(token) >= 4
                    and token not in {"make", "have", "with", "from"}
                }
                if relation_tokens & question_tokens:
                    inferred.append(relation)
        return inferred

    def parse(
        self,
        qid: str,
        question: str,
        answer_type: str | None = None,
        metadata: dict | None = None,
    ) -> StructuredQuery:
        metadata = metadata or {}
        lowered = question.casefold()
        entities, entity_offsets = self._annotated_entities(metadata)
        if not entities:
            matched = (
                self.entity_index.match_with_offsets(question)
                if self.entity_index is not None
                else []
            )
            if answer_type == "time" and self.entity_index is not None:
                qualified = self.entity_index.match_qualified_with_offsets(question)
                relation_spans = [
                    match.span()
                    for pattern, _ in _RELATION_PATTERNS
                    if (match := re.search(pattern, lowered)) is not None
                ]
                # A qualified actor span supersedes a plain country match only
                # when that country occurrence lies inside the actor phrase.
                # A second country mention elsewhere remains the event target.
                matched = [
                    item
                    for item in matched
                    if not any(
                        q_start <= item[1]
                        and item[2] <= q_end
                        and item[0] != qualified_entity
                        and not any(
                            q_start <= relation_start < q_end
                            or q_start < relation_end <= q_end
                            for relation_start, relation_end in relation_spans
                        )
                        for qualified_entity, q_start, q_end in qualified
                    )
                ]
                known = {entity for entity, _, _ in matched}
                matched.extend(
                    item for item in qualified if item[0] not in known
                )
                matched.sort(key=lambda item: (item[1], -(item[2] - item[1])))
            entities = [entity for entity, _, _ in matched]
            entity_offsets = {
                entity: start
                for entity, start, _ in matched
            }
        relations = self._relations(lowered, precise=answer_type == "time")
        dates = _DATE.findall(question)
        annotated_times = metadata.get("time", metadata.get("times", []))
        if not dates and isinstance(annotated_times, list):
            dates = [str(value) for value in annotated_times if str(value).strip()]
        annotated_granularity = str(metadata.get("time_level", "")).strip().casefold()
        if annotated_granularity not in {"day", "month", "year"}:
            annotated_granularity = (
                str(parse_time(dates[0])["granularity"]) if dates else ""
            )
        annotated_qtype = str(metadata.get("qtype", "")).strip().casefold().replace(" ", "_")
        known_qtypes = {
            "equal",
            "equal_multi",
            "before",
            "after",
            "before_last",
            "after_first",
            "first",
            "last",
        }
        if annotated_qtype in known_qtypes:
            operator = annotated_qtype
        elif "before" in lowered:
            operator = "before_last" if "last" in lowered else "before"
        elif "after" in lowered:
            operator = "after_first" if "first" in lowered else "after"
        elif "first" in lowered:
            operator = "first"
        elif "last" in lowered:
            operator = "last"
        elif "same" in lowered and any(unit in lowered for unit in ("day", "month", "year")):
            operator = "equal_multi"
        elif dates:
            operator = "equal"
        else:
            operator = None
        inferred_type = answer_type or ("time" if question.lower().startswith(("when", "in which year", "in which month")) else "entity")
        role = self._answer_role(
            question,
            inferred_type,
            entity_offsets,
        )
        topic_entities, target_entities = self._partition_entities(
            question,
            entities,
            operator if not dates else None,
            entity_offsets,
        )
        if inferred_type == "time" and re.search(
            r"\b(?:was|were)\b.*\bby\b",
            lowered,
        ):
            by_position = lowered.rfind(" by ")
            if by_position >= 0:
                topic_entities.sort(
                    key=lambda entity: (
                        entity_offsets.get(entity, len(question)) < by_position,
                        entity_offsets.get(entity, len(question)),
                    )
                )
        components = [bool(entities), bool(relations), bool(dates or operator), bool(inferred_type)]
        confidence = sum(components) / len(components)
        return StructuredQuery(
            qid=qid, topic_entities=topic_entities, target_entities=target_entities,
            relation_hints=relations,
            temporal_constraints=dates, temporal_operator=operator,
            temporal_granularity=annotated_granularity or None,
            answer_type=inferred_type,
            answer_role=role, parse_confidence=confidence, parse_source="local",
        )

    @staticmethod
    def _answer_role(
        question: str,
        answer_type: str,
        entity_offsets: dict[str, int],
    ) -> str:
        """Infer which endpoint is requested, including passive templates."""
        if answer_type == "time":
            return "time"
        lowered = question.casefold().replace("_", " ")
        # "X was rejected by who?" asks for the actor/subject.
        if re.search(r"\bby\s+(?:who|whom)\b", lowered):
            return "subject"
        if re.search(r"\bagainst\s+whom\b", lowered):
            return "object"
        if re.search(
            r"\b(?:who|which\s+(?:country|person|organisation|organization))\b"
            r".*\bsuffer(?:ed|ing|s)?\s+from\b",
            lowered,
        ):
            return "object"
        # "Who was threatened by X?" asks for the patient/object.
        if re.search(
            r"\b(?:who|which\s+(?:country|person|organisation|organization))\b"
            r".*\b(?:was|were|is|are|been)\b.*\bby\b",
            lowered,
        ):
            return "object"
        if entity_offsets:
            first_entity = min(entity_offsets.values())
            prefix = lowered[:first_entity].strip()
            # Covers "Who did X ...", "Who was X negotiating with ...", and
            # "Which country ... that X rejected?".
            if re.search(
                r"\b(?:did|does|do|that|by)$",
                prefix,
            ) or re.fullmatch(
                r"(?:to|with|for\s+)?(?:who|whom|which\s+\w+)"
                r"\s+(?:was|were|is|are|did|does|do|has|had)",
                prefix,
            ):
                return "object"
            relation_precedes_entity = any(
                re.search(pattern, prefix)
                for pattern, _ in _RELATION_PATTERNS
            )
            if (
                not relation_precedes_entity
                and re.match(
                    r"(?:who|which\s+\w+)\s+"
                    r"(?:was|were|is|are|did|does|do|has|had)\b",
                    prefix,
                )
            ):
                return "object"
        if re.search(r"\b(?:who|which\s+\w+)\s+did\b", lowered):
            return "object"
        if re.search(r"\bwho\b|\bwhich\s+country\b", lowered):
            return "subject"
        return "object"

    def _annotated_entities(
        self,
        metadata: dict,
    ) -> tuple[list[str], dict[str, int]]:
        positions = metadata.get("entity_positions", [])
        if not isinstance(positions, list):
            return [], {}
        known = (
            set(self.entity_index.entities)
            if self.entity_index is not None
            else None
        )
        located = []
        for item in positions:
            if not isinstance(item, dict):
                continue
            value = item.get("entity")
            if isinstance(value, list):
                value = value[0] if value else None
            if not isinstance(value, str) or (known is not None and value not in known):
                continue
            position = item.get("position", [])
            start = (
                int(position[0])
                if isinstance(position, list) and position
                else len(located)
            )
            located.append((start, value))
        located.sort()
        entities = list(dict.fromkeys(value for _, value in located))
        offsets = {value: start for start, value in located}
        return entities, offsets

    @staticmethod
    def _partition_entities(
        question: str,
        entities: list[str],
        operator: str | None,
        entity_offsets: dict[str, int] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Separate the event anchor from the entity whose answer is requested."""
        if len(entities) < 2 or operator not in {
            "before",
            "after",
            "before_last",
            "after_first",
            "equal_multi",
        }:
            return entities, []
        lowered = question.casefold().replace("_", " ")
        if (
            operator == "equal_multi"
            and re.match(r"\s*(?:in|on)\s+the\s+same\b", lowered)
        ):
            reference = min(
                entities,
                key=lambda entity: (entity_offsets or {}).get(entity, len(lowered)),
            )
            topics = [entity for entity in entities if entity != reference]
            return topics or entities, [reference] if topics else []
        marker = (
            lowered.find("before")
            if "before" in (operator or "")
            else lowered.find("after")
            if "after" in (operator or "")
            else max(
                lowered.rfind(" of "),
                lowered.rfind(" as "),
                lowered.rfind("same "),
            )
        )
        positioned = []
        for entity in entities:
            if entity_offsets and entity in entity_offsets:
                start = entity_offsets[entity]
            else:
                surface = entity.replace("_", " ").casefold()
                start = lowered.find(surface)
            if start >= 0:
                positioned.append((start, entity))
        if not positioned:
            return entities, []
        # Compound templates put the temporal reference after "before",
        # "after", or "as".  Absolute proximity selected the entity just
        # before the marker in questions such as "visit Iran before X" and
        # inverted the event topic and the reference anchor.
        after_marker = [
            (start, entity)
            for start, entity in positioned
            if start > marker
        ]
        reference = (
            min(after_marker, key=lambda item: item[0])[1]
            if after_marker
            else min(positioned, key=lambda item: abs(item[0] - marker))[1]
        )
        topics = [entity for entity in entities if entity != reference]
        return topics or entities, [reference] if topics else []
