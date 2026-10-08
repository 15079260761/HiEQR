"""High-precision symbolic completion for direct MultiTQ temporal questions.

The rules never inspect gold answers. They use structured questions, relation
direction, KG provenance, and temporal operators to recover candidates that
dense beam pruning can discard.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from ..data.temporal_parser import parse_time, temporal_compatible
from ..data.entity_index import EntityIndex, _DEMONYMS
from ..evaluation.answer_normalizer import normalize_answer
from ..schemas import StructuredQuery, TemporalFact


@dataclass(frozen=True)
class SymbolicCandidates:
    answers: list[str]
    facts: list[TemporalFact]
    rule: str
    prefer_symbolic: bool = False


@lru_cache(maxsize=65536)
def _entity_country(entity: str) -> str | None:
    match = re.search(r"\(([^()]*)\)$", entity.replace("_", " "))
    return normalize_answer(match.group(1)) if match else None


@lru_cache(maxsize=65536)
def _surface_key(value: str) -> str:
    return normalize_answer(value).replace("-", " ")


_TEMPORAL_ROLE_SURFACES = {
    "agriculture fishing forestry ministry": (
        "agriculture fisheries forestry ministry",
        "ministry of agriculture fisheries and forestry",
    ),
    "businessperson": ("businessman", "businesswoman", "business person"),
    "business": ("business community", "economy", "ministry of economic affairs"),
    "city mayor": ("mayor",),
    "court judge": ("judge", "chief justice"),
    "combatant": ("fighter", "fighters", "warlord"),
    "defense security ministry": (
        "defense ministry",
        "defence ministry",
        "department of defense",
        "department of defence",
        "ministry of national defence and security",
    ),
    "employee": ("staff", "worker", "workers", "employees", "officials", "servants"),
    "foreign affairs": (
        "foreign ministry",
        "foreign minister",
        "foreign policy",
        "external affairs",
        "ministry of external affairs",
    ),
    "government affiliated bank": ("state bank", "central bank"),
    "head of government": (
        "head of government",
        "government leader",
        "prime minister",
        "president",
    ),
    "immigrants": ("migrants", "immigrants"),
    "justice department ministery": ("ministry of justice", "justice ministry"),
    "insurgent": ("rebel", "rebels", "insurgency"),
    "lawmaker": ("lawmaker", "lawmakers", "legislator", "legislators", "legislature"),
    "lawyer attorney": ("lawyer", "attorney", "advocate", "attorney general"),
    "member of legislative govt": (
        "member of government",
        "member of the government",
        "government legislative council",
        "member of legislative council",
    ),
    "military": ("military",),
    "military personnel": ("military", "armed forces", "forces", "soldiers"),
    "military personnel special": ("special forces",),
    "mob": ("mafia",),
    "naval": ("navy", "naval forces"),
    "party member": ("party member", "member of the party", "party leader"),
    "peacekeeping troop": ("peacekeeping force", "peacekeepers"),
    "police": ("police",),
    "rebel group": ("rebel group", "rebels", "rebel movement"),
    "royal administration": ("royal government", "royal court", "royal administration"),
}

_TEMPORAL_QUALIFIER_SURFACES = {
    "burundi": ("burundian",),
    "cambodia": ("cambodian",),
    "denmark": ("danish",),
    "djibouti": ("djiboutian",),
    "hong kong": ("hong kong",),
    "kuwait": ("kuwaiti",),
    "libya": ("libyan",),
    "lithuania": ("lithuanian",),
    "macedonia": ("macedonian",),
    "nigeria": ("nigerian",),
    "serbia": ("serbian",),
    "south sudan": ("south sudanese",),
    "spain": ("spanish",),
    "sweden": ("swedish",),
    "switzerland": ("swiss",),
    "syria": ("syrian",),
    "zambia": ("zambian",),
}

_TEMPORAL_ENTITY_SURFACES = {
    "Council_for_National_Security": ("national security council",),
    "High_Commission_for_Refugees": ("unhcr", "high commission for refugees"),
    "Japan_Self-Defense_Forces": (
        "japan self defense forces",
        "japan self defence forces",
        "japan s self defense forces",
        "japan s self defence forces",
    ),
    "Nonaligned_Movement": ("non aligned movement",),
    "Special_Rapporteurs_of_the_United_Nations": (
        "un special rapporteurs",
        "united nations special rapporteurs",
    ),
    "Doctors_Without_Borders": ("medecins sans frontieres",),
    "U.S._1st_Marine_Expeditionary_Force": (
        "us 1st marine expeditionary force",
        "u s 1st marine expeditionary force",
    ),
    "United_Nations_High_Commissioner_for_Refugees": ("unhcr",),
}


def _fold_surface(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_like = "".join(
        character for character in decomposed
        if not unicodedata.combining(character)
    )
    return " ".join(re.findall(r"\w+", ascii_like.casefold()))


@lru_cache(maxsize=65536)
def _temporal_entity_aliases(entity: str) -> tuple[str, ...]:
    aliases = {_fold_surface(alias) for alias in EntityIndex._aliases(entity)}
    aliases.update(
        _fold_surface(alias)
        for alias in _TEMPORAL_ENTITY_SURFACES.get(entity, ())
    )
    parenthetical = re.fullmatch(
        r"\s*(.*?)\s*\(([^()]*)\)\s*",
        entity.replace("_", " "),
    )
    if parenthetical:
        base, qualifier = parenthetical.groups()
        base_key = _fold_surface(base)
        qualifier_key = _fold_surface(qualifier)
        qualifiers = {
            qualifier_key,
            *_DEMONYMS.get(qualifier_key, ()),
            *_TEMPORAL_QUALIFIER_SURFACES.get(qualifier_key, ()),
        }
        roles = {base_key, *_TEMPORAL_ROLE_SURFACES.get(base_key, ())}
        for qualifier_surface in qualifiers:
            for role in roles:
                aliases.update({
                    _fold_surface(f"{qualifier_surface} {role}"),
                    _fold_surface(f"{role} of {qualifier_surface}"),
                    _fold_surface(f"{role} from {qualifier_surface}"),
                })
        if base_key == "citizen":
            aliases.update(
                _fold_surface(EntityIndex._pluralize(demonym))
                for demonym in _DEMONYMS.get(qualifier_key, ())
            )
    return tuple(sorted(alias for alias in aliases if alias))


def _entity_surface_score(
    question: str,
    entity: str,
    *,
    folded_question: str | None = None,
) -> tuple[int, int]:
    folded = folded_question or f" {_fold_surface(question)} "
    matches = _entity_surface_matches(folded, entity)
    if not matches:
        return (0, 0)
    return matches[0][0]


def _entity_surface_matches(
    folded_question: str,
    entity: str,
) -> list[tuple[tuple[int, int], int, int]]:
    """Return exact alias scores and spans in a normalized question.

    Spans matter because two event endpoints must not be inferred from the
    same phrase (for example, ``Iraqi parliament`` cannot simultaneously
    denote both a parliament member and the legislature).
    """
    matches: list[tuple[tuple[int, int], int, int]] = []
    for alias in _temporal_entity_aliases(entity):
        needle = f" {alias} "
        start = 0
        while True:
            position = folded_question.find(needle, start)
            if position < 0:
                break
            alias_start = position + 1
            alias_end = alias_start + len(alias)
            matches.append((
                (len(alias.split()), len(alias)),
                alias_start,
                alias_end,
            ))
            start = position + 1
    matches.sort(reverse=True)
    return matches


def _best_disjoint_pair_score(
    subject_matches: list[tuple[tuple[int, int], int, int]],
    object_matches: list[tuple[tuple[int, int], int, int]],
) -> tuple[int, int, int, int]:
    best = (0, 0, 0, 0)
    for subject_score, subject_start, subject_end in subject_matches:
        for object_score, object_start, object_end in object_matches:
            if subject_start < object_end and object_start < subject_end:
                continue
            score = (
                min(subject_score[0], object_score[0]),
                subject_score[0] + object_score[0],
                min(subject_score[1], object_score[1]),
                subject_score[1] + object_score[1],
            )
            if score > best:
                best = score
    return best


def _recover_exact_time_endpoints(
    kg_index,
    question: str,
    topics: list[str],
    hints: set[str],
) -> list[str]:
    """Recover a missing endpoint from exact question/entity aliases.

    Recovery uses only question text, KG facts and relation hints.  A unique
    best exact surface match is required; ambiguous candidates fall back to
    the neural ranking.
    """
    if len(hints) != 1:
        return topics
    if topics:
        neighbor_map = {}
        for topic in topics:
            topic_neighbors = (
                kg_index.neighbors_for_relations(topic, hints)
                if hasattr(kg_index, "neighbors_for_relations")
                else kg_index.neighbors(topic)
            )
            for fact in topic_neighbors:
                neighbor_map[fact.fact_id] = fact
        neighbors = list(neighbor_map.values())
    elif hasattr(kg_index, "facts_for_relations"):
        neighbors = kg_index.facts_for_relations(hints)
    else:
        return topics
    candidates: dict[tuple[str, str], tuple[int, int, int, int, int]] = {}
    entity_matches: dict[
        str,
        list[tuple[tuple[int, int], int, int]],
    ] = {}
    folded_question = f" {_fold_surface(question)} "
    passive = bool(re.search(r"\b(?:was|were)\b.*\bby\b", question.casefold()))
    for fact in neighbors:
        if fact.relation_id not in hints:
            continue
        if fact.subject_id not in entity_matches:
            entity_matches[fact.subject_id] = _entity_surface_matches(
                folded_question, fact.subject_id
            )
        if fact.object_id not in entity_matches:
            entity_matches[fact.object_id] = _entity_surface_matches(
                folded_question, fact.object_id
            )
        pair = (fact.subject_id, fact.object_id)
        lexical_score = _best_disjoint_pair_score(
            entity_matches[fact.subject_id],
            entity_matches[fact.object_id],
        )
        if lexical_score == (0, 0, 0, 0):
            continue
        parsed_alignment = (
            len(topics) >= 2
            and _matches_entity(fact.subject_id, {_surface_key(topics[0])})
            and _matches_entity(fact.object_id, {_surface_key(topics[1])})
        )
        surface_alignment = any(
            (subject_start > object_start if passive else subject_start < object_start)
            for _, subject_start, subject_end in entity_matches[fact.subject_id]
            for _, object_start, object_end in entity_matches[fact.object_id]
            if not (
                subject_start < object_end and object_start < subject_end
            )
        )
        ordered_alignment = int(parsed_alignment or surface_alignment)
        score = lexical_score[:2] + (ordered_alignment,) + lexical_score[2:]
        if score > candidates.get(pair, (0, 0, 0, 0, 0)):
            candidates[pair] = score
    ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
    if not ranked or ranked[0][1] == (0, 0, 0, 0, 0):
        return topics
    if len(ranked) > 1 and ranked[1][1] == ranked[0][1]:
        return topics
    return list(ranked[0][0])


def _matches_entity(
    entity: str,
    references: set[str],
    *,
    allow_country_projection: bool = True,
) -> bool:
    key = _surface_key(entity)
    country = _entity_country(entity)
    return any(
        reference == key
        or (allow_country_projection and reference == country)
        for reference in references
    )


def _anchor_surface(question: str, operator: str | None) -> str | None:
    if operator not in {"before_last", "after_first", "before", "after"}:
        return None
    marker = "before" if "before" in operator else "after"
    position = question.casefold().find(marker)
    if position < 0:
        return None
    tail = question[position + len(marker):]
    value = re.split(r"[,?]", tail, maxsplit=1)[0].strip()
    value = re.sub(r"^(?:the|a|an)\s+", "", value, flags=re.I)
    return _surface_key(value) or None


def _relation_matches(fact: TemporalFact, hints: set[str]) -> bool:
    if not hints:
        return False
    fact_key = _surface_key(fact.relation_id)
    return any(fact_key == _surface_key(hint) for hint in hints)


def _direct_match(fact: TemporalFact, topics: list[str], role: str | None) -> bool:
    topic_set = set(topics)
    # Time answers require the exact event endpoints.  Projecting every
    # ``Role_(Country)`` actor to the bare country admits unrelated events and
    # corrupts first/last selection (for example, Ethiopian police vs. any
    # Ethiopian actor).  Entity-answer questions retain the older projection
    # because a country surface can itself describe a requested actor.
    strict = role == "time"
    subject_match = _matches_entity(
        fact.subject_id,
        topic_set,
        allow_country_projection=not strict,
    )
    object_match = _matches_entity(
        fact.object_id,
        topic_set,
        allow_country_projection=not strict,
    )
    if role == "subject":
        return object_match
    if role == "object":
        return subject_match
    if role == "time":
        if len(topics) >= 2:
            # Entity mentions preserve question order. For direct ICEWS events
            # this is the subject then object ("X visited/praised Y"). Requiring
            # direction prevents mirrored inverse facts from defining first/last.
            return (
                _matches_entity(
                    fact.subject_id,
                    {topics[0]},
                    allow_country_projection=False,
                )
                and _matches_entity(
                    fact.object_id,
                    {topics[1]},
                    allow_country_projection=False,
                )
            )
        return subject_match or object_match
    return subject_match or object_match


def _answer(fact: TemporalFact, query: StructuredQuery) -> str:
    if query.answer_role == "subject":
        return fact.subject_id
    if query.answer_role == "object":
        return fact.object_id
    parsed = parse_time(fact.timestamp_raw)
    value = str(parsed["normalized"] or fact.timestamp_raw)
    if query.temporal_granularity == "year":
        return value[:4]
    if query.temporal_granularity == "month":
        return value[:7]
    return value


def _time_key(fact: TemporalFact) -> str:
    parsed = parse_time(fact.timestamp_raw)
    return str(parsed["start"] or parsed["normalized"] or "")


def direct_temporal_candidates(kg_index, question: str, query: StructuredQuery) -> SymbolicCandidates:
    """Return exact direct-event candidates in temporal answer order."""
    hints = set(query.relation_hints)
    resolved_topic_entities = list(query.topic_entities)
    if query.answer_type == "time":
        resolved_topic_entities = _recover_exact_time_endpoints(
            kg_index,
            question,
            resolved_topic_entities,
            hints,
        )
    topics = [_surface_key(value) for value in resolved_topic_entities]
    if not topics or not hints:
        return SymbolicCandidates([], [], "insufficient_structure")

    pool: dict[str, TemporalFact] = {}
    for topic in resolved_topic_entities:
        neighbors = (
            kg_index.neighbors_for_relations(topic, hints)
            if hasattr(kg_index, "neighbors_for_relations")
            else kg_index.neighbors(topic)
        )
        for fact in neighbors:
            pool[fact.fact_id] = fact
    relation_facts = [
        fact for fact in pool.values()
        if _relation_matches(fact, hints)
    ]
    facts = [
        fact for fact in relation_facts
        if _direct_match(fact, topics, query.answer_role)
    ]
    if not facts:
        return SymbolicCandidates([], [], "no_direct_event")

    operator = query.temporal_operator
    ordering = operator
    lowered = question.casefold()
    if operator == "equal_multi":
        if "first" in lowered:
            ordering = "first"
        elif "last" in lowered:
            ordering = "last"

    if query.temporal_constraints:
        facts = [
            fact for fact in facts
            if temporal_compatible(
                fact.timestamp_raw,
                query.temporal_constraints,
                "equal" if operator in {"first", "last", "equal_multi"} else operator,
            )
        ]

    references = {_surface_key(value) for value in query.target_entities}
    anchor_resolved = False
    if operator in {"before_last", "after_first", "before", "after", "equal_multi"}:
        surface = _anchor_surface(question, operator)
        if surface:
            references.add(surface)
        # The reference event can use the inverse endpoint orientation of the
        # requested answer (e.g. "X praised Colombia after Nelson", where the
        # anchor is Colombia praising Nelson). Search all relation-compatible
        # topic-neighbor facts, not only answer-direction candidates.
        anchor_facts = [
            fact for fact in relation_facts
            if _matches_entity(fact.subject_id, references)
            or _matches_entity(fact.object_id, references)
        ]
        if anchor_facts:
            anchor_resolved = True
            anchor_times = [_time_key(fact) for fact in anchor_facts]
            if operator in {"before_last", "before"}:
                boundary = max(anchor_times)
                facts = [fact for fact in facts if _time_key(fact) < boundary]
            elif operator in {"after_first", "after"}:
                boundary = min(anchor_times)
                facts = [fact for fact in facts if _time_key(fact) > boundary]
            elif operator == "equal_multi":
                granularity = query.temporal_granularity or "day"
                width = {"year": 4, "month": 7, "day": 10}[granularity]
                anchors = {value[:width] for value in anchor_times}
                facts = [fact for fact in facts if _time_key(fact)[:width] in anchors]

    reverse = ordering in {"last", "before_last", "before"}
    facts.sort(key=_time_key, reverse=reverse)
    if ordering in {"first", "last", "before_last", "after_first"} and facts:
        extreme = _time_key(facts[0])
        facts = [fact for fact in facts if _time_key(fact) == extreme]

    answers = []
    selected_facts = []
    seen = set()
    for fact in facts:
        answer = _answer(fact, query)
        key = normalize_answer(answer)
        if not key or key in seen or key in set(topics) or key in references:
            continue
        seen.add(key)
        answers.append(answer)
        selected_facts.append(fact)
    exact_time_event = (
        query.answer_type == "time"
        and len(resolved_topic_entities) == 2
        and len(query.relation_hints) == 1
        and bool(answers)
    )
    prefer_symbolic = exact_time_event or (
        query.answer_type != "time" and (
            ordering in {"first", "last", "before_last", "after_first"}
            or (operator == "equal_multi" and anchor_resolved)
        )
    )
    return SymbolicCandidates(
        answers,
        selected_facts,
        f"direct_{ordering or 'event'}",
        prefer_symbolic,
    )


def merge_symbolic_answers(
    neural_ranking: list[str],
    symbolic: SymbolicCandidates,
    limit: int | None = None,
) -> list[str]:
    """Place logically exact candidates first and preserve neural fallbacks."""
    merged = []
    seen = set()
    if symbolic.prefer_symbolic:
        ordered = symbolic.answers + neural_ranking
    elif neural_ranking:
        # For non-ordinal/broad queries the symbolic set is useful for recall,
        # but its members are not uniquely ordered. Preserve the learned top-1
        # and insert exact candidates immediately below it.
        ordered = neural_ranking[:1] + symbolic.answers + neural_ranking[1:]
    else:
        ordered = symbolic.answers
    for answer in ordered:
        key = normalize_answer(answer)
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(answer)
        if limit is not None and len(merged) >= limit:
            break
    return merged
