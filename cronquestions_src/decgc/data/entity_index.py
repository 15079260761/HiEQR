"""Entity surface-form matching."""

from __future__ import annotations

import re


_DEMONYMS = {
    "austria": ("austrian",),
    "bahrain": ("bahraini",),
    "afghanistan": ("afghan",),
    "argentina": ("argentine", "argentinian"),
    "australia": ("australian",),
    "bangladesh": ("bangladeshi",),
    "belarus": ("belarusian",),
    "belgium": ("belgian",),
    "benin": ("beninese",),
    "bolivia": ("bolivian",),
    "brazil": ("brazilian",),
    "burkina faso": ("burkinabe", "burkinabé"),
    "canada": ("canadian",),
    "chile": ("chilean",),
    "china": ("chinese",),
    "colombia": ("colombian",),
    "croatia": ("croatian",),
    "cyprus": ("cypriot",),
    "czech republic": ("czech",),
    "ecuador": ("ecuadorian",),
    "egypt": ("egyptian",),
    "ethiopia": ("ethiopian",),
    "france": ("french",),
    "georgia": ("georgian",),
    "germany": ("german",),
    "ghana": ("ghanaian",),
    "greece": ("greek",),
    "hungary": ("hungarian",),
    "india": ("indian",),
    "indonesia": ("indonesian",),
    "iran": ("iranian",),
    "iraq": ("iraqi",),
    "israel": ("israeli",),
    "italy": ("italian",),
    "japan": ("japanese",),
    "kenya": ("kenyan",),
    "lebanon": ("lebanese",),
    "malaysia": ("malaysian",),
    "mali": ("malian",),
    "mexico": ("mexican",),
    "mozambique": ("mozambican",),
    "myanmar": ("burmese",),
    "netherlands": ("dutch",),
    "new zealand": ("new zealand",),
    "nigeria": ("nigerian",),
    "north korea": ("north korean",),
    "pakistan": ("pakistani",),
    "peru": ("peruvian",),
    "philippines": ("philippine", "filipino"),
    "poland": ("polish",),
    "portugal": ("portuguese",),
    "romania": ("romanian",),
    "russia": ("russian",),
    "saudi arabia": ("saudi", "saudi arabian"),
    "serbia": ("serbian",),
    "slovakia": ("slovak",),
    "slovenia": ("slovenian",),
    "somalia": ("somali",),
    "south africa": ("south african",),
    "south korea": ("south korean",),
    "sri lanka": ("sri lankan",),
    "spain": ("spanish",),
    "sudan": ("sudanese",),
    "taiwan": ("taiwanese",),
    "tanzania": ("tanzanian",),
    "thailand": ("thai",),
    "turkey": ("turkish",),
    "uganda": ("ugandan",),
    "ukraine": ("ukrainian",),
    "united kingdom": ("british",),
    "united states": ("american",),
    "occupied palestinian territories": ("palestinian",),
    "palestine": ("palestinian",),
    "ivory coast": ("ivorian",),
    "cote d ivoire": ("ivorian",),
    "venezuela": ("venezuelan",),
    "vietnam": ("vietnamese",),
    "yemen": ("yemeni",),
}


# MultiTQ questions paraphrase ICEWS actor types heavily.  These aliases are
# actor-type synonyms, not fuzzy matches: the country/organisation qualifier is
# still required, so they do not merge unrelated KG entities.
_ROLE_ALIASES = {
    "citizen": ("citizen", "citizens", "people"),
    "government": ("government", "administration"),
    "head of government": (
        "head of government",
        "government leader",
        "leader",
        "prime minister",
    ),
    "military": ("military", "armed forces"),
    "military personnel": ("military personnel", "soldiers"),
    "rebel": ("rebel", "rebels", "armed rebel", "armed rebels"),
    "armed group": ("armed group", "armed groups", "armed gang", "armed gangs"),
    "criminal": ("criminal", "criminals", "armed gang", "armed gangs"),
    "foreign affairs": (
        "foreign affairs",
        "foreign minister",
    ),
    "ministry": (
        "ministry",
        "government ministry",
        "foreign ministry",
        "ministry of foreign affairs",
    ),
    "media": ("media", "media personnel", "media staff"),
    "media personnel": ("media", "media personnel", "media staff"),
    "business": (
        "business",
        "businessman",
        "businessmen",
        "business people",
        "economy",
        "economic sector",
    ),
    # "citizens" is reserved for Citizen_(X); mapping it to Population_(X)
    # collapses two distinct ICEWS actor types. "people" is the MultiTQ
    # surface used for Population_(X).
    "population": ("population", "people"),
    "treasury finance ministry": (
        "treasury ministry",
        "finance ministry",
        "ministry of finance",
    ),
    "health ministry": ("health ministry", "ministry of health"),
    "information communication transparency ministry": (
        "information communication transparency ministry",
        "ministry of information communication transparency",
    ),
    "royal administration": (
        "royal administration",
        "king",
        "royal family",
    ),
    "legislature": ("legislature", "parliament", "lawmakers"),
    "member of parliament": (
        "member of parliament",
        "parliament member",
        "mp",
        "lawmaker",
        "lawmakers",
    ),
    "party member": ("party member", "party members"),
    "parties": ("party", "parties", "political party", "political parties"),
    "navy": ("navy", "naval forces"),
    "police": ("police", "police force"),
    "child": ("child", "children"),
    "person": ("person", "people"),
    "other authorities officials": (
        "other authorities and officials",
        "other authorities officials",
        "authorities and officials",
    ),
    "cabinet council of ministers advisors": (
        "cabinet",
        "council of ministers",
        "ministerial council of advisors",
        "cabinet council of ministers advisors",
    ),
}

_SPECIAL_ALIASES = {
    "Popular_Congress": ("people s congress", "peoples congress"),
    "Spy_Plane_(United_States)": ("us spy plane", "american spy plane"),
    "International_Government_Organizations": (
        "international governmental organisations",
        "international governmental organizations",
    ),
    "Austria_Press_Agency": ("austrian press agency",),
}


# Common geopolitical abbreviations occurring in MultiTQ question surfaces.
# These are exact aliases, not fuzzy country guesses.
_QUALIFIER_ALIASES = {
    "united states": ("us", "u s", "usa", "u s a", "american"),
    "united kingdom": ("uk", "u k", "british"),
    "united arab emirates": ("uae", "u a e", "emirati"),
    "south korea": ("rok", "south korean", "korean"),
    "north korea": ("dprk", "north korean"),
    "russia": ("russian",),
}

_TEMPORAL_QUALIFIED_ROLE_ALIASES = {
    "agriculture fishing forestry ministry": (
        "agriculture fisheries forestry ministry",
        "agriculture fisheries and forestry ministry",
    ),
    "city mayor": ("city mayor", "mayor"),
    "foreign affairs": (
        "foreign affairs",
        "foreign minister",
        "foreign ministry",
        "external affairs",
        "external affairs ministry",
    ),
    "government affiliated bank": (
        "government affiliated bank",
        "state bank",
        "central bank",
    ),
    "naval": ("naval", "navy"),
}

_ROLE_WORDS_TO_IGNORE = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "or",
    "the",
}

_ROLE_CONNECTORS = _ROLE_WORDS_TO_IGNORE | {
    "national",
    "s",
}


class EntityIndex:
    def __init__(self, entity_ids: list[str]):
        self.entities = sorted(set(entity_ids), key=len, reverse=True)
        self._trie: dict = {}
        self._qualified_by_alias: dict[str, list[tuple[str, tuple[frozenset[str], ...]]]] = {}
        self._maximum_qualifier_tokens = 1
        for entity in self.entities:
            for alias in self._aliases(entity):
                node = self._trie
                for token in alias.split():
                    node = node.setdefault(token, {})
                # Preserve the previous deterministic collision policy: the
                # first (most specific/longest entity id) owns an alias.
                node.setdefault(None, entity)
            self._register_qualified_entity(entity)

    def _register_qualified_entity(self, entity: str) -> None:
        """Index role-and-country entities for compositional surface matching.

        ICEWS stores actors such as ``Defense_/_Security_Ministry_(Taiwan)``
        while MultiTQ verbalises them as, for example, "Taiwan's Ministry of
        National Defence and Security".  Enumerating every word order is both
        brittle and incomplete.  We instead index the country qualifier and
        match the role words locally around it.
        """
        match = re.fullmatch(r"\s*(.*?)\s*\(([^()]*)\)\s*", entity.replace("_", " "))
        if match is None:
            return
        base, qualifier = match.groups()
        base_key = self._alias_key(base)
        qualifier_key = self._alias_key(qualifier)
        role_surfaces = set(_ROLE_ALIASES.get(base_key, (base_key,)))
        role_surfaces.update(_TEMPORAL_QUALIFIED_ROLE_ALIASES.get(base_key, ()))
        role_surfaces.add(base_key)
        for surface in list(role_surfaces):
            role_surfaces.update(self._spelling_variants(surface))
            if "ministry" in surface:
                role_surfaces.add(surface.replace("ministry", "department"))
            if "ministery" in surface:
                role_surfaces.add(surface.replace("ministery", "department"))
                role_surfaces.add(surface.replace("ministery", "ministry"))
            if "department" in surface:
                role_surfaces.add(surface.replace("department", "ministry"))
        role_token_sets = tuple(sorted({
            frozenset(
                token
                for token in self._normalized_tokens(surface)
                if token not in _ROLE_WORDS_TO_IGNORE
            )
            for surface in role_surfaces
        } - {frozenset()}, key=lambda value: (-len(value), sorted(value))))
        if not role_token_sets:
            return

        qualifier_aliases = {qualifier_key}
        qualifier_aliases.update(_DEMONYMS.get(qualifier_key, ()))
        qualifier_aliases.update(_QUALIFIER_ALIASES.get(qualifier_key, ()))
        for alias in qualifier_aliases:
            alias_key = self._alias_key(alias)
            if not alias_key:
                continue
            self._maximum_qualifier_tokens = max(
                self._maximum_qualifier_tokens,
                len(alias_key.split()),
            )
            self._qualified_by_alias.setdefault(alias_key, []).append(
                (entity, role_token_sets)
            )

    @staticmethod
    def _normalized_tokens(value: str) -> list[str]:
        return re.findall(
            r"\w+",
            value.replace("_", " ").casefold(),
            flags=re.UNICODE,
        )

    @classmethod
    def _alias_key(cls, value: str) -> str:
        return " ".join(cls._normalized_tokens(value))

    @staticmethod
    def _pluralize(value: str) -> str:
        words = value.split()
        if not words:
            return value
        word = words[-1]
        irregular = {
            "child": "children",
            "man": "men",
            "person": "people",
            "woman": "women",
        }
        if word in irregular:
            words[-1] = irregular[word]
            return " ".join(words)
        if word.endswith(("s", "people", "personnel")):
            return value
        if word.endswith(("ch", "sh", "x", "z")):
            words[-1] = word + "es"
        elif word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
            words[-1] = word[:-1] + "ies"
        else:
            words[-1] = word + "s"
        return " ".join(words)

    @staticmethod
    def _spelling_variants(value: str) -> set[str]:
        variants = {value}
        replacements = (
            ("organizations", "organisations"),
            ("organization", "organisation"),
            ("defense", "defence"),
            ("governmental", "government"),
        )
        for left, right in replacements:
            for current in list(variants):
                if left in current:
                    variants.add(current.replace(left, right))
                if right in current:
                    variants.add(current.replace(right, left))
        return variants

    @classmethod
    def _aliases(cls, entity: str) -> set[str]:
        surface = entity.replace("_", " ")
        aliases = set(cls._spelling_variants(cls._alias_key(surface)))
        aliases.add(cls._pluralize(cls._alias_key(surface)))
        aliases.update(_SPECIAL_ALIASES.get(entity, ()))
        surface_tokens = cls._alias_key(surface).split()
        for country, demonyms in _DEMONYMS.items():
            country_tokens = country.split()
            if surface_tokens[:len(country_tokens)] == country_tokens:
                remainder = " ".join(surface_tokens[len(country_tokens):])
                aliases.update(
                    cls._alias_key(f"{demonym} {remainder}")
                    for demonym in demonyms
                )
        parenthetical = re.fullmatch(r"\s*(.*?)\s*\(([^()]*)\)\s*", surface)
        if parenthetical:
            base, qualifier = parenthetical.groups()
            base_key = cls._alias_key(base)
            qualifier_key = cls._alias_key(qualifier)
            role_surfaces = set(_ROLE_ALIASES.get(base_key, (base_key,)))
            role_surfaces.add(cls._pluralize(base_key))
            demonyms = _DEMONYMS.get(qualifier_key, ())
            for role in role_surfaces:
                aliases.update({
                    cls._alias_key(f"{role} {qualifier_key}"),
                    cls._alias_key(f"{role} of {qualifier_key}"),
                    cls._alias_key(f"{role} of the {qualifier_key}"),
                    cls._alias_key(f"{role} in {qualifier_key}"),
                    cls._alias_key(f"{role} from {qualifier_key}"),
                    cls._alias_key(f"{qualifier_key} {role}"),
                    cls._alias_key(f"{qualifier_key} s {role}"),
                })
                aliases.update(
                    cls._alias_key(f"{demonym} {role}")
                    for demonym in demonyms
                )
            if base_key == "member of parliament":
                aliases.update(
                    cls._alias_key(f"member of the {demonym} parliament")
                    for demonym in demonyms
                )
                aliases.update(
                    cls._alias_key(f"member of {demonym} parliament")
                    for demonym in demonyms
                )
        aliases = {
            variant
            for alias in aliases
            for variant in cls._spelling_variants(alias)
        }
        return {alias for alias in aliases if alias}

    def match_with_offsets(
        self,
        question: str,
        limit: int = 16,
    ) -> list[tuple[str, int, int]]:
        if not self._trie:
            return []
        tokens = list(re.finditer(r"\w+", question, flags=re.UNICODE))
        matches = []
        seen = set()
        index = 0
        while index < len(tokens):
            node = self._trie
            best = None
            cursor = index
            while cursor < len(tokens):
                token = tokens[cursor].group(0).casefold()
                child = node.get(token)
                if child is None:
                    break
                node = child
                entity = node.get(None)
                if entity is not None:
                    best = (entity, cursor)
                cursor += 1
            if best is None:
                index += 1
                continue
            entity, end_index = best
            if entity not in seen:
                matches.append((
                    entity,
                    tokens[index].start(),
                    tokens[end_index].end(),
                ))
                seen.add(entity)
                if len(matches) == limit:
                    break
            index = end_index + 1
        return matches

    def match(self, question: str, limit: int = 16) -> list[str]:
        return [
            entity
            for entity, _, _ in self.match_with_offsets(question, limit)
        ]

    def match_qualified_with_offsets(
        self,
        question: str,
        limit: int = 8,
        window: int = 7,
    ) -> list[tuple[str, int, int]]:
        """Match compositional role-qualified actors in a local token window.

        This method is intentionally separate from :meth:`match_with_offsets`.
        The temporal QA parser opts into it only for time-answer questions, so
        existing entity-answer rankings are byte-for-byte unaffected.
        """
        tokens = list(re.finditer(r"\w+", question, flags=re.UNICODE))
        normalized = [token.group(0).casefold() for token in tokens]
        candidates: dict[str, tuple[float, int, int, int, int]] = {}
        for start in range(len(tokens)):
            for width in range(1, self._maximum_qualifier_tokens + 1):
                end = start + width
                if end > len(tokens):
                    break
                alias = " ".join(normalized[start:end])
                descriptors = self._qualified_by_alias.get(alias)
                if not descriptors:
                    continue
                left = max(0, start - window)
                right = min(len(tokens), end + window)
                local_positions: dict[str, list[int]] = {}
                for position in range(left, right):
                    local_positions.setdefault(normalized[position], []).append(position)
                local_words = set(local_positions)
                for entity, role_sets in descriptors:
                    best_score = 0.0
                    best_positions: list[int] = []
                    for role_words in role_sets:
                        overlap = role_words & local_words
                        if not overlap:
                            continue
                        # Multiword roles need at least two identifying words;
                        # a single-word role (police, ministry, military, ...)
                        # remains an exact lexical match.
                        if len(role_words) > 1 and len(overlap) < 2:
                            continue
                        coverage = len(overlap) / len(role_words)
                        score = len(overlap) + coverage + 0.01 * len(role_words)
                        if score <= best_score:
                            continue
                        qualifier_middle = (start + end - 1) / 2
                        positions = [
                            min(
                                local_positions[word],
                                key=lambda position: abs(position - qualifier_middle),
                            )
                            for word in overlap
                        ]
                        if overlap == {"military"} and any(
                            position > 0
                            and normalized[position - 1] == "conventional"
                            for position in positions
                        ):
                            continue
                        if len(positions) > 1:
                            first_role = min(positions)
                            last_role = max(positions)
                            qualifier_positions = set(range(start, end))
                            intervening = {
                                normalized[position]
                                for position in range(first_role, last_role + 1)
                                if position not in positions
                                and position not in qualifier_positions
                            }
                            if intervening - _ROLE_CONNECTORS:
                                continue
                        best_score = score
                        best_positions = positions
                    if not best_positions:
                        continue
                    span_start = min([start, *best_positions])
                    span_end = max([end - 1, *best_positions])
                    value = (
                        best_score,
                        tokens[span_start].start(),
                        tokens[span_end].end(),
                        span_start,
                        span_end,
                    )
                    existing = candidates.get(entity)
                    if existing is None or value[0] > existing[0]:
                        candidates[entity] = value

        # A role word can be visible from two nearby country mentions.  Assign
        # it to the most compact role--qualifier span ("Somali criminal"), not
        # across the intervening event predicate ("criminal threatened China").
        compact = sorted(
            candidates.items(),
            key=lambda item: (
                -item[1][0],
                item[1][4] - item[1][3],
                -(item[1][2] - item[1][1]),
                item[0],
            ),
        )
        selected = []
        occupied: set[int] = set()
        for item in compact:
            _, _, _, token_start, token_end = item[1]
            span = set(range(token_start, token_end + 1))
            if span & occupied:
                continue
            selected.append(item)
            occupied.update(span)
            if len(selected) == limit:
                break
        ordered = sorted(selected, key=lambda item: item[1][1])
        return [
            (entity, start, end)
            for entity, (_, start, end, _, _) in ordered
        ]
