from decgc.data.kg_index import KGIndex
from decgc.llm.local_parser import LocalQuestionParser
from decgc.data.entity_index import EntityIndex
from decgc.retrieval.symbolic_temporal import (
    direct_temporal_candidates,
    merge_symbolic_answers,
)
from decgc.schemas import TemporalFact


def _fact(index, subject, relation, obj, timestamp):
    return TemporalFact(
        fact_id=f"f{index}",
        subject_id=subject,
        subject_text=subject,
        relation_id=relation,
        relation_text=relation,
        object_id=obj,
        object_text=obj,
        timestamp_raw=timestamp,
        timestamp_normalized=timestamp,
        time_start=timestamp,
        time_end=timestamp,
        time_granularity="day",
        source_relative_path="kg/full.txt",
        source_row_index=index,
        source_record_hash=str(index),
    )


def test_after_first_uses_reference_event_without_gold():
    facts = [
        _fact(0, "Military_Personnel_(Azerbaijan)", "Make_a_visit", "Iran", "2013-08-20"),
        _fact(1, "Oman", "Make_a_visit", "Iran", "2013-08-23"),
        _fact(2, "France", "Make_a_visit", "Iran", "2013-09-01"),
    ]
    index = KGIndex(facts)
    parser = LocalQuestionParser(
        EntityIndex(["Iran", "Oman", "France", "Military_Personnel_(Azerbaijan)"]),
        {"Make_a_visit"},
    )
    query = parser.parse(
        "q",
        "After Azerbaijan, which country was the first to visit Iran?",
        "entity",
        {"qtype": "after_first", "time_level": "day"},
    )
    result = direct_temporal_candidates(
        index,
        "After Azerbaijan, which country was the first to visit Iran?",
        query,
    )
    assert result.answers == ["Oman"]
    assert result.prefer_symbolic


def test_nonordinal_symbolic_candidates_preserve_neural_top1():
    from decgc.retrieval.symbolic_temporal import SymbolicCandidates

    symbolic = SymbolicCandidates(["new"], [], "direct_equal", False)
    assert merge_symbolic_answers(["learned", "other"], symbolic) == [
        "learned",
        "new",
        "other",
    ]


def test_exact_time_event_prefers_symbolic_timestamp():
    facts = [
        _fact(0, "Police_(Ethiopia)", "Use_conventional_military_force", "Ethiopia", "2005-06-08"),
        _fact(1, "Police_(Ethiopia)", "Use_conventional_military_force", "Ethiopia", "2005-11-14"),
    ]
    index = KGIndex(facts)
    parser = LocalQuestionParser(
        EntityIndex(["Police_(Ethiopia)", "Ethiopia"]),
        {"Use_conventional_military_force"},
    )
    query = parser.parse(
        "q",
        "In what year did the Ethiopian police first use conventional military force against Ethiopia?",
        "time",
        {"qtype": "first_last", "time_level": "year"},
    )
    result = direct_temporal_candidates(
        index,
        "In what year did the Ethiopian police first use conventional military force against Ethiopia?",
        query,
    )
    assert query.topic_entities == ["Police_(Ethiopia)", "Ethiopia"]
    assert result.answers == ["2005"]
    assert result.prefer_symbolic
    assert merge_symbolic_answers(["2013", "2005"], result)[:2] == [
        "2005",
        "2013",
    ]


def test_time_symbolic_gate_rejects_incomplete_event_endpoints():
    result = direct_temporal_candidates(
        KGIndex([_fact(0, "France", "Make_a_visit", "Iraq", "2005-01-01")]),
        "When did someone first visit Iraq?",
        LocalQuestionParser(EntityIndex(["Iraq"]), {"Make_a_visit"}).parse(
            "q",
            "When did someone first visit Iraq?",
            "time",
            {"qtype": "first_last", "time_level": "day"},
        ),
    )
    assert result.answers == ["2005-01-01"]
    assert not result.prefer_symbolic


def test_relation_filtered_kg_neighbors_are_semantically_identical():
    facts = [
        _fact(0, "A", "visit", "B", "2005-01-01"),
        _fact(1, "A", "praise", "C", "2005-01-02"),
    ]
    index = KGIndex(facts)
    assert index.neighbors_for_relations("A", {"visit"}) == [facts[0]]
    # Simulate a KGIndex unpickled from an evaluation-version-11 cache.
    del index.relation_adjacency
    assert index.neighbors_for_relations("A", {"praise"}) == [facts[1]]


def test_time_only_qualified_actor_linking_leaves_entity_parser_unchanged():
    parser = LocalQuestionParser(
        EntityIndex([
            "China",
            "Defense_/_Security_Ministry_(Taiwan)",
        ]),
        {"Make_an_appeal_or_request"},
    )
    question = (
        "In which year did Taiwan's Ministry of National Defence and Security "
        "last make a request to China?"
    )
    time_query = parser.parse(
        "time",
        question,
        "time",
        {"qtype": "first_last", "time_level": "year"},
    )
    entity_query = parser.parse(
        "entity",
        question,
        "entity",
        {"qtype": "first_last", "time_level": "year"},
    )
    assert time_query.topic_entities == [
        "Defense_/_Security_Ministry_(Taiwan)",
        "China",
    ]
    assert entity_query.topic_entities == ["China"]


def test_passive_time_question_restores_subject_object_order():
    parser = LocalQuestionParser(
        EntityIndex([
            "China",
            "Agriculture_/_Fishing_/_Forestry_Ministry_(Vietnam)",
        ]),
        {"Make_a_visit"},
    )
    query = parser.parse(
        "q",
        "When was the last visit to China by the Vietnamese Ministry of Agriculture, Fisheries and Forestry?",
        "time",
        {"qtype": "first_last", "time_level": "day"},
    )
    assert query.topic_entities == [
        "Agriculture_/_Fishing_/_Forestry_Ministry_(Vietnam)",
        "China",
    ]


def test_compound_parser_keeps_event_topic_before_reference():
    parser = LocalQuestionParser(
        EntityIndex(["Iran", "Shura_Council"]),
        {"Make_a_visit"},
    )
    query = parser.parse(
        "q",
        "Who was the last person to visit Iran before the Shura Council?",
        "entity",
        {"qtype": "before_last", "time_level": "day"},
    )
    assert query.topic_entities == ["Iran"]
    assert query.target_entities == ["Shura_Council"]


def test_time_endpoint_recovery_cannot_reuse_one_phrase_for_both_endpoints():
    facts = [
        _fact(
            0,
            "Member_of_Parliament_(Iraq)",
            "Criticize_or_denounce",
            "Iraq",
            "2011-06-16",
        ),
        _fact(
            1,
            "Member_of_Parliament_(Iraq)",
            "Criticize_or_denounce",
            "Legislature_(Iraq)",
            "2013-03-08",
        ),
    ]
    parser = LocalQuestionParser(
        EntityIndex([
            "Member_of_Parliament_(Iraq)",
            "Legislature_(Iraq)",
            "Iraq",
        ]),
        {"Criticize_or_denounce"},
    )
    question = "In which month did a member of the Iraqi parliament last denounce Iraq?"
    query = parser.parse(
        "q",
        question,
        "time",
        {"qtype": "first_last", "time_level": "month"},
    )

    result = direct_temporal_candidates(KGIndex(facts), question, query)

    assert result.answers == ["2011-06"]
    assert result.prefer_symbolic


def test_cron_last_time_uses_interval_end_without_gold_annotation():
    index = KGIndex([_fact(0, "person", "P54", "team", "2001/2006")])
    query = LocalQuestionParser().parse(
        "q",
        "What was the last year person played for team",
        "time",
        {
            "_dataset": "CronQuestions",
            "official_entities": ["person", "team"],
            "official_times": [],
            "official_relations": ["MUST_NOT_BE_READ"],
        },
    )
    result = direct_temporal_candidates(index, "unused", query)
    assert result.answers == ["2006"]
    assert result.rule == "cron_last_time"


def test_cron_time_join_recovers_shared_interval_independent_of_entity_order():
    facts = [
        _fact(0, "reference", "P54", "team", "2000/2002"),
        _fact(1, "answer", "P54", "team", "2001/2003"),
        _fact(2, "outside", "P54", "team", "2004/2005"),
    ]
    query = LocalQuestionParser().parse(
        "q",
        "Who was a player in the team with the reference player",
        "entity",
        {
            "_dataset": "CronQuestions",
            "official_entities": ["reference", "team"],
            "official_times": [],
        },
    )
    result = direct_temporal_candidates(KGIndex(facts), "unused", query)
    assert result.answers == ["answer"]
    assert result.rule == "cron_time_join"


def test_cron_event_boundary_falls_back_to_nearest_pre_event_interval():
    facts = [
        _fact(0, "Olympics", "P793", "event", "2004"),
        _fact(1, "nearest", "P39", "position", "1999/2001"),
        _fact(2, "older", "P39", "position", "1990/1995"),
    ]
    query = LocalQuestionParser().parse(
        "q",
        "Who held the position before the Olympics",
        "entity",
        {
            "_dataset": "CronQuestions",
            "official_entities": ["position", "Olympics"],
            "official_times": [],
        },
    )
    result = direct_temporal_candidates(KGIndex(facts), "unused", query)
    assert result.answers == ["nearest"]
    assert result.rule == "cron_before_boundary"
