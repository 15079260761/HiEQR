from decgc.data.entity_index import EntityIndex
from decgc.llm.local_parser import LocalQuestionParser


def test_local_parser_preserves_month_granularity_and_first_last_direction():
    parser = LocalQuestionParser(EntityIndex(["China"]))

    month = parser.parse(
        "q1",
        "Which country visited China in December 2015?",
        "entity",
    )
    first = parser.parse("q2", "Who first visited China?", "entity")
    last = parser.parse("q3", "Who last visited China?", "entity")

    assert month.temporal_constraints == ["December 2015"]
    assert first.temporal_operator == "first"
    assert last.temporal_operator == "last"


def test_local_parser_separates_reference_entity_for_relative_time():
    parser = LocalQuestionParser(EntityIndex(["China", "Japan"]))

    prefix = parser.parse(
        "q1",
        "Before Japan, who visited China?",
        "entity",
    )
    suffix = parser.parse(
        "q2",
        "Who visited China last before Japan did?",
        "entity",
    )

    for parsed in (prefix, suffix):
        assert parsed.topic_entities == ["China"]
        assert parsed.target_entities == ["Japan"]
        assert parsed.answer_role == "subject"
    assert prefix.temporal_operator == "before"
    assert suffix.temporal_operator == "before_last"


def test_local_parser_recognizes_object_answer_role():
    parser = LocalQuestionParser(EntityIndex(["China"]))

    parsed = parser.parse("q1", "Which country did China visit?", "entity")

    assert parsed.answer_role == "object"


def test_local_parser_matches_inflected_kg_relation_names():
    parser = LocalQuestionParser(
        EntityIndex(["China"]),
        {"threaten", "engage_in_diplomatic_cooperation"},
    )

    parsed = parser.parse("q1", "Who threatened China first?", "entity")

    assert parsed.relation_hints == ["threaten"]
    assert parsed.parse_confidence == 1.0


def test_local_parser_preserves_official_multitq_ordinal_date_example():
    parser = LocalQuestionParser(EntityIndex(["Ethiopia"]))

    parsed = parser.parse(
        "q1",
        "Who expressed intent to engage in diplomatic cooperation with "
        "Ethiopia before Jun 25th, 2006?",
        "entity",
    )

    assert parsed.temporal_constraints == ["Jun 25th, 2006"]
    assert parsed.temporal_operator == "before"


def test_local_parser_uses_non_answer_dataset_annotations():
    parser = LocalQuestionParser(EntityIndex(["China"]))

    parsed = parser.parse(
        "q1",
        "Which country interacted with China?",
        "entity",
        metadata={
            "qtype": "after_first",
            "time": ["2006-06-25"],
            "time_level": "day",
        },
    )

    assert parsed.temporal_operator == "after_first"
    assert parsed.temporal_constraints == ["2006-06-25"]
    assert parsed.temporal_granularity == "day"


def test_local_parser_uses_official_entity_position_annotations_for_aliases():
    parser = LocalQuestionParser(EntityIndex(["Japan", "China"]))
    question = "Before the Japanese government, who visited PRC?"

    parsed = parser.parse(
        "q1",
        question,
        "entity",
        metadata={
            "qtype": "before",
            "entity_positions": [
                {"entity": ["Japan"], "position": [7, 30]},
                {"entity": ["China"], "position": [44, 47]},
            ],
        },
    )

    assert parsed.topic_entities == ["China"]
    assert parsed.target_entities == ["Japan"]


def test_local_parser_maps_official_condemned_example_to_kg_relation():
    parser = LocalQuestionParser(EntityIndex(["Abhisit_Vejjajiva"]))

    parsed = parser.parse(
        "q1",
        "Who condemned Abhisit Vejjajiva in May 2010?",
        "entity",
    )

    assert "criticize_or_denounce" in parsed.relation_hints
    assert parsed.temporal_constraints == ["May 2010"]


def test_local_parser_maps_official_relation_paraphrases_to_real_kg_names():
    known_relations = {
        "Praise_or_endorse",
        "Host_a_visit",
        "Make_a_visit",
        "Express_intent_to_meet_or_negotiate",
        "Engage_in_negotiation",
    }
    parser = LocalQuestionParser(
        EntityIndex(["Mali", "Yang_Hyong_Sop", "Cambodia", "China"]),
        known_relations,
    )

    commended = parser.parse(
        "q1",
        "Who commended the military of Mali?",
        "entity",
    )
    hosted = parser.parse(
        "q2",
        "Which country received the visit from Yang Hyong Sop?",
        "entity",
    )
    intent = parser.parse(
        "q3",
        "Who wanted to negotiate with China?",
        "entity",
    )
    negotiated = parser.parse(
        "q4",
        "Who negotiated with China?",
        "entity",
    )

    assert commended.relation_hints == ["Praise_or_endorse"]
    assert hosted.relation_hints == ["Host_a_visit"]
    assert intent.relation_hints == ["Express_intent_to_meet_or_negotiate"]
    assert negotiated.relation_hints == ["Engage_in_negotiation"]


def test_local_parser_covers_additional_official_relation_nouns():
    known_relations = {
        "Praise_or_endorse",
        "Criticize_or_denounce",
        "Make_pessimistic_comment",
        "Appeal",
    }
    parser = LocalQuestionParser(EntityIndex([]), known_relations)

    assert parser.parse(
        "q1",
        "Who supported Rafael Correa?",
        "entity",
    ).relation_hints == ["Praise_or_endorse"]
    assert parser.parse(
        "q2",
        "When did the ministry issue a condemnation of the citizens?",
        "time",
    ).relation_hints == ["Criticize_or_denounce"]
    assert parser.parse(
        "q3",
        "Who expressed pessimism about the government?",
        "entity",
    ).relation_hints == ["Make_pessimistic_comment"]
    assert parser.parse(
        "q4",
        "Who did the party ask for?",
        "entity",
    ).relation_hints == ["Appeal"]


def test_local_parser_handles_official_object_question_forms():
    parser = LocalQuestionParser(EntityIndex([]))

    against = parser.parse(
        "q1",
        "Against whom did the army use conventional military force?",
        "entity",
    )
    suffered = parser.parse(
        "q2",
        "Which country suffered from the unconventional violence of the rebels?",
        "entity",
    )

    assert against.answer_role == "object"
    assert suffered.answer_role == "object"


def test_local_parser_distinguishes_wh_auxiliary_entity_from_to_verb():
    parser = LocalQuestionParser(
        EntityIndex(["Mwai_Kibaki", "Benedict_XVI"]),
    )
    entity_is_actor = parser.parse(
        "q1",
        "Who was the last person Mwai Kibaki wanted to meet?",
        "entity",
        metadata={
            "entity_positions": [
                {"entity": ["Mwai_Kibaki"], "position": [24, 35]},
            ],
        },
    )
    wh_is_actor = parser.parse(
        "q2",
        "Who was the last person to visit Benedict XVI?",
        "entity",
        metadata={
            "entity_positions": [
                {"entity": ["Benedict_XVI"], "position": [33, 45]},
            ],
        },
    )

    assert entity_is_actor.answer_role == "object"
    assert wh_is_actor.answer_role == "subject"


def test_equal_multi_without_of_uses_entity_nearest_same_time_phrase_as_anchor():
    parser = LocalQuestionParser(
        EntityIndex(["Tzipi_Livni", "Egyptian_Foreign_Ministry"]),
    )
    parsed = parser.parse(
        "q1",
        "With whom did Tzipi Livni telephone the Egyptian Foreign Ministry "
        "in the same year?",
        "entity",
        metadata={
            "qtype": "equal_multi",
            "time_level": "year",
            "entity_positions": [
                {"entity": ["Tzipi_Livni"], "position": [14, 26]},
                {"entity": ["Egyptian_Foreign_Ministry"], "position": [41, 67]},
            ],
        },
    )

    assert parsed.topic_entities == ["Tzipi_Livni"]
    assert parsed.target_entities == ["Egyptian_Foreign_Ministry"]


def test_entity_index_matches_parenthetical_kg_aliases_with_offsets():
    index = EntityIndex([
        "Government_(Pakistan)",
        "Cabinet_/_Council_of_Ministers_/_Advisors_(Greece)",
    ])

    matches = index.match_with_offsets(
        "The government of Pakistan criticised the "
        "Cabinet Council of Ministers Advisors of Greece.",
    )

    assert [entity for entity, _, _ in matches] == [
        "Government_(Pakistan)",
        "Cabinet_/_Council_of_Ministers_/_Advisors_(Greece)",
    ]
    assert matches[0][1] == 4


def test_entity_index_matches_country_demonym_aliases():
    index = EntityIndex([
        "Government_(Pakistan)",
        "Military_(Mali)",
        "Ministry_of_Education_(Taiwan)",
    ])

    assert index.match(
        "The Pakistani government praised the Malian military."
    ) == [
        "Government_(Pakistan)",
        "Military_(Mali)",
    ]


def test_entity_index_matches_multitq_plural_and_preposition_actor_aliases():
    index = EntityIndex([
        "Citizen_(Uganda)",
        "Government_(Occupied_Palestinian_Territories)",
        "Police_(Ukraine)",
        "Media_(International)",
    ])

    assert index.match("the citizens of Uganda") == ["Citizen_(Uganda)"]
    assert index.match(
        "the government of the occupied Palestinian territories"
    ) == ["Government_(Occupied_Palestinian_Territories)"]
    assert index.match("the police in Ukraine") == ["Police_(Ukraine)"]
    assert index.match(
        "the media personnel of the International"
    ) == ["Media_(International)"]


def test_entity_index_matches_multitq_role_synonyms_and_missing_demonyms():
    index = EntityIndex([
        "Head_of_Government_(Greece)",
        "Rebel_(Burkina_Faso)",
        "Member_of_Parliament_(Australia)",
        "Military_(Philippines)",
        "Navy_(Sri_Lanka)",
    ])

    assert index.match(
        "the Prime Minister of Greece"
    ) == ["Head_of_Government_(Greece)"]
    assert index.match(
        "the armed rebels of Burkina Faso"
    ) == ["Rebel_(Burkina_Faso)"]
    assert index.match(
        "a member of the Australian parliament"
    ) == ["Member_of_Parliament_(Australia)"]
    assert index.match(
        "the military of the Philippines"
    ) == ["Military_(Philippines)"]
    assert index.match("the Sri Lankan Navy") == ["Navy_(Sri_Lanka)"]


def test_explicit_time_keeps_both_fact_entities_instead_of_creating_anchor():
    parser = LocalQuestionParser(EntityIndex(["Alice", "Bob"]))

    parsed = parser.parse(
        "q1",
        "When did Alice negotiate with Bob before Jun 25th, 2006?",
        "time",
        metadata={"qtype": "before", "time_level": "day"},
    )

    assert parsed.topic_entities == ["Alice", "Bob"]
    assert parsed.target_entities == []
    assert parsed.temporal_constraints == ["Jun 25th, 2006"]


def test_local_parser_accepts_official_non_padded_iso_and_comma_month():
    parser = LocalQuestionParser(EntityIndex(["Malawi"]))

    day = parser.parse(
        "q1",
        "To whom did Malawi appeal before 2012-06-8?",
        "entity",
    )
    month = parser.parse(
        "q2",
        "Who fought after Jun, 2011?",
        "entity",
    )

    assert day.temporal_constraints == ["2012-06-8"]
    assert month.temporal_constraints == ["Jun, 2011"]


def test_local_parser_recognizes_passive_object_answer_role():
    parser = LocalQuestionParser(
        EntityIndex(["Benjamin_Netanyahu", "Middle_East"]),
    )
    parsed = parser.parse(
        "q1",
        "Who was last threatened by Benjamin Netanyahu before the Middle East?",
        "entity",
        metadata={
            "qtype": "before_last",
            "entity_positions": [
                {"entity": ["Benjamin_Netanyahu"], "position": [27, 45]},
                {"entity": ["Middle_East"], "position": [57, 68]},
            ],
        },
    )

    assert parsed.answer_role == "object"
    assert parsed.topic_entities == ["Benjamin_Netanyahu"]
    assert parsed.target_entities == ["Middle_East"]


def test_local_parser_recognizes_wh_auxiliary_before_known_entity_as_object():
    parser = LocalQuestionParser(
        EntityIndex(["Seyoum_Mesfin", "Abdullah_Gül"]),
    )
    parsed = parser.parse(
        "q1",
        "Who was Seyoum Mesfin negotiating with before Abdullah Gül?",
        "entity",
        metadata={
            "qtype": "before_after",
            "entity_positions": [
                {"entity": ["Seyoum_Mesfin"], "position": [8, 21]},
                {"entity": ["Abdullah_Gül"], "position": [46, 58]},
            ],
        },
    )

    assert parsed.answer_role == "object"


def test_time_parser_maps_diplomatic_intent_templates_to_specific_relation():
    parser = LocalQuestionParser(
        EntityIndex(["China", "Thailand"]),
        {
            "Express_intent_to_cooperate",
            "Engage_in_diplomatic_cooperation",
            "Express_intent_to_engage_in_diplomatic_cooperation_(such_as_policy_support)",
        },
    )
    expected = [
        "Express_intent_to_engage_in_diplomatic_cooperation_(such_as_policy_support)"
    ]

    for question in (
        "When did China seek diplomatic cooperation with Thailand?",
        "When did China want to participate in diplomatic cooperation with Thailand?",
        "When did China want to start diplomatic cooperation with Thailand?",
    ):
        assert parser.parse("q", question, "time").relation_hints == expected


def test_cron_parser_handles_rare_relation_surfaces_without_gold_relation():
    parser = LocalQuestionParser(None, set())
    base = {
        "_dataset": "CronQuestions",
        "official_entities": ["Q_PERSON"],
        "official_times": [],
    }

    cases = {
        "Who was a camerlengo?": "P39",
        "Who received the NBA All-Rookie Team honour?": "P166",
        "Who were her peers at the company?": "P108",
    }
    for question, relation in cases.items():
        parsed = parser.parse("q", question, "entity", base)
        assert parsed.relation_hints == [relation]

    following = parser.parse(
        "q", "Who held the post following Q_PERSON?", "entity", base
    )
    assert following.temporal_operator == "after"


def test_cron_parser_strong_award_phrase_outweighs_title_inside_entity_name():
    parser = LocalQuestionParser(None, set())
    parsed = parser.parse(
        "q",
        "What was given to His Majesty Juan Carlos I, King of Spain in 1980?",
        "entity",
        {
            "_dataset": "CronQuestions",
            "official_entities": ["Q_KING"],
            "official_times": ["1980"],
        },
    )

    assert parsed.relation_hints == ["P166"]
