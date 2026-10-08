from decgc.data.sampler import stratified_sample
from decgc.data.temporal_parser import parse_time, temporal_compatible
from decgc.llm.local_parser import LocalQuestionParser
from decgc.schemas import QAExample


def test_interval_parser_uses_closed_year_ranges():
    parsed = parse_time("2004/2009")
    assert parsed["start"] == "2004-01-01"
    assert parsed["end"] == "2009-12-31"
    assert temporal_compatible("2004/2009", ["2007"], "equal")
    assert not temporal_compatible("2004/2009", ["2010"], "equal")


def test_cron_annotation_parse_does_not_read_answers():
    metadata = {
        "_dataset": "CronQuestions",
        "cron_type": "before_after",
        "annotation": {"head": "Q1", "tail": "Q2", "type": "after"},
        "official_entities": ["Q2", "Q1"],
        "official_relations": ["ORACLE_RELATION_MUST_NOT_BE_READ"],
        "official_times": [],
        "answers": ["SHOULD_NOT_BE_READ"],
    }
    query = LocalQuestionParser().parse(
        "test:0:1", "Who held the position Q2 after Q1", "entity", metadata
    )
    assert query.topic_entities == ["Q2"]
    assert query.target_entities == ["Q1"]
    assert query.relation_hints == ["P39"]
    assert "SHOULD_NOT_BE_READ" not in query.model_dump_json()
    assert "ORACLE_RELATION_MUST_NOT_BE_READ" not in query.model_dump_json()
    assert query.parse_source == "official_linking"


def test_cron_relation_templates_disambiguate_team_member_and_olympics():
    parser = LocalQuestionParser()
    team = parser.parse(
        "q1",
        "Name a team member of Alan Waddle in Liverpool FC",
        "entity",
        {"_dataset": "CronQuestions", "official_entities": ["Q1", "Q2"]},
    )
    position = parser.parse(
        "q2",
        "Who was a member of parliament during the Olympic Games",
        "entity",
        {"_dataset": "CronQuestions", "official_entities": ["Q3", "Q4"]},
    )
    assert team.relation_hints == ["P54"]
    assert position.relation_hints == ["P39"]


def test_proportional_sampling_is_deterministic():
    rows = []
    for index in range(80):
        cron_type = "simple_entity" if index < 60 else "before_after"
        rows.append(QAExample(
            qid=str(index), question=str(index), answers=["Q0"], split="train",
            question_type="Simple" if index < 60 else "Complex",
            answer_type="entity",
            raw_record={"_dataset": "CronQuestions", "cron_type": cron_type},
            source_relative_path="questions/train.pickle", source_row_index=index,
        ))
    first = stratified_sample(rows, 20, 42)
    second = stratified_sample(rows, 20, 42)
    assert [row.qid for row in first] == [row.qid for row in second]
    assert sum(row.question_type == "Simple" for row in first) == 15
    assert sum(row.question_type == "Complex" for row in first) == 5
