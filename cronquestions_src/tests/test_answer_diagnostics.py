from decgc.evaluation.metrics import answer_diagnostics


def test_answer_diagnostics_separates_ranking_errors_and_retrieval_misses():
    rows = [
        {
            "candidate_answers": ["gold", "other"],
            "gold_answers": ["gold"],
            "question_type": "single",
            "answer_type": "entity",
            "answer_role": "subject",
            "temporal_operator": "equal",
            "temporal_granularity": "day",
            "parse_source": "local",
        },
        {
            "candidate_answers": ["wrong", "gold"],
            "gold_answers": ["gold"],
            "question_type": "single",
            "answer_type": "entity",
            "answer_role": "subject",
            "temporal_operator": "equal",
            "temporal_granularity": "day",
            "parse_source": "local",
        },
        {
            "candidate_answers": ["wrong"],
            "gold_answers": ["gold"],
            "question_type": "multiple",
            "answer_type": "time",
            "answer_role": "time",
            "temporal_operator": "before_last",
            "temporal_granularity": "month",
            "parse_source": "deepseek",
        },
    ]

    diagnostics = answer_diagnostics(rows)

    assert diagnostics["overall"]["correct_count"] == 1
    assert diagnostics["overall"]["ranking_error_count"] == 1
    assert diagnostics["overall"]["retrieval_miss_count"] == 1
    assert diagnostics["overall"]["gold_rank_histogram"] == {
        "rank_1": 1,
        "rank_2_to_10": 1,
        "rank_over_10": 0,
        "missing": 1,
    }
    assert diagnostics["by_question_type"]["single"]["hits1"] == 0.5
    assert diagnostics["by_answer_type"]["time"]["candidate_answer_recall"] == 0.0
    assert diagnostics["by_temporal_operator"]["equal"]["hits10"] == 1.0
    assert diagnostics["by_temporal_granularity"]["day"]["hits1"] == 0.5
    assert diagnostics["by_answer_role"]["subject"]["hits1"] == 0.5
    assert diagnostics["by_parse_source"]["deepseek"]["retrieval_miss_count"] == 1
