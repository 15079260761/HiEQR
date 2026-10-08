from decgc.data.sampler import stratified_sample
from decgc.schemas import QAExample


def _example(index: int, qtype: str, granularity: str) -> QAExample:
    return QAExample(
        qid=str(index),
        question=f"question {index}",
        answers=["answer"],
        split="valid",
        question_type="Single",
        answer_type="entity",
        raw_record={
            "qtype": qtype,
            "time_level": granularity,
        },
        source_relative_path="questions/dev.json",
        source_row_index=index,
    )


def test_demo_sampling_stratifies_temporal_type_and_granularity():
    examples = [
        _example(1, "equal", "day"),
        _example(2, "equal", "day"),
        _example(3, "before_after", "month"),
        _example(4, "before_after", "month"),
        _example(5, "first_last", "year"),
        _example(6, "first_last", "year"),
    ]

    sample = stratified_sample(examples, 3, seed=42)

    assert {row.raw_record["qtype"] for row in sample} == {
        "equal",
        "before_after",
        "first_last",
    }
    assert {row.raw_record["time_level"] for row in sample} == {
        "day",
        "month",
        "year",
    }
