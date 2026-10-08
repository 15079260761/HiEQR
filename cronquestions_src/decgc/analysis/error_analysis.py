"""Prediction error bucketing."""

from ..evaluation.answer_normalizer import normalize_answer


def error_analysis(predictions: list[dict]) -> dict:
    errors = {"retrieval_miss": [], "ranking_error": [], "correct": []}
    for row in predictions:
        gold = {normalize_answer(item) for item in row["gold_answers"]}
        ranking = [normalize_answer(item) for item in row["candidate_answers"]]
        bucket = "correct" if ranking and ranking[0] in gold else ("ranking_error" if gold & set(ranking) else "retrieval_miss")
        errors[bucket].append(row["qid"])
    return errors

