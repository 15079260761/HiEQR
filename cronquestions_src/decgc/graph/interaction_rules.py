"""Explicit complementary-evidence cooperation and contradiction competition."""

from __future__ import annotations

from ..schemas import ChainPrefix


def cooperation(left: ChainPrefix, right: ChainPrefix) -> tuple[bool, str]:
    if not left.candidate_answer or left.candidate_answer != right.candidate_answer:
        return False, ""
    left_ids, right_ids = {fact.fact_id for fact in left.facts}, {fact.fact_id for fact in right.facts}
    if left_ids == right_ids or not (left_ids - right_ids) or not (right_ids - left_ids):
        return False, ""
    left_times, right_times = set(left.time_path), set(right.time_path)
    if left_times and right_times and max(left_times) < min(right_times) and left.relation_path == right.relation_path:
        return False, ""
    if not set(left.relation_path) & set(right.relation_path):
        return False, ""
    return True, "same_answer_with_compatible_independent_evidence"


def competition(left: ChainPrefix, right: ChainPrefix, exclusive: bool = True) -> tuple[bool, str]:
    if exclusive and left.candidate_answer and right.candidate_answer and left.candidate_answer != right.candidate_answer:
        return True, "exclusive_question_different_answers"
    shared_relation = set(left.relation_path) & set(right.relation_path)
    if shared_relation and left.time_path and right.time_path and left.time_path[-1] != right.time_path[-1]:
        return True, "key_relation_temporal_conflict"
    return False, ""

