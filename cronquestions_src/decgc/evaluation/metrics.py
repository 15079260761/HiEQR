"""Hits@K and MRR for possibly multi-answer questions."""

from .answer_normalizer import normalize_answer


def reciprocal_rank(ranking: list[str], gold: list[str]) -> float:
    targets = {normalize_answer(item) for item in gold}
    for index, item in enumerate(ranking, 1):
        if normalize_answer(item) in targets:
            return 1.0 / index
    return 0.0


def answer_metrics(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {
            "hits1": 0.0,
            "hits10": 0.0,
            "mrr": 0.0,
            "candidate_answer_recall": 0.0,
            "mean_candidate_count": 0.0,
        }
    ranks = [reciprocal_rank(row["candidate_answers"], row["gold_answers"]) for row in rows]
    return {
        "hits1": sum(rank == 1.0 for rank in ranks) / len(ranks),
        "hits10": sum(rank >= 0.1 for rank in ranks) / len(ranks),
        "mrr": sum(ranks) / len(ranks),
        "candidate_answer_recall": sum(rank > 0 for rank in ranks) / len(ranks),
        "mean_candidate_count": sum(len(row["candidate_answers"]) for row in rows) / len(rows),
    }


class _AnswerSummary:
    """Constant-memory accumulator for one diagnostic group."""

    def __init__(self) -> None:
        self.samples = 0
        self.hits1 = 0
        self.hits10 = 0
        self.reciprocal_rank_sum = 0.0
        self.retrieved = 0
        self.candidate_count_sum = 0
        self.rank_2_to_10 = 0
        self.rank_over_10 = 0
        self.found_rank_sum = 0
        self.predicted_hop_sum = 0
        self.predicted_hop_count = 0
        self.gold_hop_sum = 0
        self.gold_hop_count = 0
        self.maximum_path_multiplicity_sum = 0

    def add(self, row: dict) -> None:
        self.samples += 1
        ranking = row["candidate_answers"]
        reciprocal = reciprocal_rank(ranking, row["gold_answers"])
        self.hits1 += reciprocal == 1.0
        self.hits10 += reciprocal >= 0.1
        self.reciprocal_rank_sum += reciprocal
        self.candidate_count_sum += len(ranking)
        if reciprocal > 0:
            rank = round(1.0 / reciprocal)
            self.retrieved += 1
            self.found_rank_sum += rank
            self.rank_2_to_10 += 2 <= rank <= 10
            self.rank_over_10 += rank > 10

        hops = row.get("candidate_best_hops", {})
        counts = row.get("candidate_chain_counts", {})
        if ranking:
            predicted_key = normalize_answer(ranking[0])
            if predicted_key in hops:
                self.predicted_hop_sum += int(hops[predicted_key])
                self.predicted_hop_count += 1
        matching_gold_hops = [
            int(hops[normalize_answer(answer)])
            for answer in row["gold_answers"]
            if normalize_answer(answer) in hops
        ]
        if matching_gold_hops:
            self.gold_hop_sum += min(matching_gold_hops)
            self.gold_hop_count += 1
        self.maximum_path_multiplicity_sum += max(
            (int(value) for value in counts.values()),
            default=0,
        )

    def result(self) -> dict:
        denominator = max(1, self.samples)
        missing = self.samples - self.retrieved
        histogram = {
            "rank_1": self.hits1,
            "rank_2_to_10": self.rank_2_to_10,
            "rank_over_10": self.rank_over_10,
            "missing": missing,
        }
        return {
            "samples": self.samples,
            "hits1": self.hits1 / denominator,
            "hits10": self.hits10 / denominator,
            "mrr": self.reciprocal_rank_sum / denominator,
            "candidate_answer_recall": self.retrieved / denominator,
            "mean_candidate_count": self.candidate_count_sum / denominator,
            "correct_count": self.hits1,
            "ranking_error_count": self.retrieved - self.hits1,
            "retrieval_miss_count": missing,
            "mean_gold_rank_when_retrieved": (
                self.found_rank_sum / self.retrieved if self.retrieved else None
            ),
            "mean_top1_supporting_hop": (
                self.predicted_hop_sum / self.predicted_hop_count
                if self.predicted_hop_count else None
            ),
            "mean_gold_min_hop_when_retrieved": (
                self.gold_hop_sum / self.gold_hop_count
                if self.gold_hop_count else None
            ),
            "mean_max_paths_per_answer": (
                self.maximum_path_multiplicity_sum / denominator
            ),
            "gold_rank_histogram": histogram,
        }


def answer_diagnostics_iter(rows) -> dict:
    """Compute all answer diagnostics without retaining prediction rows."""
    group_fields = {
        "by_question_type": "question_type",
        "by_answer_type": "answer_type",
        "by_temporal_operator": "temporal_operator",
        "by_temporal_granularity": "temporal_granularity",
        "by_answer_role": "answer_role",
        "by_parse_source": "parse_source",
    }
    overall = _AnswerSummary()
    grouped: dict[str, dict[str, _AnswerSummary]] = {
        name: {} for name in group_fields
    }
    for row in rows:
        overall.add(row)
        for name, field in group_fields.items():
            value = str(row.get(field) or "unknown")
            summary = grouped[name].setdefault(value, _AnswerSummary())
            summary.add(row)
    return {
        "overall": overall.result(),
        **{
            name: {
                value: summary.result()
                for value, summary in values.items()
            }
            for name, values in grouped.items()
        },
    }


def answer_diagnostics(rows: list[dict]) -> dict:
    """Expose whether failures come from candidate recall or answer ranking."""
    return answer_diagnostics_iter(iter(rows))
