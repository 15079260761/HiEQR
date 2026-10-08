"""Traceable JSONL and deterministic human-readable evidence export."""

from __future__ import annotations

from pathlib import Path

from ..utils.io import atomic_text, write_json, write_jsonl


def chain_evidence(chain, predicted_answer: str, validation: dict) -> dict:
    history = chain.state_history
    reliabilities = [item.absolute_reliability for item in history]
    evidence_gains = []
    previous = 0.0
    for reliability in reliabilities:
        evidence_gains.append(reliability - previous)
        previous = reliability
    return {
        "chain_id": chain.chain_id, "qid": chain.qid, "predicted_answer": predicted_answer,
        "candidate_answer": chain.candidate_answer, "chain_state": chain.state, "stop_hop": chain.hop,
        "stop_reason": chain.stop_reason, "final_reliability": chain.absolute_reliability,
        "final_relative_weight": chain.relative_weight,
        "final_answer_score": chain.answer_score,
        "answer_score_contribution": chain.answer_score * chain.relative_weight,
        "reliability_trajectory": reliabilities,
        "evidence_gain_trajectory": evidence_gains,
        "relative_weight_trajectory": [item.relative_weight for item in history],
        "stop_probability_trajectory": [item.stop_probability for item in history],
        "facts": [fact.model_dump(mode="json") | {
            "hop": index + 1, "retrieval_score": chain.retrieval_scores[index],
            "reranker_score": chain.reranker_scores[index], "temporal_valid": validation["temporal_constraints_satisfied"],
        } for index, fact in enumerate(chain.facts)],
        "graph_interactions": chain.graph_interactions, "validity": validation,
    }


def write_prediction_bundle(predictions: list[dict], evidence: list[dict], audit: list[dict],
                            validation: dict, prediction_dir: Path, evidence_dir: Path) -> None:
    write_jsonl(prediction_dir / "predictions.jsonl", predictions)
    write_jsonl(evidence_dir / "supporting_evidence_chains.jsonl", evidence)
    write_jsonl(evidence_dir / "all_chain_audit.jsonl", audit)
    write_json(evidence_dir / "evidence_validation.json", validation)
    sections = []
    by_qid = {}
    for item in evidence:
        by_qid.setdefault(item["qid"], []).append(item)
    for prediction in predictions:
        sections.extend([f"# Question\n\n{prediction['question']}\n",
                         f"## Predicted Answer\n\n{prediction['predicted_answer']}\n"])
        chains = by_qid.get(prediction["qid"], [])
        if not chains:
            sections.append("No completed supporting chain; this prediction is marked as fallback.\n")
        for chain_index, chain in enumerate(chains, 1):
            sections.append(f"## Supporting Chain {chain_index}\n")
            for hop, fact in enumerate(chain["facts"], 1):
                sections.append(
                    f"Hop {hop}: ({fact['subject_text']}, {fact['relation_text']}, "
                    f"{fact['object_text']}, {fact['timestamp_raw']})\n\n"
                    f"Source: {fact['source_relative_path']} row {fact['source_row_index']}\n"
                )
            sections.append(f"Reliability trajectory: {chain['reliability_trajectory']}\n\n"
                            f"Temporal validation: {chain['validity']['temporal_constraints_satisfied']}\n")
    atomic_text(evidence_dir / "human_readable_evidence.md", "\n".join(sections))
