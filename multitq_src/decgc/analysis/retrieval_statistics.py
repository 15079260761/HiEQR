"""Retrieval score distribution summaries."""

import numpy as np


def retrieval_statistics(chains: list) -> dict:
    scores = [score for chain in chains for score in chain.retrieval_scores]
    return {"count": len(scores), "mean": float(np.mean(scores)) if scores else 0.0,
            "p95": float(np.quantile(scores, 0.95)) if scores else 0.0}

