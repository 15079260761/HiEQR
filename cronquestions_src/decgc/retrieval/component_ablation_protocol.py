"""Pure-Python protocol primitives for candidate-generation ablations."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import re


BASELINE = "baseline"
NO_DENSE_RETRIEVER = "no_dense_retriever"
NO_CROSS_ENCODER_RERANKER = "no_cross_encoder_reranker"
SUPPORTED_VARIANTS = {
    BASELINE,
    NO_DENSE_RETRIEVER,
    NO_CROSS_ENCODER_RERANKER,
}

_CACHE_NAMESPACE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def candidate_generation_variant(config) -> str:
    study = config.model_extra.get("ablation_study", {})
    variant = str(study.get("candidate_generation_variant", BASELINE))
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"Unsupported candidate-generation variant {variant!r}; "
            f"expected one of {sorted(SUPPORTED_VARIANTS)}"
        )
    return variant


def candidate_chain_root(config) -> Path:
    study = config.model_extra.get("ablation_study", {})
    namespace = study.get("candidate_cache_namespace")
    if namespace is None:
        return config.paths.cache_root / "candidate_chains"
    namespace = str(namespace)
    if not _CACHE_NAMESPACE.fullmatch(namespace):
        raise ValueError(f"Unsafe candidate-cache namespace: {namespace!r}")
    return config.paths.cache_root / "candidate_chain_ablations" / namespace


def hashed_lexical_counts(text: str, dimensions: int) -> dict[int, float]:
    """Return stable label-free unigram/bigram hashing features."""
    if dimensions <= 0:
        raise ValueError("dimensions must be positive")
    tokens = _TOKEN.findall(str(text).casefold())
    features = tokens + [
        f"{left}::{right}"
        for left, right in zip(tokens, tokens[1:])
    ]
    result: dict[int, float] = {}
    for feature, count in Counter(features).items():
        digest = hashlib.blake2b(
            feature.encode("utf-8"),
            digest_size=8,
            person=b"DECGCab1",
        ).digest()
        bucket = int.from_bytes(digest, "little") % dimensions
        result[bucket] = result.get(bucket, 0.0) + 1.0 + float(count - 1) ** 0.5
    return result
