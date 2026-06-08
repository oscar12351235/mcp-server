from __future__ import annotations

import logging
from copy import deepcopy
from typing import Protocol

from mcp_server.retrieval.hybrid import Candidate


class Reranker(Protocol):
    def score(self, query: str, candidates: list[Candidate], *, batch_size: int) -> list[float]:
        """Return one relevance score per candidate."""


class SentenceTransformersReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def score(self, query: str, candidates: list[Candidate], *, batch_size: int) -> list[float]:
        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self.model.predict(pairs, batch_size=batch_size)
        return [float(score) for score in scores]


_RERANKERS: dict[tuple[str, str], Reranker] = {}


def get_reranker(provider: str, model: str) -> Reranker:
    key = (provider, model)
    if key in _RERANKERS:
        return _RERANKERS[key]

    if provider != "local_huggingface":
        raise ValueError(f"Unsupported reranker provider: {provider}")

    reranker = SentenceTransformersReranker(model)
    _RERANKERS[key] = reranker
    return reranker


def rerank_candidates(
    query: str,
    candidates: list[Candidate],
    *,
    reranker: Reranker,
    candidate_top_k: int,
    final_top_k: int,
    batch_size: int,
) -> list[Candidate]:
    selected = [deepcopy(candidate) for candidate in candidates[:candidate_top_k]]
    scores = reranker.score(query, selected, batch_size=batch_size)

    if len(scores) != len(selected):
        raise ValueError(f"Reranker returned {len(scores)} scores for {len(selected)} candidates")

    for candidate, score in zip(selected, scores, strict=True):
        candidate.rerank_score = score
        candidate.score = score
        candidate.metadata = {
            **candidate.metadata,
            "rerank_score": score,
        }

    selected.sort(key=lambda item: item.rerank_score or 0.0, reverse=True)
    return apply_final_ranks(selected[:final_top_k])


def apply_final_ranks(candidates: list[Candidate]) -> list[Candidate]:
    ranked = [deepcopy(candidate) for candidate in candidates]

    for index, candidate in enumerate(ranked, start=1):
        candidate.final_rank = index
        candidate.metadata = {
            **candidate.metadata,
            "final_rank": index,
        }

    return ranked


def rerank_or_fallback(
    query: str,
    candidates: list[Candidate],
    *,
    enabled: bool,
    provider: str,
    model: str,
    candidate_top_k: int,
    final_top_k: int,
    batch_size: int,
    logger: logging.Logger | None = None,
    reranker: Reranker | None = None,
) -> list[Candidate]:
    if not enabled:
        return apply_final_ranks(candidates[:final_top_k])

    try:
        active_reranker = reranker or get_reranker(provider, model)
        return rerank_candidates(
            query,
            candidates,
            reranker=active_reranker,
            candidate_top_k=candidate_top_k,
            final_top_k=final_top_k,
            batch_size=batch_size,
        )
    except Exception as exc:
        if logger is not None:
            logger.warning("Reranker failed; returning fused results: %s", exc)
        return apply_final_ranks(candidates[:final_top_k])
