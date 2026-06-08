from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Candidate:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float
    source: str
    rank: int
    dense_score: float | None = None
    bm25_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    final_rank: int | None = None
    retrieval_sources: set[str] = field(default_factory=set)


def candidate_key(candidate: Candidate) -> str:
    if candidate.chunk_id:
        return candidate.chunk_id

    file_path = candidate.metadata.get("file_path", "")
    chunk_index = candidate.metadata.get("chunk_index", "")
    return f"{file_path}:{chunk_index}"


def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def fuse_candidates(
    dense_candidates: list[Candidate],
    bm25_candidates: list[Candidate],
    *,
    fused_top_k: int,
    method: str = "rrf",
    rrf_k: int = 60,
) -> list[Candidate]:
    if method != "rrf":
        raise ValueError(f"Unsupported fusion method: {method}")

    merged: dict[str, Candidate] = {}
    fused_scores: dict[str, float] = {}

    for source_candidates in (dense_candidates, bm25_candidates):
        for candidate in source_candidates:
            key = candidate_key(candidate)
            if key not in merged:
                merged[key] = Candidate(
                    chunk_id=candidate.chunk_id,
                    text=candidate.text,
                    metadata=dict(candidate.metadata),
                    score=0.0,
                    source="",
                    rank=candidate.rank,
                    dense_score=candidate.dense_score,
                    bm25_score=candidate.bm25_score,
                    fused_score=candidate.fused_score,
                    rerank_score=candidate.rerank_score,
                    final_rank=candidate.final_rank,
                    retrieval_sources=set(),
                )

            existing = merged[key]
            existing.retrieval_sources.add(candidate.source)

            if candidate.source == "dense":
                existing.dense_score = candidate.dense_score
            if candidate.source == "bm25":
                existing.bm25_score = candidate.bm25_score

            fused_scores[key] = fused_scores.get(key, 0.0) + rrf_score(candidate.rank, rrf_k)

    fused = []
    for key, candidate in merged.items():
        sources = sorted(candidate.retrieval_sources)
        candidate.source = "both" if len(sources) > 1 else sources[0]
        candidate.score = fused_scores[key]
        candidate.fused_score = candidate.score
        candidate.metadata = {**candidate.metadata}
        fused.append(candidate)

    fused.sort(key=lambda item: item.score, reverse=True)

    for index, candidate in enumerate(fused, start=1):
        candidate.rank = index
        sources = sorted(candidate.retrieval_sources)
        candidate.metadata.update(
            {
                "retrieval_source": candidate.source,
                "retrieval_sources": sources,
                "dense_score": candidate.dense_score,
                "bm25_score": candidate.bm25_score,
                "fused_score": candidate.fused_score,
                "fused_rank": index,
            }
        )

    return fused[:fused_top_k]
