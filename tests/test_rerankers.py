from __future__ import annotations

import logging

from mcp_server.retrieval.hybrid import Candidate
from mcp_server.retrieval.rerankers import rerank_candidates, rerank_or_fallback


class StaticReranker:
    def __init__(self, scores: list[float]):
        self.scores = scores

    def score(self, query: str, candidates: list[Candidate], *, batch_size: int) -> list[float]:
        return self.scores[: len(candidates)]


class FailingReranker:
    def score(self, query: str, candidates: list[Candidate], *, batch_size: int) -> list[float]:
        raise RuntimeError("reranker unavailable")


def _candidate(chunk_id: str, text: str, fused_score: float, rank: int) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        text=text,
        metadata={
            "chunk_id": chunk_id,
            "file_path": f"notes/{chunk_id}.md",
            "fused_score": fused_score,
            "fused_rank": rank,
        },
        score=fused_score,
        source="both",
        rank=rank,
        dense_score=0.5,
        bm25_score=2.0,
        fused_score=fused_score,
        retrieval_sources={"dense", "bm25"},
    )


def test_reranker_preserves_candidate_metadata():
    candidates = [_candidate("a", "alpha", 0.2, 1)]

    results = rerank_candidates(
        "query",
        candidates,
        reranker=StaticReranker([0.91]),
        candidate_top_k=10,
        final_top_k=5,
        batch_size=2,
    )

    assert results[0].metadata["file_path"] == "notes/a.md"
    assert results[0].metadata["fused_score"] == 0.2
    assert results[0].metadata["rerank_score"] == 0.91
    assert results[0].metadata["final_rank"] == 1


def test_reranker_sorts_by_score():
    candidates = [
        _candidate("a", "alpha", 0.3, 1),
        _candidate("b", "beta", 0.2, 2),
        _candidate("c", "gamma", 0.1, 3),
    ]

    results = rerank_candidates(
        "query",
        candidates,
        reranker=StaticReranker([0.1, 0.9, 0.2]),
        candidate_top_k=3,
        final_top_k=3,
        batch_size=2,
    )

    assert [candidate.chunk_id for candidate in results] == ["b", "c", "a"]
    assert [candidate.metadata["final_rank"] for candidate in results] == [1, 2, 3]


def test_reranker_failure_falls_back_to_fusion(caplog):
    candidates = [
        _candidate("a", "alpha", 0.3, 1),
        _candidate("b", "beta", 0.2, 2),
    ]

    with caplog.at_level(logging.WARNING):
        results = rerank_or_fallback(
            "query",
            candidates,
            enabled=True,
            provider="local_huggingface",
            model="BAAI/bge-reranker-v2-m3",
            candidate_top_k=2,
            final_top_k=2,
            batch_size=2,
            logger=logging.getLogger("test"),
            reranker=FailingReranker(),
        )

    assert [candidate.chunk_id for candidate in results] == ["a", "b"]
    assert "Reranker failed; returning fused results" in caplog.text
    assert "rerank_score" not in results[0].metadata
    assert results[0].metadata["final_rank"] == 1


def test_final_top_k_applied_after_reranking():
    candidates = [
        _candidate("a", "alpha", 0.3, 1),
        _candidate("b", "beta", 0.2, 2),
        _candidate("c", "gamma", 0.1, 3),
    ]

    results = rerank_or_fallback(
        "query",
        candidates,
        enabled=True,
        provider="local_huggingface",
        model="BAAI/bge-reranker-v2-m3",
        candidate_top_k=3,
        final_top_k=2,
        batch_size=2,
        reranker=StaticReranker([0.4, 0.9, 0.8]),
    )

    assert [candidate.chunk_id for candidate in results] == ["b", "c"]
    assert len(results) == 2


def test_reranker_can_be_disabled():
    candidates = [
        _candidate("a", "alpha", 0.3, 1),
        _candidate("b", "beta", 0.2, 2),
    ]

    results = rerank_or_fallback(
        "query",
        candidates,
        enabled=False,
        provider="local_huggingface",
        model="BAAI/bge-reranker-v2-m3",
        candidate_top_k=2,
        final_top_k=1,
        batch_size=2,
        reranker=FailingReranker(),
    )

    assert [candidate.chunk_id for candidate in results] == ["a"]
    assert "rerank_score" not in results[0].metadata
    assert results[0].metadata["final_rank"] == 1
