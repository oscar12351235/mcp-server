from __future__ import annotations

from llama_index.core import Document

from mcp_server.retrieval.bm25_store import bm25_retrieve, load_bm25_index, write_bm25_index
from mcp_server.retrieval.hybrid import Candidate, fuse_candidates


def test_rrf_fusion_deduplicates_chunks():
    dense = [
        Candidate(
            chunk_id="a",
            text="dense text",
            metadata={"chunk_id": "a"},
            score=0.8,
            source="dense",
            rank=1,
            dense_score=0.8,
            retrieval_sources={"dense"},
        )
    ]
    bm25 = [
        Candidate(
            chunk_id="a",
            text="bm25 text",
            metadata={"chunk_id": "a"},
            score=3.0,
            source="bm25",
            rank=1,
            bm25_score=3.0,
            retrieval_sources={"bm25"},
        )
    ]

    fused = fuse_candidates(dense, bm25, fused_top_k=10)

    assert len(fused) == 1
    assert fused[0].metadata["retrieval_source"] == "both"
    assert fused[0].metadata["dense_score"] == 0.8
    assert fused[0].metadata["bm25_score"] == 3.0


def test_bm25_retrieves_exact_function_name(tmp_path):
    index_path = tmp_path / "bm25.jsonl"
    documents = [
        Document(
            text="def _read_gitlab_file_raw(project_id, file_path): pass",
            metadata={"chunk_id": "function", "symbol_name": "_read_gitlab_file_raw"},
        ),
        Document(
            text="general architecture notes",
            metadata={"chunk_id": "notes"},
        ),
    ]
    write_bm25_index(index_path, documents)

    results = bm25_retrieve("_read_gitlab_file_raw", load_bm25_index(index_path), top_k=2)

    assert results[0].chunk_id == "function"
    assert results[0].metadata["symbol_name"] == "_read_gitlab_file_raw"
    assert results[0].source == "bm25"


def test_bm25_retrieves_exact_file_path(tmp_path):
    index_path = tmp_path / "bm25.jsonl"
    documents = [
        Document(
            text="Configuration notes",
            metadata={"chunk_id": "path", "file_path": "src/mcp_server/mcp_server.py"},
        ),
        Document(text="unrelated text", metadata={"chunk_id": "other"}),
    ]
    write_bm25_index(index_path, documents)

    results = bm25_retrieve("src/mcp_server/mcp_server.py", load_bm25_index(index_path), top_k=2)

    assert results[0].chunk_id == "path"


def test_bm25_retrieves_symbol_name_from_metadata_only(tmp_path):
    index_path = tmp_path / "bm25.jsonl"
    documents = [
        Document(
            text="implementation details without the function token",
            metadata={
                "chunk_id": "function",
                "symbol_name": "_read_gitlab_file_raw",
                "symbol_type": "function",
            },
        ),
        Document(text="unrelated text", metadata={"chunk_id": "other"}),
    ]
    write_bm25_index(index_path, documents)

    results = bm25_retrieve("_read_gitlab_file_raw", load_bm25_index(index_path), top_k=2)

    assert results[0].chunk_id == "function"
    assert results[0].metadata["symbol_name"] == "_read_gitlab_file_raw"


def test_bm25_retrieves_section_path_from_metadata_only(tmp_path):
    index_path = tmp_path / "bm25.jsonl"
    documents = [
        Document(
            text="content does not contain the heading token",
            metadata={"chunk_id": "section", "section_path": "Architecture / MCP Tools"},
        ),
        Document(text="unrelated text", metadata={"chunk_id": "other"}),
    ]
    write_bm25_index(index_path, documents)

    results = bm25_retrieve("MCP Tools", load_bm25_index(index_path), top_k=2)

    assert results[0].chunk_id == "section"


def test_bm25_retrieves_key_path_from_metadata_only(tmp_path):
    index_path = tmp_path / "bm25.jsonl"
    documents = [
        Document(
            text="configuration value without the queried path",
            metadata={"chunk_id": "key", "key_path": "retrieval.defaults.knowledge_top_k"},
        ),
        Document(text="unrelated text", metadata={"chunk_id": "other"}),
    ]
    write_bm25_index(index_path, documents)

    results = bm25_retrieve(
        "retrieval.defaults.knowledge_top_k",
        load_bm25_index(index_path),
        top_k=2,
    )

    assert results[0].chunk_id == "key"


def test_hybrid_includes_dense_and_bm25_sources():
    dense = [
        Candidate(
            chunk_id="semantic",
            text="semantic concept",
            metadata={"chunk_id": "semantic"},
            score=0.7,
            source="dense",
            rank=1,
            dense_score=0.7,
            retrieval_sources={"dense"},
        )
    ]
    bm25 = [
        Candidate(
            chunk_id="exact",
            text="exact token",
            metadata={"chunk_id": "exact"},
            score=2.0,
            source="bm25",
            rank=1,
            bm25_score=2.0,
            retrieval_sources={"bm25"},
        )
    ]

    fused = fuse_candidates(dense, bm25, fused_top_k=10)
    sources = {candidate.metadata["retrieval_source"] for candidate in fused}

    assert sources == {"dense", "bm25"}
