from __future__ import annotations

from mcp_server.retrieval.bm25_store import bm25_retrieve, load_bm25_index, write_bm25_index
from mcp_server.retrieval.code_chunker import chunk_python_code, infer_file_role
from mcp_server.retrieval.code_loader import load_code_documents
from mcp_server.retrieval.code_retrieval import prefer_source_code_candidates
from mcp_server.retrieval.hybrid import Candidate

CLASS_SOURCE = '''
import os
from pathlib import Path

mcp = FastMCP("personal-assistant")

class Worker:
    """Worker docs."""

    def run(self):
        """Run docs."""
        return Path(os.getcwd())

    def stop(self):
        return "stopped"

def build_worker():
    return Worker()
'''.strip()


def test_python_code_chunker_extracts_classes_functions_methods_and_imports(tmp_path):
    path = tmp_path / "repo" / "src" / "service.py"
    path.parent.mkdir(parents=True)
    path.write_text(CLASS_SOURCE + "\n", encoding="utf-8")

    chunks = chunk_python_code(
        path,
        repo_name="repo",
        relative_path="src/service.py",
        max_chars=2000,
        overlap_chars=0,
    )
    by_symbol = {chunk.metadata.get("symbol_name"): chunk for chunk in chunks}

    assert "FastMCP" in by_symbol["mcp"].text
    assert by_symbol["mcp"].metadata["symbol_type"] == "module"
    assert by_symbol["__imports__"].metadata["imports"] == "os, pathlib.Path"
    assert by_symbol["Worker"].metadata["symbol_type"] == "class"
    assert by_symbol["Worker"].metadata["docstring"] == "Worker docs."
    assert "def run" not in by_symbol["Worker"].text
    assert "methods: run, stop" in by_symbol["Worker"].text
    assert by_symbol["run"].metadata["symbol_type"] == "method"
    assert by_symbol["run"].metadata["parent_symbol"] == "Worker"
    assert by_symbol["run"].metadata["qualified_symbol"] == "Worker.run"
    assert by_symbol["run"].metadata["start_line"] < by_symbol["run"].metadata["end_line"]
    assert by_symbol["build_worker"].metadata["symbol_type"] == "function"
    assert by_symbol["build_worker"].metadata["file_path"] == "src/service.py"
    assert by_symbol["build_worker"].metadata["repo_name"] == "repo"


def test_full_class_chunks_can_include_method_bodies(tmp_path):
    path = tmp_path / "repo" / "src" / "service.py"
    path.parent.mkdir(parents=True)
    path.write_text(CLASS_SOURCE + "\n", encoding="utf-8")

    chunks = chunk_python_code(
        path,
        repo_name="repo",
        relative_path="src/service.py",
        max_chars=2000,
        overlap_chars=0,
        include_full_class_chunks=True,
    )
    by_symbol = {chunk.metadata.get("symbol_name"): chunk for chunk in chunks}

    assert "def run" in by_symbol["Worker"].text
    assert "def stop" in by_symbol["Worker"].text


def test_method_chunks_can_be_suppressed(tmp_path):
    path = tmp_path / "repo" / "src" / "service.py"
    path.parent.mkdir(parents=True)
    path.write_text(CLASS_SOURCE + "\n", encoding="utf-8")

    chunks = chunk_python_code(
        path,
        repo_name="repo",
        relative_path="src/service.py",
        max_chars=2000,
        overlap_chars=0,
        include_method_chunks=False,
    )
    symbols = {chunk.metadata.get("symbol_name") for chunk in chunks}

    assert "Worker" in symbols
    assert "run" not in symbols
    assert "stop" not in symbols


def test_code_loader_excludes_venv_storage_git_and_node_modules(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "src").mkdir()
    (repo / ".venv").mkdir()
    (repo / "storage").mkdir()
    (repo / "node_modules").mkdir()
    (repo / ".git" / "hooks").mkdir()
    (repo / "src" / "app.py").write_text("def keep_me():\n    return 1\n", encoding="utf-8")
    (repo / ".venv" / "bad.py").write_text("def skip_venv(): pass\n", encoding="utf-8")
    (repo / "storage" / "bad.py").write_text("def skip_storage(): pass\n", encoding="utf-8")
    (repo / "node_modules" / "bad.py").write_text("def skip_node(): pass\n", encoding="utf-8")
    (repo / ".git" / "hooks" / "bad.py").write_text("def skip_git(): pass\n", encoding="utf-8")

    result = load_code_documents(
        {
            "code_retrieval": {
                "enabled": True,
                "default_profile": "projects",
                "profiles": {"projects": {"paths": [str(repo)], "excluded": []}},
                "allowed_extensions": [".py"],
            }
        },
        max_chars=2000,
        overlap_chars=0,
    )

    assert {doc.metadata["file_path"] for doc in result.documents} == {"src/app.py"}
    assert any(skipped == ".venv/bad.py" for skipped in result.skipped)
    assert any(skipped == "storage/bad.py" for skipped in result.skipped)
    assert any(skipped == "node_modules/bad.py" for skipped in result.skipped)
    assert any(skipped == ".git/hooks/bad.py" for skipped in result.skipped)


def test_code_bm25_retrieves_symbol_from_metadata(tmp_path):
    repo = tmp_path / "repo"
    path = repo / "src" / "service.py"
    path.parent.mkdir(parents=True)
    path.write_text("def unrelated_name():\n    return 'body text'\n", encoding="utf-8")
    chunks = chunk_python_code(
        path,
        repo_name="repo",
        relative_path="src/service.py",
        max_chars=2000,
        overlap_chars=0,
    )
    docs = []
    for chunk in chunks:
        doc = chunk.text
        if chunk.metadata.get("symbol_name") == "unrelated_name":
            chunk.metadata["symbol_name"] = "exact_symbol_only_in_metadata"
        docs.append((doc, dict(chunk.metadata)))

    from llama_index.core import Document

    index_path = tmp_path / "code_bm25.jsonl"
    write_bm25_index(
        index_path,
        [Document(text=text, metadata=metadata) for text, metadata in docs],
    )

    results = bm25_retrieve(
        "exact_symbol_only_in_metadata",
        load_bm25_index(index_path),
        top_k=3,
    )

    assert results[0].metadata["symbol_name"] == "exact_symbol_only_in_metadata"
    assert results[0].metadata["file_path"] == "src/service.py"


def test_infer_file_role():
    assert infer_file_role("tests/test_app.py") == "test"
    assert infer_file_role("pyproject.toml") == "config"
    assert infer_file_role("docs/guide.md") == "docs"
    assert infer_file_role("scripts/run.sh") == "scripts"
    assert infer_file_role("src/app.py") == "source"


def test_code_retrieval_prefers_source_over_docs_for_similar_scores():
    docs = Candidate(
        chunk_id="docs",
        text="docs",
        metadata={"file_role": "docs", "fused_score": 0.02},
        score=0.02,
        source="bm25",
        rank=1,
        fused_score=0.02,
        retrieval_sources={"bm25"},
    )
    source = Candidate(
        chunk_id="source",
        text="source",
        metadata={"file_role": "source", "fused_score": 0.018},
        score=0.018,
        source="bm25",
        rank=2,
        fused_score=0.018,
        retrieval_sources={"bm25"},
    )

    ranked = prefer_source_code_candidates([docs, source])

    assert [candidate.chunk_id for candidate in ranked] == ["source", "docs"]


def test_code_retrieval_boosts_source_exact_query_terms():
    docs = Candidate(
        chunk_id="docs",
        text="server tools overview",
        metadata={"file_role": "docs", "fused_score": 0.03},
        score=0.03,
        source="bm25",
        rank=1,
        fused_score=0.03,
        retrieval_sources={"bm25"},
    )
    source = Candidate(
        chunk_id="source",
        text='mcp = FastMCP("personal-assistant")',
        metadata={"file_role": "source", "fused_score": 0.01, "symbol_name": "mcp"},
        score=0.01,
        source="bm25",
        rank=2,
        fused_score=0.01,
        retrieval_sources={"bm25"},
    )

    ranked = prefer_source_code_candidates([docs, source], query="FastMCP server tools")

    assert [candidate.chunk_id for candidate in ranked] == ["source", "docs"]


def _role_candidate(role: str, score: float = 0.01) -> Candidate:
    return Candidate(
        chunk_id=role,
        text=role,
        metadata={"file_role": role, "fused_score": score},
        score=score,
        source="bm25",
        rank=1,
        fused_score=score,
        retrieval_sources={"bm25"},
    )


def test_test_query_intent_ranks_tests_above_source():
    ranked = prefer_source_code_candidates(
        [_role_candidate("source"), _role_candidate("test")],
        query="pytest fixture behavior",
    )

    assert [candidate.chunk_id for candidate in ranked] == ["test", "source"]


def test_config_query_intent_ranks_config_above_source():
    ranked = prefer_source_code_candidates(
        [_role_candidate("source"), _role_candidate("config")],
        query="pyproject yaml config",
    )

    assert [candidate.chunk_id for candidate in ranked] == ["config", "source"]


def test_docs_query_intent_ranks_docs_above_source():
    ranked = prefer_source_code_candidates(
        [_role_candidate("source"), _role_candidate("docs")],
        query="architecture readme guide",
    )

    assert [candidate.chunk_id for candidate in ranked] == ["docs", "source"]


def test_script_query_intent_ranks_scripts_above_source():
    ranked = prefer_source_code_candidates(
        [_role_candidate("source"), _role_candidate("scripts")],
        query="bash launcher script",
    )

    assert [candidate.chunk_id for candidate in ranked] == ["scripts", "source"]
