from __future__ import annotations

from pathlib import Path

from mcp_server.retrieval.chunkers import chunk_file, chunks_to_documents

FIXTURES = Path(__file__).parent / "fixtures"


def test_markdown_chunker_preserves_heading_path():
    chunks = chunk_file(FIXTURES / "markdown" / "sample.md")

    section_paths = {chunk.metadata.get("section_path") for chunk in chunks}

    assert "Black-Scholes Notes > Greeks > Delta" in section_paths
    assert "Black-Scholes Notes > Volatility" in section_paths

    delta_chunk = next(
        chunk
        for chunk in chunks
        if chunk.metadata.get("section_path") == "Black-Scholes Notes > Greeks > Delta"
    )
    assert delta_chunk.metadata["source_type"] == "markdown"
    assert delta_chunk.metadata["chunk_type"] == "markdown_section"
    assert delta_chunk.metadata["extension"] == ".md"
    assert delta_chunk.metadata["chunk_index"] >= 0
    assert delta_chunk.metadata["content_hash"]


def test_python_chunker_extracts_function_and_class_names():
    chunks = chunk_file(FIXTURES / "code" / "sample.py")

    symbols = {
        chunk.metadata.get("symbol_name"): chunk.metadata
        for chunk in chunks
        if chunk.metadata.get("symbol_name")
    }

    assert symbols["load_config"]["symbol_type"] == "function"
    assert symbols["Worker"]["symbol_type"] == "class"
    assert symbols["load_config"]["source_type"] == "code"
    assert symbols["load_config"]["language"] == "python"
    assert symbols["load_config"]["start_line"] > 0
    assert symbols["load_config"]["end_line"] >= symbols["load_config"]["start_line"]


def test_generic_chunker_limits_size():
    chunks = chunk_file(FIXTURES / "text" / "sample.txt", max_chars=90, overlap_chars=10)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 130 for chunk in chunks)
    assert all(chunk.metadata["chunk_type"] == "text_block" for chunk in chunks)
    assert all(chunk.metadata["source_type"] == "text" for chunk in chunks)


def test_config_chunker_preserves_key_paths(tmp_path):
    config_path = tmp_path / "sample.yaml"
    config_path.write_text(
        "retrieval:\n  dense_top_k: 5\n  nested:\n    enabled: true\n",
        encoding="utf-8",
    )

    chunks = chunk_file(config_path)
    key_paths = {chunk.metadata.get("key_path") for chunk in chunks}

    assert "retrieval.dense_top_k" in key_paths
    assert "retrieval.nested.enabled" in key_paths
    assert all(chunk.metadata["source_type"] == "config" for chunk in chunks)


def test_chunks_convert_to_documents():
    chunks = chunk_file(FIXTURES / "markdown" / "sample.md")
    documents = chunks_to_documents(chunks)

    assert len(documents) == len(chunks)
    assert documents[0].text == chunks[0].text
    assert documents[0].metadata["file_path"].endswith("sample.md")
    assert documents[0].metadata["chunk_id"] == chunks[0].chunk_id
