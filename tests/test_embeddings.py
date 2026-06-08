from __future__ import annotations

import pytest

from mcp_server.retrieval.embeddings import (
    build_embedding_backend,
    embedding_profile_warnings,
)


def test_build_local_huggingface_backend_for_bge_without_loading_model():
    backend = build_embedding_backend(
        "local_bge_small",
        {
            "provider": "local_huggingface",
            "model": "BAAI/bge-small-en-v1.5",
            "dimension": 384,
            "normalize": True,
        },
    )

    assert backend.name == "local_bge_small"
    assert backend.provider == "local_huggingface"
    assert backend.model == "BAAI/bge-small-en-v1.5"
    assert backend.dimension == 384


def test_build_local_huggingface_backend_for_qwen_without_loading_model():
    backend = build_embedding_backend(
        "local_qwen3_4b",
        {
            "provider": "local_huggingface",
            "model": "Qwen/Qwen3-Embedding-4B",
            "dimension": 2560,
            "normalize": True,
            "instruction": "Represent this document for retrieval.",
        },
    )

    assert backend.name == "local_qwen3_4b"
    assert backend.model == "Qwen/Qwen3-Embedding-4B"
    assert backend.dimension == 2560


def test_unsupported_embedding_provider_has_actionable_error():
    with pytest.raises(ValueError, match="Voyage remains a future optional backend"):
        build_embedding_backend(
            "voyage",
            {
                "provider": "voyage",
                "model": "voyage-code-3",
                "dimension": 1024,
            },
        )


def test_qwen8b_warning_is_explicit():
    warnings = embedding_profile_warnings({"model": "Qwen/Qwen3-Embedding-8B"})

    assert any("8B" in warning and "heavy" in warning for warning in warnings)
