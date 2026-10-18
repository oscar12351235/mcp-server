from __future__ import annotations

import builtins

import pytest

from mcp_server.retrieval.embeddings import (
    build_embedding_backend,
    embedding_profile_warnings,
)


def test_build_local_huggingface_backend_for_bge_without_voyage_key(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)

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


def test_build_voyage_backend_without_loading_model():
    backend = build_embedding_backend(
        "voyage_code_3",
        {
            "provider": "voyage",
            "model": "voyage-code-3",
            "dimension": 1024,
        },
    )

    assert backend.name == "voyage_code_3"
    assert backend.provider == "voyage"
    assert backend.model == "voyage-code-3"


def test_voyage_backend_requires_api_key_when_loaded(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    backend = build_embedding_backend(
        "voyage_code_3",
        {
            "provider": "voyage",
            "model": "voyage-code-3",
            "dimension": 1024,
        },
    )

    with pytest.raises(RuntimeError, match="VOYAGE_API_KEY is missing"):
        backend.build_llamaindex_embedding()


def test_voyage_backend_creation_does_not_import_voyageai(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "voyageai":
            raise AssertionError("voyageai should not be imported during backend creation")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    backend = build_embedding_backend(
        "voyage_code_3",
        {
            "provider": "voyage",
            "model": "voyage-code-3",
            "dimension": 1024,
        },
    )

    assert backend.provider == "voyage"


def test_unsupported_embedding_provider_has_actionable_error():
    with pytest.raises(ValueError, match="Supported providers: local_huggingface, voyage"):
        build_embedding_backend(
            "unknown",
            {
                "provider": "unknown",
                "model": "unknown-model",
                "dimension": 1024,
            },
        )


def test_qwen8b_warning_is_explicit():
    warnings = embedding_profile_warnings({"model": "Qwen/Qwen3-Embedding-8B"})

    assert any("8B" in warning and "heavy" in warning for warning in warnings)


def test_voyage_warning_is_explicit():
    warnings = embedding_profile_warnings({"provider": "voyage", "model": "voyage-code-3"})

    assert any("cloud API" in warning for warning in warnings)
