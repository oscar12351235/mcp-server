from __future__ import annotations

import importlib
import sys

EXPECTED_TOOLS = {
    "collection_status",
    "embedding_config_status",
    "embedding_status",
    "config_status",
    "knowledge_profile_status",
    "index_status",
    "refresh_indexes",
    "rebuild_indexes",
    "scan_index_changes",
    "list_indexed_files",
    "list_changed_files",
    "retrieve_skills",
    "retrieve_knowledge",
    "retrieve_knowledge_debug",
    "retrieve_code",
    "list_code_repos",
    "code_index_status",
    "read_code_symbol",
    "retrieve_memory",
    "read_local_file",
    "list_local_files",
    "get_gitlab_file",
    "list_gitlab_repo",
    "propose_memory_update",
    "propose_new_skill",
}


def expected_knowledge_collection(server, profile: str) -> str:
    return (
        f"knowledge_{profile}_local_bge_small_local_huggingface_"
        f"baai_bge_small_en_v1_5_384_structure_v1_{server._retrieval_config_fingerprint()}"
    )


def import_server_with_profile(monkeypatch, profile: str):
    monkeypatch.setenv("MCP_KNOWLEDGE_PROFILE", profile)
    sys.modules.pop("mcp_server.mcp_server", None)
    return importlib.import_module("mcp_server.mcp_server")


def test_indexed_profile_selects_indexed_paths_and_collection(monkeypatch):
    server = import_server_with_profile(monkeypatch, "indexed")

    assert server.KNOWLEDGE_PROFILE == "indexed"
    assert server._get_knowledge_paths() == [
        "/Users/oscarbonnet/Documents/80–89 Knowledge Base/87 Indexed"
    ]

    status = server.collection_status()

    assert status["knowledge"]["profile"] == "indexed"
    assert status["knowledge"]["collection"] == expected_knowledge_collection(server, "indexed")


def test_full_profile_selects_full_paths_and_collection(monkeypatch):
    server = import_server_with_profile(monkeypatch, "full")

    assert server.KNOWLEDGE_PROFILE == "full"
    assert server._get_knowledge_paths() == ["/Users/oscarbonnet/Documents/80–89 Knowledge Base"]

    status = server.collection_status()

    assert status["knowledge"]["profile"] == "full"
    assert status["knowledge"]["collection"] == expected_knowledge_collection(server, "full")


def test_expected_tools_are_registered(monkeypatch):
    server = import_server_with_profile(monkeypatch, "indexed")

    registered_tools = set(server.mcp._tool_manager._tools)

    assert EXPECTED_TOOLS <= registered_tools


def test_embedding_config_status_reports_active_profile(monkeypatch):
    server = import_server_with_profile(monkeypatch, "indexed")

    status = server.embedding_config_status()

    assert status["active_profile"] == "local_bge_small"
    assert status["embedding_profile"] == "local_bge_small"
    assert status["provider"] == "local_huggingface"
    assert status["model"] == "BAAI/bge-small-en-v1.5"
    assert status["dimension"] == 384
    assert status["chunking_profile"] == "structure_v1"
    assert len(status["embedding_fingerprint"]) == 16
    assert len(status["retrieval_fingerprint"]) == 16
    assert status["collections"]["knowledge"].startswith("knowledge_indexed_local_bge_small_")


def test_changing_embedding_model_changes_collection_name(monkeypatch):
    server = import_server_with_profile(monkeypatch, "indexed")
    original_name = server._collection_name("knowledge")

    monkeypatch.setitem(
        server.CONFIG["embedding"]["profiles"],
        "alternate",
        {
            "provider": "local_huggingface",
            "model": "BAAI/bge-base-en-v1.5",
            "dimension": 768,
            "normalize": True,
        },
    )
    monkeypatch.setitem(server.CONFIG["embedding"], "active_profile", "alternate")

    changed_name = server._collection_name("knowledge")

    assert changed_name != original_name
    assert "bge_base_en_v1_5" in changed_name
    assert "768" in changed_name


def test_changing_only_chunking_config_changes_collection_name(monkeypatch):
    server = import_server_with_profile(monkeypatch, "indexed")
    original_name = server._collection_name("knowledge")

    monkeypatch.setitem(
        server.CONFIG["chunking"]["profiles"]["structure_v1"],
        "chunk_overlap",
        200,
    )

    changed_name = server._collection_name("knowledge")

    assert changed_name != original_name
    assert changed_name.startswith(
        "knowledge_indexed_local_bge_small_local_huggingface_"
        "baai_bge_small_en_v1_5_384_structure_v1_"
    )


def test_changing_only_embedding_normalize_changes_collection_name(monkeypatch):
    server = import_server_with_profile(monkeypatch, "indexed")
    original_name = server._collection_name("knowledge")

    monkeypatch.setitem(
        server.CONFIG["embedding"]["profiles"]["local_bge_small"],
        "normalize",
        False,
    )

    changed_name = server._collection_name("knowledge")

    assert changed_name != original_name
    assert changed_name.startswith(
        "knowledge_indexed_local_bge_small_local_huggingface_"
        "baai_bge_small_en_v1_5_384_structure_v1_"
    )


def test_reranker_final_top_k_is_default_when_enabled(monkeypatch):
    server = import_server_with_profile(monkeypatch, "indexed")

    monkeypatch.setitem(server.CONFIG["reranker"], "enabled", True)
    monkeypatch.setitem(server.CONFIG["reranker"], "final_top_k", 6)
    monkeypatch.setitem(server.CONFIG["retrieval"], "final_top_k", 11)

    assert server._knowledge_final_top_k(None) == 6


def test_explicit_top_k_overrides_reranker_final_top_k(monkeypatch):
    server = import_server_with_profile(monkeypatch, "indexed")

    monkeypatch.setitem(server.CONFIG["reranker"], "enabled", True)
    monkeypatch.setitem(server.CONFIG["reranker"], "final_top_k", 6)

    assert server._knowledge_final_top_k(4) == 4


def test_retrieval_final_top_k_is_default_when_reranker_disabled(monkeypatch):
    server = import_server_with_profile(monkeypatch, "indexed")

    monkeypatch.setitem(server.CONFIG["reranker"], "enabled", False)
    monkeypatch.setitem(server.CONFIG["reranker"], "final_top_k", 6)
    monkeypatch.setitem(server.CONFIG["retrieval"], "final_top_k", 11)

    assert server._knowledge_final_top_k(None) == 11


def test_qwen3_4b_profile_changes_collection_name(monkeypatch):
    server = import_server_with_profile(monkeypatch, "indexed")

    monkeypatch.setitem(server.CONFIG["embedding"], "active_profile", "local_qwen3_4b")

    collection_name = server._collection_name("knowledge")
    status = server.embedding_config_status()

    assert "local_qwen3_4b" in collection_name
    assert "qwen_qwen3_embedding_4b" in collection_name
    assert "2560" in collection_name
    assert status["active_profile"] == "local_qwen3_4b"
    assert status["model"] == "Qwen/Qwen3-Embedding-4B"
    assert status["dimension"] == 2560
    assert status["collections"]["knowledge"] == collection_name


def test_qwen3_8b_profile_reports_heavy_model_warning(monkeypatch):
    server = import_server_with_profile(monkeypatch, "indexed")

    monkeypatch.setitem(server.CONFIG["embedding"], "active_profile", "local_qwen3_8b")

    status = server.embedding_status()

    assert status["active_profile"] == "local_qwen3_8b"
    assert any("8B" in warning and "heavy" in warning for warning in status["warnings"])


def test_changing_include_full_class_chunks_changes_code_identity(monkeypatch):
    server = import_server_with_profile(monkeypatch, "indexed")
    original_collection = server._code_collection_name()
    original_bm25_path = server._bm25_index_path(original_collection)
    original_fingerprint = server._code_retrieval_fingerprint()

    monkeypatch.setitem(
        server.CONFIG["code_retrieval"]["chunking"],
        "include_full_class_chunks",
        True,
    )

    changed_collection = server._code_collection_name()

    assert changed_collection != original_collection
    assert server._bm25_index_path(changed_collection) != original_bm25_path
    assert server._code_retrieval_fingerprint() != original_fingerprint


def test_changing_include_method_chunks_changes_code_identity(monkeypatch):
    server = import_server_with_profile(monkeypatch, "indexed")
    original_collection = server._code_collection_name()
    original_bm25_path = server._bm25_index_path(original_collection)
    original_fingerprint = server._code_retrieval_fingerprint()

    monkeypatch.setitem(
        server.CONFIG["code_retrieval"]["chunking"],
        "include_method_chunks",
        False,
    )

    changed_collection = server._code_collection_name()
    status = server.code_index_status()

    assert changed_collection != original_collection
    assert server._bm25_index_path(changed_collection) != original_bm25_path
    assert status["fingerprint"] != original_fingerprint
    assert status["chunking"]["include_method_chunks"] is False
