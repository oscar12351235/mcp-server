from __future__ import annotations

import importlib
import sys

EXPECTED_TOOLS = {
    "collection_status",
    "knowledge_profile_status",
    "index_status",
    "refresh_indexes",
    "rebuild_indexes",
    "retrieve_skills",
    "retrieve_knowledge",
    "retrieve_memory",
    "read_local_file",
    "list_local_files",
    "get_gitlab_file",
    "list_gitlab_repo",
    "propose_memory_update",
    "propose_new_skill",
}


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
    assert status["knowledge"]["collection"] == "knowledge_indexed"


def test_full_profile_selects_full_paths_and_collection(monkeypatch):
    server = import_server_with_profile(monkeypatch, "full")

    assert server.KNOWLEDGE_PROFILE == "full"
    assert server._get_knowledge_paths() == ["/Users/oscarbonnet/Documents/80–89 Knowledge Base"]

    status = server.collection_status()

    assert status["knowledge"]["profile"] == "full"
    assert status["knowledge"]["collection"] == "knowledge_full"


def test_expected_tools_are_registered(monkeypatch):
    server = import_server_with_profile(monkeypatch, "indexed")

    registered_tools = set(server.mcp._tool_manager._tools)

    assert EXPECTED_TOOLS <= registered_tools
