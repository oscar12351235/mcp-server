from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import chromadb
import requests
import yaml
from dotenv import load_dotenv
from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.llms import MockLLM
from llama_index.vector_stores.chroma import ChromaVectorStore
from mcp.server.fastmcp import FastMCP

from mcp_server.retrieval.bm25_store import (
    bm25_retrieve,
    load_bm25_index,
    write_bm25_index,
)
from mcp_server.retrieval.chunkers import chunk_file, chunk_markdown, chunks_to_documents
from mcp_server.retrieval.code_loader import configured_code_repos, load_code_documents
from mcp_server.retrieval.code_retrieval import (
    find_symbol_candidate,
    format_code_candidates,
    retrieve_code_candidates,
)
from mcp_server.retrieval.embeddings import (
    build_embedding_backend,
    embedding_profile_warnings,
)
from mcp_server.retrieval.hybrid import Candidate, fuse_candidates
from mcp_server.retrieval.manifest import (
    detect_changes,
    get_files_for_profile,
    init_manifest_db,
    manifest_chunk_from_metadata,
    manifest_file_from_scan,
    scan_paths,
    upsert_chunk,
    upsert_file,
)
from mcp_server.retrieval.rerankers import rerank_or_fallback
from mcp_server.retrieval.token_budget import (
    check_token_budget,
    estimate_tokens,
    new_run_id,
    record_token_ledger,
    token_budget_from_config,
    token_ledger_summary,
)

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

# Important for MCP stdio:
# never print normal logs to stdout, because stdout is used by the MCP protocol.
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="[%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]

load_dotenv(PROJECT_ROOT / ".env")

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    required_sections = ["gitlab", "knowledge", "memory", "retrieval", "security"]
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ValueError(f"Missing required config sections: {', '.join(missing_sections)}")

    retrieval = config["retrieval"]
    if "defaults" not in retrieval or "limits" not in retrieval:
        raise ValueError("retrieval.defaults and retrieval.limits are required")

    embedding = config.setdefault("embedding", {})
    profiles = embedding.setdefault("profiles", {})

    if not profiles:
        legacy_model = retrieval.get("embedding_model")
        if not legacy_model:
            raise ValueError("embedding.profiles or retrieval.embedding_model is required")

        profiles["legacy"] = {
            "provider": "local_huggingface",
            "model": legacy_model,
            "dimension": "default",
            "normalize": True,
        }
        embedding.setdefault("active_profile", "legacy")

    active_embedding_profile = embedding.get("active_profile")
    if not active_embedding_profile:
        raise ValueError("embedding.active_profile is required")
    if active_embedding_profile not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(
            f"Unknown embedding profile: {active_embedding_profile}. "
            f"Available profiles: {available}"
        )

    active_embedding = profiles[active_embedding_profile]
    for key in ("provider", "model"):
        if not active_embedding.get(key):
            raise ValueError(f"embedding profile {active_embedding_profile} requires {key}")

    provider = active_embedding["provider"]
    if provider not in {"local_huggingface", "voyage"}:
        raise ValueError(
            f"Embedding provider is configured but not implemented in this phase: {provider}. "
            "Supported providers: local_huggingface, voyage."
        )

    chunking = config.setdefault("chunking", {})
    chunking_profiles = chunking.setdefault(
        "profiles",
        {
            "generic_v1": {
                "strategy": "generic",
                "chunk_size": 1024,
                "chunk_overlap": 150,
            }
        },
    )
    chunking.setdefault("active_profile", "generic_v1")

    active_chunking_profile = chunking["active_profile"]
    if active_chunking_profile not in chunking_profiles:
        available = ", ".join(sorted(chunking_profiles))
        raise ValueError(
            f"Unknown chunking profile: {active_chunking_profile}. "
            f"Available profiles: {available}"
        )

    return config


CONFIG = _validate_config(CONFIG)

KNOWLEDGE_PROFILE = os.environ.get(
    "MCP_KNOWLEDGE_PROFILE",
    CONFIG["knowledge"].get("default_profile", "indexed"),
)

GITLAB_TOKEN = os.environ.get("GITLAB_TOKEN")
GITLAB_BASE_URL = os.environ.get(
    "GITLAB_BASE_URL",
    CONFIG["gitlab"]["base_url"],
).rstrip("/")


# ---------------------------------------------------------------------
# MCP app
# ---------------------------------------------------------------------
# creates the MCP server instance
mcp = FastMCP("personal-assistant")


# ---------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


# These functions prevent the MCP server from reading/writing arbitrary files or GitLab repos.
def _assert_allowed_read(path: str | Path) -> Path:
    resolved = _resolve(path)

    for forbidden in CONFIG["security"].get("never_read", []):
        forbidden_path = _resolve(forbidden)
        if resolved == forbidden_path or _is_inside(resolved, forbidden_path):
            raise PermissionError(f"Blocked by never_read policy: {resolved}")

    allowed_roots = [_resolve(p) for p in CONFIG["security"]["allowed_read_roots"]]

    if not any(resolved == root or _is_inside(resolved, root) for root in allowed_roots):
        raise PermissionError(f"Read path is not allowlisted: {resolved}")

    return resolved


def _assert_allowed_write(path: str | Path) -> Path:
    resolved = _resolve(path)

    for forbidden in CONFIG["security"].get("never_write", []):
        forbidden_path = _resolve(forbidden)
        if resolved == forbidden_path or _is_inside(resolved, forbidden_path):
            raise PermissionError(f"Blocked by never_write policy: {resolved}")

    allowed_roots = [_resolve(p) for p in CONFIG["security"]["allowed_write_roots"]]

    if not any(resolved == root or _is_inside(resolved, root) for root in allowed_roots):
        raise PermissionError(f"Write path is not allowlisted: {resolved}")

    return resolved


def _get_knowledge_paths(profile: str | None = None) -> list[str]:
    selected_profile = profile or KNOWLEDGE_PROFILE

    profiles = CONFIG["knowledge"].get("profiles")

    if profiles:
        if selected_profile not in profiles:
            available = ", ".join(sorted(profiles))
            raise ValueError(
                f"Unknown knowledge profile: {selected_profile}. "
                f"Available profiles: {available}"
            )

        paths = profiles[selected_profile].get("paths", [])

    else:
        # Backward compatibility with old config:
        # knowledge:
        #   paths:
        #     - ...
        paths = CONFIG["knowledge"].get("paths", [])

    if not paths:
        raise ValueError(f"No knowledge paths configured for profile: {selected_profile}")

    return [str(path) for path in paths]


def _collection_count(name: str) -> int:
    try:
        collection = chroma_client.get_collection(name)
        return collection.count()
    except Exception:
        return 0


def _safe_collection_component(text: str | int | bool | None) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", str(text or "default"))
    return clean.strip("_").lower() or "default"


def _active_embedding_profile_name() -> str:
    return str(CONFIG["embedding"]["active_profile"])


def _active_embedding_config() -> dict[str, Any]:
    profile = _active_embedding_profile_name()
    return dict(CONFIG["embedding"]["profiles"][profile])


def _active_embedding_provider() -> str:
    return str(_active_embedding_config().get("provider", ""))


def _active_chunking_profile_name() -> str:
    return str(CONFIG["chunking"]["active_profile"])


def _active_chunking_config() -> dict[str, Any]:
    profile = _active_chunking_profile_name()
    return dict(CONFIG["chunking"]["profiles"][profile])


def _embedding_config_fingerprint() -> str:
    payload = {
        "profile": _active_embedding_profile_name(),
        **_active_embedding_config(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _retrieval_config_fingerprint() -> str:
    payload = {
        "embedding_profile": _active_embedding_profile_name(),
        "embedding": _active_embedding_config(),
        "chunking_profile": _active_chunking_profile_name(),
        "chunking": _active_chunking_config(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _active_code_profile_config() -> dict[str, Any]:
    code_config = _code_retrieval_config()
    profiles = code_config.get("profiles", {})
    return dict(profiles.get(_active_code_profile_name(), {}))


def _active_code_chunking_config() -> dict[str, Any]:
    chunking = dict(_code_retrieval_config().get("chunking", {}))
    return {
        "include_full_class_chunks": bool(chunking.get("include_full_class_chunks", False)),
        "include_method_chunks": bool(chunking.get("include_method_chunks", True)),
    }


def _code_retrieval_fingerprint() -> str:
    code_config = _code_retrieval_config()
    code_profile = _active_code_profile_config()
    payload = {
        "embedding_profile": _active_embedding_profile_name(),
        "embedding": _active_embedding_config(),
        "chunking_profile": _active_chunking_profile_name(),
        "chunking": _active_chunking_config(),
        "code_profile": _active_code_profile_name(),
        "code_paths": code_profile.get("paths", []),
        "code_excluded": code_profile.get("excluded", code_config.get("excluded", [])),
        "code_allowed_extensions": code_config.get("allowed_extensions", []),
        "code_chunking": _active_code_chunking_config(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _collection_name(domain: str, profile: str | None = None) -> str:
    allowed_domains = {"skills", "knowledge", "memory", "code"}
    if domain not in allowed_domains:
        available = ", ".join(sorted(allowed_domains))
        raise ValueError(f"Unknown collection domain: {domain}. Available domains: {available}")

    embedding_cfg = _active_embedding_config()
    profile_component = profile or (KNOWLEDGE_PROFILE if domain == "knowledge" else "global")
    dimension = embedding_cfg.get("dimension", "default")

    parts = [
        domain,
        profile_component,
        _active_embedding_profile_name(),
        embedding_cfg["provider"],
        embedding_cfg["model"],
        dimension,
        _active_chunking_profile_name(),
        _retrieval_config_fingerprint(),
    ]

    return "_".join(_safe_collection_component(part) for part in parts)


def _code_collection_name() -> str:
    embedding_cfg = _active_embedding_config()
    parts = [
        "code",
        _active_code_profile_name(),
        _active_embedding_profile_name(),
        embedding_cfg["provider"],
        embedding_cfg["model"],
        embedding_cfg.get("dimension", "default"),
        _active_chunking_profile_name(),
        _code_retrieval_fingerprint(),
    ]
    return "_".join(_safe_collection_component(part) for part in parts)


def _collection_metadata(domain: str, profile: str | None = None) -> dict[str, str]:
    embedding_cfg = _active_embedding_config()
    return {
        "domain": domain,
        "knowledge_profile": profile or (KNOWLEDGE_PROFILE if domain == "knowledge" else "global"),
        "embedding_profile": _active_embedding_profile_name(),
        "embedding_provider": str(embedding_cfg["provider"]),
        "embedding_model": str(embedding_cfg["model"]),
        "embedding_dimension": str(embedding_cfg.get("dimension", "default")),
        "embedding_fingerprint": _embedding_config_fingerprint(),
        "retrieval_fingerprint": _retrieval_config_fingerprint(),
        "code_retrieval_fingerprint": (_code_retrieval_fingerprint() if domain == "code" else ""),
        "chunking_profile": _active_chunking_profile_name(),
    }


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "untitled"


def _bounded_top_k(value: int | None, default: int, maximum: int) -> int:
    if value is None:
        return default
    return max(1, min(value, maximum))


def _get_default_top_k(kind: str) -> int:
    return int(CONFIG["retrieval"]["defaults"][f"{kind}_top_k"])


def _get_max_top_k(kind: str) -> int:
    return int(CONFIG["retrieval"]["limits"][f"{kind}_top_k_max"])


def _bm25_enabled() -> bool:
    return bool(CONFIG["retrieval"].get("bm25_enabled", False))


def _code_retrieval_config() -> dict[str, Any]:
    return dict(CONFIG.get("code_retrieval", {}))


def _code_retrieval_enabled() -> bool:
    return bool(_code_retrieval_config().get("enabled", False))


def _active_code_profile_name() -> str:
    return str(_code_retrieval_config().get("default_profile", "projects"))


def _dense_top_k() -> int:
    return int(CONFIG["retrieval"].get("dense_top_k", _get_default_top_k("knowledge")))


def _bm25_top_k() -> int:
    return int(CONFIG["retrieval"].get("bm25_top_k", _get_default_top_k("knowledge")))


def _fused_top_k() -> int:
    return int(CONFIG["retrieval"].get("fused_top_k", _get_max_top_k("knowledge")))


def _fusion_method() -> str:
    return str(CONFIG["retrieval"].get("fusion", "rrf"))


def _reranker_config() -> dict[str, Any]:
    return dict(CONFIG.get("reranker", {}))


def _reranker_enabled() -> bool:
    return bool(_reranker_config().get("enabled", False))


def _reranker_provider() -> str:
    return str(_reranker_config().get("provider", "local_huggingface"))


def _reranker_model() -> str:
    return str(_reranker_config().get("model", "BAAI/bge-reranker-v2-m3"))


def _reranker_candidate_top_k() -> int:
    return int(_reranker_config().get("candidate_top_k", _fused_top_k()))


def _reranker_final_top_k() -> int:
    return int(_reranker_config().get("final_top_k", CONFIG["retrieval"].get("final_top_k", 8)))


def _reranker_batch_size() -> int:
    return int(_reranker_config().get("batch_size", 8))


def _retrieval_final_top_k() -> int:
    return int(CONFIG["retrieval"].get("final_top_k", _get_default_top_k("knowledge")))


def _default_final_top_k() -> int:
    if _reranker_enabled():
        return _reranker_final_top_k()
    return _retrieval_final_top_k()


def _knowledge_final_top_k(top_k: int | None = None) -> int:
    maximum = _get_max_top_k("knowledge")
    return _bounded_top_k(top_k, min(_default_final_top_k(), maximum), maximum)


def _reranker_status() -> dict[str, Any]:
    return {
        "enabled": _reranker_enabled(),
        "provider": _reranker_provider(),
        "model": _reranker_model(),
        "candidate_top_k": _reranker_candidate_top_k(),
        "final_top_k": _reranker_final_top_k(),
        "batch_size": _reranker_batch_size(),
    }


def _is_voyage_active() -> bool:
    return _active_embedding_provider() == "voyage"


def _voyage_config() -> dict[str, Any]:
    return dict(CONFIG.get("voyage", {}))


def _estimate_changed_embedding_tokens(profile: str | None = None) -> dict[str, Any]:
    status = scan_index_changes(profile)
    changed_files = (
        status["added"]
        + status["edited"]
        + status.get("config_changed", [])
        + [item for item in status.get("moved", []) if item.get("new_path")]
    )
    estimated_tokens_total = 0
    estimated_files: list[dict[str, Any]] = []
    for item in changed_files:
        path = item.get("path") or item.get("new_path")
        if not path:
            continue
        try:
            chunks = chunk_file(
                path,
                max_chars=_chunking_limits()[0],
                overlap_chars=_chunking_limits()[1],
                base_metadata={"chunking_profile": _active_chunking_profile_name()},
            )
            tokens = sum(estimate_tokens(chunk.text) for chunk in chunks)
            chunk_count = len(chunks)
        except Exception:
            try:
                tokens = max(1, Path(path).stat().st_size // 4)
            except Exception:
                tokens = 0
            chunk_count = 1 if tokens else 0

        estimated_tokens_total += tokens
        estimated_files.append(
            {
                "path": path,
                "status": item.get("status"),
                "estimated_tokens": tokens,
                "estimated_chunks": chunk_count,
            }
        )

    ledger = token_ledger_summary(MANIFEST_DB_PATH)
    budget = token_budget_from_config(CONFIG)
    budget_check = check_token_budget(
        estimated_tokens=estimated_tokens_total,
        historical_tokens=int(ledger["total_tokens_actual"]),
        budget=budget,
    )
    return {
        "profile": status["profile"],
        "provider": _active_embedding_provider(),
        "embedding_profile": _active_embedding_profile_name(),
        "model": _active_embedding_config().get("model"),
        "collection_name": status["collection_name"],
        "estimated_tokens": estimated_tokens_total,
        "estimated_files": estimated_files,
        "counts": status["counts"],
        "budget": budget_check,
        "dry_run": True,
    }


def _bm25_index_path(collection_name: str) -> Path:
    return BM25_PATH / f"{_safe_collection_component(collection_name)}.jsonl"


def _assert_allowed_gitlab_project(project_id: str) -> str:
    """
    Only allow GitLab access to explicitly configured projects.

    By default, this allows only the configured skills repo.
    You may add `security.allowed_gitlab_projects` in config.yaml later.
    """
    configured_skills_project = str(CONFIG["gitlab"]["skills"]["project_id"])

    allowed = CONFIG["security"].get(
        "allowed_gitlab_projects",
        [configured_skills_project],
    )

    allowed_as_str = {str(p) for p in allowed}

    if str(project_id) not in allowed_as_str:
        raise PermissionError(f"GitLab project is not allowlisted: {project_id}")

    return str(project_id)


def _gitlab_project_api_id(project_id: str) -> str:
    """
    GitLab accepts either:
    - numeric project ID, e.g. "123456789"
    - URL-encoded full path, e.g. "oscar12351235%2Fskills"

    If the user accidentally gives "oscar12351235/skills",
    encode it safely.
    """
    project_id = str(project_id)

    if re.fullmatch(r"\d+", project_id):
        return project_id

    if "%2f" in project_id.lower():
        return project_id

    return quote(project_id, safe="")


# ---------------------------------------------------------------------
# GitLab helpers
# ---------------------------------------------------------------------


def _gitlab_headers() -> dict[str, str]:
    if not GITLAB_TOKEN:
        raise RuntimeError("GITLAB_TOKEN is missing from .env")

    return {"PRIVATE-TOKEN": GITLAB_TOKEN}


def _gitlab_url(path: str) -> str:
    return f"{GITLAB_BASE_URL}/{path.lstrip('/')}"


def _gitlab_get(path: str, params: dict[str, Any] | None = None) -> Any:
    response = requests.get(
        _gitlab_url(path),
        headers=_gitlab_headers(),
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _gitlab_post(
    path: str, json: dict[str, Any] | None = None, data: dict[str, Any] | None = None
) -> Any:
    response = requests.post(
        _gitlab_url(path),
        headers=(
            {**_gitlab_headers(), "Content-Type": "application/json"} if json else _gitlab_headers()
        ),
        json=json,
        data=data,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _gitlab_put(path: str, json: dict[str, Any]) -> Any:
    response = requests.put(
        _gitlab_url(path),
        headers={**_gitlab_headers(), "Content-Type": "application/json"},
        json=json,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _list_gitlab_tree_raw(
    project_id: str,
    repo_path: str = "",
    branch: str = "main",
) -> list[dict[str, Any]]:
    project_id = _assert_allowed_gitlab_project(project_id)
    api_project_id = _gitlab_project_api_id(project_id)

    results: list[dict[str, Any]] = []
    page = 1

    while True:
        params: dict[str, str | int] = {
            "path": repo_path,
            "ref": branch,
            "recursive": "true",
            "per_page": 100,
            "page": page,
        }
        response = requests.get(
            _gitlab_url(f"projects/{api_project_id}/repository/tree"),
            headers=_gitlab_headers(),
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        batch = response.json()
        results.extend(batch)

        next_page = response.headers.get("X-Next-Page")
        if not next_page:
            break

        page = int(next_page)

    return results


def _read_gitlab_file_raw(
    project_id: str,
    file_path: str,
    branch: str = "main",
) -> str:
    project_id = _assert_allowed_gitlab_project(project_id)
    api_project_id = _gitlab_project_api_id(project_id)

    encoded_path = quote(file_path, safe="")

    data = _gitlab_get(
        f"projects/{api_project_id}/repository/files/{encoded_path}",
        params={"ref": branch},
    )

    content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    return content


def _create_gitlab_branch(project_id: str, branch: str, ref: str) -> None:
    project_id = _assert_allowed_gitlab_project(project_id)
    api_project_id = _gitlab_project_api_id(project_id)

    response = requests.post(
        _gitlab_url(f"projects/{api_project_id}/repository/branches"),
        headers=_gitlab_headers(),
        data={"branch": branch, "ref": ref},
        timeout=30,
    )

    if response.status_code in (200, 201):
        return

    # 400 can mean branch already exists. For our timestamped branches this is unlikely,
    # but we keep this tolerant to avoid breaking harmless reruns.
    if response.status_code == 400 and "already exists" in response.text.lower():
        return

    response.raise_for_status()


# ---------------------------------------------------------------------
# LlamaIndex / Chroma setup
# ---------------------------------------------------------------------

# We only use LlamaIndex for retrieval, not answer generation.
Settings.llm = MockLLM()

CHROMA_PATH = PROJECT_ROOT / "storage" / "chroma"
CHROMA_PATH.mkdir(parents=True, exist_ok=True)
MANIFEST_DB_PATH = PROJECT_ROOT / "storage" / "index_manifest.sqlite"
BM25_PATH = PROJECT_ROOT / "storage" / "bm25"
BM25_PATH.mkdir(parents=True, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

INDEXABLE_SUFFIXES = {
    ".md",
    ".markdown",
    ".txt",
    ".pdf",
    ".py",
    ".yaml",
    ".yml",
    ".epub",
    ".docx",
    ".json",
    ".html",
}

IGNORED_INDEX_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "storage",
}


skills_index: VectorStoreIndex | None = None
knowledge_index: VectorStoreIndex | None = None
memory_index: VectorStoreIndex | None = None
code_index: VectorStoreIndex | None = None
INDEX_STATUS: dict[str, str] = {
    "skills": "not built yet",
    "knowledge": "not built yet",
    "memory": "not built yet",
    "code": "not built yet",
}
_EMBEDDING_MODEL_READY = False
_EMBEDDING_MODEL_FINGERPRINT: str | None = None


def _ensure_embedding_model() -> None:
    global _EMBEDDING_MODEL_FINGERPRINT, _EMBEDDING_MODEL_READY

    fingerprint = _embedding_config_fingerprint()
    if _EMBEDDING_MODEL_READY and _EMBEDDING_MODEL_FINGERPRINT == fingerprint:
        return

    backend = build_embedding_backend(
        _active_embedding_profile_name(),
        _active_embedding_config(),
    )
    Settings.embed_model = backend.build_llamaindex_embedding()
    _EMBEDDING_MODEL_READY = True
    _EMBEDDING_MODEL_FINGERPRINT = fingerprint


def _chunking_limits() -> tuple[int, int]:
    chunking_cfg = _active_chunking_config()
    max_chars = int(
        chunking_cfg.get(
            "chunk_size_chars",
            chunking_cfg.get("chunk_size", 3000),
        )
    )
    overlap_chars = int(
        chunking_cfg.get(
            "chunk_overlap_chars",
            chunking_cfg.get("chunk_overlap", 300),
        )
    )
    return max_chars, overlap_chars


def _get_collection(
    name: str,
    force_rebuild: bool = False,
    metadata: dict[str, str] | None = None,
):
    if force_rebuild:
        try:
            chroma_client.delete_collection(name)
            logger.info("Deleted existing Chroma collection: %s", name)
        except Exception:
            pass

    return chroma_client.get_or_create_collection(name, metadata=metadata)


def _build_index(
    collection_name: str,
    docs: list[Document],
    force_rebuild: bool = False,
    collection_metadata: dict[str, str] | None = None,
) -> VectorStoreIndex:
    collection = _get_collection(
        collection_name,
        force_rebuild=force_rebuild,
        metadata=collection_metadata,
    )
    existing_count = collection.count()

    if _is_voyage_active() and not docs and existing_count == 0:
        raise RuntimeError(
            "Refusing to create placeholder vectors with a Voyage embedding profile. "
            "Run index_changes(apply=False) to estimate cost, then "
            "index_changes(apply=True, confirm=True) for explicit Voyage indexing."
        )

    _ensure_embedding_model()

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if existing_count > 0 and not force_rebuild:
        logger.info(
            "Using existing Chroma collection '%s' with %s vectors.",
            collection_name,
            existing_count,
        )
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
        )

    if not docs:
        docs = [
            Document(
                text="No documents loaded.",
                metadata={"source": "empty", "collection": collection_name},
            )
        ]

    logger.info(
        "Building Chroma collection '%s' from %s documents.",
        collection_name,
        len(docs),
    )

    return VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        show_progress=False,
    )


def _load_single_local_file(path: Path, source_type: str) -> list[Document]:
    max_chars, overlap_chars = _chunking_limits()
    base_metadata = {
        "source_group": source_type,
        "chunking_profile": _active_chunking_profile_name(),
    }

    try:
        chunks = chunk_file(
            path,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            base_metadata=base_metadata,
        )
        return chunks_to_documents(chunks)

    except Exception as exc:
        logger.warning(
            "Structure-aware chunking failed for %s; falling back to SimpleDirectoryReader: %s",
            path,
            exc,
        )

    try:
        loaded = SimpleDirectoryReader(input_files=[str(path)]).load_data()
    except Exception as exc:
        logger.warning("Could not load file %s: %s", path, exc)
        return []

    for index, doc in enumerate(loaded):
        text = doc.get_content()
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        doc.metadata.update(
            {
                "source_type": source_type,
                "source_group": source_type,
                "file_path": str(path),
                "extension": path.suffix.lower(),
                "chunk_index": index,
                "chunk_type": "reader_fallback",
                "content_hash": content_hash,
                "text_hash": content_hash,
                "chunking_profile": _active_chunking_profile_name(),
            }
        )

    return loaded


def _load_local_documents(paths: list[str], source_type: str) -> list[Document]:
    documents: list[Document] = []

    for raw_path in paths:
        root = _assert_allowed_read(raw_path)

        if not root.exists():
            logger.warning("Configured path does not exist: %s", root)
            continue

        if root.is_file():
            if root.suffix.lower() in INDEXABLE_SUFFIXES:
                documents.extend(_load_single_local_file(root, source_type))
            continue

        for path in root.rglob("*"):
            if any(part in IGNORED_INDEX_DIRS for part in path.parts):
                continue

            if not path.is_file():
                continue

            if path.suffix.lower() not in INDEXABLE_SUFFIXES:
                continue

            try:
                _assert_allowed_read(path)
            except PermissionError as exc:
                logger.warning("Skipping blocked file %s: %s", path, exc)
                continue

            documents.extend(_load_single_local_file(path, source_type))

    return documents


def _load_code_documents() -> list[Document]:
    if not _code_retrieval_enabled():
        return []

    max_chars, overlap_chars = _chunking_limits()
    result = load_code_documents(
        CONFIG,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        validate_path=_assert_allowed_read,
    )
    return result.documents


def _load_gitlab_skills() -> list[Document]:
    skills_cfg = CONFIG["gitlab"]["skills"]

    project_id = str(skills_cfg["project_id"])
    branch = skills_cfg["branch"]
    skills_path = skills_cfg["skills_path"]

    try:
        items = _list_gitlab_tree_raw(project_id, skills_path, branch)
    except requests.HTTPError as exc:
        logger.warning(
            "Could not load GitLab skills from project=%s path=%s branch=%s: %s",
            project_id,
            skills_path,
            branch,
            exc,
        )
        return []

    docs: list[Document] = []
    max_chars, overlap_chars = _chunking_limits()

    for item in items:
        if item.get("type") != "blob":
            continue

        file_path = item["path"]

        if not file_path.endswith(".md"):
            continue

        try:
            content = _read_gitlab_file_raw(project_id, file_path, branch)
        except requests.HTTPError as exc:
            logger.warning("Could not read GitLab file %s: %s", file_path, exc)
            continue

        chunks = chunk_markdown(
            file_path,
            content,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            base_metadata={
                "source_group": "skills",
                "chunking_profile": _active_chunking_profile_name(),
                "project_id": project_id,
                "branch": branch,
            },
        )
        for doc in chunks_to_documents(chunks):
            doc.metadata.update(
                {
                    "source_type": "gitlab_skill",
                    "project_id": project_id,
                    "file_path": file_path,
                    "branch": branch,
                }
            )
            docs.append(doc)

    return docs


def _get_profile_collection_name(profile: str) -> str:
    return _collection_name("knowledge", profile=profile)


def _scan_profile_files(profile: str) -> list[Any]:
    paths = _get_knowledge_paths(profile)
    allowed_roots = [_assert_allowed_read(path) for path in paths]
    return scan_paths(
        allowed_roots,
        INDEXABLE_SUFFIXES,
        IGNORED_INDEX_DIRS,
        validate_path=_assert_allowed_read,
    )


def _estimate_scan_chunks(scanned_files: list[Any]) -> tuple[int, int]:
    max_chars, overlap_chars = _chunking_limits()
    estimated_chunks = 0
    estimated_tokens = 0

    for scanned in scanned_files:
        try:
            chunks = chunk_file(
                scanned.path,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
                base_metadata={"chunking_profile": _active_chunking_profile_name()},
            )
        except Exception:
            estimated_chunks += 1
            estimated_tokens += max(1, scanned.size_bytes // 4)
            continue

        estimated_chunks += len(chunks)
        estimated_tokens += sum(max(1, len(chunk.text) // 4) for chunk in chunks)

    return estimated_chunks, estimated_tokens


def _serialize_manifest_file(row: Any) -> dict[str, Any]:
    return {
        "file_id": row.file_id,
        "profile": row.profile,
        "path": row.path,
        "normalized_path": row.normalized_path,
        "size_bytes": row.size_bytes,
        "mtime_ns": row.mtime_ns,
        "content_hash": row.content_hash,
        "extension": row.extension,
        "chunking_profile": row.chunking_profile,
        "embedding_profile": row.embedding_profile,
        "collection_name": row.collection_name,
        "embedding_config_fingerprint": row.embedding_config_fingerprint,
        "last_seen_at": row.last_seen_at,
        "last_indexed_at": row.last_indexed_at,
        "status": row.status,
    }


def _record_manifest_for_documents(
    *,
    profile: str,
    documents: list[Document],
    collection_name: str,
) -> None:
    scanned_files = _scan_profile_files(profile)
    scanned_by_path = {scanned.normalized_path: scanned for scanned in scanned_files}
    docs_by_path: dict[str, list[Document]] = {}

    for doc in documents:
        file_path = doc.metadata.get("file_path")
        if not file_path:
            continue
        try:
            normalized_path = str(Path(str(file_path)).expanduser().resolve())
        except Exception:
            continue
        docs_by_path.setdefault(normalized_path, []).append(doc)

    embedding_cfg = _active_embedding_config()
    embedding_dimension = embedding_cfg.get("dimension")
    if not isinstance(embedding_dimension, int):
        embedding_dimension = None

    for normalized_path, docs in docs_by_path.items():
        scanned = scanned_by_path.get(normalized_path)
        if not scanned:
            continue

        file_row = manifest_file_from_scan(
            scanned,
            profile=profile,
            collection_name=collection_name,
            embedding_config_fingerprint=_retrieval_config_fingerprint(),
            chunking_profile=_active_chunking_profile_name(),
            embedding_profile=_active_embedding_profile_name(),
            last_indexed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        upsert_file(MANIFEST_DB_PATH, file_row)

        for index, doc in enumerate(docs):
            metadata = dict(doc.metadata)
            text_hash = str(metadata.get("text_hash") or metadata.get("content_hash") or "")
            if not text_hash:
                text_hash = hashlib.sha256(doc.get_content().encode("utf-8")).hexdigest()[:16]

            chunk_row = manifest_chunk_from_metadata(
                file_row=file_row,
                chunk_index=int(metadata.get("chunk_index", index)),
                chunk_hash=text_hash,
                text_hash=text_hash,
                metadata=metadata,
                embedding_provider=str(embedding_cfg.get("provider")),
                embedding_model=str(embedding_cfg.get("model")),
                embedding_dimension=embedding_dimension,
                embedding_profile=_active_embedding_profile_name(),
                chunking_profile=_active_chunking_profile_name(),
            )
            upsert_chunk(MANIFEST_DB_PATH, chunk_row)


def _observability_context(
    *,
    domain: str,
    mode: str,
    collection_name: str,
) -> dict[str, Any]:
    return {
        "retrieval_mode": mode,
        "domain": domain,
        "collection": collection_name,
        "embedding_profile": _active_embedding_profile_name(),
        "embedding_provider": _active_embedding_provider(),
        "embedding_model": _active_embedding_config().get("model"),
        "embedding_fingerprint": _embedding_config_fingerprint(),
        "chunking_profile": _active_chunking_profile_name(),
        "bm25_enabled": _bm25_enabled(),
        "reranker_enabled": _reranker_enabled(),
        "reranker_model": _reranker_model() if _reranker_enabled() else None,
        "fusion_method": _fusion_method(),
        "knowledge_profile": KNOWLEDGE_PROFILE if domain == "knowledge" else None,
    }


def _result_scores_from_metadata(metadata: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "final_score": float(score),
        "dense_score": metadata.get("dense_score"),
        "bm25_score": metadata.get("bm25_score"),
        "fused_score": metadata.get("fused_score"),
        "rerank_score": metadata.get("rerank_score"),
        "fused_rank": metadata.get("fused_rank"),
        "final_rank": metadata.get("final_rank"),
    }


def _enrich_result(
    item: dict[str, Any],
    *,
    context: dict[str, Any] | None,
    rank: int,
) -> dict[str, Any]:
    metadata = item.get("metadata") or {}
    enriched = dict(item)
    enriched.setdefault("rank", rank)
    enriched.setdefault("source_path", metadata.get("file_path"))
    enriched.setdefault("source_type", metadata.get("source_type") or metadata.get("source_group"))
    enriched.setdefault("scores", _result_scores_from_metadata(metadata, item.get("score", 0.0)))
    if context is not None:
        enriched["retrieval_mode"] = context["retrieval_mode"]
        enriched["collection"] = context["collection"]
        enriched["embedding_profile"] = context["embedding_profile"]
        enriched["reranker_enabled"] = context["reranker_enabled"]
    return enriched


def _format_nodes(
    nodes: list[Any],
    max_chars: int = 1800,
    *,
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for rank, node_with_score in enumerate(nodes, start=1):
        node = node_with_score.node
        text = node.get_content()

        item = {
            "score": float(node_with_score.score or 0.0),
            "metadata": node.metadata,
            "text": text[:max_chars],
        }
        output.append(_enrich_result(item, context=context, rank=rank))

    return output


def _dense_candidates(nodes: list[Any]) -> list[Candidate]:
    candidates: list[Candidate] = []

    for rank, node_with_score in enumerate(nodes, start=1):
        node = node_with_score.node
        metadata = dict(node.metadata)
        chunk_id = str(
            metadata.get("chunk_id")
            or f"{metadata.get('file_path', '')}:{metadata.get('chunk_index', '')}"
        )
        score = float(node_with_score.score or 0.0)
        candidates.append(
            Candidate(
                chunk_id=chunk_id,
                text=node.get_content(),
                metadata=metadata,
                score=score,
                source="dense",
                rank=rank,
                dense_score=score,
                retrieval_sources={"dense"},
            )
        )

    return candidates


def _format_candidates(
    candidates: list[Candidate],
    max_chars: int = 1800,
    *,
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, start=1):
        item = {
            "score": float(candidate.score),
            "metadata": candidate.metadata,
            "text": candidate.text[:max_chars],
        }
        output.append(_enrich_result(item, context=context, rank=rank))
    return output


def _retrieve_knowledge_hybrid(query: str, final_top_k: int) -> list[dict[str, Any]]:
    assert knowledge_index is not None

    knowledge_collection_name = _collection_name("knowledge")
    dense_k = max(final_top_k, _dense_top_k())
    bm25_k = max(final_top_k, _bm25_top_k())
    fused_k = max(final_top_k, _fused_top_k())

    dense_retriever = knowledge_index.as_retriever(similarity_top_k=dense_k)
    dense_candidates = _dense_candidates(dense_retriever.retrieve(query))

    bm25_documents = load_bm25_index(_bm25_index_path(knowledge_collection_name))
    bm25_candidates = bm25_retrieve(query, bm25_documents, top_k=bm25_k)

    fused = fuse_candidates(
        dense_candidates,
        bm25_candidates,
        fused_top_k=fused_k,
        method=_fusion_method(),
    )
    reranked = rerank_or_fallback(
        query,
        fused,
        enabled=_reranker_enabled(),
        provider=_reranker_provider(),
        model=_reranker_model(),
        candidate_top_k=max(final_top_k, _reranker_candidate_top_k()),
        final_top_k=final_top_k,
        batch_size=_reranker_batch_size(),
        logger=logger,
    )

    mode = "hybrid+rerank" if _reranker_enabled() else "hybrid"
    context = _observability_context(
        domain="knowledge",
        mode=mode,
        collection_name=knowledge_collection_name,
    )
    return _format_candidates(reranked, context=context)


def _knowledge_debug_payload(query: str, final_top_k: int) -> dict[str, Any]:
    assert knowledge_index is not None

    knowledge_collection_name = _collection_name("knowledge")
    dense_k = max(final_top_k, _dense_top_k())
    bm25_k = max(final_top_k, _bm25_top_k())
    fused_k = max(final_top_k, _fused_top_k())

    dense_retriever = knowledge_index.as_retriever(similarity_top_k=dense_k)
    dense_candidates = _dense_candidates(dense_retriever.retrieve(query))
    bm25_candidates = bm25_retrieve(
        query,
        load_bm25_index(_bm25_index_path(knowledge_collection_name)),
        top_k=bm25_k,
    )
    fused_candidates = fuse_candidates(
        dense_candidates,
        bm25_candidates,
        fused_top_k=fused_k,
        method=_fusion_method(),
    )
    reranked_candidates = rerank_or_fallback(
        query,
        fused_candidates,
        enabled=_reranker_enabled(),
        provider=_reranker_provider(),
        model=_reranker_model(),
        candidate_top_k=max(final_top_k, _reranker_candidate_top_k()),
        final_top_k=final_top_k,
        batch_size=_reranker_batch_size(),
        logger=logger,
    )

    return {
        "query": query,
        "collection": knowledge_collection_name,
        "bm25_enabled": _bm25_enabled(),
        "bm25_index_path": str(_bm25_index_path(knowledge_collection_name)),
        "reranker": _reranker_status(),
        "dense_candidates": _format_candidates(dense_candidates[:dense_k]),
        "bm25_candidates": _format_candidates(bm25_candidates[:bm25_k]),
        "fused_candidates": _format_candidates(fused_candidates[:final_top_k]),
        "reranked_candidates": _format_candidates(reranked_candidates),
    }


def _rebuild_all_indexes(force_rebuild: bool = False) -> dict[str, str]:
    global code_index, skills_index, knowledge_index, memory_index, INDEX_STATUS

    if force_rebuild and _is_voyage_active():
        raise RuntimeError(
            "Refusing full rebuild with Voyage embedding profile. Use index_changes(apply=False) "
            "to estimate cost, then index_changes(apply=True, confirm=True) for explicit "
            "incremental indexing."
        )

    knowledge_paths = _get_knowledge_paths()
    skills_collection_name = _collection_name("skills")
    knowledge_collection_name = _collection_name("knowledge")
    memory_collection_name = _collection_name("memory")
    code_collection_name = _code_collection_name()

    if force_rebuild:
        logger.info("Force rebuild enabled. Loading source documents...")

        logger.info("Loading GitLab skills...")
        skills_docs = _load_gitlab_skills()

        logger.info("Loading local knowledge...")
        knowledge_docs = _load_local_documents(knowledge_paths, "knowledge")

        logger.info("Loading local memory...")
        memory_docs = _load_local_documents([CONFIG["memory"]["local_path"]], "memory")

        logger.info("Loading local code repos...")
        code_docs = _load_code_documents()
    else:
        skills_docs = []
        knowledge_docs = []
        memory_docs = []
        code_docs = []

    logger.info("Building/loading skills index...")
    skills_index = _build_index(
        skills_collection_name,
        skills_docs,
        force_rebuild=force_rebuild,
        collection_metadata=_collection_metadata("skills"),
    )

    logger.info("Building/loading knowledge index...")
    knowledge_index = _build_index(
        knowledge_collection_name,
        knowledge_docs,
        force_rebuild=force_rebuild,
        collection_metadata=_collection_metadata("knowledge"),
    )

    logger.info("Building/loading memory index...")
    memory_index = _build_index(
        memory_collection_name,
        memory_docs,
        force_rebuild=force_rebuild,
        collection_metadata=_collection_metadata("memory"),
    )

    logger.info("Building/loading code index...")
    code_index = _build_index(
        code_collection_name,
        code_docs,
        force_rebuild=force_rebuild,
        collection_metadata=_collection_metadata("code", profile=_active_code_profile_name()),
    )

    if force_rebuild:
        logger.info("Recording indexed knowledge files in manifest...")
        _record_manifest_for_documents(
            profile=KNOWLEDGE_PROFILE,
            documents=knowledge_docs,
            collection_name=knowledge_collection_name,
        )
        logger.info("Writing BM25 knowledge index...")
        write_bm25_index(_bm25_index_path(knowledge_collection_name), knowledge_docs)
        logger.info("Writing BM25 code index...")
        write_bm25_index(_bm25_index_path(code_collection_name), code_docs)

    INDEX_STATUS = {
        "skills_collection": skills_collection_name,
        "knowledge_collection": knowledge_collection_name,
        "memory_collection": memory_collection_name,
        "code_collection": code_collection_name,
        "code_enabled": str(_code_retrieval_enabled()),
        "code_profile": _active_code_profile_name(),
        "code_bm25_index_path": str(_bm25_index_path(code_collection_name)),
        "knowledge_profile": KNOWLEDGE_PROFILE,
        "knowledge_paths": ", ".join(knowledge_paths),
        "embedding_profile": _active_embedding_profile_name(),
        "embedding_fingerprint": _embedding_config_fingerprint(),
        "retrieval_fingerprint": _retrieval_config_fingerprint(),
        "chunking_profile": _active_chunking_profile_name(),
        "bm25_enabled": str(_bm25_enabled()),
        "bm25_index_path": str(_bm25_index_path(knowledge_collection_name)),
        "reranker_enabled": str(_reranker_enabled()),
        "reranker_provider": _reranker_provider(),
        "reranker_model": _reranker_model(),
        "mode": "force_rebuild" if force_rebuild else "load_existing_or_empty",
    }

    logger.info("Index status: %s", INDEX_STATUS)

    return INDEX_STATUS


def _ensure_indexes() -> None:
    if skills_index is None or knowledge_index is None or memory_index is None:
        _rebuild_all_indexes()


def _ensure_code_index(force_rebuild: bool = False) -> None:
    global code_index

    if not _code_retrieval_enabled():
        raise RuntimeError("Code retrieval is disabled in config.yaml")
    if code_index is not None and not force_rebuild:
        return

    code_collection_name = _code_collection_name()
    code_docs = _load_code_documents() if force_rebuild else []
    code_index = _build_index(
        code_collection_name,
        code_docs,
        force_rebuild=force_rebuild,
        collection_metadata=_collection_metadata("code", profile=_active_code_profile_name()),
    )
    if force_rebuild:
        write_bm25_index(_bm25_index_path(code_collection_name), code_docs)


# ---------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------


@mcp.tool()
def embedding_config_status() -> dict[str, Any]:
    """Show active embedding and chunking config without exposing secrets."""
    embedding_cfg = _active_embedding_config()
    chunking_cfg = _active_chunking_config()

    return {
        "active_profile": _active_embedding_profile_name(),
        "embedding_profile": _active_embedding_profile_name(),
        "provider": embedding_cfg["provider"],
        "model": embedding_cfg["model"],
        "dimension": embedding_cfg.get("dimension", "default"),
        "normalize": embedding_cfg.get("normalize"),
        "embedding_fingerprint": _embedding_config_fingerprint(),
        "retrieval_fingerprint": _retrieval_config_fingerprint(),
        "chunking_profile": _active_chunking_profile_name(),
        "chunking": chunking_cfg,
        "warnings": embedding_profile_warnings(embedding_cfg),
        "collections": {
            "skills": _collection_name("skills"),
            "knowledge": _collection_name("knowledge"),
            "memory": _collection_name("memory"),
        },
    }


@mcp.tool()
def embedding_status() -> dict[str, Any]:
    """Show active embedding profile and model-specific collection names."""
    return embedding_config_status()


@mcp.tool()
def config_status() -> dict[str, Any]:
    """Show active non-secret config profiles."""
    return {
        "knowledge_profile": KNOWLEDGE_PROFILE,
        "knowledge_paths": _get_knowledge_paths(),
        "embedding": embedding_config_status(),
    }


@mcp.tool()
def retrieval_config_status() -> dict[str, Any]:
    """Show active retrieval pipeline config: top_k values, BM25, fusion, reranker."""
    retrieval_cfg = CONFIG.get("retrieval", {})
    return {
        "bm25_enabled": _bm25_enabled(),
        "fusion_method": _fusion_method(),
        "dense_top_k": _dense_top_k(),
        "bm25_top_k": _bm25_top_k(),
        "fused_top_k": _fused_top_k(),
        "retrieval_final_top_k": _retrieval_final_top_k(),
        "default_final_top_k": _default_final_top_k(),
        "defaults": dict(retrieval_cfg.get("defaults", {})),
        "limits": dict(retrieval_cfg.get("limits", {})),
        "reranker": _reranker_status(),
        "retrieval_fingerprint": _retrieval_config_fingerprint(),
        "code_retrieval_enabled": _code_retrieval_enabled(),
        "code_retrieval_fingerprint": _code_retrieval_fingerprint(),
    }


@mcp.tool()
def index_manifest_status(profile: str | None = None) -> dict[str, Any]:
    """Show indexing manifest status: db path, profile, file/chunk counts."""
    db_exists = MANIFEST_DB_PATH.exists()
    payload: dict[str, Any] = {
        "manifest_db_path": str(MANIFEST_DB_PATH),
        "exists": db_exists,
    }
    if not db_exists:
        return payload

    selected_profile = profile or KNOWLEDGE_PROFILE
    try:
        files = get_files_for_profile(MANIFEST_DB_PATH, selected_profile)
    except Exception as exc:
        payload["error"] = f"Failed to read manifest: {exc}"
        return payload

    by_status: dict[str, int] = {}
    by_extension: dict[str, int] = {}
    last_indexed_at: str | None = None
    for row in files:
        by_status[row.status] = by_status.get(row.status, 0) + 1
        ext = row.extension or "<none>"
        by_extension[ext] = by_extension.get(ext, 0) + 1
        if row.last_indexed_at and (
            last_indexed_at is None or row.last_indexed_at > last_indexed_at
        ):
            last_indexed_at = row.last_indexed_at

    payload.update(
        {
            "profile": selected_profile,
            "total_files": len(files),
            "by_status": by_status,
            "by_extension": by_extension,
            "last_indexed_at": last_indexed_at,
            "collection_name": _get_profile_collection_name(selected_profile),
        }
    )
    return payload


@mcp.tool()
def runtime_status() -> dict[str, Any]:
    """Aggregate runtime diagnostics: profile, embeddings, collections, indexes, reranker."""
    return {
        "knowledge_profile": KNOWLEDGE_PROFILE,
        "knowledge_paths": _get_knowledge_paths(),
        "embedding_profile": _active_embedding_profile_name(),
        "embedding_provider": _active_embedding_provider(),
        "embedding_model": _active_embedding_config().get("model"),
        "chunking_profile": _active_chunking_profile_name(),
        "bm25_enabled": _bm25_enabled(),
        "reranker": _reranker_status(),
        "code_retrieval_enabled": _code_retrieval_enabled(),
        "loaded_indexes": {
            "skills": skills_index is not None,
            "knowledge": knowledge_index is not None,
            "memory": memory_index is not None,
            "code": code_index is not None,
        },
        "collections": collection_status(),
        "index_status": INDEX_STATUS,
    }


@mcp.tool()
def collection_status() -> dict[str, Any]:
    """Show Chroma collection names and vector counts."""
    skills_collection_name = _collection_name("skills")
    knowledge_collection_name = _collection_name("knowledge")
    memory_collection_name = _collection_name("memory")
    code_collection_name = _code_collection_name()

    return {
        "skills": {
            "collection": skills_collection_name,
            "vectors": _collection_count(skills_collection_name),
            "embedding": _collection_metadata("skills"),
        },
        "knowledge": {
            "collection": knowledge_collection_name,
            "profile": KNOWLEDGE_PROFILE,
            "vectors": _collection_count(knowledge_collection_name),
            "paths": _get_knowledge_paths(),
            "embedding": _collection_metadata("knowledge"),
        },
        "memory": {
            "collection": memory_collection_name,
            "vectors": _collection_count(memory_collection_name),
            "embedding": _collection_metadata("memory"),
        },
        "code": {
            "collection": code_collection_name,
            "profile": _active_code_profile_name(),
            "enabled": _code_retrieval_enabled(),
            "vectors": _collection_count(code_collection_name),
            "embedding": _collection_metadata("code", profile=_active_code_profile_name()),
        },
    }


@mcp.tool()
def list_code_repos() -> list[dict[str, Any]]:
    """List local repositories configured for code retrieval."""
    repos = configured_code_repos(CONFIG, validate_path=_assert_allowed_read)
    return [{"repo_name": repo.name, "path": str(repo.path)} for repo in repos]


@mcp.tool()
def code_index_status() -> dict[str, Any]:
    """Show code retrieval index status."""
    collection_name = _code_collection_name()
    bm25_documents = load_bm25_index(_bm25_index_path(collection_name))
    files = {
        f"{doc.metadata.get('repo_name')}:{doc.metadata.get('file_path')}" for doc in bm25_documents
    }
    repos = sorted({str(doc.metadata.get("repo_name")) for doc in bm25_documents})

    return {
        "enabled": _code_retrieval_enabled(),
        "profile": _active_code_profile_name(),
        "fingerprint": _code_retrieval_fingerprint(),
        "chunking": _active_code_chunking_config(),
        "collection": collection_name,
        "vectors": _collection_count(collection_name),
        "bm25_index_path": str(_bm25_index_path(collection_name)),
        "bm25_chunks": len(bm25_documents),
        "indexed_files": len(files),
        "indexed_repos": repos,
        "configured_repos": list_code_repos(),
        "excluded": CONFIG.get("code_retrieval", {})
        .get("profiles", {})
        .get(_active_code_profile_name(), {})
        .get("excluded", []),
    }


@mcp.tool()
def scan_index_changes(profile: str | None = None) -> dict[str, Any]:
    """Dry-run scan for knowledge files that need indexing."""
    selected_profile = profile or KNOWLEDGE_PROFILE
    collection_name = _get_profile_collection_name(selected_profile)
    init_manifest_db(MANIFEST_DB_PATH)

    scanned_files = _scan_profile_files(selected_profile)
    manifest_files = get_files_for_profile(MANIFEST_DB_PATH, selected_profile)
    changes = detect_changes(
        selected_profile,
        scanned_files,
        manifest_files,
        collection_name=collection_name,
        embedding_config_fingerprint=_retrieval_config_fingerprint(),
        chunking_profile=_active_chunking_profile_name(),
        embedding_profile=_active_embedding_profile_name(),
    )
    estimated_chunks, estimated_tokens = _estimate_scan_chunks(scanned_files)

    return {
        "profile": selected_profile,
        "manifest_path": str(MANIFEST_DB_PATH),
        "collection_name": collection_name,
        "embedding_config_fingerprint": _retrieval_config_fingerprint(),
        "files_scanned": len(scanned_files),
        "added": changes["added"],
        "edited": changes["edited"],
        "config_changed": changes["config_changed"],
        "deleted": changes["deleted"],
        "moved": changes["moved"],
        "counts": {
            "added": len(changes["added"]),
            "edited": len(changes["edited"]),
            "config_changed": len(changes["config_changed"]),
            "deleted": len(changes["deleted"]),
            "moved": len(changes["moved"]),
        },
        "estimated_chunks": estimated_chunks,
        "estimated_tokens": estimated_tokens,
        "dry_run": True,
    }


@mcp.tool()
def list_indexed_files(profile: str | None = None) -> list[dict[str, Any]]:
    """List file-level manifest rows for a knowledge profile."""
    selected_profile = profile or KNOWLEDGE_PROFILE
    files = get_files_for_profile(MANIFEST_DB_PATH, selected_profile)
    return [_serialize_manifest_file(row) for row in files]


@mcp.tool()
def list_changed_files(profile: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """List changed files grouped by added, edited, deleted, and moved."""
    status = scan_index_changes(profile)
    return {
        "added": status["added"],
        "edited": status["edited"],
        "config_changed": status["config_changed"],
        "deleted": status["deleted"],
        "moved": status["moved"],
    }


@mcp.tool()
def embedding_budget_status() -> dict[str, Any]:
    """Show embedding token budget and ledger summary."""
    budget = token_budget_from_config(CONFIG)
    ledger = token_ledger_summary(MANIFEST_DB_PATH)
    return {
        "provider": _active_embedding_provider(),
        "embedding_profile": _active_embedding_profile_name(),
        "model": _active_embedding_config().get("model"),
        "voyage_active": _is_voyage_active(),
        "voyage": {
            "require_explicit_apply": bool(_voyage_config().get("require_explicit_apply", True)),
            "max_tokens_per_run_default": budget.max_tokens_per_run,
            "warn_at_total_tokens": budget.warning_threshold,
            "hard_stop_at_total_tokens": budget.hard_stop_threshold,
        },
        "ledger": ledger,
    }


@mcp.tool()
def estimate_embedding_cost(profile: str | None = None) -> dict[str, Any]:
    """Dry-run changed-file embedding token estimate."""
    estimate = _estimate_changed_embedding_tokens(profile)
    run_id = new_run_id()
    record_token_ledger(
        MANIFEST_DB_PATH,
        run_id=run_id,
        provider=estimate["provider"],
        model=str(estimate["model"]),
        usage_type="document_embedding_estimate",
        tokens=int(estimate["estimated_tokens"]),
        estimated_or_actual="estimated",
        notes=f"profile={estimate['profile']}; dry_run=true",
    )
    return {
        **estimate,
        "run_id": run_id,
        "ledger_recorded": True,
    }


@mcp.tool()
def index_changes(
    profile: str | None = None,
    apply: bool = False,
    confirm: bool = False,
) -> dict[str, Any]:
    """
    Dry-run changed-file indexing by default.

    Voyage apply requires apply=True and confirm=True. This prevents accidental
    cloud embedding of a full library.
    """
    estimate = _estimate_changed_embedding_tokens(profile)
    run_id = new_run_id()
    record_token_ledger(
        MANIFEST_DB_PATH,
        run_id=run_id,
        provider=estimate["provider"],
        model=str(estimate["model"]),
        usage_type="document_embedding_estimate",
        tokens=int(estimate["estimated_tokens"]),
        estimated_or_actual="estimated",
        notes=f"profile={estimate['profile']}; apply={apply}",
    )

    if not apply:
        return {
            **estimate,
            "run_id": run_id,
            "apply": False,
            "applied": False,
            "message": "Dry run only. Re-run with apply=True and confirm=True to request indexing.",
        }

    if not _is_voyage_active():
        return {
            **estimate,
            "run_id": run_id,
            "apply": True,
            "applied": False,
            "message": "index_changes(apply=True) is only enabled for Voyage profiles.",
        }

    if _voyage_config().get("require_explicit_apply", True) and not confirm:
        return {
            **estimate,
            "run_id": run_id,
            "apply": True,
            "applied": False,
            "message": "Voyage indexing requires confirm=True.",
        }

    if not estimate["budget"]["allowed"]:
        return {
            **estimate,
            "run_id": run_id,
            "apply": True,
            "applied": False,
            "message": "Estimated tokens exceed configured Voyage budget limits.",
        }

    if not os.environ.get("VOYAGE_API_KEY"):
        raise RuntimeError("VOYAGE_API_KEY is missing. Set it in .env before Voyage indexing.")

    return {
        **estimate,
        "run_id": run_id,
        "apply": True,
        "applied": False,
        "message": (
            "Voyage budget and confirmation checks passed. Per-file Voyage vector upserts "
            "are intentionally not executed by this safety patch."
        ),
    }


@mcp.tool()
def refresh_indexes(force_rebuild: bool = False) -> dict[str, str]:
    """
    Load existing indexes or rebuild them.

    If force_rebuild=True, source files are reparsed and Chroma collections are rebuilt.
    If force_rebuild=False, existing Chroma collections are reused when available.
    """
    return _rebuild_all_indexes(force_rebuild=force_rebuild)


@mcp.tool()
def rebuild_indexes() -> dict[str, str]:
    """Force a full rebuild of skills, knowledge, and memory indexes."""
    return _rebuild_all_indexes(force_rebuild=True)


@mcp.tool()
def knowledge_profile_status() -> dict[str, Any]:
    """Show the active knowledge profile and paths."""
    return {
        "active_profile": KNOWLEDGE_PROFILE,
        "paths": _get_knowledge_paths(),
    }


@mcp.tool()
def index_status() -> dict[str, str]:
    """Show current index status."""
    return INDEX_STATUS


@mcp.tool()
def retrieve_skills(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Retrieve relevant GitLab skill files for a task."""
    _ensure_indexes()

    assert skills_index is not None

    k = _bounded_top_k(
        top_k,
        _get_default_top_k("skills"),
        _get_max_top_k("skills"),
    )

    retriever = skills_index.as_retriever(similarity_top_k=k)
    nodes = retriever.retrieve(query)

    context = _observability_context(
        domain="skills",
        mode="dense",
        collection_name=_collection_name("skills"),
    )
    return _format_nodes(nodes, context=context)


@mcp.tool()
def retrieve_knowledge(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Retrieve relevant local knowledge from books, articles, and notes."""
    _ensure_indexes()

    assert knowledge_index is not None

    k = _knowledge_final_top_k(top_k)

    if _bm25_enabled():
        return _retrieve_knowledge_hybrid(query, k)

    dense_k = max(k, _reranker_candidate_top_k())
    retriever = knowledge_index.as_retriever(similarity_top_k=dense_k)
    dense_candidates = _dense_candidates(retriever.retrieve(query))
    reranked = rerank_or_fallback(
        query,
        dense_candidates,
        enabled=_reranker_enabled(),
        provider=_reranker_provider(),
        model=_reranker_model(),
        candidate_top_k=max(k, _reranker_candidate_top_k()),
        final_top_k=k,
        batch_size=_reranker_batch_size(),
        logger=logger,
    )

    mode = "dense+rerank" if _reranker_enabled() else "dense"
    context = _observability_context(
        domain="knowledge",
        mode=mode,
        collection_name=_collection_name("knowledge"),
    )
    return _format_candidates(reranked, context=context)


@mcp.tool()
def retrieve_knowledge_debug(query: str, top_k: int | None = None) -> dict[str, Any]:
    """Retrieve knowledge with dense, BM25, and fused diagnostic candidates."""
    _ensure_indexes()

    assert knowledge_index is not None

    k = _knowledge_final_top_k(top_k)

    if _bm25_enabled():
        return _knowledge_debug_payload(query, k)

    dense_k = max(k, _reranker_candidate_top_k())
    dense_retriever = knowledge_index.as_retriever(similarity_top_k=dense_k)
    dense_candidates = _dense_candidates(dense_retriever.retrieve(query))
    reranked_candidates = rerank_or_fallback(
        query,
        dense_candidates,
        enabled=_reranker_enabled(),
        provider=_reranker_provider(),
        model=_reranker_model(),
        candidate_top_k=max(k, _reranker_candidate_top_k()),
        final_top_k=k,
        batch_size=_reranker_batch_size(),
        logger=logger,
    )
    return {
        "query": query,
        "collection": _collection_name("knowledge"),
        "bm25_enabled": False,
        "reranker": _reranker_status(),
        "dense_candidates": _format_candidates(dense_candidates),
        "bm25_candidates": [],
        "fused_candidates": _format_candidates(dense_candidates),
        "reranked_candidates": _format_candidates(reranked_candidates),
    }


@mcp.tool()
def retrieve_code(
    query: str,
    repo: str | None = None,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Retrieve relevant local repository code chunks with file paths and line ranges."""
    _ensure_code_index()

    assert code_index is not None

    k = _knowledge_final_top_k(top_k)
    dense_k = max(k, _dense_top_k())
    bm25_k = max(k, _bm25_top_k())
    fused_k = max(k, _fused_top_k())
    collection_name = _code_collection_name()

    dense_retriever = code_index.as_retriever(similarity_top_k=dense_k)
    candidates = retrieve_code_candidates(
        query,
        dense_nodes=dense_retriever.retrieve(query),
        bm25_index_path=str(_bm25_index_path(collection_name)),
        repo=repo,
        bm25_top_k=bm25_k,
        fused_top_k=fused_k,
        fusion_method=_fusion_method(),
        reranker_enabled=_reranker_enabled(),
        reranker_provider=_reranker_provider(),
        reranker_model=_reranker_model(),
        reranker_candidate_top_k=max(k, _reranker_candidate_top_k()),
        final_top_k=k,
        reranker_batch_size=_reranker_batch_size(),
        logger=logger,
    )

    mode = "hybrid+rerank" if _reranker_enabled() else "hybrid"
    context = _observability_context(
        domain="code",
        mode=mode,
        collection_name=collection_name,
    )
    formatted = format_code_candidates(candidates["reranked"])
    return [
        _enrich_result(item, context=context, rank=rank)
        for rank, item in enumerate(formatted, start=1)
    ]


@mcp.tool()
def read_code_symbol(
    repo: str,
    symbol_name: str,
    file_path: str | None = None,
) -> dict[str, Any]:
    """Read exact source lines for an indexed code symbol."""
    collection_name = _code_collection_name()
    bm25_candidates = bm25_retrieve(
        symbol_name,
        load_bm25_index(_bm25_index_path(collection_name)),
        top_k=200,
    )
    candidate = find_symbol_candidate(
        bm25_candidates,
        repo=repo,
        symbol_name=symbol_name,
        file_path=file_path,
    )
    if candidate is None:
        raise FileNotFoundError(f"Indexed symbol not found: {repo}:{symbol_name}")

    metadata = candidate.metadata
    absolute_path = _assert_allowed_read(str(metadata.get("absolute_file_path", "")))
    start_line = int(metadata.get("start_line", 1))
    end_line = int(metadata.get("end_line", start_line))
    lines = absolute_path.read_text(encoding="utf-8", errors="replace").splitlines()
    source = "\n".join(lines[start_line - 1 : end_line])

    return {
        "repo": repo,
        "symbol_name": symbol_name,
        "file_path": metadata.get("file_path"),
        "absolute_file_path": str(absolute_path),
        "start_line": start_line,
        "end_line": end_line,
        "metadata": metadata,
        "source": source,
    }


@mcp.tool()
def retrieve_memory(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Retrieve relevant long-term memory about preferences, projects, and decisions."""
    _ensure_indexes()

    assert memory_index is not None

    k = _bounded_top_k(
        top_k,
        _get_default_top_k("memory"),
        _get_max_top_k("memory"),
    )

    retriever = memory_index.as_retriever(similarity_top_k=k)
    nodes = retriever.retrieve(query)

    context = _observability_context(
        domain="memory",
        mode="dense",
        collection_name=_collection_name("memory"),
    )
    return _format_nodes(nodes, context=context)


@mcp.tool()
def read_local_file(path: str) -> str:
    """Read an allowlisted local text file."""
    resolved = _assert_allowed_read(path)

    if not resolved.is_file():
        raise FileNotFoundError(f"Not a file: {resolved}")

    return resolved.read_text(encoding="utf-8", errors="replace")


@mcp.tool()
def list_local_files(directory: str) -> list[str]:
    """List files inside an allowlisted local directory."""
    resolved = _assert_allowed_read(directory)

    if not resolved.is_dir():
        raise NotADirectoryError(f"Not a directory: {resolved}")

    return sorted(str(p) for p in resolved.iterdir())


@mcp.tool()
def get_gitlab_file(
    file_path: str,
    project_id: str | None = None,
    branch: str | None = None,
) -> str:
    """Read a file from an allowlisted GitLab project."""
    skills_cfg = CONFIG["gitlab"]["skills"]

    selected_project_id = str(project_id or skills_cfg["project_id"])
    selected_branch = str(branch or skills_cfg["branch"])

    return _read_gitlab_file_raw(selected_project_id, file_path, selected_branch)


@mcp.tool()
def list_gitlab_repo(
    project_id: str | None = None,
    repo_path: str = "",
    branch: str | None = None,
) -> list[dict[str, Any]]:
    """List a GitLab repository tree for an allowlisted project."""
    skills_cfg = CONFIG["gitlab"]["skills"]

    selected_project_id = str(project_id or skills_cfg["project_id"])
    selected_branch = str(branch or skills_cfg["branch"])

    return _list_gitlab_tree_raw(selected_project_id, repo_path, selected_branch)


@mcp.tool()
def propose_memory_update(category: str, content: str, reason: str) -> str:
    """
    Create a local memory proposal.

    This does not directly update permanent memory.
    """
    allowed_categories = {
        "profile",
        "projects",
        "decisions",
        "preferences",
        "working_style",
    }

    clean_category = _slugify(category)

    if clean_category not in allowed_categories:
        clean_category = "profile"

    proposal_dir = _assert_allowed_write(CONFIG["memory"]["proposal_path"])
    proposal_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_path = proposal_dir / f"{timestamp}-{clean_category}.md"

    proposal = f"""---
status: proposed
category: {clean_category}
created_at: {timestamp}
---

# Memory Proposal

## Content

{content}

## Reason

{reason}
"""

    resolved = _assert_allowed_write(file_path)
    resolved.write_text(proposal, encoding="utf-8")

    return f"Memory proposal created: {resolved}"


@mcp.tool()
def propose_new_skill(name: str, content: str, reason: str) -> dict[str, Any]:
    """
    Create a proposed skill in GitLab using a branch and merge request.

    This does not push directly to main.
    """
    if not CONFIG["security"]["allow_gitlab_write"]:
        raise PermissionError("GitLab write is disabled in config.yaml")

    if CONFIG["security"]["allow_direct_main_write"]:
        raise PermissionError("Direct main writes are intentionally blocked.")

    skills_cfg = CONFIG["gitlab"]["skills"]

    project_id = str(skills_cfg["project_id"])
    api_project_id = _gitlab_project_api_id(_assert_allowed_gitlab_project(project_id))

    base_branch = skills_cfg["branch"]
    proposal_path = skills_cfg["proposal_path"]

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    slug = _slugify(name)

    branch = f"agent/skill-proposal-{timestamp}-{slug}"
    file_path = f"{proposal_path}/{timestamp}-{slug}.md"

    full_content = f"""---
status: proposed
created_by: mcp-agent
created_at: {timestamp}
reason: {reason}
---

{content}
"""

    _create_gitlab_branch(project_id, branch, base_branch)

    encoded_path = quote(file_path, safe="")

    _gitlab_post(
        f"projects/{api_project_id}/repository/files/{encoded_path}",
        json={
            "branch": branch,
            "content": full_content,
            "commit_message": f"Propose new skill: {name}",
        },
    )

    mr = _gitlab_post(
        f"projects/{api_project_id}/merge_requests",
        json={
            "source_branch": branch,
            "target_branch": base_branch,
            "title": f"Propose skill: {name}",
            "description": f"Reason:\n\n{reason}",
            "remove_source_branch": True,
        },
    )

    return {
        "branch": branch,
        "file_path": file_path,
        "merge_request": mr.get("web_url", mr),
    }


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
