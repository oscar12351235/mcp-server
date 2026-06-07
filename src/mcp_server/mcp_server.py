from __future__ import annotations

import base64
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
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from mcp.server.fastmcp import FastMCP

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

chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))


skills_index: VectorStoreIndex | None = None
knowledge_index: VectorStoreIndex | None = None
memory_index: VectorStoreIndex | None = None
INDEX_STATUS: dict[str, str] = {
    "skills": "not built yet",
    "knowledge": "not built yet",
    "memory": "not built yet",
}
_EMBEDDING_MODEL_READY = False


def _ensure_embedding_model() -> None:
    global _EMBEDDING_MODEL_READY

    if _EMBEDDING_MODEL_READY:
        return

    Settings.embed_model = HuggingFaceEmbedding(model_name=CONFIG["retrieval"]["embedding_model"])
    _EMBEDDING_MODEL_READY = True


def _get_collection(name: str, force_rebuild: bool = False):
    if force_rebuild:
        try:
            chroma_client.delete_collection(name)
            logger.info("Deleted existing Chroma collection: %s", name)
        except Exception:
            pass

    return chroma_client.get_or_create_collection(name)


def _build_index(
    collection_name: str,
    docs: list[Document],
    force_rebuild: bool = False,
) -> VectorStoreIndex:
    _ensure_embedding_model()

    collection = _get_collection(collection_name, force_rebuild=force_rebuild)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    existing_count = collection.count()

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
    try:
        loaded = SimpleDirectoryReader(
            input_files=[str(path)],
        ).load_data()

        for doc in loaded:
            doc.metadata["source_type"] = source_type
            doc.metadata["file_path"] = str(path)

        return loaded

    except Exception as exc:
        logger.warning("Could not load file %s: %s", path, exc)
        return []


def _load_local_documents(paths: list[str], source_type: str) -> list[Document]:
    documents: list[Document] = []

    allowed_suffixes = {
        ".md",
        ".txt",
        ".pdf",
        ".py",
        ".yaml",
        ".yml",
        ".png",
        ".epub",
        ".docx",
        ".csv",
        ".xlsx",
        ".parquet",
        ".json",
        ".html",
    }

    ignored_dirs = {
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

    for raw_path in paths:
        root = _assert_allowed_read(raw_path)

        if not root.exists():
            logger.warning("Configured path does not exist: %s", root)
            continue

        if root.is_file():
            if root.suffix.lower() in allowed_suffixes:
                documents.extend(_load_single_local_file(root, source_type))
            continue

        for path in root.rglob("*"):
            if any(part in ignored_dirs for part in path.parts):
                continue

            if not path.is_file():
                continue

            if path.suffix.lower() not in allowed_suffixes:
                continue

            try:
                _assert_allowed_read(path)
            except PermissionError as exc:
                logger.warning("Skipping blocked file %s: %s", path, exc)
                continue

            documents.extend(_load_single_local_file(path, source_type))

    return documents


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

        docs.append(
            Document(
                text=content,
                metadata={
                    "source_type": "gitlab_skill",
                    "project_id": project_id,
                    "file_path": file_path,
                    "branch": branch,
                },
            )
        )

    return docs


def _format_nodes(nodes: list[Any], max_chars: int = 1800) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for node_with_score in nodes:
        node = node_with_score.node
        text = node.get_content()

        output.append(
            {
                "score": float(node_with_score.score or 0.0),
                "metadata": node.metadata,
                "text": text[:max_chars],
            }
        )

    return output


def _rebuild_all_indexes(force_rebuild: bool = False) -> dict[str, str]:
    global skills_index, knowledge_index, memory_index, INDEX_STATUS

    knowledge_paths = _get_knowledge_paths()
    knowledge_collection_name = f"knowledge_{KNOWLEDGE_PROFILE}"

    if force_rebuild:
        logger.info("Force rebuild enabled. Loading source documents...")

        logger.info("Loading GitLab skills...")
        skills_docs = _load_gitlab_skills()

        logger.info("Loading local knowledge...")
        knowledge_docs = _load_local_documents(knowledge_paths, "knowledge")

        logger.info("Loading local memory...")
        memory_docs = _load_local_documents([CONFIG["memory"]["local_path"]], "memory")
    else:
        skills_docs = []
        knowledge_docs = []
        memory_docs = []

    logger.info("Building/loading skills index...")
    skills_index = _build_index(
        "skills",
        skills_docs,
        force_rebuild=force_rebuild,
    )

    logger.info("Building/loading knowledge index...")
    knowledge_index = _build_index(
        knowledge_collection_name,
        knowledge_docs,
        force_rebuild=force_rebuild,
    )

    logger.info("Building/loading memory index...")
    memory_index = _build_index(
        "memory",
        memory_docs,
        force_rebuild=force_rebuild,
    )

    INDEX_STATUS = {
        "skills_collection": "skills",
        "knowledge_collection": knowledge_collection_name,
        "memory_collection": "memory",
        "knowledge_profile": KNOWLEDGE_PROFILE,
        "knowledge_paths": ", ".join(knowledge_paths),
        "mode": "force_rebuild" if force_rebuild else "load_existing_or_empty",
    }

    logger.info("Index status: %s", INDEX_STATUS)

    return INDEX_STATUS


def _ensure_indexes() -> None:
    if skills_index is None or knowledge_index is None or memory_index is None:
        _rebuild_all_indexes()


# ---------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------


@mcp.tool()
def collection_status() -> dict[str, Any]:
    """Show Chroma collection names and vector counts."""
    knowledge_collection_name = f"knowledge_{KNOWLEDGE_PROFILE}"

    return {
        "skills": {
            "collection": "skills",
            "vectors": _collection_count("skills"),
        },
        "knowledge": {
            "collection": knowledge_collection_name,
            "profile": KNOWLEDGE_PROFILE,
            "vectors": _collection_count(knowledge_collection_name),
            "paths": _get_knowledge_paths(),
        },
        "memory": {
            "collection": "memory",
            "vectors": _collection_count("memory"),
        },
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

    return _format_nodes(nodes)


@mcp.tool()
def retrieve_knowledge(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Retrieve relevant local knowledge from books, articles, and notes."""
    _ensure_indexes()

    assert knowledge_index is not None

    k = _bounded_top_k(
        top_k,
        _get_default_top_k("knowledge"),
        _get_max_top_k("knowledge"),
    )

    retriever = knowledge_index.as_retriever(similarity_top_k=k)
    nodes = retriever.retrieve(query)

    return _format_nodes(nodes)


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

    return _format_nodes(nodes)


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
