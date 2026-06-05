from __future__ import annotations

import base64
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import chromadb
import requests
import yaml
from dotenv import load_dotenv
from llama_index.core import Document, Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from mcp.server.fastmcp import FastMCP


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

CONFIG_PATH = ROOT / "config.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

GITLAB_TOKEN = __import__("os").environ.get("GITLAB_TOKEN")
GITLAB_BASE_URL = __import__("os").environ.get(
    "GITLAB_BASE_URL",
    CONFIG["gitlab"]["base_url"],
).rstrip("/")

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

    allowed_roots = [_resolve(p) for p in CONFIG["security"]["allowed_write_roots"]]
    if not any(resolved == root or _is_inside(resolved, root) for root in allowed_roots):
        raise PermissionError(f"Write path is not allowlisted: {resolved}")

    return resolved


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "untitled"


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


def _gitlab_post(path: str, json: dict[str, Any]) -> Any:
    response = requests.post(
        _gitlab_url(path),
        headers={**_gitlab_headers(), "Content-Type": "application/json"},
        json=json,
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
    return _gitlab_get(
        f"projects/{project_id}/repository/tree",
        params={
            "path": repo_path,
            "ref": branch,
            "recursive": "true",
            "per_page": 100,
        },
    )


def _read_gitlab_file_raw(
    project_id: str,
    file_path: str,
    branch: str = "main",
) -> str:
    encoded_path = quote(file_path, safe="")
    data = _gitlab_get(
        f"projects/{project_id}/repository/files/{encoded_path}",
        params={"ref": branch},
    )
    content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    return content


def _gitlab_file_exists(project_id: str, file_path: str, branch: str) -> bool:
    encoded_path = quote(file_path, safe="")
    response = requests.get(
        _gitlab_url(f"projects/{project_id}/repository/files/{encoded_path}"),
        headers=_gitlab_headers(),
        params={"ref": branch},
        timeout=30,
    )
    return response.status_code == 200


def _create_gitlab_branch(project_id: str, branch: str, ref: str) -> None:
    response = requests.post(
        _gitlab_url(f"projects/{project_id}/repository/branches"),
        headers=_gitlab_headers(),
        data={"branch": branch, "ref": ref},
        timeout=30,
    )

    if response.status_code not in (200, 201, 400):
        response.raise_for_status()


# ---------------------------------------------------------------------
# LlamaIndex setup
# ---------------------------------------------------------------------

Settings.embed_model = HuggingFaceEmbedding(
    model_name=CONFIG["retrieval"]["embedding_model"]
)
Settings.llm = None

CHROMA_PATH = ROOT / "storage" / "chroma"
CHROMA_PATH.mkdir(parents=True, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))


def _fresh_collection(name: str):
    try:
        chroma_client.delete_collection(name)
    except Exception:
        pass
    return chroma_client.get_or_create_collection(name)


def _build_index(collection_name: str, docs: list[Document]) -> VectorStoreIndex:
    if not docs:
        docs = [
            Document(
                text="No documents loaded.",
                metadata={"source": "empty", "collection": collection_name},
            )
        ]

    collection = _fresh_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        show_progress=True,
    )


def _load_local_documents(paths: list[str], source_type: str) -> list[Document]:
    documents: list[Document] = []

    for raw_path in paths:
        path = _assert_allowed_read(raw_path)

        if not path.exists():
            continue

        reader = SimpleDirectoryReader(
            input_dir=str(path),
            recursive=True,
            required_exts=[
                ".md",
                ".txt",
                ".pdf",
                ".docx",
                ".py",
                ".yaml",
                ".yml",
            ],
        )

        loaded = reader.load_data()
        for doc in loaded:
            doc.metadata["source_type"] = source_type
            documents.append(doc)

    return documents


def _load_gitlab_skills() -> list[Document]:
    skills_cfg = CONFIG["gitlab"]["skills"]
    project_id = str(skills_cfg["project_id"])
    branch = skills_cfg["branch"]
    skills_path = skills_cfg["skills_path"]

    items = _list_gitlab_tree_raw(project_id, skills_path, branch)
    docs: list[Document] = []

    for item in items:
        if item.get("type") != "blob":
            continue

        file_path = item["path"]
        if not file_path.endswith(".md"):
            continue

        content = _read_gitlab_file_raw(project_id, file_path, branch)
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
    output = []

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


def _rebuild_all_indexes() -> dict[str, str]:
    global skills_index, knowledge_index, memory_index

    skills_docs = _load_gitlab_skills()
    knowledge_docs = _load_local_documents(CONFIG["knowledge"]["paths"], "knowledge")
    memory_docs = _load_local_documents([CONFIG["memory"]["local_path"]], "memory")

    skills_index = _build_index("skills", skills_docs)
    knowledge_index = _build_index("knowledge", knowledge_docs)
    memory_index = _build_index("memory", memory_docs)

    return {
        "skills": f"{len(skills_docs)} docs indexed",
        "knowledge": f"{len(knowledge_docs)} docs indexed",
        "memory": f"{len(memory_docs)} docs indexed",
    }


skills_index: VectorStoreIndex
knowledge_index: VectorStoreIndex
memory_index: VectorStoreIndex

INDEX_STATUS = _rebuild_all_indexes()


# ---------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------

@mcp.tool()
def retrieve_skills(query: str) -> list[dict[str, Any]]:
    """Retrieve relevant GitLab skill files for a task."""
    retriever = skills_index.as_retriever(
        similarity_top_k=CONFIG["retrieval"]["skills_top_k"]
    )
    nodes = retriever.retrieve(query)
    return _format_nodes(nodes)


@mcp.tool()
def retrieve_knowledge(query: str) -> list[dict[str, Any]]:
    """Retrieve relevant local knowledge from books, articles, and notes."""
    retriever = knowledge_index.as_retriever(
        similarity_top_k=CONFIG["retrieval"]["knowledge_top_k"]
    )
    nodes = retriever.retrieve(query)
    return _format_nodes(nodes)


@mcp.tool()
def retrieve_memory(query: str) -> list[dict[str, Any]]:
    """Retrieve relevant long-term memory about preferences, projects, and decisions."""
    retriever = memory_index.as_retriever(
        similarity_top_k=CONFIG["retrieval"]["memory_top_k"]
    )
    nodes = retriever.retrieve(query)
    return _format_nodes(nodes)


@mcp.tool()
def read_local_file(path: str) -> str:
    """Read an allowlisted local file."""
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
def get_gitlab_file(project_id: str, file_path: str, branch: str = "main") -> str:
    """Read a file from an allowed GitLab project."""
    return _read_gitlab_file_raw(project_id, file_path, branch)


@mcp.tool()
def list_gitlab_repo(
    project_id: str,
    repo_path: str = "",
    branch: str = "main",
) -> list[dict[str, Any]]:
    """List a GitLab repository tree."""
    return _list_gitlab_tree_raw(project_id, repo_path, branch)


@mcp.tool()
def propose_memory_update(category: str, content: str, reason: str) -> str:
    """
    Create a local memory proposal.
    This does not directly update permanent memory.
    """
    allowed_categories = {"profile", "projects", "decisions", "preferences", "working_style"}
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
    Does not push directly to main.
    """
    if not CONFIG["security"]["allow_gitlab_write"]:
        raise PermissionError("GitLab write is disabled in config.yaml")

    if CONFIG["security"]["allow_direct_main_write"]:
        raise PermissionError("Direct main writes are intentionally blocked.")

    skills_cfg = CONFIG["gitlab"]["skills"]

    project_id = str(skills_cfg["project_id"])
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
        f"projects/{project_id}/repository/files/{encoded_path}",
        json={
            "branch": branch,
            "content": full_content,
            "commit_message": f"Propose new skill: {name}",
        },
    )

    mr = _gitlab_post(
        f"projects/{project_id}/merge_requests",
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


@mcp.tool()
def refresh_indexes() -> dict[str, str]:
    """Rebuild skills, knowledge, and memory retrieval indexes."""
    return _rebuild_all_indexes()


@mcp.tool()
def index_status() -> dict[str, str]:
    """Show current index status."""
    return INDEX_STATUS


if __name__ == "__main__":
    mcp.run()