from __future__ import annotations

import fnmatch
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llama_index.core import Document

from mcp_server.retrieval.chunkers import chunks_to_documents
from mcp_server.retrieval.code_chunker import chunk_code_file

DEFAULT_EXCLUDED = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "storage",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
}


@dataclass(frozen=True)
class CodeRepo:
    name: str
    path: Path


@dataclass(frozen=True)
class CodeLoadResult:
    repos: list[CodeRepo]
    files: list[Path]
    documents: list[Document]
    skipped: list[str]


def _matches_excluded(relative_path: str, parts: tuple[str, ...], excluded: set[str]) -> bool:
    for part in parts:
        if part in DEFAULT_EXCLUDED or part in excluded:
            return True

    return any(
        fnmatch.fnmatch(relative_path, pattern) or fnmatch.fnmatch(f"/{relative_path}", pattern)
        for pattern in excluded
    )


def _repo_root(path: Path) -> Path:
    if path.is_file():
        return path.parent
    if (path / ".git").exists():
        return path
    return path


def configured_code_repos(
    config: dict[str, Any],
    *,
    validate_path: Callable[[Path], Path] | None = None,
) -> list[CodeRepo]:
    code_config = config.get("code_retrieval", {})
    profile_name = code_config.get("default_profile", "projects")
    profile = code_config.get("profiles", {}).get(profile_name, {})
    raw_paths = profile.get("paths", code_config.get("paths", []))

    repos: list[CodeRepo] = []
    for raw_path in raw_paths:
        root = Path(raw_path).expanduser().resolve()
        if validate_path is not None:
            root = validate_path(root)
        if not root.exists():
            continue

        if (root / ".git").exists() or root.is_file():
            repo_path = _repo_root(root)
            repos.append(CodeRepo(name=repo_path.name, path=repo_path))
            continue

        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / ".git").exists():
                if validate_path is not None:
                    child = validate_path(child)
                repos.append(CodeRepo(name=child.name, path=child))

    unique: dict[str, CodeRepo] = {}
    for repo in repos:
        unique[str(repo.path)] = repo
    return list(unique.values())


def load_code_documents(
    config: dict[str, Any],
    *,
    max_chars: int,
    overlap_chars: int,
    validate_path: Callable[[Path], Path] | None = None,
) -> CodeLoadResult:
    code_config = config.get("code_retrieval", {})
    profile_name = code_config.get("default_profile", "projects")
    profile = code_config.get("profiles", {}).get(profile_name, {})
    allowed_extensions = {
        str(extension).lower()
        for extension in code_config.get(
            "allowed_extensions",
            [".py", ".md", ".yaml", ".yml", ".toml", ".json", ".sql", ".sh"],
        )
    }
    excluded = set(profile.get("excluded", code_config.get("excluded", [])))
    chunking_config = code_config.get("chunking", {})
    include_full_class_chunks = bool(chunking_config.get("include_full_class_chunks", False))
    include_method_chunks = bool(chunking_config.get("include_method_chunks", True))

    repos = configured_code_repos(config, validate_path=validate_path)
    documents: list[Document] = []
    files: list[Path] = []
    skipped: list[str] = []

    for repo in repos:
        for path in sorted(repo.path.rglob("*")):
            if not path.is_file():
                continue
            try:
                relative_path = path.relative_to(repo.path).as_posix()
            except ValueError:
                continue

            if _matches_excluded(relative_path, path.relative_to(repo.path).parts, excluded):
                skipped.append(relative_path)
                continue
            if path.suffix.lower() not in allowed_extensions:
                continue

            if validate_path is not None:
                path = validate_path(path)

            chunks = chunk_code_file(
                path,
                repo_name=repo.name,
                relative_path=relative_path,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
                include_full_class_chunks=include_full_class_chunks,
                include_method_chunks=include_method_chunks,
            )
            documents.extend(chunks_to_documents(chunks))
            files.append(path)

    return CodeLoadResult(repos=repos, files=files, documents=documents, skipped=skipped)
