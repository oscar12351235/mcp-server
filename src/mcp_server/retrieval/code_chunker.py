from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from mcp_server.retrieval.chunkers import (
    Chunk,
    _finalize_chunks,
    chunk_config,
    chunk_markdown,
    chunk_text,
)

LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".sql": "sql",
    ".sh": "shell",
}

CONFIG_FILENAMES = {
    "config.yaml",
    "config.yml",
    "pyproject.toml",
    "poetry.lock",
    ".gitlab-ci.yml",
}


def language_for_path(path: str | Path) -> str:
    return LANGUAGE_BY_EXTENSION.get(Path(path).suffix.lower(), "text")


def infer_file_role(relative_path: str | Path) -> str:
    path = Path(relative_path)
    parts = set(path.parts)
    name = path.name

    if "tests" in parts or name.startswith("test_"):
        return "test"
    if name in CONFIG_FILENAMES or path.suffix.lower() in {".yaml", ".yml", ".toml", ".json"}:
        return "config"
    if "docs" in parts or path.suffix.lower() == ".md":
        return "docs"
    if "scripts" in parts or path.suffix.lower() == ".sh":
        return "scripts"
    return "source"


def _import_names(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    module = node.module or ""
    return [f"{module}.{alias.name}" if module else alias.name for alias in node.names]


def _module_imports(tree: ast.Module) -> list[str]:
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            imports.extend(_import_names(node))
    return imports


def _node_source(lines: list[str], node: ast.AST) -> str:
    start_line = getattr(node, "lineno", 1)
    end_line = getattr(node, "end_lineno", start_line)
    return "\n".join(lines[start_line - 1 : end_line])


def _module_statement_symbol(path: Path, node: ast.AST) -> str:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                return target.id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return path.stem


def _class_summary(lines: list[str], node: ast.ClassDef) -> str:
    start_line = getattr(node, "lineno", 1)
    header = lines[start_line - 1].strip()
    parts = [header]
    docstring = ast.get_docstring(node)
    if docstring:
        parts.append(f'"""{docstring}"""')

    method_names = [
        child.name
        for child in node.body
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    if method_names:
        parts.append("methods: " + ", ".join(method_names))

    return "\n".join(parts)


def _base_code_metadata(
    path: Path,
    relative_path: str,
    repo_name: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_type": "code",
        "repo_name": repo_name,
        "file_path": relative_path,
        "absolute_file_path": str(path),
        "extension": path.suffix.lower(),
        "language": language_for_path(path),
        "file_role": infer_file_role(relative_path),
    }
    if extra:
        metadata.update({key: value for key, value in extra.items() if value is not None})
    return metadata


def chunk_python_code(
    path: str | Path,
    *,
    repo_name: str,
    relative_path: str,
    max_chars: int = 3000,
    overlap_chars: int = 300,
    include_full_class_chunks: bool = False,
    include_method_chunks: bool = True,
) -> list[Chunk]:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return chunk_text(
            path,
            text,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            source_type="code",
            chunk_type="python_parse_fallback",
            base_metadata=_base_code_metadata(
                path,
                relative_path,
                repo_name,
                {"symbol_type": "module"},
            ),
        )

    imports = _module_imports(tree)
    imports_text = ", ".join(imports)
    chunks: list[Chunk] = []
    module_nodes = [
        node
        for node in tree.body
        if not isinstance(
            node,
            ast.Import | ast.ImportFrom | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        )
        and hasattr(node, "lineno")
    ]

    if module_nodes:
        for node in module_nodes:
            start_line = getattr(node, "lineno", 1)
            end_line = getattr(node, "end_lineno", start_line)
            chunks.append(
                Chunk(
                    text=_node_source(lines, node),
                    source_path=str(path),
                    metadata=_base_code_metadata(
                        path,
                        relative_path,
                        repo_name,
                        {
                            "chunk_type": "python_module_statement",
                            "symbol_type": "module",
                            "symbol_name": _module_statement_symbol(path, node),
                            "start_line": start_line,
                            "end_line": end_line,
                            "imports": imports_text,
                        },
                    ),
                )
            )

    if imports:
        import_nodes = [node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom)]
        start_line = min(getattr(node, "lineno", 1) for node in import_nodes)
        end_line = max(
            getattr(node, "end_lineno", getattr(node, "lineno", 1)) for node in import_nodes
        )
        chunks.append(
            Chunk(
                text="\n".join(lines[start_line - 1 : end_line]),
                source_path=str(path),
                metadata=_base_code_metadata(
                    path,
                    relative_path,
                    repo_name,
                    {
                        "chunk_type": "python_imports",
                        "symbol_type": "imports",
                        "symbol_name": "__imports__",
                        "start_line": start_line,
                        "end_line": end_line,
                        "imports": imports_text,
                    },
                ),
            )
        )

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_text = (
                _node_source(lines, node)
                if include_full_class_chunks
                else _class_summary(lines, node)
            )
            chunks.append(
                Chunk(
                    text=class_text,
                    source_path=str(path),
                    metadata=_base_code_metadata(
                        path,
                        relative_path,
                        repo_name,
                        {
                            "chunk_type": "python_symbol",
                            "symbol_type": "class",
                            "symbol_name": node.name,
                            "start_line": node.lineno,
                            "end_line": getattr(node, "end_lineno", node.lineno),
                            "imports": imports_text,
                            "docstring": ast.get_docstring(node),
                        },
                    ),
                )
            )
            if include_method_chunks:
                for child in node.body:
                    if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                        chunks.append(
                            Chunk(
                                text=_node_source(lines, child),
                                source_path=str(path),
                                metadata=_base_code_metadata(
                                    path,
                                    relative_path,
                                    repo_name,
                                    {
                                        "chunk_type": "python_symbol",
                                        "symbol_type": "method",
                                        "symbol_name": child.name,
                                        "parent_symbol": node.name,
                                        "qualified_symbol": f"{node.name}.{child.name}",
                                        "start_line": child.lineno,
                                        "end_line": getattr(child, "end_lineno", child.lineno),
                                        "imports": imports_text,
                                        "docstring": ast.get_docstring(child),
                                    },
                                ),
                            )
                        )
            continue

        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            chunks.append(
                Chunk(
                    text=_node_source(lines, node),
                    source_path=str(path),
                    metadata=_base_code_metadata(
                        path,
                        relative_path,
                        repo_name,
                        {
                            "chunk_type": "python_symbol",
                            "symbol_type": "function",
                            "symbol_name": node.name,
                            "start_line": node.lineno,
                            "end_line": getattr(node, "end_lineno", node.lineno),
                            "imports": imports_text,
                            "docstring": ast.get_docstring(node),
                        },
                    ),
                )
            )

    if not chunks:
        chunks.append(
            Chunk(
                text=text,
                source_path=str(path),
                metadata=_base_code_metadata(
                    path,
                    relative_path,
                    repo_name,
                    {
                        "chunk_type": "python_module",
                        "symbol_type": "module",
                        "symbol_name": path.stem,
                        "start_line": 1,
                        "end_line": len(lines),
                        "imports": imports_text,
                    },
                ),
            )
        )

    return _finalize_chunks(chunks)


def chunk_code_file(
    path: str | Path,
    *,
    repo_name: str,
    relative_path: str,
    max_chars: int = 3000,
    overlap_chars: int = 300,
    include_full_class_chunks: bool = False,
    include_method_chunks: bool = True,
) -> list[Chunk]:
    path = Path(path)
    extension = path.suffix.lower()
    metadata = _base_code_metadata(path, relative_path, repo_name)

    if extension == ".py":
        return chunk_python_code(
            path,
            repo_name=repo_name,
            relative_path=relative_path,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            include_full_class_chunks=include_full_class_chunks,
            include_method_chunks=include_method_chunks,
        )

    if extension in {".md", ".markdown"}:
        return chunk_markdown(
            path,
            path.read_text(encoding="utf-8", errors="replace"),
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            base_metadata=metadata,
        )

    if extension in {".yaml", ".yml", ".json"}:
        return chunk_config(
            path,
            path.read_text(encoding="utf-8", errors="replace"),
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            base_metadata=metadata,
        )

    return chunk_text(
        path,
        path.read_text(encoding="utf-8", errors="replace"),
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        source_type="code",
        chunk_type="code_text",
        base_metadata=metadata,
    )
