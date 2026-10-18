from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz
import yaml
from llama_index.core import Document


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: str = ""
    chunk_index: int = 0
    chunk_id: str | None = None
    content_hash: str | None = None


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _chunk_id(source_path: str, chunk_index: int, content_hash: str) -> str:
    payload = f"{source_path}:{chunk_index}:{content_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _source_type_for_extension(extension: str) -> str:
    if extension in {".md", ".markdown"}:
        return "markdown"
    if extension == ".py":
        return "code"
    if extension == ".pdf":
        return "pdf"
    if extension in {".yaml", ".yml", ".json"}:
        return "config"
    return "text"


def _base_metadata(
    path: str | Path,
    source_type: str,
    extension: str,
    chunk_type: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_type": source_type,
        "file_path": str(path),
        "extension": extension,
        "chunk_type": chunk_type,
    }
    if extra:
        metadata.update({key: value for key, value in extra.items() if value is not None})
    return metadata


def _finalize_chunks(chunks: list[Chunk]) -> list[Chunk]:
    finalized: list[Chunk] = []

    for chunk in chunks:
        text = chunk.text.strip()
        if not text:
            continue

        content_hash = _text_hash(text)
        chunk.chunk_index = len(finalized)
        chunk.text = text
        chunk.content_hash = content_hash
        chunk.chunk_id = _chunk_id(chunk.source_path, chunk.chunk_index, content_hash)
        chunk.metadata["chunk_index"] = chunk.chunk_index
        chunk.metadata["content_hash"] = content_hash
        chunk.metadata["text_hash"] = content_hash
        chunk.metadata["chunk_id"] = chunk.chunk_id
        finalized.append(chunk)

    return finalized


def _split_blocks(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n+", text) if block.strip()]


def _split_large_text(
    text: str,
    max_chars: int,
    overlap_chars: int,
    *,
    split_pattern: str = r"\n\s*\n+",
) -> list[str]:
    blocks = [block.strip() for block in re.split(split_pattern, text) if block.strip()]
    if not blocks:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current = ""

    for block in blocks:
        separator = "\n\n" if current else ""
        candidate = f"{current}{separator}{block}"

        if len(candidate) <= max_chars or not current:
            current = candidate
            continue

        chunks.append(current)
        overlap = current[-overlap_chars:].strip() if overlap_chars > 0 else ""
        current = f"{overlap}\n\n{block}" if overlap else block

    if current:
        chunks.append(current)

    return chunks


def _chunk_with_metadata(
    text: str,
    metadata: dict[str, Any],
    source_path: str,
    max_chars: int,
    overlap_chars: int,
) -> list[Chunk]:
    parts = _split_large_text(text, max_chars, overlap_chars)
    return [Chunk(text=part, metadata=dict(metadata), source_path=source_path) for part in parts]


def chunk_markdown(
    path: str | Path,
    text: str,
    *,
    max_chars: int = 3000,
    overlap_chars: int = 300,
    base_metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    source_path = str(path)
    extension = Path(path).suffix.lower()
    heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_section_path = ""
    current_heading_level: int | None = None
    chunks: list[Chunk] = []

    def flush_section() -> None:
        if not current_lines:
            return
        section_text = "\n".join(current_lines).strip()
        metadata = _base_metadata(
            source_path,
            "markdown",
            extension,
            "markdown_section",
            {
                "section_path": current_section_path,
                "heading_level": current_heading_level,
                **(base_metadata or {}),
            },
        )
        chunks.extend(
            _chunk_with_metadata(section_text, metadata, source_path, max_chars, overlap_chars)
        )

    for line in text.splitlines():
        match = heading_re.match(line)
        if match:
            flush_section()
            current_lines = [line]
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack = [(lvl, name) for lvl, name in heading_stack if lvl < level]
            heading_stack.append((level, title))
            current_section_path = " > ".join(name for _, name in heading_stack)
            current_heading_level = level
            continue

        current_lines.append(line)

    flush_section()

    if not chunks and text.strip():
        metadata = _base_metadata(
            source_path,
            "markdown",
            extension,
            "markdown_section",
            base_metadata,
        )
        chunks.extend(_chunk_with_metadata(text, metadata, source_path, max_chars, overlap_chars))

    return _finalize_chunks(chunks)


def chunk_text(
    path: str | Path,
    text: str,
    *,
    max_chars: int = 3000,
    overlap_chars: int = 300,
    source_type: str = "text",
    chunk_type: str = "text_block",
    base_metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    source_path = str(path)
    extension = Path(path).suffix.lower()
    text = re.sub(r"[ \t]+", " ", text)
    metadata = _base_metadata(
        source_path,
        source_type,
        extension,
        chunk_type,
        base_metadata,
    )
    chunks = _chunk_with_metadata(text, metadata, source_path, max_chars, overlap_chars)
    return _finalize_chunks(chunks)


def _node_source(lines: list[str], node: ast.AST) -> str:
    start_line = getattr(node, "lineno", 1)
    end_line = getattr(node, "end_lineno", start_line)
    return "\n".join(lines[start_line - 1 : end_line])


def chunk_python(
    path: str | Path,
    text: str,
    *,
    max_chars: int = 3000,
    overlap_chars: int = 300,
    base_metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    source_path = str(path)
    extension = Path(path).suffix.lower()
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
            base_metadata={"language": "python", **(base_metadata or {})},
        )

    chunks: list[Chunk] = []
    import_nodes: list[ast.AST] = [
        node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom)
    ]

    if import_nodes:
        start_line = min(getattr(node, "lineno", 1) for node in import_nodes)
        import_end_lines = [
            getattr(node, "end_lineno", getattr(node, "lineno", 1)) for node in import_nodes
        ]
        end_line = max(import_end_lines)
        metadata = _base_metadata(
            source_path,
            "code",
            extension,
            "python_imports",
            {
                "language": "python",
                "symbol_type": "imports",
                "start_line": start_line,
                "end_line": end_line,
                **(base_metadata or {}),
            },
        )
        chunks.extend(
            _chunk_with_metadata(
                "\n".join(lines[start_line - 1 : end_line]),
                metadata,
                source_path,
                max_chars,
                overlap_chars,
            )
        )

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue

        symbol_type = "class" if isinstance(node, ast.ClassDef) else "function"
        start_line = getattr(node, "lineno", 1)
        end_line = getattr(node, "end_lineno", start_line)
        metadata = _base_metadata(
            source_path,
            "code",
            extension,
            "python_symbol",
            {
                "language": "python",
                "symbol_type": symbol_type,
                "symbol_name": node.name,
                "start_line": start_line,
                "end_line": end_line,
                **(base_metadata or {}),
            },
        )
        chunks.extend(
            _chunk_with_metadata(
                _node_source(lines, node),
                metadata,
                source_path,
                max_chars,
                overlap_chars,
            )
        )

    if not chunks:
        return chunk_text(
            path,
            text,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            source_type="code",
            chunk_type="python_module",
            base_metadata={"language": "python", **(base_metadata or {})},
        )

    return _finalize_chunks(chunks)


def _flatten_config_items(value: Any, path: str = "") -> list[tuple[str, str]]:
    if isinstance(value, dict):
        items: list[tuple[str, str]] = []
        for key, nested in value.items():
            nested_path = f"{path}.{key}" if path else str(key)
            items.extend(_flatten_config_items(nested, nested_path))
        return items
    if isinstance(value, list):
        items = []
        for index, nested in enumerate(value):
            nested_path = f"{path}[{index}]" if path else f"[{index}]"
            items.extend(_flatten_config_items(nested, nested_path))
        return items
    return [(path, json.dumps(value, ensure_ascii=False))]


def chunk_config(
    path: str | Path,
    text: str,
    *,
    max_chars: int = 3000,
    overlap_chars: int = 300,
    base_metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    source_path = str(path)
    extension = Path(path).suffix.lower()

    try:
        parsed = json.loads(text) if extension == ".json" else yaml.safe_load(text)
    except Exception:
        parsed = None

    if parsed is None:
        return chunk_text(
            path,
            text,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            source_type="config",
            chunk_type="config_text",
            base_metadata=base_metadata,
        )

    chunks: list[Chunk] = []
    for key_path, value_text in _flatten_config_items(parsed):
        line = f"{key_path}: {value_text}" if key_path else value_text
        metadata = _base_metadata(
            source_path,
            "config",
            extension,
            "config_key",
            {
                "key_path": key_path,
                **(base_metadata or {}),
            },
        )
        chunks.extend(_chunk_with_metadata(line, metadata, source_path, max_chars, overlap_chars))

    return _finalize_chunks(chunks)


def chunk_pdf(
    path: str | Path,
    *,
    max_chars: int = 3000,
    overlap_chars: int = 300,
    base_metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    source_path = str(path)
    extension = Path(path).suffix.lower()
    chunks: list[Chunk] = []

    with fitz.open(source_path) as pdf:
        for page_index, page in enumerate(pdf):
            text = page.get_text("text").strip()
            if not text:
                continue
            metadata = _base_metadata(
                source_path,
                "pdf",
                extension,
                "pdf_page",
                {
                    "page_start": page_index + 1,
                    "page_end": page_index + 1,
                    "parser": "pymupdf",
                    **(base_metadata or {}),
                },
            )
            chunks.extend(
                _chunk_with_metadata(
                    text,
                    metadata,
                    source_path,
                    max_chars,
                    overlap_chars,
                )
            )

    return _finalize_chunks(chunks)


def chunk_file(
    path: str | Path,
    *,
    max_chars: int = 3000,
    overlap_chars: int = 300,
    base_metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    path = Path(path)
    extension = path.suffix.lower()

    if extension == ".pdf":
        return chunk_pdf(
            path,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            base_metadata=base_metadata,
        )

    text = path.read_text(encoding="utf-8", errors="replace")

    if extension in {".md", ".markdown"}:
        return chunk_markdown(
            path,
            text,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            base_metadata=base_metadata,
        )
    if extension == ".py":
        return chunk_python(
            path,
            text,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            base_metadata=base_metadata,
        )
    if extension in {".yaml", ".yml", ".json"}:
        return chunk_config(
            path,
            text,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            base_metadata=base_metadata,
        )

    return chunk_text(
        path,
        text,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        source_type=_source_type_for_extension(extension),
        base_metadata=base_metadata,
    )


def chunks_to_documents(chunks: list[Chunk]) -> list[Document]:
    return [Document(text=chunk.text, metadata=chunk.metadata) for chunk in chunks]
