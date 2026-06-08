from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llama_index.core import Document

from mcp_server.retrieval.hybrid import Candidate

TOKEN_RE = re.compile(r"[a-zA-Z0-9_./:=-]+")
SEARCHABLE_METADATA_FIELDS = (
    "file_path",
    "section_path",
    "symbol_name",
    "symbol_type",
    "key_path",
    "chunk_type",
    "source_type",
    "extension",
    "language",
    "repo_name",
    "qualified_symbol",
    "parent_symbol",
    "file_role",
    "imports",
)


@dataclass(frozen=True)
class BM25Document:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []

    for match in TOKEN_RE.finditer(text.lower()):
        token = match.group(0)
        tokens.append(token)
        tokens.extend(part for part in re.split(r"[^a-z0-9_]+", token) if part)

    return tokens


def _metadata_value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(part for item in value if (part := _metadata_value_to_text(item)))
    if isinstance(value, dict):
        return " ".join(
            part for key in sorted(value) if (part := _metadata_value_to_text(value[key]))
        )
    return str(value).strip()


def bm25_searchable_text(content: str, metadata: dict[str, Any]) -> str:
    parts = [content.strip()] if content.strip() else []

    for field in SEARCHABLE_METADATA_FIELDS:
        value = _metadata_value_to_text(metadata.get(field))
        if value:
            parts.append(value)

    return "\n".join(parts)


def document_to_bm25_document(document: Document) -> BM25Document:
    metadata = dict(document.metadata)
    text = bm25_searchable_text(document.get_content(), metadata)
    chunk_id = str(
        metadata.get("chunk_id")
        or f"{metadata.get('file_path', '')}:{metadata.get('chunk_index', '')}"
    )
    return BM25Document(chunk_id=chunk_id, text=text, metadata=metadata)


def write_bm25_index(path: str | Path, documents: list[Document]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            bm25_document = document_to_bm25_document(document)
            handle.write(
                json.dumps(
                    {
                        "chunk_id": bm25_document.chunk_id,
                        "text": bm25_document.text,
                        "metadata": bm25_document.metadata,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def load_bm25_index(path: str | Path) -> list[BM25Document]:
    path = Path(path)
    if not path.exists():
        return []

    documents = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            documents.append(
                BM25Document(
                    chunk_id=str(data["chunk_id"]),
                    text=str(data["text"]),
                    metadata=dict(data["metadata"]),
                )
            )

    return documents


def bm25_retrieve(
    query: str,
    documents: list[BM25Document],
    *,
    top_k: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[Candidate]:
    if not documents:
        return []

    tokenized_docs = [tokenize(document.text) for document in documents]
    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    doc_count = len(documents)
    doc_lengths = [len(tokens) for tokens in tokenized_docs]
    avg_doc_length = sum(doc_lengths) / max(1, doc_count)
    doc_freq: Counter[str] = Counter()

    for tokens in tokenized_docs:
        doc_freq.update(set(tokens))

    scores: list[tuple[int, float]] = []

    for index, tokens in enumerate(tokenized_docs):
        if not tokens:
            continue

        frequencies = Counter(tokens)
        doc_length = doc_lengths[index]
        score = 0.0

        for token in query_tokens:
            freq = frequencies.get(token, 0)
            if freq == 0:
                continue

            df = doc_freq[token]
            idf = math.log(1.0 + ((doc_count - df + 0.5) / (df + 0.5)))
            denominator = freq + k1 * (1.0 - b + b * (doc_length / avg_doc_length))
            score += idf * ((freq * (k1 + 1.0)) / denominator)

        if score > 0:
            scores.append((index, score))

    scores.sort(key=lambda item: item[1], reverse=True)

    candidates: list[Candidate] = []
    for rank, (index, score) in enumerate(scores[:top_k], start=1):
        document = documents[index]
        candidates.append(
            Candidate(
                chunk_id=document.chunk_id,
                text=document.text,
                metadata=dict(document.metadata),
                score=score,
                source="bm25",
                rank=rank,
                bm25_score=score,
                retrieval_sources={"bm25"},
            )
        )

    return candidates
