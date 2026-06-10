from __future__ import annotations

import ast
import re
from typing import Any

from mcp_server.retrieval.bm25_store import bm25_retrieve, load_bm25_runtime_index
from mcp_server.retrieval.hybrid import Candidate, fuse_candidates
from mcp_server.retrieval.rerankers import apply_final_ranks, rerank_or_fallback


def dense_candidates_from_nodes(nodes: list[Any]) -> list[Candidate]:
    candidates: list[Candidate] = []

    for rank, node_with_score in enumerate(nodes, start=1):
        node = node_with_score.node
        metadata = dict(node.metadata)
        chunk_id = str(
            metadata.get("chunk_id")
            or f"{metadata.get('repo_name', '')}:{metadata.get('file_path', '')}:"
            f"{metadata.get('chunk_index', '')}"
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


def format_code_candidates(
    candidates: list[Candidate],
    *,
    max_chars: int = 1800,
) -> list[dict[str, Any]]:
    return [
        {
            "score": float(candidate.score),
            "metadata": candidate.metadata,
            "text": candidate.text[:max_chars],
        }
        for candidate in candidates
    ]


def filter_candidates_by_repo(candidates: list[Candidate], repo: str | None) -> list[Candidate]:
    if not repo:
        return candidates
    return [
        candidate
        for candidate in candidates
        if str(candidate.metadata.get("repo_name", "")).lower() == repo.lower()
    ]


def _query_terms(query: str) -> set[str]:
    return {term.lower() for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", query)}


def _candidate_search_text(candidate: Candidate) -> str:
    metadata = candidate.metadata
    parts = [
        candidate.text,
        str(metadata.get("file_path", "")),
        str(metadata.get("symbol_name", "")),
        str(metadata.get("qualified_symbol", "")),
        str(metadata.get("imports", "")),
    ]
    return " ".join(parts).lower()


ROLE_INTENT_TERMS = {
    "test": {"test", "tests", "pytest", "fixture", "fixtures", "mock", "mocks"},
    "config": {"config", "yaml", "yml", "toml", "pyproject", "poetry", "settings"},
    "docs": {"docs", "doc", "readme", "architecture", "phase", "guide"},
    "scripts": {"script", "scripts", "bash", "shell", "launcher"},
}


def _role_intent(query: str) -> str | None:
    terms = _query_terms(query)
    for role, role_terms in ROLE_INTENT_TERMS.items():
        if terms & role_terms:
            return role
    return None


def prefer_source_code_candidates(
    candidates: list[Candidate],
    *,
    query: str = "",
) -> list[Candidate]:
    default_role_weights = {
        "source": 1.35,
        "test": 1.15,
        "scripts": 1.05,
        "config": 0.95,
        "docs": 0.75,
    }
    intent = _role_intent(query)
    role_weights = dict(default_role_weights)
    if intent:
        role_weights[intent] = max(role_weights.get(intent, 1.0), 1.6)

    default_priority = ["source", "test", "scripts", "config", "docs"]
    if intent and intent in default_priority:
        priority_order = [intent] + [role for role in default_priority if role != intent]
    else:
        priority_order = default_priority
    role_priority = {role: index for index, role in enumerate(priority_order)}

    adjusted: list[Candidate] = []
    exact_terms = _query_terms(query)
    for candidate in candidates:
        role = str(candidate.metadata.get("file_role", "source"))
        weight = role_weights.get(role, 1.0)
        if role in {"source", "test", "scripts"} and exact_terms:
            search_text = _candidate_search_text(candidate)
            if any(term in search_text for term in exact_terms):
                weight *= 3.0
        # Rerankers can emit negative scores; dividing keeps "boost" meaning
        # "closer to the top" regardless of sign.
        if candidate.score >= 0:
            candidate.score *= weight
        else:
            candidate.score /= weight
        if candidate.fused_score is not None:
            candidate.fused_score *= weight
            candidate.metadata["fused_score"] = candidate.fused_score
        adjusted.append(candidate)

    adjusted.sort(
        key=lambda candidate: (
            role_priority.get(str(candidate.metadata.get("file_role", "source")), 2),
            -candidate.score,
        )
    )
    for rank, candidate in enumerate(adjusted, start=1):
        candidate.rank = rank
        candidate.metadata["fused_rank"] = rank
    return adjusted


def retrieve_code_candidates(
    query: str,
    *,
    dense_nodes: list[Any],
    bm25_index_path: str,
    repo: str | None,
    bm25_top_k: int,
    fused_top_k: int,
    fusion_method: str,
    reranker_enabled: bool,
    reranker_provider: str,
    reranker_model: str,
    reranker_candidate_top_k: int,
    final_top_k: int,
    reranker_batch_size: int,
    bm25_enabled: bool = True,
    logger: Any = None,
) -> dict[str, list[Candidate]]:
    dense_candidates = filter_candidates_by_repo(dense_candidates_from_nodes(dense_nodes), repo)
    bm25_candidates: list[Candidate] = []
    if bm25_enabled:
        bm25_index = load_bm25_runtime_index(bm25_index_path)
        bm25_candidates = filter_candidates_by_repo(
            bm25_retrieve(query, bm25_index, top_k=bm25_top_k),
            repo,
        )
        for term in _query_terms(query):
            bm25_candidates.extend(
                filter_candidates_by_repo(
                    bm25_retrieve(term, bm25_index, top_k=max(3, bm25_top_k // 6)),
                    repo,
                )
            )
    fused_candidates = fuse_candidates(
        dense_candidates,
        bm25_candidates,
        fused_top_k=fused_top_k,
        method=fusion_method,
    )
    fused_candidates = prefer_source_code_candidates(fused_candidates, query=query)
    # Rerank a pool larger than final_top_k, then re-apply role preference so the
    # cross-encoder cannot demote source files below docs again.
    rerank_pool_top_k = max(final_top_k, reranker_candidate_top_k)
    reranked_candidates = rerank_or_fallback(
        query,
        fused_candidates,
        enabled=reranker_enabled,
        provider=reranker_provider,
        model=reranker_model,
        candidate_top_k=reranker_candidate_top_k,
        final_top_k=rerank_pool_top_k,
        batch_size=reranker_batch_size,
        logger=logger,
    )
    if reranker_enabled:
        reranked_candidates = prefer_source_code_candidates(reranked_candidates, query=query)
    reranked_candidates = apply_final_ranks(reranked_candidates[:final_top_k])

    return {
        "dense": dense_candidates,
        "bm25": bm25_candidates,
        "fused": fused_candidates,
        "reranked": reranked_candidates,
    }


def locate_python_symbol(source: str, symbol_name: str) -> tuple[int, int] | None:
    """Locate a def/class/assignment by name in current source, returning 1-based lines.

    Accepts plain names ("retrieve_code"), qualified names ("Worker.run"),
    and the special "__imports__" symbol. Returns None when the symbol no
    longer exists, which callers should treat as a stale index.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    plain_name = symbol_name.rsplit(".", 1)[-1]

    if symbol_name == "__imports__":
        import_nodes = [node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom)]
        if not import_nodes:
            return None
        start = min(node.lineno for node in import_nodes)
        end = max(getattr(node, "end_lineno", node.lineno) for node in import_nodes)
        return start, end

    def assigned_names(node: ast.AST) -> list[str]:
        if isinstance(node, ast.Assign):
            return [target.id for target in node.targets if isinstance(target, ast.Name)]
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            return [node.target.id]
        return []

    def qualified(node: ast.AST, parents: list[str]) -> str:
        name = getattr(node, "name", "")
        return ".".join([*parents, name]) if parents else name

    matches: list[tuple[bool, int, int]] = []

    def visit(body: list[ast.stmt], parents: list[str]) -> None:
        for node in body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                if node.name == plain_name:
                    exact = qualified(node, parents) == symbol_name
                    matches.append((exact, node.lineno, getattr(node, "end_lineno", node.lineno)))
                if isinstance(node, ast.ClassDef):
                    visit(node.body, [*parents, node.name])
            elif plain_name in assigned_names(node):
                matches.append((not parents, node.lineno, getattr(node, "end_lineno", node.lineno)))

    visit(tree.body, [])
    if not matches:
        return None

    _, start, end = max(matches, key=lambda item: item[0])
    return start, end


def find_symbol_candidate(
    candidates: list[Candidate],
    *,
    repo: str,
    symbol_name: str,
    file_path: str | None = None,
) -> Candidate | None:
    for candidate in candidates:
        metadata = candidate.metadata
        if metadata.get("repo_name") != repo:
            continue
        symbols = {
            str(metadata.get("symbol_name", "")),
            str(metadata.get("qualified_symbol", "")),
        }
        if symbol_name not in symbols:
            continue
        if file_path and metadata.get("file_path") != file_path:
            continue
        return candidate
    return None
