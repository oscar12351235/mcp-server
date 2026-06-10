"""Run a fixed query set through the MCP retrieval tools and save results.

Usage:
    poetry run python scripts/evaluate_retrieval.py
    poetry run python scripts/evaluate_retrieval.py --queries eval/queries.yaml
    poetry run python scripts/evaluate_retrieval.py --query-ids pca_markets,hybrid_fusion
    poetry run python scripts/evaluate_retrieval.py --top-k 8 --tag finance
    poetry run python scripts/evaluate_retrieval.py --dry-run  # list queries without running

Output:
    eval/results/YYYYMMDD-HHMMSS-<embedding_profile>-<knowledge_profile>.json

The script imports the MCP server module directly (it does not spawn the MCP
server). It exercises retrieve_skills / retrieve_knowledge / retrieve_code /
retrieve_memory exactly as an MCP client would, then writes the structured
results to disk for manual inspection.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_QUERIES_PATH = PROJECT_ROOT / "eval" / "queries.yaml"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

DOMAIN_DISPATCH = {
    "skills": "retrieve_skills",
    "knowledge": "retrieve_knowledge",
    "code": "retrieve_code",
    "memory": "retrieve_memory",
}


def _load_queries(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    queries = data.get("queries") if isinstance(data, dict) else None
    if not queries:
        raise ValueError(f"No 'queries' list found in {path}")
    return queries


def _select_queries(
    queries: list[dict[str, Any]],
    *,
    query_ids: set[str] | None,
    tag: str | None,
    domain: str | None,
) -> list[dict[str, Any]]:
    selected = []
    for query in queries:
        if query_ids is not None and query.get("id") not in query_ids:
            continue
        if tag and tag not in (query.get("tags") or []):
            continue
        if domain and query.get("domain") != domain:
            continue
        selected.append(query)
    return selected


def _expected_terms_hits(results: list[dict[str, Any]], expected_terms: list[str]) -> list[str]:
    if not expected_terms:
        return []
    haystacks = []
    for item in results:
        text = item.get("text", "") or ""
        metadata = item.get("metadata") or {}
        haystacks.append(
            " ".join(
                [
                    text,
                    json.dumps(metadata, default=str),
                    item.get("source_path") or "",
                ]
            ).lower()
        )
    hits = []
    for term in expected_terms:
        lowered = term.lower()
        if any(lowered in haystack for haystack in haystacks):
            hits.append(term)
    return hits


def _expected_paths_hits(results: list[dict[str, Any]], expected_paths_any: list[str]) -> list[str]:
    if not expected_paths_any:
        return []
    paths = [
        (item.get("source_path") or (item.get("metadata") or {}).get("file_path") or "").lower()
        for item in results
    ]
    hits = []
    for candidate in expected_paths_any:
        lowered = candidate.lower()
        if any(lowered in path for path in paths):
            hits.append(candidate)
    return hits


def _summarize_result(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") or {}
    text = item.get("text") or ""
    return {
        "rank": item.get("rank"),
        "score": item.get("score"),
        "scores": item.get("scores"),
        "source_path": item.get("source_path") or metadata.get("file_path"),
        "retrieval_mode": item.get("retrieval_mode"),
        "collection": item.get("collection"),
        "embedding_profile": item.get("embedding_profile"),
        "reranker_enabled": item.get("reranker_enabled"),
        "symbol_name": metadata.get("symbol_name"),
        "symbol_type": metadata.get("symbol_type"),
        "start_line": metadata.get("start_line"),
        "end_line": metadata.get("end_line"),
        "chunk_index": metadata.get("chunk_index"),
        "text_preview": text[:400],
    }


def _run_one_query(server: Any, query: dict[str, Any], top_k: int | None) -> dict[str, Any]:
    domain = query.get("domain", "knowledge")
    fn_name = DOMAIN_DISPATCH.get(domain)
    if fn_name is None:
        return {
            "query_id": query.get("id"),
            "domain": domain,
            "error": f"Unsupported domain: {domain}",
        }

    fn = getattr(server, fn_name)
    started = time.perf_counter()
    error: str | None = None
    results: list[dict[str, Any]] = []
    try:
        results = fn(query["query"], top_k=top_k) if top_k else fn(query["query"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = int((time.perf_counter() - started) * 1000)

    expected_terms = query.get("expected_terms") or []
    expected_paths = query.get("expected_paths_any") or []
    terms_hit = _expected_terms_hits(results, expected_terms)
    paths_hit = _expected_paths_hits(results, expected_paths)

    head = results[0] if results else {}
    return {
        "query_id": query.get("id"),
        "query": query.get("query"),
        "domain": domain,
        "tags": query.get("tags") or [],
        "top_k_requested": top_k,
        "latency_ms": latency_ms,
        "error": error,
        "candidate_count": len(results),
        "retrieval_mode": head.get("retrieval_mode"),
        "collection": head.get("collection"),
        "embedding_profile": head.get("embedding_profile"),
        "reranker_enabled": head.get("reranker_enabled"),
        "expected_terms": expected_terms,
        "expected_terms_found": terms_hit,
        "expected_terms_hit_rate": (
            round(len(terms_hit) / len(expected_terms), 3) if expected_terms else None
        ),
        "expected_paths_any": expected_paths,
        "expected_paths_found": paths_hit,
        "expected_paths_hit": bool(paths_hit) if expected_paths else None,
        "top_results": [_summarize_result(item) for item in results],
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    errored = sum(1 for r in results if r.get("error"))
    term_runs = [r for r in results if r.get("expected_terms_hit_rate") is not None]
    avg_term_hit = (
        round(sum(r["expected_terms_hit_rate"] for r in term_runs) / len(term_runs), 3)
        if term_runs
        else None
    )
    path_runs = [r for r in results if r.get("expected_paths_hit") is not None]
    avg_path_hit = (
        round(sum(1 for r in path_runs if r["expected_paths_hit"]) / len(path_runs), 3)
        if path_runs
        else None
    )
    latencies = [r["latency_ms"] for r in results if isinstance(r.get("latency_ms"), int)]
    return {
        "queries_total": total,
        "queries_with_errors": errored,
        "mean_expected_terms_hit_rate": avg_term_hit,
        "mean_expected_paths_hit": avg_path_hit,
        "latency_ms": {
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
            "mean": int(sum(latencies) / len(latencies)) if latencies else None,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--query-ids",
        type=str,
        default=None,
        help="Comma-separated query ids to run (default: all).",
    )
    parser.add_argument("--tag", type=str, default=None, help="Filter queries by tag.")
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        choices=sorted(DOMAIN_DISPATCH),
        help="Filter queries by retrieval domain.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override top_k for every query (defaults to per-domain config).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List selected queries and exit without running retrieval.",
    )
    parser.add_argument(
        "--retrieval-mode",
        type=str,
        default=None,
        choices=["dense", "dense_rerank", "hybrid", "hybrid_rerank"],
        help="Override BM25/reranker toggles in-process (config.yaml is not modified).",
    )
    parser.add_argument(
        "--disable-bm25",
        action="store_true",
        help="Disable BM25 hybrid retrieval for this run only.",
    )
    parser.add_argument(
        "--disable-reranker",
        action="store_true",
        help="Disable the reranker for this run only.",
    )
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="Free-form note appended to the run metadata.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    queries = _load_queries(args.queries)
    query_ids = set(args.query_ids.split(",")) if args.query_ids else None
    selected = _select_queries(
        queries,
        query_ids=query_ids,
        tag=args.tag,
        domain=args.domain,
    )

    if not selected:
        print("No queries matched the given filters.", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"Would run {len(selected)} queries:")
        for q in selected:
            print(f"  - {q.get('id'):<28} [{q.get('domain')}] {q.get('query')}")
        return 0

    print(f"Loading MCP server module... ({len(selected)} queries selected)", file=sys.stderr)
    from mcp_server import mcp_server as server  # noqa: E402

    # In-process retrieval mode overrides; persisted config.yaml is untouched.
    if args.retrieval_mode:
        server.CONFIG["retrieval"]["bm25_enabled"] = args.retrieval_mode.startswith("hybrid")
        server.CONFIG["reranker"]["enabled"] = args.retrieval_mode.endswith("rerank")
    if args.disable_bm25:
        server.CONFIG["retrieval"]["bm25_enabled"] = False
    if args.disable_reranker:
        server.CONFIG["reranker"]["enabled"] = False

    server._ensure_indexes()
    if any(q.get("domain") == "code" for q in selected) and server._code_retrieval_enabled():
        server._ensure_code_index()

    runtime = server.runtime_status()
    retrieval_cfg = server.retrieval_config_status()

    print("Running queries...", file=sys.stderr)
    started = time.perf_counter()
    per_query = []
    for query in selected:
        print(f"  - {query.get('id')}", file=sys.stderr)
        per_query.append(_run_one_query(server, query, args.top_k))
    total_ms = int((time.perf_counter() - started) * 1000)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    embedding_profile = runtime.get("embedding_profile", "unknown")
    knowledge_profile = runtime.get("knowledge_profile", "unknown")
    run_id = f"{timestamp}-{embedding_profile}-{knowledge_profile}"

    payload = {
        "run_id": run_id,
        "started_at": timestamp,
        "duration_ms": total_ms,
        "note": args.note,
        "filters": {
            "query_ids": sorted(query_ids) if query_ids else None,
            "tag": args.tag,
            "domain": args.domain,
            "top_k_override": args.top_k,
            "retrieval_mode_override": args.retrieval_mode,
            "disable_bm25": args.disable_bm25,
            "disable_reranker": args.disable_reranker,
        },
        "runtime": runtime,
        "retrieval_config": retrieval_cfg,
        "aggregate": _aggregate(per_query),
        "results": per_query,
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / f"{run_id}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out_path}", file=sys.stderr)

    agg = payload["aggregate"]
    print(
        f"Done. queries={agg['queries_total']} errors={agg['queries_with_errors']} "
        f"mean_terms_hit={agg['mean_expected_terms_hit_rate']} "
        f"mean_path_hit={agg['mean_expected_paths_hit']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
