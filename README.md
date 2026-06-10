# Personal MCP Server

Local MCP server for GitLab-hosted skills, local knowledge retrieval, code-aware repository retrieval, and persistent memory proposals.

Designed to be launched by Claude Code, Cursor, Codex, VS Code Copilot, PyCharm AI, or any MCP-compatible stdio client. The LLM is provided by the client; this server only does retrieval, file access, and proposal writing.

## Retrieval pipeline

```text
query
  ↓
dense retrieval (local ChromaDB) + BM25 lexical retrieval
  ↓
reciprocal rank fusion (RRF)
  ↓
optional cross-encoder reranking (BAAI/bge-reranker-v2-m3)
  ↓
role-aware re-weighting for code (source > tests > scripts > config > docs)
  ↓
final top-k chunks with full score breakdown
```

Four retrieval domains, each with its own Chroma collection:

* **skills** — markdown skills from the GitLab skills repo (dense only)
* **knowledge** — local notes/books/papers from the active knowledge profile (hybrid + rerank)
* **memory** — local markdown memory files (dense only)
* **code** — symbol-level chunks from configured local repos (hybrid + rerank, AST-aware)

Collection names embed the embedding profile, provider, model, dimension, chunking profile, and a config fingerprint, so changing any retrieval-relevant config automatically creates a new collection instead of silently reusing stale vectors. `cleanup_stale_collections` removes leftovers from old configs.

## Main components

* `src/mcp_server/mcp_server.py` — MCP server and tool definitions
* `src/mcp_server/retrieval/` — chunkers, BM25 store, hybrid fusion, rerankers, manifest, embeddings, code retrieval, token budget
* `scripts/evaluate_retrieval.py` — retrieval evaluation runner
* `eval/queries.yaml` — fixed evaluation query set; results land in `eval/results/` (gitignored), deliberate baselines in `eval/baselines/`
* `config.yaml` — paths, profiles, retrieval settings, security allowlists
* `.env` — secrets (`GITLAB_TOKEN`, optional `VOYAGE_API_KEY`); never committed, keep mode 600
* `storage/` — ChromaDB vectors, BM25 indexes, manifest SQLite; never committed

## Profiles

**Knowledge profiles** (`MCP_KNOWLEDGE_PROFILE`): `indexed` (curated daily set, default) or `full` (entire knowledge base; slow with raw PDFs).

**Embedding profiles** (`embedding.active_profile` in config.yaml):

* `local_bge_small` — default; fast local baseline (BAAI/bge-small-en-v1.5, 384d)
* `local_qwen3_4b` / `local_qwen3_8b` — optional heavier local models; never downloaded unless activated
* `voyage_*` — optional cloud embeddings, guarded (see Voyage safety below)

## Running

```bash
poetry run personal-mcp                          # default profile
MCP_KNOWLEDGE_PROFILE=indexed poetry run personal-mcp
```

Recommended launchers for MCP clients (in `~/.local/bin`): `personal-mcp-indexed` (daily) and `personal-mcp-full`. Both use the project venv and keep stdout clean for the MCP protocol; logs go to stderr.

## MCP tools

Retrieval:

* `retrieve_skills(query, top_k)` — GitLab skill chunks
* `retrieve_knowledge(query, top_k)` — local knowledge (hybrid + rerank)
* `retrieve_knowledge_debug(query, top_k)` — dense/BM25/fused/reranked candidate diagnostics
* `retrieve_code(query, repo, top_k)` — code chunks with file paths and line ranges
* `read_code_symbol(repo, symbol_name, file_path)` — exact current source for a symbol; re-locates Python symbols by AST and refuses to return stale slices
* `retrieve_memory(query, top_k)` — memory chunks

Files and GitLab:

* `read_local_file`, `list_local_files` — allowlisted local access
* `get_gitlab_file`, `list_gitlab_repo` — allowlisted GitLab access
* `propose_memory_update(category, content, reason)` — writes a proposal into the memory `_proposals` folder only
* `propose_new_skill(name, content, reason)` — creates a branch + merge request into the skills repo `_proposals` path; never writes to main

Indexing and status:

* `refresh_indexes(force_rebuild)` / `rebuild_indexes()` — load or fully rebuild collections (rebuild also rewrites BM25 indexes and the manifest)
* `scan_index_changes`, `list_changed_files`, `list_indexed_files`, `index_manifest_status` — manifest-based change detection (scan-only; does not embed or mutate Chroma)
* `index_changes(apply, confirm)` — dry-run token estimates; Voyage apply is intentionally stubbed pending incremental upserts
* `estimate_embedding_cost`, `embedding_budget_status` — token estimates and ledger
* `collection_status` (includes a `placeholder` flag for empty collections), `code_index_status`, `runtime_status`, `retrieval_config_status`, `embedding_config_status`, `knowledge_profile_status`, `list_code_repos`, `cleanup_stale_collections(dry_run)`

## Evaluation

```bash
poetry run python scripts/evaluate_retrieval.py --dry-run
poetry run python scripts/evaluate_retrieval.py --top-k 5 --note "baseline"
poetry run python scripts/evaluate_retrieval.py --domain code --retrieval-mode hybrid
poetry run python scripts/evaluate_retrieval.py --disable-reranker
```

`--retrieval-mode dense|dense_rerank|hybrid|hybrid_rerank`, `--disable-bm25`, and `--disable-reranker` override toggles in-process only; `config.yaml` is never modified. Results include expected-term/path hit rates, latencies, and the full runtime config for reproducibility.

## Security model

* Allowlisted `allowed_read_roots` / `allowed_write_roots` plus `never_read` / `never_write` blocklists (includes `.env`, `.venv`, `storage/`, and `config.yaml` itself)
* GitLab access restricted to allowlisted project IDs; writes only via branch + MR (`allow_direct_main_write: false`)
* Memory changes are proposals first (`memory/_proposals`); permanent memory is human-curated

### Voyage (cloud embedding) safety

* Voyage is never the default and is impossible to trigger accidentally:
  * building the backend object does not import `voyageai` or need an API key
  * `rebuild_indexes()` is refused while a Voyage profile is active
  * `index_changes` requires `apply=True` **and** `confirm=True`, checks token budgets against the ledger, and fails clearly if `VOYAGE_API_KEY` is missing
  * dry-runs record estimated tokens only; budget checks count actual tokens

## Development

```bash
poetry run black .
poetry run ruff check .
poetry run mypy .
poetry run pytest
```

## Current limitations

* Indexing is scan-only incremental: `scan_index_changes` detects added/edited/deleted/moved files cheaply, but applying changes still requires a full `rebuild_indexes()`. Per-file Chroma upserts (and the real Voyage apply path) are the next planned phase.
* BM25 term statistics are cached per process per index file, not persisted to disk.
* Full knowledge mode can be slow because large PDFs produce many chunks.
* Spreadsheet/image/table parsing is out of scope for the main RAG pipeline.
