# Agent Build Instructions for Claude/Codex

This file is intended for AI coding agents working on this repository.

## How to use the docs

Read in this order:

1. `docs/ARCHITECTURE.md`
2. `docs/phases/README.md`
3. the phase file requested by the user
4. current code in `src/mcp_server/`
5. `config.yaml`
6. `pyproject.toml`

Do not implement multiple phases at once unless explicitly instructed.

---

## Default development rules

1. Preserve the current working MCP server behavior.
2. Do not broaden file permissions unless explicitly asked.
3. Do not commit `.env`, `.venv`, or `storage/`.
4. Do not delete user files.
5. Keep launch scripts stable.
6. Keep indexed and full knowledge profiles separate.
7. Never mix embeddings from different models in the same Chroma collection.
8. Use dry-run modes before destructive or expensive operations.
9. Add diagnostics before adding complexity.
10. Prefer small, testable changes.

---

## Current priority order

The safest order is:

```text
1. Connect MCP client with current simple retrieval.
2. Harden config and collection naming.
3. Add structure-aware chunking.
4. Add manifest and incremental indexing.
5. Add BM25 hybrid retrieval.
6. Add reranking.
7. Add Qwen embeddings.
8. Add code-aware repo retrieval.
9. Add optional Voyage backend.
10. Add evaluation and observability.
```

---

## Before modifying code

Run:

```bash
cd "/Users/oscarbonnet/PycharmProjects/mcp-server"
poetry run python -m py_compile src/mcp_server/mcp_server.py
```

Check launcher scripts:

```bash
which personal-mcp-indexed
which personal-mcp-full
```

---

## After modifying code

At minimum run:

```bash
poetry run python -m py_compile src/mcp_server/mcp_server.py
```

If tests exist:

```bash
poetry run pytest
```

If changing retrieval/indexing:

```bash
MCP_KNOWLEDGE_PROFILE=indexed poetry run python - <<'PY'
from mcp_server.mcp_server import collection_status
print(collection_status())
PY
```

---

## Expected safety behavior

Any tool that writes should:

```text
- check allowed_write_roots
- check never_write
- default to proposal/dry-run where possible
- avoid direct permanent mutation
```

Any tool that embeds or indexes large files should:

```text
- support dry-run
- show files affected
- show estimated chunks/tokens
- require apply=True for changes
```
