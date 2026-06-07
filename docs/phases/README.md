# Implementation Phases Overview

This folder contains phase-by-phase implementation guides for evolving the MCP/RAG server from the current dense-only prototype into a local-first hybrid retrieval system.

## Phase order

```text
PHASE_00_BASELINE_AND_CONNECTION
  ↓
PHASE_01_CONFIG_AND_COLLECTION_SAFETY
  ↓
PHASE_02_STRUCTURE_AWARE_CHUNKING
  ↓
PHASE_03_INCREMENTAL_INDEXING_MANIFEST
  ↓
PHASE_04_BM25_HYBRID_RETRIEVAL
  ↓
PHASE_05_RERANKING
  ↓
PHASE_06_LOCAL_QWEN_EMBEDDINGS
  ↓
PHASE_07_CODE_AWARE_REPO_RETRIEVAL
  ↓
PHASE_08_VOYAGE_OPTIONAL_BACKEND
  ↓
PHASE_09_EVALUATION_AND_OBSERVABILITY
```

The phases are ordered to minimize risk. Do not jump directly to Qwen, reranking, or Voyage until the MCP client loop and baseline retrieval are working.

## Output standard for every phase

Every phase should produce:

```text
- code changes
- config changes if needed
- diagnostics or MCP tools to verify behavior
- tests or manual verification commands
- updated docs
- no silent permission expansion
```

## Agent instructions

When Claude/Codex uses these files as context, it should:

1. Read `docs/ARCHITECTURE.md` first.
2. Read the current phase file.
3. Implement only the current phase unless asked otherwise.
4. Preserve backward compatibility where practical.
5. Prefer small commits and testable steps.
6. Never delete or rewrite `.env`, `.venv`, `storage/`, or user files.
