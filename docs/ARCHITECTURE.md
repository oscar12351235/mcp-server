# MCP/RAG Repo Architecture

## 0. Purpose

This repository implements a **local MCP server** that gives IDE agents such as Claude Code, Codex, VS Code Copilot, and PyCharm AI controlled access to:

- reusable skills stored in a GitLab repository;
- local knowledge files, notes, books, articles, and indexed summaries;
- local long-term memory files and memory proposals;
- selected local project files;
- retrieval tools backed by LlamaIndex, ChromaDB, embeddings, BM25, and reranking.

The system is designed to be **local-first**, **security-aware**, and **upgradeable** from the current dense-only proof of concept to a stronger hybrid retrieval stack.

The intended long-term target is:

```text
MCP client: Claude Code / Codex / Copilot / PyCharm AI
        ↓
MCP server: local Python process
        ↓
Tools: skills, knowledge, memory, GitLab, files, diagnostics
        ↓
Retrieval pipeline:
  file scanners → structure-aware chunkers → embedding backend
  + BM25 lexical index → hybrid fusion → reranker → top context
        ↓
Stores:
  ChromaDB vector collections
  BM25 index store
  SQLite manifest / indexing ledger
  GitLab skills repo
  local memory repo/folder
```

The LLM is **not** hosted inside this repo. Claude/Codex/Copilot/PyCharm is the LLM. This repo exposes tools and retrieval context.

---

## 1. Current repo layout

Current expected layout:

```text
mcp-server/
  src/
    mcp_server/
      __init__.py
      __main__.py
      mcp_server.py

  config.yaml
  .env                  # not committed
  pyproject.toml
  poetry.lock
  README.md

  storage/              # not committed
    chroma/

  docs/
    ARCHITECTURE.md
    phases/
      PHASE_00_BASELINE_AND_CONNECTION.md
      PHASE_01_CONFIG_AND_COLLECTION_SAFETY.md
      PHASE_02_STRUCTURE_AWARE_CHUNKING.md
      PHASE_03_INCREMENTAL_INDEXING_MANIFEST.md
      PHASE_04_BM25_HYBRID_RETRIEVAL.md
      PHASE_05_RERANKING.md
      PHASE_06_LOCAL_QWEN_EMBEDDINGS.md
      PHASE_07_CODE_AWARE_REPO_RETRIEVAL.md
      PHASE_08_VOYAGE_OPTIONAL_BACKEND.md
      PHASE_09_EVALUATION_AND_OBSERVABILITY.md
```

Future target layout after refactor:

```text
mcp-server/
  src/
    mcp_server/
      __init__.py
      __main__.py
      mcp_server.py              # thin app entrypoint only

      config/
        loader.py
        schema.py

      security/
        paths.py
        gitlab.py

      gitlab/
        client.py
        skills.py

      loaders/
        base.py
        markdown_loader.py
        pdf_loader.py
        code_loader.py
        table_loader.py
        repo_loader.py

      chunking/
        base.py
        markdown_chunker.py
        pdf_chunker.py
        python_ast_chunker.py
        latex_chunker.py
        generic_text_chunker.py

      embeddings/
        base.py
        local_huggingface.py
        qwen.py
        voyage.py
        registry.py

      indexing/
        manifest.py
        scanner.py
        change_detector.py
        indexer.py
        collections.py
        token_budget.py

      retrieval/
        dense.py
        bm25.py
        fusion.py
        reranker.py
        pipeline.py

      tools/
        diagnostics.py
        skills_tools.py
        knowledge_tools.py
        memory_tools.py
        gitlab_tools.py
        file_tools.py

  storage/
    chroma/
    bm25/
    manifest.sqlite
    logs/
```

Do not refactor into this structure all at once. The phases document the safe build order.

---

## 2. Current runtime architecture

### 2.1 MCP server

The MCP server is created with:

```python
mcp = FastMCP("personal-assistant")
```

Every function decorated with `@mcp.tool()` becomes callable by an MCP client.

Current MCP tools include:

```text
collection_status()
refresh_indexes(force_rebuild=False)
rebuild_indexes()
knowledge_profile_status()
index_status()
retrieve_skills(query, top_k=None)
retrieve_knowledge(query, top_k=None)
retrieve_memory(query, top_k=None)
read_local_file(path)
list_local_files(directory)
get_gitlab_file(file_path, project_id=None, branch=None)
list_gitlab_repo(project_id=None, repo_path="", branch=None)
propose_memory_update(category, content, reason)
propose_new_skill(name, content, reason)
```

### 2.2 Launcher scripts

The server should be launched through stable scripts in:

```text
/Users/oscarbonnet/.local/bin/personal-mcp-indexed
/Users/oscarbonnet/.local/bin/personal-mcp-full
```

Their role is to provide stable executable commands for MCP clients.

They set:

```bash
MCP_KNOWLEDGE_PROFILE=indexed
```

or:

```bash
MCP_KNOWLEDGE_PROFILE=full
```

then execute:

```bash
/Users/oscarbonnet/PycharmProjects/mcp-server/.venv/bin/python -m mcp_server
```

MCP clients should call the launcher scripts directly. The user may run them manually for startup checks, but should not type into the terminal after launch because the server expects JSON-RPC messages from an MCP client.

---

## 3. Knowledge profiles

The server supports at least two knowledge profiles:

```yaml
knowledge:
  default_profile: "indexed"

  profiles:
    indexed:
      paths:
        - "/Users/oscarbonnet/Documents/80–89 Knowledge Base/87 Indexed"

    full:
      paths:
        - "/Users/oscarbonnet/Documents/80–89 Knowledge Base"
```

### `indexed`

Fast daily mode. It should contain curated files:

```text
87 Indexed/
  finance/
  math/
  computer_science/
  machine_learning/
  project_notes/
```

Prefer:

```text
.md summaries
.txt notes
selected PDFs only
code/documentation snippets
clean exported notes
```

### `full`

Comprehensive slow mode. It indexes the whole knowledge base. Use only when the user explicitly needs full-library search.

---

## 4. Current retrieval architecture

Current retrieval path:

```text
source files / GitLab files
        ↓
LlamaIndex SimpleDirectoryReader
        ↓
LlamaIndex Document objects
        ↓
HuggingFaceEmbedding model
        ↓
ChromaDB local persistent vector store
        ↓
VectorStoreIndex retriever
        ↓
MCP tool response
        ↓
Claude/Codex/Copilot generates final answer
```

### ChromaDB role

ChromaDB stores vectors locally in:

```text
storage/chroma/
```

Current collections:

```text
skills
knowledge_indexed
knowledge_full
memory
```

Future collections must include embedding backend/model/dimension, e.g.:

```text
knowledge_indexed_qwen3_embedding_4b_2560
knowledge_indexed_qwen3_embedding_8b_4096
knowledge_indexed_voyage_4_large_1024
```

This prevents querying vectors produced by one embedding model with another embedding model.

### LlamaIndex role

LlamaIndex is currently used for:

- loading files through `SimpleDirectoryReader`;
- representing content as `Document` objects;
- wrapping ChromaDB through `ChromaVectorStore`;
- building `VectorStoreIndex` objects;
- creating retrievers with `index.as_retriever()`.

LlamaIndex is **not** used as an LLM. The code sets:

```python
Settings.llm = None
```

### Embedding model role

The embedding model converts text into vectors. Current baseline:

```yaml
retrieval:
  embedding_model: "BAAI/bge-small-en-v1.5"
```

Target local-first options:

```text
Qwen/Qwen3-Embedding-4B   # default target for higher-quality local retrieval
Qwen/Qwen3-Embedding-8B   # high-quality but heavier mode
BAAI/bge-small-en-v1.5    # fast baseline
BAAI/bge-m3               # fallback multilingual/local option
```

Cloud fallback options may include Voyage models but should remain optional.

---

## 5. Target retrieval architecture

The target system should not rely on dense retrieval alone.

Target pipeline:

```text
User query
  ↓
query normalization / optional query expansion
  ↓
Dense retrieval with embedding model
  +
BM25 lexical retrieval
  ↓
Candidate fusion, usually Reciprocal Rank Fusion
  ↓
Cross-encoder reranking with BGE-reranker-v2-m3
  ↓
Return top N chunks with metadata to MCP client
```

Why hybrid retrieval is needed:

- dense embeddings capture semantic similarity;
- BM25 captures exact terms, filenames, class names, symbols, equations, package names, tickers, and error messages;
- reranking improves final precision.

---

## 6. Code-aware repo retrieval

Repos should not be indexed exactly like books.

Code retrieval needs:

```text
repo scanner
language detection
AST chunking for Python
symbol extraction
file role metadata
path-aware BM25
exact read_local_file fallback
```

For Python, chunks should align with:

```text
module imports
classes
functions
methods
top-level constants
configuration files
tests
```

Metadata should include:

```yaml
source_type: code
repo_name: mcp-server
language: python
file_path: src/mcp_server/mcp_server.py
symbol_type: function
symbol_name: retrieve_knowledge
parent_symbol: null
start_line: 123
end_line: 151
```

A good code retrieval answer should usually retrieve candidates first, then call `read_local_file()` or a future `read_code_symbol()` tool for exact inspection.

---

## 7. Skills repo architecture

The skills repo is separate from this repo.

Expected GitLab structure:

```text
skills/
  active/
    architecture_review.md
    python_debugging.md
    refactor_planning.md
    test_design.md

  _proposals/
    .gitkeep

usage/
  usage_log.jsonl

registry.yaml
README.md
```

The MCP server reads active skills from:

```yaml
gitlab:
  skills:
    project_id: "<numeric GitLab project ID>"
    branch: "main"
    skills_path: "skills/active"
    proposal_path: "skills/_proposals"
    write_mode: "merge_request"
```

Skill proposals should never be written directly to `main`. They should go to a branch and merge request.

---

## 8. Memory architecture

Memory is local and proposal-based.

Recommended layout:

```text
/Users/oscarbonnet/Documents/40–49 Coding & Data/44 MCP Memory/
  profile/
    preferences.md
    working_style.md

  projects/
    mcp-server.md
    trading-research.md

  decisions/
    architecture-decisions.md

  _proposals/
    20260607-203000-preferences.md
```

The MCP tool `propose_memory_update()` writes to `_proposals`. The user should review and move/merge memory proposals manually.

---

## 9. Security model

The server must not have unrestricted access to the machine.

Security rules:

```text
allowed_read_roots: what MCP may read
allowed_write_roots: what MCP may write
never_read: hard read blocks
never_write: hard write blocks
allowed_gitlab_projects: GitLab project allowlist
```

Important examples:

```yaml
never_read:
  - "/Users/oscarbonnet/.ssh"
  - "/Users/oscarbonnet/.gnupg"
  - "/Users/oscarbonnet/Library"
  - "/Users/oscarbonnet/Downloads"
  - "/Users/oscarbonnet/Desktop"

never_write:
  - "/Users/oscarbonnet/PycharmProjects/mcp-server/.env"
  - "/Users/oscarbonnet/PycharmProjects/mcp-server/.venv"
  - "/Users/oscarbonnet/PycharmProjects/mcp-server/storage"
  - "/Users/oscarbonnet/PycharmProjects/python-project-template"
```

The agent should never weaken security just to make a feature easier to implement.

---

## 10. Indexing lifecycle

Current lifecycle:

```text
First build:
  rebuild_indexes()
  → force_rebuild=True
  → parse files
  → embed chunks
  → store in Chroma

Normal session:
  refresh_indexes()
  → force_rebuild=False
  → reuse existing Chroma collections

After file changes:
  rebuild_indexes()
  → full rebuild for current profile
```

Future lifecycle:

```text
scan_index_changes()
  → detect added/edited/deleted/moved files
  → estimate cost

index_changes(apply=False)
  → dry run, no embedding

index_changes(apply=True)
  → update only changed chunks
```

---

## 11. Incremental indexing target

The target incremental indexer should use SQLite:

```text
storage/manifest.sqlite
```

Tables:

```text
files
chunks
embedding_runs
retrieval_eval_runs
token_ledger
```

A file is changed if:

```text
added: path exists now but not in manifest
deleted: path in manifest but no longer exists
edited: same path but content_hash changed
moved: content_hash same but path changed
config changed: parser/chunker/embedding settings changed
```

Only changed chunks should be embedded again.

---

## 12. Desired long-term modules

### `config/`

Owns config loading and validation.

### `security/`

Owns path and GitLab allowlist checks.

### `loaders/`

Reads source files into raw documents.

### `chunking/`

Turns raw documents into semantically meaningful chunks.

### `embeddings/`

Abstracts local HuggingFace, Qwen, and Voyage backends.

### `indexing/`

Owns file scanning, manifest state, change detection, and Chroma upserts/deletes.

### `retrieval/`

Owns dense retrieval, BM25 retrieval, fusion, reranking, and final context selection.

### `tools/`

Owns MCP tool definitions and maps external tool calls to internal services.

---

## 13. Non-goals

Do not implement these prematurely:

```text
cloud Chroma
unrestricted full-home file access
auto-writing permanent memory
Voyage as default without token accounting
auto-summarizing every file blindly
large refactor before MCP client connection works
```

---

## 14. Build principles for Claude/Codex

When using this repo as context, an AI coding agent should follow these principles:

1. Preserve the current working MCP tools unless explicitly refactoring.
2. Prefer small modules with narrow responsibilities.
3. Never broaden filesystem permissions silently.
4. Keep indexed and full profiles separate.
5. Never mix vectors from different embedding models in the same collection.
6. Add tests or diagnostic commands for every new indexing/retrieval component.
7. Avoid silent re-embedding of large corpora.
8. Use dry-run modes before destructive or expensive indexing operations.
9. Keep `.env`, `.venv`, and `storage/` uncommitted.
10. When uncertain, add a diagnostic tool rather than guessing.
