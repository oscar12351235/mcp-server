# Phase 07 — Code-Aware Repository Retrieval

## Objective

Add a dedicated retrieval path for code repositories instead of treating repos as generic documents.

Code retrieval must support:

```text
exact symbol lookup
file path lookup
semantic code search
error/debug search
test/source/config distinction
read exact file after candidate retrieval
```

---

## Desired architecture

```text
repo scanner
  ↓
code-aware loaders
  ↓
language-specific chunkers
  ↓
code dense index + code BM25 index
  ↓
code retrieval tool
  ↓
optional read_local_file/read_code_symbol tool
```

Do not mix code repos blindly into the general knowledge index.

---

## Config

Add:

```yaml
code_retrieval:
  enabled: true
  default_profile: "projects"

  profiles:
    projects:
      paths:
        - "/Users/oscarbonnet/PycharmProjects"
      excluded:
        - "mcp-server/.venv"
        - "mcp-server/storage"
        - "python-project-template"
        - "**/__pycache__"
        - "**/.git"
        - "**/node_modules"

  allowed_extensions:
    - ".py"
    - ".md"
    - ".yaml"
    - ".yml"
    - ".toml"
    - ".json"
    - ".sql"
    - ".sh"
```

---

## Files/modules to create

```text
src/mcp_server/loaders/repo_loader.py
src/mcp_server/chunking/python_ast_chunker.py
src/mcp_server/retrieval/code_pipeline.py
src/mcp_server/tools/code_tools.py
```

---

## Code chunk metadata

Every code chunk should include:

```yaml
source_type: code
repo_name: mcp-server
language: python
file_path: src/mcp_server/mcp_server.py
file_role: source
symbol_type: function
symbol_name: retrieve_knowledge
parent_symbol: null
start_line: 552
end_line: 571
imports:
  - chromadb
  - requests
```

`file_role` should be inferred:

```text
source: src/, package files
test: tests/, test_*.py
config: pyproject.toml, config.yaml, .gitlab-ci.yml
docs: README.md, docs/
scripts: scripts/, .sh
```

---

## MCP tools to add

### `retrieve_code(query, repo=None, top_k=None)`

Hybrid retrieval over code chunks.

### `list_code_repos()`

Lists indexed repos.

### `read_code_symbol(repo, symbol_name, file_path=None)`

Reads exact symbol if indexed metadata has line ranges.

### `code_index_status()`

Shows indexed repos, files, chunks, and collections.

---

## Implementation steps

### Step 1 — Repo scanner

Scan configured repo roots.

For each repo:

```text
find .git root if available
infer repo_name from folder
apply excluded patterns
collect allowed extensions
```

### Step 2 — Python AST chunker

Use `ast.parse()`.

Extract:

```text
imports chunk
classes
functions
methods
module constants if useful
```

Store line ranges with:

```python
node.lineno
node.end_lineno
```

### Step 3 — Documentation chunking

For repo `.md` files, reuse Markdown chunker.

### Step 4 — Config chunking

For `.yaml`, `.toml`, `.json`, chunk by top-level keys or whole file if small.

### Step 5 — Build code-specific indexes

Use separate collections:

```text
code_projects_<embedding_profile>_<chunking_profile>
code_bm25_projects
```

### Step 6 — Add retrieval tool

`retrieve_code()` should use hybrid retrieval and return chunks with line numbers.

### Step 7 — Add read symbol tool

`read_code_symbol()` should:

```text
look up symbol metadata
open file safely
return exact line range with context
```

---

## Connection to the rest of the system

Code retrieval should be separate from knowledge retrieval:

```text
retrieve_knowledge → notes/books/articles
retrieve_code      → repos/source/tests/config
read_local_file    → exact file read when needed
```

The LLM can combine both:

```text
retrieve_skills("python debugging")
retrieve_code("where is _read_gitlab_file_raw")
read_code_symbol(...)
```

---

## Desired outputs

```text
- Python source chunks by function/class.
- Code BM25 exact symbol search.
- retrieve_code() MCP tool.
- read_code_symbol() MCP tool.
- Code index status diagnostics.
```

---

## Acceptance criteria

```text
[ ] Query for a function name retrieves the function.
[ ] Query for an error message retrieves relevant code/config.
[ ] Query for a concept retrieves semantically related code.
[ ] Returned code chunks include file path and line range.
[ ] `read_code_symbol` returns exact source lines.
[ ] Excluded directories are not indexed.
```
