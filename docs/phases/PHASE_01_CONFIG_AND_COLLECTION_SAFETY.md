# Phase 01 — Config, Collection Naming, and Safety Hardening

## Objective

Make configuration explicit, safe, and model-aware before adding advanced retrieval.

This phase prevents dangerous bugs such as:

```text
mixing vectors from different embedding models
using full profile accidentally
writing outside allowlisted paths
silently indexing unsupported file types
reusing stale Chroma collections after config changes
```

---

## Desired architecture after this phase

```text
config.yaml
  ↓
validated config object
  ↓
collection naming function
  ↓
collection names include:
    domain + profile + embedding provider + model + dimension + chunking version
```

Example collection names:

```text
skills_local_bge_small_en_v1_5_384_chunk_v1
knowledge_indexed_local_bge_small_en_v1_5_384_chunk_v1
knowledge_indexed_qwen3_embedding_4b_2560_chunk_v2
memory_local_bge_small_en_v1_5_384_chunk_v1
```

---

## Files to modify/create

Current minimal implementation can stay in:

```text
src/mcp_server/mcp_server.py
config.yaml
```

Target modular implementation later:

```text
src/mcp_server/config/loader.py
src/mcp_server/config/schema.py
src/mcp_server/indexing/collections.py
src/mcp_server/security/paths.py
```

---

## Step-by-step implementation

### Step 1 — Add embedding config section

Current config has:

```yaml
retrieval:
  embedding_model: "BAAI/bge-small-en-v1.5"
```

Evolve to:

```yaml
embedding:
  active_profile: "local_bge_small"

  profiles:
    local_bge_small:
      provider: "local_huggingface"
      model: "BAAI/bge-small-en-v1.5"
      dimension: 384
      normalize: true

    local_qwen3_4b:
      provider: "local_huggingface"
      model: "Qwen/Qwen3-Embedding-4B"
      dimension: 2560
      normalize: true

    local_qwen3_8b:
      provider: "local_huggingface"
      model: "Qwen/Qwen3-Embedding-8B"
      dimension: 4096
      normalize: true

    voyage_4_large:
      provider: "voyage"
      model: "voyage-4-large"
      dimension: 1024
      input_type_document: "document"
      input_type_query: "query"
      normalize: false
```

Keep backward compatibility during transition:

```text
If `embedding` does not exist, read `retrieval.embedding_model`.
```

### Step 2 — Add chunking version to config

Add:

```yaml
chunking:
  active_profile: "generic_v1"

  profiles:
    generic_v1:
      strategy: "generic"
      chunk_size: 1024
      chunk_overlap: 150

    markdown_headings_v1:
      strategy: "markdown_headings"
      max_chunk_chars: 4000

    python_ast_v1:
      strategy: "python_ast"
      include_docstrings: true
      include_imports: true
```

### Step 3 — Implement safe collection naming

Create function:

```python
def _safe_collection_component(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
```

Create function:

```python
def _collection_name(domain: str, profile: str | None = None) -> str:
    profile = profile or KNOWLEDGE_PROFILE
    embedding_profile = CONFIG.get("embedding", {}).get("active_profile", "legacy")
    embedding_cfg = CONFIG.get("embedding", {}).get("profiles", {}).get(embedding_profile, {})
    model = embedding_cfg.get("model", CONFIG["retrieval"].get("embedding_model", "unknown"))
    dimension = str(embedding_cfg.get("dimension", "default"))
    chunking_profile = CONFIG.get("chunking", {}).get("active_profile", "chunk_v1")

    parts = [
        domain,
        profile if domain == "knowledge" else "global",
        embedding_profile,
        model,
        dimension,
        chunking_profile,
    ]

    return "_".join(_safe_collection_component(p) for p in parts if p)
```

Then replace hardcoded collection names:

```python
"skills"
"memory"
f"knowledge_{KNOWLEDGE_PROFILE}"
```

with:

```python
_collection_name("skills")
_collection_name("knowledge")
_collection_name("memory")
```

### Step 4 — Add config diagnostics tool

Add MCP tool:

```python
@mcp.tool()
def config_status() -> dict[str, Any]:
    """Show active config profiles without exposing secrets."""
    return {
        "knowledge_profile": KNOWLEDGE_PROFILE,
        "knowledge_paths": _get_knowledge_paths(),
        "embedding_profile": CONFIG.get("embedding", {}).get("active_profile", "legacy"),
        "embedding_model": CONFIG.get("embedding", {}).get("profiles", {}).get(
            CONFIG.get("embedding", {}).get("active_profile", ""), {}
        ).get("model", CONFIG["retrieval"].get("embedding_model")),
        "chunking_profile": CONFIG.get("chunking", {}).get("active_profile", "legacy"),
    }
```

Never return `.env` values or tokens.

### Step 5 — Harden allowed suffixes

Do not load all file types through one generic reader.

Recommended default:

```python
allowed_suffixes = {
    ".md",
    ".txt",
    ".pdf",
    ".py",
    ".yaml",
    ".yml",
    ".json",
    ".html",
    ".docx",
    ".epub",
}
```

Do not include these until dedicated loaders exist:

```text
.png
.csv
.xlsx
.parquet
```

Add dedicated loaders in later phases.

---

## Desired outputs

```text
- Config supports embedding profiles.
- Config supports chunking profiles.
- Chroma collection names are model/profile/chunking safe.
- Diagnostic config_status() tool exists.
- Unsupported file types are not silently indexed.
```

---

## Acceptance criteria

```text
[ ] collection_status() shows model-specific collection names.
[ ] Changing embedding profile changes collection name.
[ ] indexed and full profiles never share the same knowledge collection.
[ ] config_status() does not expose secrets.
[ ] No `.env` value is returned by any MCP tool.
[ ] Syntax check passes.
```

---

## Tests and commands

```bash
poetry run python -m py_compile src/mcp_server/mcp_server.py
```

```bash
MCP_KNOWLEDGE_PROFILE=indexed poetry run python - <<'PY'
from mcp_server.mcp_server import collection_status
print(collection_status())
PY
```

If `config_status()` exists:

```bash
MCP_KNOWLEDGE_PROFILE=indexed poetry run python - <<'PY'
from mcp_server.mcp_server import config_status
print(config_status())
PY
```
