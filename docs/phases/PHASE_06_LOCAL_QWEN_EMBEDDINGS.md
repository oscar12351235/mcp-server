# Phase 06 — Local Qwen3 Embedding Backend

## Objective

Add a local Qwen3 embedding backend for higher-quality technical retrieval while preserving the existing local BGE baseline.

Target embedding options:

```text
BAAI/bge-small-en-v1.5      # fast baseline
Qwen/Qwen3-Embedding-4B     # target default high-quality local model
Qwen/Qwen3-Embedding-8B     # heavier high-quality mode
BAAI/bge-m3                 # fallback multilingual/local mode
```

Do not remove the existing embedding backend. Add Qwen as a configurable backend.

---

## Desired architecture

```text
config.yaml embedding profile
  ↓
embedding registry
  ↓
embedding backend object
  ↓
indexing pipeline embeds documents
  ↓
retrieval pipeline embeds queries
  ↓
Chroma collection named by model/dimension/profile
```

---

## Files/modules to create

```text
src/mcp_server/embeddings/base.py
src/mcp_server/embeddings/local_huggingface.py
src/mcp_server/embeddings/registry.py
```

Optional later:

```text
src/mcp_server/embeddings/qwen.py
```

But Qwen can initially be handled by `local_huggingface.py` if it loads through SentenceTransformers or Transformers.

---

## Config

Add or evolve to:

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
      instruction: "Represent this document for retrieval."

    local_qwen3_8b:
      provider: "local_huggingface"
      model: "Qwen/Qwen3-Embedding-8B"
      dimension: 4096
      normalize: true
      instruction: "Represent this document for retrieval."
```

If the specific model requires custom prompt/instruction handling, implement it inside the backend.

---

## Embedding backend interface

Create a protocol/base class:

```python
from typing import Protocol

class EmbeddingBackend(Protocol):
    name: str
    model: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...
```

This isolates the rest of the system from model-specific code.

---

## Implementation steps

### Step 1 — Extract current embedding config

Current code sets:

```python
Settings.embed_model = HuggingFaceEmbedding(model_name=CONFIG["retrieval"]["embedding_model"])
```

Replace direct config access with:

```python
embedding_cfg = get_active_embedding_config(CONFIG)
Settings.embed_model = build_llamaindex_embedding(embedding_cfg)
```

### Step 2 — Add collection naming by embedding profile

Collection names must include:

```text
embedding provider
model name
dimension
chunking profile
knowledge profile
```

Example:

```text
knowledge_indexed_local_qwen3_4b_qwen_qwen3_embedding_4b_2560_chunk_v2
```

### Step 3 — Add diagnostic tool

Add:

```python
@mcp.tool()
def embedding_status() -> dict[str, Any]:
    ...
```

Expected output:

```json
{
  "active_profile": "local_qwen3_4b",
  "provider": "local_huggingface",
  "model": "Qwen/Qwen3-Embedding-4B",
  "dimension": 2560,
  "collections": {
    "knowledge": "knowledge_indexed_..."
  }
}
```

### Step 4 — Rebuild only indexed first

After changing embedding profile, rebuild indexed mode only:

```bash
MCP_KNOWLEDGE_PROFILE=indexed poetry run python - <<'PY'
from mcp_server.mcp_server import _rebuild_all_indexes
print(_rebuild_all_indexes(force_rebuild=True))
PY
```

Do not run full mode until performance is acceptable.

### Step 5 — Compare retrieval quality

Create a retrieval test set:

```text
retrieval_eval/queries.yaml
```

Example:

```yaml
queries:
  - id: pca_markets
    query: "Explain PCA in equity markets"
    expected_terms:
      - PCA
      - principal components
      - equity

  - id: black_scholes_greeks
    query: "Find notes about Black-Scholes delta and gamma"
    expected_terms:
      - delta
      - gamma
      - Black-Scholes
```

Run same queries with BGE and Qwen. Compare top results manually first.

---

## Connection to the rest of the system

The embedding backend feeds:

```text
document embedding during indexing
query embedding during retrieval
collection naming
manifest compatibility checks
```

Changing embedding model requires new collection names or a full rebuild.

Do not query an old collection with a new embedding model.

---

## Desired outputs

```text
- Embedding profiles in config.
- Current BGE baseline still works.
- Qwen3-4B profile can be selected.
- Collection names include embedding model/dimension.
- embedding_status() tool exists.
- Indexed profile can be rebuilt with Qwen.
```

---

## Acceptance criteria

```text
[ ] BGE indexed collection still retrieves.
[ ] Qwen3-4B collection builds separately.
[ ] Changing embedding profile changes collection name.
[ ] No vector mixing occurs.
[ ] Retrieval evaluation queries can be run on both profiles.
[ ] If Qwen model is too slow or fails, BGE fallback still works.
```
