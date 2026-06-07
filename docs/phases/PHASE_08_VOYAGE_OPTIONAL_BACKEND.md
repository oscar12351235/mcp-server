# Phase 08 — Optional Voyage Embedding Backend

## Objective

Keep Voyage embeddings as an optional cloud/API backend without making them the default.

Voyage is useful for:

```text
benchmarking against local models
cloud fallback if local Qwen is too slow
specialized code/finance embeddings if selected
high-quality retrieval experiments
```

But it introduces:

```text
API key dependency
token/cost tracking
network dependency
need for strict incremental indexing
```

Do not enable Voyage by default.

---

## Prerequisites

Before this phase, the repo should already have:

```text
[ ] model/profile-safe collection names
[ ] incremental indexing manifest
[ ] token ledger
[ ] scan_index_changes dry-run
[ ] index_changes(apply=True)
[ ] no default full rebuilds
```

Do not add Voyage before these are done.

---

## Config

Add:

```yaml
embedding:
  active_profile: "local_qwen3_4b"

  profiles:
    voyage_4_large:
      provider: "voyage"
      model: "voyage-4-large"
      dimension: 1024
      input_type_document: "document"
      input_type_query: "query"
      batch_size: 64

    voyage_code_3:
      provider: "voyage"
      model: "voyage-code-3"
      dimension: 1024
      input_type_document: "document"
      input_type_query: "query"
      batch_size: 64

voyage:
  require_explicit_apply: true
  max_tokens_per_run_default: 2000000
  max_tokens_per_file_default: 500000
  warn_at_total_tokens: 150000000
  hard_stop_at_total_tokens: 190000000
```

`.env`:

```env
VOYAGE_API_KEY=...
```

Never commit `.env`.

---

## Files/modules to create

```text
src/mcp_server/embeddings/voyage.py
src/mcp_server/indexing/token_budget.py
src/mcp_server/tools/budget_tools.py
```

---

## Voyage backend interface

Implement the same backend interface as local embeddings:

```python
class VoyageEmbeddingBackend:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...
```

Do not expose Voyage-specific details to retrieval pipeline.

---

## Token budget flow

Before embedding any document chunks:

```text
chunks to embed
  ↓
estimate tokens
  ↓
check per-run and total budget
  ↓
if safe, allow apply=True
  ↓
call Voyage API
  ↓
record actual/estimated usage in token_ledger
```

### Required MCP tools

```text
embedding_budget_status()
scan_embedding_cost(profile=None)
index_changes(profile=None, apply=False)
```

`index_changes` must default to dry-run.

---

## Implementation steps

### Step 1 — Add Voyage dependency

```bash
poetry add voyageai
```

### Step 2 — Load API key

In backend:

```python
api_key = os.environ.get("VOYAGE_API_KEY")
if not api_key:
    raise RuntimeError("VOYAGE_API_KEY missing")
```

### Step 3 — Implement document/query input types

Documents should use:

```text
input_type=document
```

Queries should use:

```text
input_type=query
```

Do not accidentally embed documents as queries.

### Step 4 — Add batching

Batch document chunks.

Rules:

```text
batch by count
batch by token estimate
retry transient errors
stop on budget errors
```

### Step 5 — Record ledger rows

For every embedding batch:

```text
run_id
provider=voyage
model
tokens
usage_type=document_embedding or query_embedding
estimated_or_actual
```

### Step 6 — Protect full profile

If profile is `full`, require explicit confirmation or config flag.

Example:

```text
index_changes(profile="full", apply=True, confirm_full_profile=True)
```

---

## Connection to the rest of the system

Voyage backend should plug into the same embedding registry as local models.

```text
EmbeddingBackend interface
  ├── LocalHuggingFaceBackend
  ├── QwenBackend
  └── VoyageBackend
```

The manifest and collection naming system must treat Voyage as a separate incompatible embedding profile.

Example collection:

```text
knowledge_indexed_voyage_4_large_1024_chunk_v2
```

---

## Desired outputs

```text
- Voyage backend exists but disabled by default.
- VOYAGE_API_KEY is required only when using Voyage profile.
- Dry-run estimates tokens before embedding.
- Token ledger records usage.
- Voyage collection names are separate from local collections.
```

---

## Acceptance criteria

```text
[ ] Local embeddings still work without VOYAGE_API_KEY.
[ ] Selecting Voyage profile without API key fails clearly.
[ ] Dry-run cost estimate works without embedding.
[ ] apply=True embeds only changed chunks.
[ ] Token ledger is updated.
[ ] Full profile cannot accidentally burn budget.
```
