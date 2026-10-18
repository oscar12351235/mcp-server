# Phase 04 — BM25 Hybrid Retrieval

## Objective

Add lexical retrieval with BM25 and combine it with dense vector retrieval.

Dense embeddings are good for semantic similarity. BM25 is better for exact technical tokens such as:

```text
function names
class names
file names
error messages
package names
tickers
formula names
theorem names
LaTeX symbols
configuration keys
```

The target is not to replace dense retrieval. The target is hybrid retrieval:

```text
dense retrieval + BM25 retrieval → fusion → candidates
```

---

## Desired architecture

```text
query
  ↓
Dense retriever: Chroma + embedding model
  ↓
Dense candidates

query
  ↓
BM25 retriever: lexical index
  ↓
BM25 candidates

Dense + BM25 candidates
  ↓
Fusion, usually Reciprocal Rank Fusion
  ↓
Top fused candidates
  ↓
Optional reranker in next phase
```

---

## Files/modules to create

```text
src/mcp_server/retrieval/bm25.py
src/mcp_server/retrieval/dense.py
src/mcp_server/retrieval/fusion.py
src/mcp_server/retrieval/pipeline.py
storage/bm25/
```

If staying inside current file temporarily, implement internal helper functions first, then refactor later.

---

## BM25 index design

The BM25 store must use the same chunk IDs as the vector store.

Each BM25 document should store:

```json
{
  "chunk_id": "...",
  "text": "...",
  "metadata": {
    "file_path": "...",
    "source_type": "...",
    "section_path": "...",
    "symbol_name": "..."
  }
}
```

This lets dense and BM25 results merge cleanly.

---

## Retrieval config

Add:

```yaml
retrieval:
  dense_top_k: 60
  bm25_top_k: 60
  fused_top_k: 80
  final_top_k: 8
  fusion: "rrf"

  defaults:
    skills_top_k: 3
    knowledge_top_k: 8
    memory_top_k: 5

  limits:
    skills_top_k_max: 10
    knowledge_top_k_max: 30
    memory_top_k_max: 10
```

Maintain backward compatibility with current `top_k` tool arguments.

---

## Reciprocal Rank Fusion

Use RRF to combine rankings:

```python
def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)
```

For each candidate:

```text
score = sum(rrf_score(rank_in_each_retriever))
```

Benefits:

```text
simple
robust
no need to calibrate dense and BM25 scores
works well as a first fusion method
```

---

## Implementation steps

### Step 1 — Persist BM25 documents

When indexing chunks, write BM25 corpus data to:

```text
storage/bm25/<collection_name>.jsonl
```

Each line:

```json
{"chunk_id": "...", "text": "...", "metadata": {...}}
```

Later, this can be optimized with a real BM25 library/store.

### Step 2 — Build BM25 retriever

Options:

```text
LlamaIndex BM25Retriever
rank_bm25 library
custom simple BM25 wrapper
```

Prefer a simple implementation first if integration is faster.

### Step 3 — Add dense retrieval wrapper

Wrap current vector retrieval into:

```python
dense_retrieve(query, collection_name, top_k) -> list[Candidate]
```

Candidate should include:

```text
chunk_id
text
metadata
score
source="dense"
rank
```

### Step 4 — Add BM25 retrieval wrapper

```python
bm25_retrieve(query, collection_name, top_k) -> list[Candidate]
```

### Step 5 — Add fusion

```python
fuse_candidates(dense_candidates, bm25_candidates, method="rrf") -> list[Candidate]
```

Deduplicate by `chunk_id`.

If same chunk appears in both dense and BM25:

```text
merge metadata
keep both scores
compute fused score
```

### Step 6 — Add hybrid retrieval tool

Keep old `retrieve_knowledge()` but internally switch to hybrid retrieval when enabled.

Optional new diagnostic tool:

```python
@mcp.tool()
def retrieve_knowledge_debug(query: str, top_k: int | None = None) -> dict[str, Any]:
    ...
```

Return:

```json
{
  "query": "...",
  "dense_candidates": [...],
  "bm25_candidates": [...],
  "fused_candidates": [...]
}
```

### Step 7 — Add tests

Minimum tests:

```text
test_rrf_fusion_deduplicates_chunks
test_bm25_retrieves_exact_function_name
test_hybrid_includes_dense_and_bm25_sources
test_retrieve_knowledge_returns_final_top_k
```

---

## Connection to the rest of the system

The BM25 index should be built from the same chunks as Chroma.

```text
chunker output
  ↓
chunk_id
  ├── Chroma vector index
  └── BM25 lexical index
```

The manifest should track both:

```text
chunks.vector_id
chunks.bm25_doc_id or chunk_id
```

For deleted files:

```text
delete vectors from Chroma
delete BM25 docs from BM25 store
mark manifest rows deleted
```

---

## Desired outputs

After this phase:

```text
- Every indexed chunk has both a vector entry and BM25 entry.
- retrieve_knowledge() uses hybrid retrieval.
- Exact technical queries improve.
- Debug tool can show dense vs BM25 contributions.
```

---

## Acceptance criteria

```text
[ ] Query for an exact function name retrieves the correct code chunk.
[ ] Query for an exact file path retrieves the correct file/chunk.
[ ] Query for a semantic concept still works via dense retrieval.
[ ] Fusion deduplicates identical chunk IDs.
[ ] BM25 index is profile/collection-specific.
[ ] Deleting/reindexing chunks updates BM25 too.
```
