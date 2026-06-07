# Phase 05 — Reranking with BGE-reranker-v2-m3

## Objective

Add a reranking stage after dense+BM25 hybrid retrieval.

First-stage retrieval should retrieve broad candidates. The reranker should decide which candidates are actually most relevant to the query.

Target pipeline:

```text
query
  ↓
dense top 60 + BM25 top 60
  ↓
RRF fusion top 80
  ↓
BGE-reranker-v2-m3 reranks candidates
  ↓
final top 5–12 chunks returned to MCP client
```

---

## Why reranking matters

Dense retrieval and BM25 are approximate first-stage retrievers.

They can retrieve:

```text
chunks with similar vocabulary but wrong meaning
chunks from the wrong section
chunks that mention a term but are not explanatory
near-duplicates
```

A cross-encoder reranker scores `(query, passage)` pairs directly. This usually improves top results, especially for technical retrieval.

---

## Files/modules to create

```text
src/mcp_server/retrieval/reranker.py
src/mcp_server/retrieval/pipeline.py
```

Optional:

```text
src/mcp_server/models/reranker_registry.py
```

---

## Config

Add:

```yaml
reranker:
  enabled: true
  provider: "local_huggingface"
  model: "BAAI/bge-reranker-v2-m3"
  candidates_top_k: 50
  final_top_k: 8
  batch_size: 8
```

Keep ability to disable:

```yaml
reranker:
  enabled: false
```

---

## Candidate object

The reranker should receive candidates after fusion.

Candidate fields:

```python
@dataclass
class Candidate:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    dense_score: float | None = None
    bm25_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    rank: int | None = None
    sources: list[str] = field(default_factory=list)
```

---

## Implementation steps

### Step 1 — Add reranker loader

Implement a lazy loader:

```python
_reranker_model = None

def get_reranker():
    global _reranker_model
    if _reranker_model is None:
        ... load BGE-reranker-v2-m3 ...
    return _reranker_model
```

Do not load reranker at server startup unless needed.

### Step 2 — Implement rerank function

```python
def rerank(query: str, candidates: list[Candidate], top_k: int) -> list[Candidate]:
    ...
```

Input pairs:

```text
[(query, candidate.text), ...]
```

Output:

```text
same candidates with rerank_score set
sorted by rerank_score descending
```

### Step 3 — Add fallback if reranker fails

If reranker cannot load or errors:

```text
log warning
return fused candidates without reranking
```

Do not break MCP retrieval.

### Step 4 — Add debug output

`retrieve_knowledge_debug()` should show:

```json
{
  "fused_rank": 3,
  "reranked_rank": 1,
  "rerank_score": 0.87,
  "file_path": "...",
  "section_path": "..."
}
```

### Step 5 — Add tests

Minimum tests:

```text
test_reranker_preserves_candidate_metadata
test_reranker_sorts_by_score
test_reranker_failure_falls_back_to_fusion
test_final_top_k_applied_after_reranking
```

Use a mock reranker in tests to avoid loading large models.

---

## Connection to the rest of the system

Reranking comes after fusion:

```text
dense retrieval
BM25 retrieval
  ↓
fusion
  ↓
reranker
  ↓
format_nodes / MCP response
```

Do not run reranking over the entire corpus. Only rerank top fused candidates.

---

## Desired outputs

After this phase:

```text
- reranker can be enabled/disabled in config
- retrieve_knowledge returns reranked results
- debug tool shows fused and reranked scores
- failures fall back gracefully
```

---

## Acceptance criteria

```text
[ ] Reranker is lazy-loaded.
[ ] Config can disable reranking.
[ ] Hybrid candidates are reranked before final output.
[ ] Retrieval still works if reranker fails.
[ ] Rerank scores appear in debug output.
```
