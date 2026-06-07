# Phase 09 — Retrieval Evaluation and Observability

## Objective

Add systematic evaluation and diagnostics so retrieval improvements are measurable.

Without evaluation, changes to embeddings, chunking, BM25, and reranking are subjective.

Target:

```text
fixed query set
expected results/terms
retrieval debug output
per-stage scores
timing metrics
index size metrics
model/profile comparison
```

---

## Desired architecture

```text
retrieval_eval/queries.yaml
  ↓
eval runner
  ↓
run retrieval pipeline for each query
  ↓
store results
  ↓
compare models/profiles/configs
```

---

## Files/modules to create

```text
retrieval_eval/queries.yaml
retrieval_eval/README.md
src/mcp_server/eval/runner.py
src/mcp_server/eval/metrics.py
src/mcp_server/tools/eval_tools.py
storage/eval_runs/
```

---

## Query set format

Create:

```yaml
queries:
  - id: pca_markets
    query: "Explain PCA in equity markets"
    domain: finance
    expected_terms:
      - PCA
      - principal components
      - equity
    expected_paths_any:
      - "finance"
      - "pca"

  - id: black_scholes_greeks
    query: "Find notes about Black-Scholes delta and gamma"
    domain: finance
    expected_terms:
      - Black-Scholes
      - delta
      - gamma

  - id: gitlab_file_reader
    query: "Where is the GitLab file reading function implemented?"
    domain: code
    expected_terms:
      - _read_gitlab_file_raw
      - base64
      - repository/files

  - id: bm25_exact_error
    query: "Advanced encoding not implemented pypdf cmap"
    domain: debugging
    expected_terms:
      - pypdf
      - cmap
      - encoding
```

---

## Evaluation output

Each run should write:

```text
storage/eval_runs/YYYYMMDD-HHMMSS-<profile>.json
```

Example output:

```json
{
  "run_id": "20260607-210000-indexed-qwen4b",
  "profile": "indexed",
  "embedding_profile": "local_qwen3_4b",
  "retrieval_config": {
    "dense_top_k": 60,
    "bm25_top_k": 60,
    "reranker_enabled": true
  },
  "results": [
    {
      "query_id": "pca_markets",
      "query": "Explain PCA in equity markets",
      "top_results": [...],
      "expected_terms_found": ["PCA", "principal components"],
      "latency_ms": 1240
    }
  ]
}
```

---

## MCP tools to add

### `run_retrieval_eval(profile=None, query_ids=None)`

Runs eval set.

### `compare_eval_runs(run_id_a, run_id_b)`

Compares configs/models.

### `retrieval_debug(query, domain="knowledge")`

Shows:

```text
dense candidates
BM25 candidates
fused candidates
reranked candidates
final returned chunks
latency per stage
```

---

## Metrics

Start simple:

```text
expected_terms_hit_rate
expected_path_hit_rate
manual_relevance_score
latency_ms
candidate_count
```

Later:

```text
MRR@k
Recall@k
nDCG@k
```

For a personal system, manual scoring is acceptable and useful.

---

## Observability diagnostics

Add tools:

```text
collection_status()
config_status()
embedding_status()
index_manifest_status()
bm25_status()
runtime_status()
```

`runtime_status()` should show:

```json
{
  "knowledge_profile": "indexed",
  "embedding_profile": "local_qwen3_4b",
  "reranker_enabled": true,
  "collections": {...},
  "loaded_indexes": {...}
}
```

---

## Implementation steps

### Step 1 — Create query set

Start with 10–20 queries across:

```text
finance
math
CS/ML
code/debugging
MCP architecture
personal memory
```

### Step 2 — Add retrieval_debug

This is more useful than aggregate metrics at first.

### Step 3 — Add eval runner

Run fixed queries and save JSON results.

### Step 4 — Compare embedding models

Run same queries with:

```text
BGE small
Qwen3 4B
Qwen3 8B if feasible
Voyage optional
```

### Step 5 — Compare retrieval pipeline changes

Compare:

```text
dense-only
hybrid dense+BM25
hybrid+reranker
```

---

## Desired outputs

```text
- retrieval_eval/queries.yaml exists.
- retrieval_debug() shows stages.
- eval runner writes JSON results.
- model/pipeline comparisons are possible.
```

---

## Acceptance criteria

```text
[ ] Same query set can run across two embedding profiles.
[ ] Debug output includes dense/BM25/fusion/rerank stages.
[ ] Eval results are saved to storage/eval_runs.
[ ] Latency is reported.
[ ] Top result metadata includes source path and chunk ID.
```
