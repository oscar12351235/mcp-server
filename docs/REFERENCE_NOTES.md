# Reference Notes

These notes record external assumptions used in the architecture documents. Verify model cards and API docs again before implementation, because embedding and reranking model details may change.

## Qwen3 Embedding

Use Qwen3 embedding models as optional local high-quality embedding backends. The working target profiles are:

```text
Qwen/Qwen3-Embedding-4B
Qwen/Qwen3-Embedding-8B
```

Before implementation, check the current Hugging Face model card for:

```text
supported dimensions
context length
recommended inference code
memory requirements
instruction/query formatting
quantized variants
```

## BGE reranker

The intended local reranker is:

```text
BAAI/bge-reranker-v2-m3
```

Before implementation, check the model card for:

```text
recommended inference library
batching behavior
score interpretation
GPU/CPU feasibility
```

## BM25

BM25 is used as the lexical retrieval component. It should complement dense retrieval and improve exact-token retrieval for code symbols, filenames, formulas, error messages, and technical terms.

## Voyage optional backend

Voyage should remain optional and disabled by default. If implemented, it must use:

```text
explicit provider profile
API key in .env only
separate collection names
manifest-based incremental indexing
token ledger and dry-run cost estimates
```

Do not make Voyage the default embedding provider unless explicitly requested.
