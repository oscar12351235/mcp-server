# Phase 02 — Structure-Aware Chunking

## Objective

Replace naive file/page-level loading with chunks that preserve document structure.

The goal is to improve retrieval quality before adding more expensive components such as reranking or larger embedding models.

Current issue:

```text
A PDF book may produce hundreds or thousands of page-level documents.
A Python file may be chunked without respecting functions/classes.
A Markdown file may lose section hierarchy.
```

Target behavior:

```text
Markdown chunks follow headings.
PDF chunks preserve page/chapter/section metadata where possible.
Python chunks follow AST symbols.
LaTeX/math chunks preserve theorem/definition/proof blocks.
Generic text chunks use stable chunking with overlap.
```

---

## Desired architecture

```text
file scanner
  ↓
loader by file type
  ↓
structure-aware chunker by file type
  ↓
normalized Chunk objects
  ↓
embedding/indexing pipeline
```

Target modules:

```text
src/mcp_server/chunking/base.py
src/mcp_server/chunking/markdown_chunker.py
src/mcp_server/chunking/python_ast_chunker.py
src/mcp_server/chunking/pdf_chunker.py
src/mcp_server/chunking/latex_chunker.py
src/mcp_server/chunking/generic_text_chunker.py
```

---

## Chunk object contract

Create a normalized internal object, even if still converted to LlamaIndex `Document` later.

Suggested dataclass:

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: str = ""
    chunk_index: int = 0
    chunk_id: str | None = None
    content_hash: str | None = None
```

Each chunk must have enough metadata to answer:

```text
Where did this come from?
What kind of document is it?
What section/symbol does it represent?
Can I read the original file if needed?
```

---

## Markdown chunking

### Input

Files:

```text
.md
.markdown
```

### Desired chunking

Split by heading hierarchy:

```text
# Title
## Section
### Subsection
```

Preserve heading path in metadata:

```yaml
source_type: markdown
section_path: "Title > Section > Subsection"
heading_level: 3
file_path: ...
```

### Implementation steps

1. Read Markdown file as text.
2. Parse headings with regex.
3. Create chunks per section.
4. If a section is too large, split inside it using paragraph boundaries.
5. Preserve heading path in every subchunk.
6. Convert chunks to LlamaIndex `Document` objects.

### Desired output

For a file:

```markdown
# Black-Scholes Notes
## Greeks
### Delta
...
```

A retrieved chunk should include metadata:

```json
{
  "source_type": "markdown",
  "file_path": ".../black_scholes.md",
  "section_path": "Black-Scholes Notes > Greeks > Delta"
}
```

---

## PDF chunking

### Input

Files:

```text
.pdf
```

### Recommended approach

Do not rely on raw page-level parsing forever.

Preferred strategy:

```text
PDF → Markdown/text extraction → section-aware chunks
```

Use local parsers first:

```text
PyMuPDF
pymupdf4llm
```

Do not require LlamaParse by default because that adds a cloud/API dependency.

### Implementation steps

1. Keep current SimpleDirectoryReader fallback.
2. Add optional PyMuPDF parser.
3. Extract text per page with page numbers.
4. If markdown conversion is available, parse headings.
5. Attach metadata:

```yaml
source_type: pdf
file_path: ...
page_start: 12
page_end: 14
parser: pymupdf
```

6. For books, prefer manually curated summaries in `/87 Indexed` instead of raw full books.

### Desired output

A retrieved chunk from a PDF should include:

```json
{
  "source_type": "pdf",
  "file_path": ".../Hull.pdf",
  "page_start": 123,
  "page_end": 124,
  "parser": "pymupdf"
}
```

---

## Python code chunking

### Input

Files:

```text
.py
```

### Desired chunking

Use Python AST. Chunk by:

```text
module imports
class definitions
function definitions
methods
large top-level blocks
```

### Metadata

```yaml
source_type: code
language: python
file_path: ...
symbol_type: function
symbol_name: retrieve_knowledge
parent_symbol: null
start_line: 123
end_line: 151
```

### Implementation steps

1. Parse with `ast.parse()`.
2. Extract top-level imports as one chunk.
3. Extract each class/function node using line numbers.
4. For class nodes, either:
   - keep whole class if small; or
   - split methods into separate chunks while preserving parent class metadata.
5. Include docstrings and decorators.
6. Store symbol name and line range.
7. Convert to `Document` objects.

### Desired output

Query:

```text
How is GitLab file reading implemented?
```

Should retrieve chunk with metadata:

```json
{
  "symbol_name": "_read_gitlab_file_raw",
  "symbol_type": "function",
  "file_path": "src/mcp_server/mcp_server.py"
}
```

---

## LaTeX / math chunking

### Input

Files:

```text
.tex
.md containing LaTeX math
```

### Desired chunking

Detect blocks:

```text
definition
theorem
lemma
proposition
corollary
proof
example
equation
```

### Metadata

```yaml
source_type: latex
math_object_type: theorem
label: optional-stopping
section_path: "Martingales > Stopping Times"
```

### Implementation steps

1. Add `.tex` to allowed suffixes only when LaTeX chunker exists.
2. Detect environments with regex.
3. Keep theorem/proof pairs close when possible.
4. Preserve labels and section titles.
5. Avoid splitting equations away from explanations.

---

## Generic text chunking

For `.txt`, `.json`, `.html`, and fallback formats:

1. Read text.
2. Clean excessive whitespace.
3. Split by paragraphs.
4. Merge paragraphs up to target size.
5. Add overlap only at paragraph boundaries.

Recommended defaults:

```yaml
chunk_size_chars: 3000
chunk_overlap_chars: 300
```

---

## Connection to the rest of the system

Chunkers feed the same downstream indexing pipeline:

```text
Chunk objects
  ↓
LlamaIndex Document objects
  ↓
embedding backend
  ↓
ChromaDB
  ↓
BM25 index later
  ↓
retrieval pipeline
```

Do not make chunkers aware of Chroma, BM25, or MCP. They should only produce chunks.

---

## Acceptance criteria

```text
[ ] Markdown headings preserved in metadata.
[ ] Python functions/classes retrieved as symbols, not arbitrary text slices.
[ ] PDF chunks contain page metadata.
[ ] Chunks have stable IDs or enough data to generate stable IDs.
[ ] Generic fallback still works.
[ ] Existing MCP tools continue working.
```

---

## Test cases

Create small fixtures:

```text
tests/fixtures/markdown/sample.md
tests/fixtures/code/sample.py
tests/fixtures/text/sample.txt
```

Expected tests:

```text
test_markdown_chunker_preserves_heading_path
test_python_chunker_extracts_function_name
test_generic_chunker_limits_size
test_chunks_convert_to_documents
```
