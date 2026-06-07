# Phase 03 — Incremental Indexing and Manifest

## Objective

Prevent unnecessary re-indexing and re-embedding by tracking what has already been indexed.

This phase is required before using expensive/cloud embedding backends and highly recommended before indexing large local corpora.

Current behavior:

```text
rebuild_indexes()
  → parse all files
  → embed all chunks
  → recreate collections
```

Target behavior:

```text
scan files
  → detect added/edited/deleted/moved files
  → parse only affected files
  → embed only new/changed chunks
  → delete removed vectors
  → update manifest
```

---

## Desired architecture

```text
profile paths
  ↓
file scanner
  ↓
manifest.sqlite
  ↓
change detector
  ↓
chunker only for changed files
  ↓
embedding backend only for changed chunks
  ↓
Chroma upsert/delete
  ↓
manifest update
```

---

## Files/modules to create

```text
src/mcp_server/indexing/manifest.py
src/mcp_server/indexing/scanner.py
src/mcp_server/indexing/change_detector.py
src/mcp_server/indexing/indexer.py
src/mcp_server/indexing/collections.py
src/mcp_server/indexing/token_budget.py

storage/manifest.sqlite
```

---

## SQLite schema

Use SQLite, not JSON, for reliability and queryability.

### `files`

```sql
CREATE TABLE IF NOT EXISTS files (
    file_id TEXT PRIMARY KEY,
    profile TEXT NOT NULL,
    path TEXT NOT NULL,
    normalized_path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    extension TEXT NOT NULL,
    parser_name TEXT,
    parser_version TEXT,
    chunking_profile TEXT,
    embedding_profile TEXT,
    collection_name TEXT,
    last_seen_at TEXT,
    last_indexed_at TEXT,
    status TEXT NOT NULL
);
```

### `chunks`

```sql
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    file_id TEXT NOT NULL,
    profile TEXT NOT NULL,
    path TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_hash TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    token_count INTEGER,
    embedding_provider TEXT,
    embedding_model TEXT,
    embedding_dimension INTEGER,
    collection_name TEXT NOT NULL,
    vector_id TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT,
    updated_at TEXT,
    FOREIGN KEY(file_id) REFERENCES files(file_id)
);
```

### `embedding_runs`

```sql
CREATE TABLE IF NOT EXISTS embedding_runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    profile TEXT NOT NULL,
    mode TEXT NOT NULL,
    embedding_profile TEXT NOT NULL,
    collection_name TEXT NOT NULL,
    files_scanned INTEGER DEFAULT 0,
    files_added INTEGER DEFAULT 0,
    files_edited INTEGER DEFAULT 0,
    files_deleted INTEGER DEFAULT 0,
    files_moved INTEGER DEFAULT 0,
    chunks_embedded INTEGER DEFAULT 0,
    tokens_embedded INTEGER DEFAULT 0,
    status TEXT NOT NULL
);
```

### `token_ledger`

```sql
CREATE TABLE IF NOT EXISTS token_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    run_id TEXT,
    provider TEXT,
    model TEXT,
    usage_type TEXT,
    tokens INTEGER,
    estimated_or_actual TEXT,
    file_id TEXT,
    chunk_id TEXT,
    notes TEXT
);
```

Even for local embeddings, track token estimates because it helps decide whether a future cloud backend is safe.

---

## What counts as change?

### Added file

```text
Current scan sees a path/content_hash not in manifest.
```

Action:

```text
parse → chunk → embed → add vectors → insert manifest rows
```

### Edited file

```text
Same normalized path exists but content_hash changed.
```

Action:

```text
delete old vectors for file_id
parse → chunk → embed → insert new chunks
update file row
```

### Deleted file

```text
Manifest has file, but current scan does not.
```

Action:

```text
delete vectors for file_id
mark file/chunks deleted
```

### Moved/renamed file

```text
Old path missing, new path appears, content_hash same.
```

Action:

```text
update path metadata
avoid re-embedding if chunk hashes unchanged
```

### Config-changed file

If parser, chunking profile, embedding profile, or collection name changed:

```text
old vectors incompatible
re-index affected file or whole collection
```

---

## Stable ID strategy

### File ID

Use content hash + profile or content hash only depending on deduplication policy.

Recommended:

```text
file_id = sha256(profile + content_hash)
```

### Chunk ID

Recommended:

```text
chunk_id = sha256(
  profile + collection_name + content_hash + chunk_index + chunk_hash + embedding_profile + chunking_profile
)
```

### Vector ID

Use:

```text
vector_id = chunk_id
```

This makes Chroma deletes/upserts deterministic.

---

## MCP tools to add

### `scan_index_changes(profile=None)`

Dry-run tool. It should not embed anything.

Returns:

```json
{
  "profile": "indexed",
  "files_scanned": 12,
  "added": [...],
  "edited": [...],
  "deleted": [...],
  "moved": [...],
  "estimated_chunks": 104,
  "estimated_tokens": 180000
}
```

### `index_changes(profile=None, apply=False)`

Default `apply=False` returns dry-run.

Only with `apply=True` does it update Chroma and manifest.

### `list_indexed_files(profile=None)`

Returns file-level manifest rows.

### `list_indexed_chunks(path)`

Returns chunks for a file.

### `delete_vectors_for_file(path, apply=False)`

Manual cleanup tool. Must default to dry-run.

---

## Implementation steps

### Step 1 — Create manifest module

Implement:

```python
init_manifest_db(path)
upsert_file(row)
upsert_chunk(row)
mark_file_deleted(file_id)
get_files_for_profile(profile)
get_chunks_for_file(file_id)
```

### Step 2 — Create scanner

Implement:

```python
scan_paths(paths, allowed_suffixes, ignored_dirs) -> list[ScannedFile]
```

Each scanned file includes:

```text
path
normalized_path
size_bytes
mtime_ns
content_hash
extension
```

Use SHA-256 content hash.

### Step 3 — Create change detector

Implement:

```python
detect_changes(profile, scanned_files, manifest_files) -> ChangeSet
```

### Step 4 — Create dry-run MCP tool

Implement `scan_index_changes` first.

Do not implement embedding yet.

### Step 5 — Implement vector deletes

For deleted/edited files, delete vectors by stored `vector_id` from Chroma.

### Step 6 — Implement changed-file indexing

For added/edited files:

```text
load file → chunk → embed → Chroma upsert → manifest update
```

### Step 7 — Add tests

Minimum tests:

```text
test_added_file_detected
test_edited_file_detected_by_hash
test_deleted_file_detected
test_moved_file_detected_by_same_hash
test_dry_run_does_not_modify_manifest
test_apply_updates_manifest
```

---

## Connection to the rest of the system

After this phase:

- `rebuild_indexes()` remains as emergency full rebuild.
- `scan_index_changes()` becomes the normal pre-index command.
- `index_changes(apply=True)` becomes the normal update command.
- Retrieval tools continue to use Chroma collections as before.
- BM25 phase will reuse the same manifest and chunk IDs.

---

## Desired outputs

```text
- manifest.sqlite created
- scan_index_changes() works
- index_changes(apply=False) works
- index_changes(apply=True) updates only changed files
- Chroma vector IDs are deterministic
- deleted files are removed from Chroma
- moved files do not force re-embedding if content is unchanged
```

---

## Acceptance criteria

```text
[ ] Adding one .md file embeds only that file.
[ ] Editing one file re-embeds only that file.
[ ] Deleting one file removes its vectors.
[ ] Moving one file updates path metadata without re-embedding if possible.
[ ] Dry run never writes to Chroma or SQLite.
[ ] Manifest survives process restarts.
```
