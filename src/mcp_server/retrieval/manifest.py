from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ScannedFile:
    path: str
    normalized_path: str
    size_bytes: int
    mtime_ns: int
    content_hash: str
    extension: str


@dataclass(frozen=True)
class ManifestFile:
    file_id: str
    profile: str
    path: str
    normalized_path: str
    size_bytes: int
    mtime_ns: int
    content_hash: str
    extension: str
    parser_name: str | None
    parser_version: str | None
    chunking_profile: str | None
    embedding_profile: str | None
    collection_name: str | None
    embedding_config_fingerprint: str | None
    last_seen_at: str | None
    last_indexed_at: str | None
    status: str


@dataclass(frozen=True)
class ManifestChunk:
    chunk_id: str
    file_id: str
    profile: str
    path: str
    chunk_index: int
    chunk_hash: str
    text_hash: str
    token_count: int | None
    embedding_provider: str | None
    embedding_model: str | None
    embedding_dimension: int | None
    collection_name: str
    embedding_config_fingerprint: str
    vector_id: str
    metadata_json: str
    status: str
    created_at: str | None
    updated_at: str | None


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def content_hash_for_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def file_id_for(profile: str, normalized_path: str) -> str:
    payload = f"{profile}:{normalized_path}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def vector_id_for(
    profile: str,
    collection_name: str,
    content_hash: str,
    chunk_index: int,
    chunk_hash: str,
    embedding_profile: str,
    chunking_profile: str,
) -> str:
    payload = (
        f"{profile}:{collection_name}:{content_hash}:{chunk_index}:"
        f"{chunk_hash}:{embedding_profile}:{chunking_profile}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def init_manifest_db(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
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
                embedding_config_fingerprint TEXT,
                last_seen_at TEXT,
                last_indexed_at TEXT,
                status TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_files_profile_status
                ON files(profile, status);
            CREATE INDEX IF NOT EXISTS idx_files_profile_path
                ON files(profile, normalized_path);
            CREATE INDEX IF NOT EXISTS idx_files_profile_hash
                ON files(profile, content_hash);

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
                embedding_config_fingerprint TEXT NOT NULL,
                vector_id TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT,
                updated_at TEXT,
                FOREIGN KEY(file_id) REFERENCES files(file_id)
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_file_status
                ON chunks(file_id, status);
            CREATE INDEX IF NOT EXISTS idx_chunks_profile_path
                ON chunks(profile, path);

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
            """
        )


def scan_paths(
    paths: Sequence[str | Path],
    allowed_suffixes: set[str],
    ignored_dirs: set[str],
    validate_path: Callable[[Path], Any] | None = None,
) -> list[ScannedFile]:
    scanned: list[ScannedFile] = []

    for raw_path in paths:
        root = Path(raw_path).expanduser().resolve()
        if not root.exists():
            continue

        candidates = [root] if root.is_file() else root.rglob("*")

        for path in candidates:
            if any(part in ignored_dirs for part in path.parts):
                continue
            if not path.is_file():
                continue

            extension = path.suffix.lower()
            if extension not in allowed_suffixes:
                continue

            if validate_path:
                validate_path(path)

            stat = path.stat()
            scanned.append(
                ScannedFile(
                    path=str(path),
                    normalized_path=normalize_path(path),
                    size_bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    content_hash=content_hash_for_path(path),
                    extension=extension,
                )
            )

    return sorted(scanned, key=lambda item: item.normalized_path)


def _row_to_file(row: sqlite3.Row) -> ManifestFile:
    return ManifestFile(**dict(row))


def _row_to_chunk(row: sqlite3.Row) -> ManifestChunk:
    return ManifestChunk(**dict(row))


def get_files_for_profile(db_path: str | Path, profile: str) -> list[ManifestFile]:
    init_manifest_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT * FROM files
            WHERE profile = ?
            ORDER BY normalized_path
            """,
            (profile,),
        ).fetchall()
    return [_row_to_file(row) for row in rows]


def get_chunks_for_file(db_path: str | Path, file_id: str) -> list[ManifestChunk]:
    init_manifest_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT * FROM chunks
            WHERE file_id = ?
            ORDER BY chunk_index
            """,
            (file_id,),
        ).fetchall()
    return [_row_to_chunk(row) for row in rows]


def upsert_file(db_path: str | Path, row: ManifestFile) -> None:
    init_manifest_db(db_path)
    data = asdict(row)
    columns = list(data)
    placeholders = ", ".join("?" for _ in columns)
    assignments = ", ".join(f"{column} = excluded.{column}" for column in columns)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            f"""
            INSERT INTO files ({", ".join(columns)})
            VALUES ({placeholders})
            ON CONFLICT(file_id) DO UPDATE SET {assignments}
            """,
            [data[column] for column in columns],
        )


def upsert_chunk(db_path: str | Path, row: ManifestChunk) -> None:
    init_manifest_db(db_path)
    data = asdict(row)
    columns = list(data)
    placeholders = ", ".join("?" for _ in columns)
    assignments = ", ".join(f"{column} = excluded.{column}" for column in columns)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            f"""
            INSERT INTO chunks ({", ".join(columns)})
            VALUES ({placeholders})
            ON CONFLICT(chunk_id) DO UPDATE SET {assignments}
            """,
            [data[column] for column in columns],
        )


def mark_file_deleted(db_path: str | Path, file_id: str) -> None:
    init_manifest_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE files SET status = ? WHERE file_id = ?", ("deleted", file_id))
        conn.execute("UPDATE chunks SET status = ? WHERE file_id = ?", ("deleted", file_id))


def detect_changes(
    profile: str,
    scanned_files: list[ScannedFile],
    manifest_files: list[ManifestFile],
    *,
    collection_name: str | None = None,
    embedding_config_fingerprint: str | None = None,
    chunking_profile: str | None = None,
    embedding_profile: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    active_manifest = [
        item for item in manifest_files if item.profile == profile and item.status == "active"
    ]
    scanned_by_path = {item.normalized_path: item for item in scanned_files}
    manifest_by_path = {item.normalized_path: item for item in active_manifest}

    added_candidates = [
        item for item in scanned_files if item.normalized_path not in manifest_by_path
    ]
    deleted_candidates = [
        item for item in active_manifest if item.normalized_path not in scanned_by_path
    ]

    deleted_by_hash: dict[str, list[ManifestFile]] = {}
    for item in deleted_candidates:
        deleted_by_hash.setdefault(item.content_hash, []).append(item)

    moved: list[dict[str, Any]] = []
    added: list[dict[str, Any]] = []
    moved_new_paths: set[str] = set()
    moved_old_file_ids: set[str] = set()

    for scanned in added_candidates:
        candidates = deleted_by_hash.get(scanned.content_hash, [])
        if candidates:
            old = candidates.pop(0)
            moved.append(
                {
                    "file_id": old.file_id,
                    "old_path": old.path,
                    "new_path": scanned.path,
                    "old_normalized_path": old.normalized_path,
                    "new_normalized_path": scanned.normalized_path,
                    "content_hash": scanned.content_hash,
                    "extension": scanned.extension,
                    "status": "moved",
                }
            )
            moved_new_paths.add(scanned.normalized_path)
            moved_old_file_ids.add(old.file_id)
        else:
            added.append({**asdict(scanned), "status": "added"})

    edited = []
    config_changed = []
    for normalized_path, scanned in scanned_by_path.items():
        manifest = manifest_by_path.get(normalized_path)
        if manifest and manifest.content_hash != scanned.content_hash:
            edited.append(
                {
                    "file_id": manifest.file_id,
                    "path": scanned.path,
                    "normalized_path": scanned.normalized_path,
                    "old_content_hash": manifest.content_hash,
                    "content_hash": scanned.content_hash,
                    "size_bytes": scanned.size_bytes,
                    "mtime_ns": scanned.mtime_ns,
                    "extension": scanned.extension,
                    "status": "edited",
                }
            )
            continue

        if manifest and _manifest_config_changed(
            manifest,
            collection_name=collection_name,
            embedding_config_fingerprint=embedding_config_fingerprint,
            chunking_profile=chunking_profile,
            embedding_profile=embedding_profile,
        ):
            config_changed.append(
                {
                    "file_id": manifest.file_id,
                    "path": scanned.path,
                    "normalized_path": scanned.normalized_path,
                    "content_hash": scanned.content_hash,
                    "extension": scanned.extension,
                    "status": "config_changed",
                    "reason": "indexing configuration changed",
                    "current": {
                        "collection_name": collection_name,
                        "embedding_config_fingerprint": embedding_config_fingerprint,
                        "chunking_profile": chunking_profile,
                        "embedding_profile": embedding_profile,
                    },
                    "manifest": {
                        "collection_name": manifest.collection_name,
                        "embedding_config_fingerprint": manifest.embedding_config_fingerprint,
                        "chunking_profile": manifest.chunking_profile,
                        "embedding_profile": manifest.embedding_profile,
                    },
                }
            )

    deleted = [
        {
            "file_id": item.file_id,
            "path": item.path,
            "normalized_path": item.normalized_path,
            "content_hash": item.content_hash,
            "extension": item.extension,
            "status": "deleted",
        }
        for item in deleted_candidates
        if item.file_id not in moved_old_file_ids
    ]

    return {
        "added": [item for item in added if item["normalized_path"] not in moved_new_paths],
        "edited": edited,
        "config_changed": config_changed,
        "deleted": deleted,
        "moved": moved,
    }


def _manifest_config_changed(
    manifest: ManifestFile,
    *,
    collection_name: str | None,
    embedding_config_fingerprint: str | None,
    chunking_profile: str | None,
    embedding_profile: str | None,
) -> bool:
    expected = {
        "collection_name": collection_name,
        "embedding_config_fingerprint": embedding_config_fingerprint,
        "chunking_profile": chunking_profile,
        "embedding_profile": embedding_profile,
    }
    actual = {
        "collection_name": manifest.collection_name,
        "embedding_config_fingerprint": manifest.embedding_config_fingerprint,
        "chunking_profile": manifest.chunking_profile,
        "embedding_profile": manifest.embedding_profile,
    }

    return any(value is not None and actual[key] != value for key, value in expected.items())


def manifest_file_from_scan(
    scanned: ScannedFile,
    *,
    profile: str,
    collection_name: str,
    embedding_config_fingerprint: str,
    chunking_profile: str,
    embedding_profile: str,
    status: str = "active",
    last_indexed_at: str | None = None,
) -> ManifestFile:
    now = utc_now()
    return ManifestFile(
        file_id=file_id_for(profile, scanned.normalized_path),
        profile=profile,
        path=scanned.path,
        normalized_path=scanned.normalized_path,
        size_bytes=scanned.size_bytes,
        mtime_ns=scanned.mtime_ns,
        content_hash=scanned.content_hash,
        extension=scanned.extension,
        parser_name=None,
        parser_version=None,
        chunking_profile=chunking_profile,
        embedding_profile=embedding_profile,
        collection_name=collection_name,
        embedding_config_fingerprint=embedding_config_fingerprint,
        last_seen_at=now,
        last_indexed_at=last_indexed_at,
        status=status,
    )


def manifest_chunk_from_metadata(
    *,
    file_row: ManifestFile,
    chunk_index: int,
    chunk_hash: str,
    text_hash: str,
    metadata: dict[str, Any],
    embedding_provider: str | None,
    embedding_model: str | None,
    embedding_dimension: int | None,
    embedding_profile: str,
    chunking_profile: str,
) -> ManifestChunk:
    vector_id = vector_id_for(
        file_row.profile,
        file_row.collection_name or "",
        file_row.content_hash,
        chunk_index,
        chunk_hash,
        embedding_profile,
        chunking_profile,
    )
    return ManifestChunk(
        chunk_id=metadata.get("chunk_id") or vector_id,
        file_id=file_row.file_id,
        profile=file_row.profile,
        path=file_row.path,
        chunk_index=chunk_index,
        chunk_hash=chunk_hash,
        text_hash=text_hash,
        token_count=None,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        collection_name=file_row.collection_name or "",
        embedding_config_fingerprint=file_row.embedding_config_fingerprint or "",
        vector_id=vector_id,
        metadata_json=json.dumps(metadata, sort_keys=True, separators=(",", ":")),
        status="active",
        created_at=utc_now(),
        updated_at=utc_now(),
    )
