from __future__ import annotations

from pathlib import Path

from mcp_server.retrieval.manifest import (
    detect_changes,
    get_files_for_profile,
    init_manifest_db,
    manifest_file_from_scan,
    scan_paths,
    upsert_file,
)


def scan_tmp(root: Path):
    return scan_paths([root], {".md", ".txt"}, ignored_dirs=set())


def index_scan(db_path: Path, root: Path, profile: str = "indexed") -> None:
    for scanned in scan_tmp(root):
        upsert_file(
            db_path,
            manifest_file_from_scan(
                scanned,
                profile=profile,
                collection_name="knowledge_indexed_test",
                embedding_config_fingerprint="fingerprint",
                chunking_profile="structure_v1",
                embedding_profile="local_bge_small",
                last_indexed_at="2026-06-08T00:00:00Z",
            ),
        )


def changes_for(db_path: Path, root: Path, profile: str = "indexed"):
    scanned = scan_tmp(root)
    manifest_files = get_files_for_profile(db_path, profile)
    return detect_changes(
        profile,
        scanned,
        manifest_files,
        collection_name="knowledge_indexed_test",
        embedding_config_fingerprint="fingerprint",
        chunking_profile="structure_v1",
        embedding_profile="local_bge_small",
    )


def test_added_file_detected(tmp_path):
    db_path = tmp_path / "manifest.sqlite"
    init_manifest_db(db_path)
    (tmp_path / "added.md").write_text("# Added\n", encoding="utf-8")

    changes = changes_for(db_path, tmp_path)

    assert [item["path"] for item in changes["added"]] == [str(tmp_path / "added.md")]
    assert changes["edited"] == []
    assert changes["config_changed"] == []
    assert changes["deleted"] == []
    assert changes["moved"] == []


def test_edited_file_detected_by_hash(tmp_path):
    db_path = tmp_path / "manifest.sqlite"
    file_path = tmp_path / "note.md"
    file_path.write_text("# Before\n", encoding="utf-8")
    index_scan(db_path, tmp_path)

    file_path.write_text("# After\n", encoding="utf-8")
    changes = changes_for(db_path, tmp_path)

    assert changes["added"] == []
    assert len(changes["edited"]) == 1
    assert changes["config_changed"] == []
    assert changes["edited"][0]["path"] == str(file_path)
    assert changes["edited"][0]["old_content_hash"] != changes["edited"][0]["content_hash"]


def test_deleted_file_detected(tmp_path):
    db_path = tmp_path / "manifest.sqlite"
    file_path = tmp_path / "deleted.md"
    file_path.write_text("# Deleted\n", encoding="utf-8")
    index_scan(db_path, tmp_path)

    file_path.unlink()
    changes = changes_for(db_path, tmp_path)

    assert changes["added"] == []
    assert changes["edited"] == []
    assert changes["config_changed"] == []
    assert len(changes["deleted"]) == 1
    assert changes["deleted"][0]["path"] == str(file_path)


def test_moved_file_detected_by_same_hash(tmp_path):
    db_path = tmp_path / "manifest.sqlite"
    old_path = tmp_path / "old.md"
    new_path = tmp_path / "new.md"
    old_path.write_text("# Same content\n", encoding="utf-8")
    index_scan(db_path, tmp_path)

    old_path.rename(new_path)
    changes = changes_for(db_path, tmp_path)

    assert changes["added"] == []
    assert changes["deleted"] == []
    assert changes["config_changed"] == []
    assert len(changes["moved"]) == 1
    assert changes["moved"][0]["old_path"] == str(old_path)
    assert changes["moved"][0]["new_path"] == str(new_path)


def test_dry_run_detection_does_not_modify_manifest(tmp_path):
    db_path = tmp_path / "manifest.sqlite"
    init_manifest_db(db_path)
    (tmp_path / "added.md").write_text("# Added\n", encoding="utf-8")

    changes = changes_for(db_path, tmp_path)
    files_after_scan = get_files_for_profile(db_path, "indexed")

    assert len(changes["added"]) == 1
    assert files_after_scan == []


def test_manifest_persists_across_connections(tmp_path):
    db_path = tmp_path / "manifest.sqlite"
    file_path = tmp_path / "persisted.md"
    file_path.write_text("# Persisted\n", encoding="utf-8")
    index_scan(db_path, tmp_path)

    files = get_files_for_profile(db_path, "indexed")

    assert len(files) == 1
    assert files[0].path == str(file_path)
    assert files[0].status == "active"


def test_duplicate_files_with_same_content_create_two_manifest_rows(tmp_path):
    db_path = tmp_path / "manifest.sqlite"
    first_path = tmp_path / "first.md"
    second_path = tmp_path / "second.md"
    first_path.write_text("# Same\n", encoding="utf-8")
    second_path.write_text("# Same\n", encoding="utf-8")
    index_scan(db_path, tmp_path)

    files = get_files_for_profile(db_path, "indexed")

    assert len(files) == 2
    assert files[0].content_hash == files[1].content_hash
    assert files[0].file_id != files[1].file_id


def test_changing_only_config_fingerprint_marks_config_changed(tmp_path):
    db_path = tmp_path / "manifest.sqlite"
    file_path = tmp_path / "note.md"
    file_path.write_text("# Stable\n", encoding="utf-8")
    index_scan(db_path, tmp_path)

    scanned = scan_tmp(tmp_path)
    manifest_files = get_files_for_profile(db_path, "indexed")
    changes = detect_changes(
        "indexed",
        scanned,
        manifest_files,
        collection_name="knowledge_indexed_test",
        embedding_config_fingerprint="new-fingerprint",
        chunking_profile="structure_v1",
        embedding_profile="local_bge_small",
    )

    assert changes["added"] == []
    assert changes["edited"] == []
    assert len(changes["config_changed"]) == 1
    assert changes["config_changed"][0]["path"] == str(file_path)
    assert changes["config_changed"][0]["status"] == "config_changed"


def test_dry_run_config_changed_detection_does_not_modify_manifest(tmp_path):
    db_path = tmp_path / "manifest.sqlite"
    file_path = tmp_path / "note.md"
    file_path.write_text("# Stable\n", encoding="utf-8")
    index_scan(db_path, tmp_path)
    files_before = get_files_for_profile(db_path, "indexed")

    scanned = scan_tmp(tmp_path)
    manifest_files = get_files_for_profile(db_path, "indexed")
    changes = detect_changes(
        "indexed",
        scanned,
        manifest_files,
        collection_name="different_collection",
        embedding_config_fingerprint="fingerprint",
        chunking_profile="structure_v1",
        embedding_profile="local_bge_small",
    )
    files_after = get_files_for_profile(db_path, "indexed")

    assert len(changes["config_changed"]) == 1
    assert files_after == files_before
