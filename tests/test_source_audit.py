import json

from backend.ingestion.source_audit import audit_notes


def test_source_audit_reports_saved_file_coverage(services, tmp_path):
    folder = tmp_path / "saved-notes"
    folder.mkdir()
    (folder / "indexed.txt").write_text("original text")
    record = services.ingestion.ingest("opaque.txt", b"original text")
    (services.settings.upload_directory / "notes_filename_mapping.json").write_text(
        json.dumps({record["document_id"]: "indexed.txt"})
    )
    (folder / "new.txt").write_text("not imported")
    (folder / "empty.txt").write_text("   ")
    (folder / "binary").write_bytes(b"\xff\xfe")
    audit = audit_notes(services.ingestion, folder)
    assert audit["counts"]["indexed"] == 1
    assert audit["counts"]["not_imported"] == 1
    assert audit["counts"]["empty_or_binary"] == 1
    assert audit["counts"]["unreadable_or_non_utf8"] == 1
    (folder / "indexed.txt").write_text("changed after import")
    assert audit_notes(services.ingestion, folder)["counts"]["changed_since_import"] == 1


def test_missing_folder_and_unsaved_limit_are_explicit(services, tmp_path):
    result = audit_notes(services.ingestion, tmp_path / "missing")
    assert result["folder_exists"] is False
    assert "Unsaved editor buffers" in result["message"]
