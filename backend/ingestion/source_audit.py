"""Read-only coverage audit of the user-approved Downloads notes folder."""

import hashlib


def audit_notes(ingestion, source=None):
    source = source or ingestion.settings.notes_source_directory
    records = ingestion.manifest.all()
    indexed_hashes = {record["sha256"] for record in records}
    names = ingestion.source_names()
    known_names = {name for name in names.values() if isinstance(name, str)}
    entries = []
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if (
                not path.is_file()
                or path.is_symlink()
                or any(part.startswith(".") for part in path.relative_to(source).parts)
            ):
                continue
            name = str(path.relative_to(source))
            try:
                raw = path.read_bytes()
                text = raw.decode("utf-8-sig")
                if not text.strip() or "\x00" in text:
                    status = "empty_or_binary"
                elif hashlib.sha256(raw).hexdigest() in indexed_hashes:
                    status = "indexed"
                elif name in known_names:
                    status = "changed_since_import"
                else:
                    status = "not_imported"
            except (OSError, UnicodeError):
                status = "unreadable_or_non_utf8"
            entries.append(dict(source_name=name, status=status))
    return dict(
        source_folder=str(source),
        folder_exists=source.is_dir(),
        files=entries,
        counts={
            status: sum(entry["status"] == status for entry in entries)
            for status in [
                "indexed",
                "changed_since_import",
                "not_imported",
                "empty_or_binary",
                "unreadable_or_non_utf8",
            ]
        },
        message="This audits saved files only. Unsaved editor buffers are not automatically monitored or imported.",
    )
