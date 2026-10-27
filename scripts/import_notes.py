"""Import UTF-8 notes, including extensionless Notepad++ buffers, locally only."""

import argparse
import hashlib
import json
from pathlib import Path

from backend.config import Settings
from backend.services import Services
from backend.state import ProcessLease


def main():
    settings = Settings()
    if settings.allow_gemini_document_context or settings.data_profile != "local":
        raise SystemExit("Use python -m scripts.run_profile local index-notes")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, nargs="?", default=settings.notes_source_directory)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace a previously imported source by its original relative name",
    )
    args = parser.parse_args()
    source = args.folder
    if not source.is_dir():
        raise SystemExit("Notes folder missing")
    services = Services(settings)
    counts = {"ready": 0, "duplicate": 0, "skipped": 0, "error": 0}
    existing_mapping = settings.upload_directory / "notes_filename_mapping.json"
    mapping = json.loads(existing_mapping.read_text()) if existing_mapping.exists() else {}
    active_names = services.ingestion.source_names()
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        try:
            content = path.read_bytes()
            text = content.decode("utf-8-sig")
            if not text.strip() or "\x00" in text:
                counts["skipped"] += 1
                continue
            # Buffer names sometimes contain pasted commands. Use neutral labels to avoid logging them.
            label = (
                "local-note-"
                + hashlib.sha256(str(path.relative_to(source)).encode()).hexdigest()[:12]
                + ".txt"
            )
            previous = [
                record
                for record in services.ingestion.manifest.all()
                if active_names.get(record["document_id"]) == str(path.relative_to(source))
            ]
            if args.replace_existing and len(previous) > 1:
                raise ValueError(
                    "Multiple active imports share this source name; select a replacement in the UI"
                )
            replacement = previous[0]["document_id"] if args.replace_existing and previous else None
            record = services.ingestion.ingest(label, content, replacement)
            counts[record["status"]] += 1
            mapping[record["document_id"]] = str(path.relative_to(source))
        except UnicodeDecodeError:
            counts["skipped"] += 1
        except Exception as exc:
            counts["error"] += 1
            print(
                json.dumps({"file_index": sum(counts.values()), "error_type": type(exc).__name__}),
                flush=True,
            )
    mapping_path = settings.upload_directory / "notes_filename_mapping.json"
    mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2))
    mapping_path.chmod(0o600)
    print(json.dumps(counts), flush=True)
    raise SystemExit(1 if counts["error"] else 0)


if __name__ == "__main__":
    with ProcessLease(Settings().upload_directory / ".owner.lock"):
        main()
