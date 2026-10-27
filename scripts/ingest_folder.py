"""Local-only import: run while the API is stopped (single database owner)."""

import argparse
import json
from pathlib import Path

from backend.config import Settings
from backend.ingestion.loader import SUPPORTED
from backend.services import Services
from backend.state import ProcessLease


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace the active import with the same filename; rejects ambiguous names",
    )
    args = parser.parse_args()
    if not args.folder.is_dir():
        parser.error("Folder not found")
    services = Services(Settings())
    failed = False
    for path in sorted(args.folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            try:
                if path.stat().st_size > services.settings.max_upload_size_mb * 1024 * 1024:
                    raise ValueError("File exceeds MAX_UPLOAD_SIZE_MB")
                matches = [
                    record
                    for record in services.ingestion.manifest.all()
                    if record["filename"] == path.name
                ]
                if args.replace_existing and len(matches) > 1:
                    raise ValueError(
                        "Ambiguous filename; replace an explicit document ID in the UI"
                    )
                replacement = (
                    matches[0]["document_id"] if args.replace_existing and matches else None
                )
                result = services.ingestion.ingest(path.name, path.read_bytes(), replacement)
                print(json.dumps(result), flush=True)
            except Exception as exc:
                print(
                    json.dumps(
                        {"filename": path.name, "status": "error", "error_type": type(exc).__name__}
                    ),
                    flush=True,
                )
                failed = True
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    with ProcessLease(Settings().upload_directory / ".owner.lock"):
        main()
