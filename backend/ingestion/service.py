import hashlib
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

from backend.ingestion.chunker import Chunker
from backend.ingestion.loader import extract, safe_filename
from backend.manifest import Manifest


class IngestionService:
    def __init__(self, settings, store, embedder):
        self.settings, self.store, self.embedder = settings, store, embedder
        self.lock = RLock()
        settings.upload_directory.mkdir(parents=True, exist_ok=True)
        self.directory = settings.upload_directory / store.name
        self.directory.mkdir(exist_ok=True)
        self.manifest = Manifest(self.directory / "manifest.sqlite3")
        self.chunker = Chunker(settings.chunk_size, settings.chunk_overlap)
        # Recover interrupted ingestion/deletion. One backend process owns this local database.
        with self.lock:
            for record in self.manifest.all(False):
                if record["status"] in {"processing", "deleting"}:
                    self._remove(record["document_id"])
            records = self.manifest.all(False)
            ready = {r["document_id"] for r in records if r["status"] == "ready"}
            retained = {r["document_id"] for r in records}
            for document_id in set(store.list_document_ids()) - ready:
                store.delete(document_id)
            for path in self.directory.glob("*.bin"):
                if path.stem not in retained:
                    path.unlink(missing_ok=True)
            counts = Counter(
                m["document_id"] for m in store.collection.get(include=["metadatas"])["metadatas"]
            )
            self.recovery_issues = []
            for record in records:
                if record["status"] != "ready":
                    continue
                identifier = record["document_id"]
                if not (self.directory / f"{identifier}.bin").is_file():
                    self.recovery_issues.append(
                        {"document_id": identifier, "issue": "missing_original"}
                    )
                if counts[identifier] != record["chunk_count"]:
                    self.recovery_issues.append(
                        {"document_id": identifier, "issue": "chunk_count_mismatch"}
                    )

    def _remove(self, document_id):
        self.store.delete(document_id)
        (self.directory / f"{document_id}.bin").unlink(missing_ok=True)
        self.manifest.delete(document_id)

    def ingest(self, filename: str, content: bytes, replace_document_id: str | None = None):
        start = time.perf_counter()
        filename = safe_filename(filename)
        if not content or len(content) > self.settings.max_upload_size_mb * 1024 * 1024:
            raise ValueError("File is empty or exceeds MAX_UPLOAD_SIZE_MB")
        digest = hashlib.sha256(content).hexdigest()
        with self.lock:
            previous = self.manifest.get(replace_document_id) if replace_document_id else None
            if replace_document_id and (not previous or previous["status"] != "ready"):
                raise KeyError(replace_document_id)
            existing = previous if previous else self.manifest.find_hash(digest)
            if existing and existing["sha256"] == digest:
                return {**existing, "status": "duplicate"}
            document_id = str(uuid4())
            path = self.directory / f"{document_id}.bin"
            record = dict(
                document_id=document_id,
                filename=filename,
                sha256=digest,
                size_bytes=len(content),
                character_count=0,
                chunk_count=0,
                duration_seconds=0,
                status="processing",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            record.update(
                source_id=previous.get("source_id", previous["document_id"])
                if previous
                else document_id,
                version=previous.get("version", 1) + 1 if previous else 1,
                supersedes=previous["document_id"] if previous else None,
            )
            try:
                self.manifest.save(record)
                with path.open("wb") as file:
                    import os

                    file.write(content)
                    file.flush()
                    os.fsync(file.fileno())
                sections = extract(path, Path(filename).suffix.lower())
                chunks = self.chunker.split(sections, document_id, filename)
                if not chunks:
                    raise ValueError("No chunks extracted")
                # Embed and store batches to bound peak memory on large manuals.
                for start_index in range(0, len(chunks), 128):
                    batch = chunks[start_index : start_index + 128]
                    self.store.insert(
                        batch, self.embedder.documents([chunk.text for chunk in batch])
                    )
                record.update(
                    character_count=sum(len(s.text) for s in sections),
                    chunk_count=len(chunks),
                    duration_seconds=time.perf_counter() - start,
                    status="ready",
                )
                self.manifest.activate(record, previous)
            except Exception:
                self._remove(document_id)
                raise
            # Publication committed; a later cleanup failure must not undo the new version.
            # Startup recovery removes vectors belonging to superseded records.
            if previous:
                try:
                    self.store.delete(previous["document_id"])
                except Exception:
                    import logging

                    logging.getLogger(__name__).warning("Deferred superseded vector cleanup")
            return record

    def versions(self, document_id: str):
        with self.lock:
            record = self.manifest.get(document_id)
            if not record:
                raise KeyError(document_id)
            return self.manifest.versions(record.get("source_id", document_id))

    def delete(self, document_id):
        with self.lock:
            record = self.manifest.get(document_id)
            if not record:
                raise KeyError(document_id)
            record["status"] = "deleting"
            self.manifest.save(record)
            self._remove(document_id)

    def content(self, document_id):
        """Read the original imported copy; never concatenate overlapping chunks."""
        with self.lock:
            record = self.manifest.get(document_id)
            if not record or record["status"] not in {"ready", "superseded"}:
                raise KeyError(document_id)
            path = self.directory / f"{record['document_id']}.bin"
            suffix = Path(record["filename"]).suffix.lower()
            if suffix in {".txt", ".md"}:
                # Decode bytes directly to preserve original line endings and blank lines.
                text = path.read_bytes().decode("utf-8-sig")
                content_format = "original_text"
            else:
                text = "\n\n".join(section.text for section in extract(path, suffix))
                content_format = "extracted_text"
            source_name = self.source_names().get(document_id)
            return dict(
                document_id=document_id,
                filename=record["filename"],
                source_name=source_name,
                content=text,
                character_count=len(text),
                content_format=content_format,
            )

    def source_names(self):
        import json

        path = self.settings.upload_directory / "notes_filename_mapping.json"
        try:
            mapping = json.loads(path.read_text())
            if not isinstance(mapping, dict):
                return {}
            for record in self.manifest.all(False):
                original_name = mapping.get(record.get("source_id"))
                if original_name:
                    mapping.setdefault(record["document_id"], original_name)
            return mapping
        except (OSError, ValueError):
            return {}
