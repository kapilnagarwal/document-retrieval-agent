"""SQLite metadata repository with explicit connection ownership."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DocumentRecord = dict[str, Any]


class Manifest:
    def __init__(self, path: Path):
        self.path = path
        with self.connection() as db:
            db.execute("BEGIN")
            db.execute(
                "CREATE TABLE IF NOT EXISTS documents "
                "(id TEXT PRIMARY KEY, hash TEXT, status TEXT, record TEXT)"
            )
            # Versions may legitimately reuse bytes; migrate the original unique hash.
            definition = db.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='documents'"
            ).fetchone()[0]
            if "hash text unique" in definition.lower():
                db.execute(
                    "CREATE TABLE documents_migration "
                    "(id TEXT PRIMARY KEY, hash TEXT, status TEXT, record TEXT)"
                )
                db.execute("INSERT INTO documents_migration SELECT * FROM documents")
                db.execute("DROP TABLE documents")
                db.execute("ALTER TABLE documents_migration RENAME TO documents")
            db.execute("CREATE INDEX IF NOT EXISTS documents_hash ON documents(hash)")
            db.execute("CREATE INDEX IF NOT EXISTS documents_status ON documents(status)")

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _save(db: sqlite3.Connection, record: DocumentRecord) -> None:
        db.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET hash=excluded.hash, "
            "status=excluded.status, record=excluded.record",
            (record["document_id"], record["sha256"], record["status"], json.dumps(record)),
        )

    def save(self, record: DocumentRecord) -> None:
        with self.connection() as db:
            self._save(db, record)

    def activate(self, record: DocumentRecord, previous: DocumentRecord | None) -> None:
        """Publish a fully indexed replacement and retire its predecessor atomically."""
        with self.connection() as db:
            self._save(db, record)
            if previous:
                self._save(db, {**previous, "status": "superseded"})

    def all(self, ready: bool = True) -> list[DocumentRecord]:
        with self.connection() as db:
            sql = "SELECT record FROM documents" + (" WHERE status='ready'" if ready else "")
            return [json.loads(row[0]) for row in db.execute(sql)]

    def find_hash(self, digest: str) -> DocumentRecord | None:
        with self.connection() as db:
            row = db.execute(
                "SELECT record FROM documents WHERE hash=? AND status='ready' LIMIT 1",
                (digest,),
            ).fetchone()
            return json.loads(row[0]) if row else None

    def get(self, document_id: str) -> DocumentRecord | None:
        with self.connection() as db:
            row = db.execute("SELECT record FROM documents WHERE id=?", (document_id,)).fetchone()
            return json.loads(row[0]) if row else None

    def versions(self, source_id: str) -> list[DocumentRecord]:
        records = [
            record
            for record in self.all(False)
            if record.get("source_id", record["document_id"]) == source_id
            and record["status"] in {"ready", "superseded"}
        ]
        return sorted(records, key=lambda record: record.get("version", 1))

    def delete(self, document_id: str) -> None:
        with self.connection() as db:
            db.execute("DELETE FROM documents WHERE id=?", (document_id,))
