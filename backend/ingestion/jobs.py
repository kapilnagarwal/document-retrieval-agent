"""A durable serialized ingestion queue; unfinished jobs replay after restart."""

import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from threading import Event, Thread
from uuid import uuid4

from backend.ingestion.loader import safe_filename
from backend.state import now


class IndexJobs:
    def __init__(self, ingestion):
        self.ingestion = ingestion
        self.directory = ingestion.directory / "jobs"
        self.directory.mkdir(exist_ok=True)
        self.path = self.directory / "jobs.sqlite3"
        self.stop = Event()
        self.thread = None
        self.last_error = None
        with self.db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, filename TEXT, digest TEXT, status TEXT,
                created_at TEXT, result TEXT, error TEXT)""")
            db.execute("UPDATE jobs SET status='queued' WHERE status='running'")
            retained = {
                row["id"] for row in db.execute("SELECT id FROM jobs WHERE status!='complete'")
            }
        for payload in self.directory.glob("*.payload"):
            if payload.stem not in retained:
                payload.unlink(missing_ok=True)

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def submit(self, filename, content):
        filename = safe_filename(filename)
        if not content or len(content) > self.ingestion.settings.max_upload_size_mb * 1024 * 1024:
            raise ValueError("File is empty or exceeds MAX_UPLOAD_SIZE_MB")
        if filename.lower().rsplit(".", 1)[-1] not in {"pdf", "docx", "txt", "md"}:
            raise ValueError("Supported formats: PDF, DOCX, TXT, MD")
        digest = hashlib.sha256(content).hexdigest()
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM jobs WHERE digest=? AND status IN ('queued','running')", (digest,)
            ).fetchone()
            if row:
                return dict(row)
            identifier = str(uuid4())
            # Publish the durable payload before the queue row. Unreferenced payloads
            # are harmless and removed on the next startup.
            path = self.directory / f"{identifier}.payload"
            with path.open("wb") as file:
                import os

                file.write(content)
                file.flush()
                os.fsync(file.fileno())
            db.execute(
                "INSERT INTO jobs VALUES(?,?,?,?,?,?,?)",
                (identifier, filename, digest, "queued", now(), None, None),
            )
        return self.get(identifier)

    def get(self, identifier):
        with self.db() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (identifier,)).fetchone()
        if not row:
            raise KeyError(identifier)
        result = dict(row)
        result["result"] = json.loads(result["result"]) if result["result"] else None
        return result

    def list(self):
        with self.db() as db:
            identifiers = [
                row["id"]
                for row in db.execute("SELECT id FROM jobs ORDER BY created_at DESC LIMIT 100")
            ]
        return [self.get(identifier) for identifier in identifiers]

    def retry(self, identifier):
        with self.db() as db:
            row = db.execute("SELECT status FROM jobs WHERE id=?", (identifier,)).fetchone()
            if not row:
                raise KeyError(identifier)
            if not (self.directory / f"{identifier}.payload").is_file():
                raise ValueError("Queued payload is missing; upload the original document again")
            if row["status"] != "failed":
                raise ValueError("Only failed jobs can be retried")
            db.execute("UPDATE jobs SET status='queued',error=NULL WHERE id=?", (identifier,))
        return self.get(identifier)

    def run_one(self):
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if not row:
                return False
            db.execute("UPDATE jobs SET status='running' WHERE id=?", (row["id"],))
        try:
            content = (self.directory / f"{row['id']}.payload").read_bytes()
            if hashlib.sha256(content).hexdigest() != row["digest"]:
                raise ValueError(
                    "Queued payload checksum mismatch; upload the original document again"
                )
            result = self.ingestion.ingest(row["filename"], content)
            with self.db() as db:
                db.execute(
                    "UPDATE jobs SET status='complete',result=? WHERE id=?",
                    (json.dumps(result), row["id"]),
                )
            (self.directory / f"{row['id']}.payload").unlink(missing_ok=True)
        except Exception as exc:
            logging.getLogger(__name__).warning("Index job failed (%s)", type(exc).__name__)
            error = (
                str(exc)
                if isinstance(exc, ValueError)
                else "Indexing failed; check disk space/model availability, then retry."
            )
            with self.db() as db:
                db.execute("UPDATE jobs SET status='failed',error=? WHERE id=?", (error, row["id"]))
        return True

    def start(self):
        def work():
            while not self.stop.is_set():
                try:
                    worked = self.run_one()
                    self.last_error = None
                    if not worked:
                        self.stop.wait(0.5)
                except Exception as exc:
                    self.last_error = "Index queue unavailable; check disk space and backend logs."
                    logging.getLogger(__name__).error(
                        "Index worker temporarily unavailable (%s)", type(exc).__name__
                    )
                    self.stop.wait(2)

        self.thread = Thread(target=work, name="index-jobs", daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join()  # Finish the current atomic publication before releasing ownership.
