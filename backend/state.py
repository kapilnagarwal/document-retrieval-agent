"""Durable single-user chat state and an exclusive process lease."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import uuid4


def now():
    return datetime.now(timezone.utc).isoformat()


class ProcessLease:
    def __init__(self, path):
        self.path = path
        self.file = None

    def __enter__(self):
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+")
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise RuntimeError(
                "This data profile is already open by another process. Stop it before restarting or indexing from the CLI."
            ) from None
        return self

    def __exit__(self, *args):
        self.file.close()


class ChatStore:
    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, settings TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS turns (
                    request_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL
                    REFERENCES conversations(id) ON DELETE CASCADE,
                    question TEXT NOT NULL, response TEXT, status TEXT NOT NULL,
                    created_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS turns_conversation ON turns(conversation_id, created_at);
            """)
            # A process died before committing an answer. Preserve the question for retry.
            db.execute("UPDATE turns SET status='interrupted' WHERE status='running'")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA journal_mode=WAL")
            with db:
                yield db
        finally:
            db.close()

    def create(self, title, settings):
        identifier, stamp = str(uuid4()), now()
        with self.connection() as db:
            db.execute(
                "INSERT INTO conversations VALUES(?,?,?,?,?)",
                (identifier, title[:120], json.dumps(settings), stamp, stamp),
            )
        return self.get(identifier)

    def list(self, limit=100):
        with self.connection() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT id,title,created_at,updated_at FROM conversations ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                )
            ]

    def get(self, identifier):
        with self.connection() as db:
            row = db.execute("SELECT * FROM conversations WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise KeyError(identifier)
            turns = [
                dict(turn)
                for turn in db.execute(
                    "SELECT * FROM turns WHERE conversation_id=? ORDER BY created_at,rowid",
                    (identifier,),
                )
            ]
        return {
            **dict(row),
            "settings": json.loads(row["settings"]),
            "turns": [
                {**t, "response": json.loads(t["response"]) if t["response"] else None}
                for t in turns
            ],
        }

    def begin(self, identifier, request_id, question):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM conversations WHERE id=?", (identifier,)).fetchone():
                raise KeyError(identifier)
            row = db.execute("SELECT * FROM turns WHERE request_id=?", (request_id,)).fetchone()
            if row:
                if row["conversation_id"] != identifier or row["question"] != question:
                    raise ValueError("Request ID belongs to another question")
                if row["status"] == "complete":
                    return json.loads(row["response"])
                if row["status"] == "running":
                    raise ValueError("This request is already running")
                if db.execute(
                    "SELECT 1 FROM turns WHERE conversation_id=? AND status='running'",
                    (identifier,),
                ).fetchone():
                    raise ValueError("Wait for this conversation's current response")
                db.execute("UPDATE turns SET status='running' WHERE request_id=?", (request_id,))
            else:
                if db.execute(
                    "SELECT 1 FROM turns WHERE conversation_id=? AND status='running'",
                    (identifier,),
                ).fetchone():
                    raise ValueError("Wait for this conversation's current response")
                db.execute(
                    "INSERT INTO turns VALUES(?,?,?,?,?,?)",
                    (request_id, identifier, question, None, "running", now()),
                )
        return None

    def finish(self, request_id, response=None):
        with self.connection() as db:
            db.execute(
                "UPDATE turns SET response=?,status=? WHERE request_id=?",
                (
                    json.dumps(response) if response else None,
                    "complete" if response else "failed",
                    request_id,
                ),
            )
            db.execute(
                "UPDATE conversations SET updated_at=? WHERE id=(SELECT conversation_id FROM turns WHERE request_id=?)",
                (now(), request_id),
            )

    def delete(self, identifier):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute(
                "SELECT 1 FROM turns WHERE conversation_id=? AND status='running'", (identifier,)
            ).fetchone():
                raise ValueError("Wait for generation before deleting this conversation")
            db.execute("DELETE FROM conversations WHERE id=?", (identifier,))
