import json

import pytest
from fastapi.testclient import TestClient

from backend.ingestion.jobs import IndexJobs
from backend.main import create_app
from backend.rag.verification import parse_audit, verify_exact_facts
from backend.state import ChatStore, ProcessLease


def test_saved_chat_restart_history_and_idempotent_retry(services, monkeypatch):
    services.ingestion.ingest(
        "example.txt", b"Alpha deployment supports TLS encryption and certificate validation."
    )
    seen = []
    original = services.pipeline.answer

    def capture(body, **kwargs):
        seen.append(body.history)
        return original(body, **kwargs)

    monkeypatch.setattr(services.pipeline, "answer", capture)
    with TestClient(create_app(services.settings, services)) as client:
        chat = client.post("/conversations", json={"title": "TLS"}).json()["id"]
        first = {
            "conversation_id": chat,
            "request_id": "request-one",
            "question": "Alpha TLS encryption",
        }
        response = client.post("/chat", json=first)
        assert response.status_code == 200
        assert client.post("/chat", json=first).json() == response.json()
        assert len(seen) == 1
        assert client.post("/chat", json={**first, "question": "different"}).status_code == 409
        assert (
            client.post(
                "/chat", json={**first, "request_id": "two", "question": "certificate validation"}
            ).status_code
            == 200
        )
        assert seen[1][0].question == first["question"]
        assert client.post("/chat", json={**first, "scope": "pdf"}).status_code == 409
        saved = ChatStore(services.chats.path).get(chat)
        assert len(saved["turns"]) == 2
        assert saved["turns"][0]["response"]["sources"][0]["sha256"]
        assert client.delete("/conversations/" + chat).status_code == 200
        assert client.get("/conversations/" + chat).status_code == 404


def test_stream_draft_final_and_failed_turn_recovery(services, monkeypatch):
    services.ingestion.ingest("example.txt", b"Alpha deployment supports TLS encryption.")

    def stream(prompt, emit):
        emit("delta", {"text": "Alpha TLS [S1]"})
        return "Alpha TLS [S1]", {"total_tokens": 3}

    monkeypatch.setattr(services.pipeline.generator, "generate_stream", stream, raising=False)
    with TestClient(create_app(services.settings, services)) as client:
        chat = client.post("/conversations", json={}).json()["id"]
        payload = {"conversation_id": chat, "request_id": "stream-one", "question": "Alpha TLS"}
        response = client.post("/chat/stream", json=payload)
        events = [json.loads(line) for line in response.text.splitlines()]
        assert "delta" in [e["event"] for e in events]
        assert events[-1]["event"] == "final"
        assert events[-1]["data"]["conversation_id"] == chat
        assert client.get("/conversations/" + chat).json()["turns"][0]["status"] == "complete"

        def fail(*args):
            raise RuntimeError("private exception must not leak")

        monkeypatch.setattr(services.pipeline.generator, "generate_stream", fail)
        failed = client.post("/chat/stream", json={**payload, "request_id": "failed"})
        assert "private exception" not in failed.text
        assert json.loads(failed.text.splitlines()[-1])["event"] == "error"
        assert client.get("/conversations/" + chat).json()["turns"][-1]["status"] == "failed"


def test_interrupted_chat_retries_without_duplicate(services):
    chat = services.chats.create("interrupted", {"scope": "all", "document_ids": []})["id"]
    services.chats.begin(chat, "request", "question")
    with pytest.raises(ValueError, match="current response"):
        services.chats.begin(chat, "another", "question")
    recovered = ChatStore(services.chats.path)
    assert recovered.get(chat)["turns"][0]["status"] == "interrupted"
    recovered.begin(chat, "request", "question")
    recovered.finish("request", {"question": "question", "answer": "recovered"})
    assert len(recovered.get(chat)["turns"]) == 1


def test_index_replay_failure_and_retry(services, monkeypatch):
    job = services.jobs.submit("queued.txt", b"Queued document for recovery and indexing.")
    assert (
        services.jobs.submit("queued.txt", b"Queued document for recovery and indexing.")["id"]
        == job["id"]
    )
    with services.jobs.db() as db:
        db.execute("UPDATE jobs SET status='running' WHERE id=?", (job["id"],))
    queue = IndexJobs(services.ingestion)
    assert queue.get(job["id"])["status"] == "queued"
    ingest = services.ingestion.ingest
    monkeypatch.setattr(
        services.ingestion,
        "ingest",
        lambda *args: (_ for _ in ()).throw(RuntimeError("disk issue")),
    )
    queue.run_one()
    assert queue.get(job["id"])["status"] == "failed"
    assert list(queue.directory.glob("*.payload"))
    monkeypatch.setattr(services.ingestion, "ingest", ingest)
    queue.retry(job["id"])
    queue.run_one()
    assert queue.get(job["id"])["status"] == "complete"
    assert len(services.ingestion.manifest.all()) == 1
    assert not list(queue.directory.glob("*.payload"))
    # Crash after publication but before completion: replay is content-idempotent.
    replay = queue.submit("queued.txt", b"Queued document for recovery and indexing.")
    queue.run_one()
    assert queue.get(replay["id"])["result"]["status"] == "duplicate"
    assert len(services.ingestion.manifest.all()) == 1


def test_exclusive_owner(tmp_path):
    with ProcessLease(tmp_path / "owner.lock"):
        with pytest.raises(RuntimeError, match="already open"):
            with ProcessLease(tmp_path / "owner.lock"):
                pass
    with ProcessLease(tmp_path / "owner.lock"):
        pass


def test_exact_verification_checks_the_cited_source():
    chunks = [
        {
            "text": "SecNegotiateRspData: selectedSecCapability TLS. R17+ returns 403 REQUESTED_PURPOSE_NOT_ALLOWED."
        },
        {"text": "R16 returns 400 MANDATORY_QUERY_PARAM_INCORRECT."},
    ]
    assert not verify_exact_facts("R17+ returns 403 REQUESTED_PURPOSE_NOT_ALLOWED [S1]", chunks)
    assert verify_exact_facts("R17+ returns 400 MANDATORY_QUERY_PARAM_INCORRECT [S1]", chunks)
    assert verify_exact_facts("SecNegotiateRspData includes supportedSecCapability [S1]", chunks)
    assert parse_audit('{"supported":true,"checks":[{"source":"S1","quote":"invented"}]}', chunks)
    assert parse_audit("invalid JSON", chunks)
    assert not parse_audit(
        '{"supported":true,"checks":[{"source":"S2","quote":"R16 returns 400"}]}', chunks
    )


def test_backup_verification_detects_tampering_and_refuses_live_owner(tmp_path):
    from scripts.backup_profile import backup, verify

    source = tmp_path / "profile"
    (source / "uploads").mkdir(parents=True)
    (source / "uploads" / "original.bin").write_bytes(b"original document")
    destination = tmp_path / "backup"
    with ProcessLease(source / "uploads" / ".owner.lock"):
        with pytest.raises(RuntimeError):
            backup(source, destination)
    backup(source, destination)
    assert verify(destination) == 1
    (destination / "uploads" / "original.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="verification failed"):
        verify(destination)
    with pytest.raises(ValueError):
        backup(source, source / "recursive")


def test_adjacent_fields_keep_source_chunk_provenance(services, monkeypatch):
    from backend.models.schemas import ChatRequest

    services.ingestion.ingest(
        "definition.txt",
        b"ExampleResponse fields:\nsender: receiver\n"
        + b"Detailed context for the response. " * 3
        + b"\nexpiryAction: disconnect",
    )
    result = services.pipeline.answer(ChatRequest(question="Describe ExampleResponse fields"))
    evidence = "\n".join(c["text"] for c in result["retrieved_chunks"])
    assert "expiryAction" in evidence
    assert any(len(s["context_chunk_ids"]) > 1 for s in result["sources"])


def test_startup_reports_missing_original_without_deleting_ready_record(services):
    from backend.ingestion.service import IngestionService

    record = services.ingestion.ingest("original.txt", b"Original source with a known reference.")
    (services.ingestion.directory / (record["document_id"] + ".bin")).unlink()
    recovered = IngestionService(services.settings, services.store, services.embedder)
    assert recovered.recovery_issues == [
        {"document_id": record["document_id"], "issue": "missing_original"}
    ]
    assert recovered.manifest.get(record["document_id"])["status"] == "ready"


def test_queued_payload_corruption_is_not_published(services):
    job = services.jobs.submit("original.txt", b"Original source with a known reference.")
    (services.jobs.directory / (job["id"] + ".payload")).write_bytes(b"Corrupted content")
    services.jobs.run_one()
    assert services.jobs.get(job["id"])["status"] == "failed"
    assert "checksum mismatch" in services.jobs.get(job["id"])["error"]
    assert services.ingestion.manifest.all() == []


def test_failed_retry_and_delete_cannot_overlap_another_running_turn(services):
    chat = services.chats.create("Concurrency", {"scope": "all", "document_ids": []})["id"]
    services.chats.begin(chat, "failed", "first question")
    services.chats.finish("failed")
    services.chats.begin(chat, "running", "second question")
    with pytest.raises(ValueError, match="current response"):
        services.chats.begin(chat, "failed", "first question")
    with pytest.raises(ValueError, match="generation"):
        services.chats.delete(chat)
    services.chats.finish("running")
    services.chats.delete(chat)
    with pytest.raises(KeyError):
        services.chats.begin(chat, "after-delete", "third question")
