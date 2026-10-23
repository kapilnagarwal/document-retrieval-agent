"""Regression coverage for evidence, source versions, resource ownership, and locks."""

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.manifest import Manifest
from backend.models.schemas import ChatRequest
from backend.rag.evidence import assess_evidence
from backend.services import Services
from scripts.evaluate_retrieval import score_results


def test_connections_close_and_transactions_roll_back(tmp_path):
    manifest = Manifest(tmp_path / "metadata.sqlite3")
    with manifest.connection() as connection:
        connection.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
    with pytest.raises(RuntimeError):
        with manifest.connection() as failed:
            failed.execute("INSERT INTO documents VALUES ('x', 'hash', 'ready', '{}')")
            raise RuntimeError("abort")
    assert manifest.get("x") is None
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        failed.execute("SELECT 1")


def test_legacy_manifest_migration_preserves_records(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    record = {"document_id": "old", "sha256": "same", "status": "ready"}
    connection = sqlite3.connect(path)
    try:
        with connection:
            connection.execute(
                "CREATE TABLE documents (id TEXT PRIMARY KEY, hash TEXT UNIQUE, status TEXT, record TEXT)"
            )
            connection.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?)",
                ("old", "same", "ready", json.dumps(record)),
            )
    finally:
        connection.close()
    manifest = Manifest(path)
    assert manifest.get("old") == record
    manifest.save({**record, "document_id": "new"})
    assert len(manifest.all()) == 2
    assert Manifest(path).get("new")["sha256"] == "same"


def test_replacement_versions_are_atomic_and_survive_restart(services):
    first = services.ingestion.ingest("guide.txt", b"RAG original reference")
    second = services.ingestion.ingest("guide.txt", b"RAG updated reference", first["document_id"])
    assert second["source_id"] == first["source_id"]
    assert second["version"] == 2 and second["supersedes"] == first["document_id"]
    assert services.ingestion.manifest.all() == [second]
    assert services.ingestion.content(first["document_id"])["content"] == "RAG original reference"
    # Reverting source bytes creates a new version, rather than hitting a unique-hash constraint.
    third = services.ingestion.ingest("guide.txt", b"RAG original reference", second["document_id"])
    assert third["version"] == 3
    reopened = Services(services.settings, services.embedder)
    assert [r["status"] for r in reopened.ingestion.versions(third["document_id"])] == [
        "superseded",
        "superseded",
        "ready",
    ]
    assert reopened.ingestion.content(first["document_id"])["content"] == "RAG original reference"
    assert set(reopened.store.list_document_ids()) == {third["document_id"]}


def test_failed_replacement_preserves_previous_active_source(services, monkeypatch):
    original = services.ingestion.ingest("guide.txt", b"RAG original reference")

    def fail(texts):
        raise RuntimeError("embedding failure")

    monkeypatch.setattr(services.embedder, "documents", fail)
    with pytest.raises(RuntimeError, match="embedding failure"):
        services.ingestion.ingest("guide.txt", b"RAG new reference", original["document_id"])
    assert services.ingestion.manifest.all() == [original]
    assert services.store.list_document_ids() == [original["document_id"]]
    assert len(list(services.ingestion.directory.glob("*.bin"))) == 1


def test_cleanup_failure_keeps_new_version_active_and_search_excludes_old(services, monkeypatch):
    first = services.ingestion.ingest("guide.txt", b"RAG old reference")
    delete = services.store.delete

    def fail_old(identifier):
        if identifier == first["document_id"]:
            raise RuntimeError("cleanup interrupted")
        delete(identifier)

    monkeypatch.setattr(services.store, "delete", fail_old)
    current = services.ingestion.ingest("guide.txt", b"RAG new reference", first["document_id"])
    with TestClient(create_app(services.settings, services)) as client:
        results = client.post("/search", json={"query": "RAG"}).json()
        assert {h["document_id"] for h in results["chunks"]} == {current["document_id"]}
    reopened = Services(services.settings, services.embedder)
    assert reopened.store.list_document_ids() == [current["document_id"]]


def test_replacement_api_and_version_history(services):
    first = services.ingestion.ingest("guide.txt", b"RAG original")
    with TestClient(create_app(services.settings, services)) as client:
        url = "/documents/" + first["document_id"]
        response = client.put(url, files={"file": ("guide.txt", b"RAG updated")})
        assert response.status_code == 200
        assert response.json()["version"] == 2
        assert len(client.get(url + "/versions").json()) == 2
        assert client.put(url, files={"file": ("guide.txt", b"stale update")}).status_code == 404
        new_url = "/documents/" + response.json()["document_id"]
        assert client.put(new_url, files={"file": ("guide.txt", b"")}).status_code == 422
        assert client.get("/documents/missing/versions").status_code == 404
        assert len(client.get("/documents").json()) == 1


def test_unrelated_nearest_passages_do_not_trigger_generation(services, monkeypatch):
    services.ingestion.ingest("garden.txt", b"Tomatoes grow in moist garden soil.")

    def fail(prompt):
        pytest.fail("Unrelated evidence must not be sent to the generator")

    monkeypatch.setattr(services.pipeline.generator, "generate", fail)
    result = services.pipeline.answer(
        ChatRequest(question="Explain TLS certificate validation", debug=True)
    )
    assert result["retrieved_chunks"]  # Deliberately nonempty nearest-neighbor results.
    assert result["answer_status"] == "insufficient_evidence"
    assert result["generation_seconds"] == 0 and not result["sources"]
    assert result["evidence"]["reason"] == "insufficient_keyword_or_literal_support"


def test_literal_constraints_and_uncited_answer_status(services, monkeypatch):
    assessment = assess_evidence(
        'podman "1.0.4-3"', [{"chunk_id": "wrong", "text": "podman 1.0.4-30"}], 0.5
    )
    assert not assessment.sufficient
    services.ingestion.ingest("rag.txt", b"RAG retrieves supporting passages")
    monkeypatch.setattr(
        services.pipeline.generator, "generate", lambda prompt: ("Uncited answer", None)
    )
    result = services.pipeline.answer(ChatRequest(question="What is RAG?"))
    assert result["answer_status"] == "unverified"


def test_slow_generation_releases_lock_and_preserves_source_snapshot(services, monkeypatch):
    original = services.ingestion.ingest("rag.txt", b"RAG retrieves supporting passages")
    started, release = Event(), Event()

    def slow(prompt):
        started.set()
        assert release.wait(5), "Test did not release generator"
        return "RAG retrieves supporting passages [S1]", None

    monkeypatch.setattr(services.pipeline.generator, "generate", slow)
    with ThreadPoolExecutor(max_workers=2) as executor:
        answer = executor.submit(services.pipeline.answer, ChatRequest(question="What is RAG?"))
        try:
            assert started.wait(3)
            deletion = executor.submit(services.ingestion.delete, original["document_id"])
            deletion.result(timeout=2)  # Would time out if generation held the ingestion lock.
        finally:
            release.set()
        result = answer.result(timeout=3)
    assert result["answer_status"] == "answered"
    assert result["sources"][0]["document_id"] == original["document_id"]
    assert (
        result["sources"][0]["sha256"]
        == hashlib.sha256(b"RAG retrieves supporting passages").hexdigest()
    )


def test_evaluation_metrics_count_missing_sources_and_false_support():
    cases = [{"id": "positive", "expected": ["a", "b"]}, {"id": "negative", "expected": []}]
    metrics = score_results(
        cases,
        {"positive": ["noise", "a"], "negative": ["noise"]},
        {"positive": True, "negative": True},
    )
    assert metrics["recall_at_k"] == 0.5
    assert metrics["mrr_at_k"] == 0.5
    assert metrics["false_support_rate"] == 1.0
