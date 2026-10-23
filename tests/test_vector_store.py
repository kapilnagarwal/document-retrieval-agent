import pytest

from backend.rag.vector_store import VectorStore
from backend.services import Services


def test_persistence_duplicate_and_delete(services):
    result = services.ingestion.ingest(
        "guide.txt", b"RAG retrieves passages from a vector database."
    )
    identifier = result["document_id"]
    assert (
        services.ingestion.ingest("renamed.txt", b"RAG retrieves passages from a vector database.")[
            "status"
        ]
        == "duplicate"
    )
    reopened = VectorStore(
        services.settings.chroma_persist_directory, services.settings.embedding_model, 120, 20
    )
    assert reopened.count() == result["chunk_count"]
    assert reopened.by_document(identifier)["documents"]
    services.ingestion.delete(identifier)
    assert reopened.count() == 0
    assert not services.ingestion.manifest.all()


def test_failed_ingestion_rolls_back(services, monkeypatch):
    original = services.store.insert

    def broken(chunks, vectors):
        original(chunks, vectors)
        raise RuntimeError("simulated storage failure")

    monkeypatch.setattr(services.store, "insert", broken)
    with pytest.raises(RuntimeError):
        services.ingestion.ingest("guide.txt", b"some reference text")
    assert services.store.count() == 0
    assert services.ingestion.manifest.all(False) == []
    assert not list(services.ingestion.directory.glob("*.bin"))


def test_restart_recovers_pending(services):
    result = services.ingestion.ingest("guide.txt", b"reference text")
    result["status"] = "deleting"
    services.ingestion.manifest.save(result)
    Services(services.settings, services.embedder)
    assert not services.ingestion.manifest.all(False)
    assert services.store.count() == 0
