import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.rag.document_finder import document_phrase, wants_document

QUERY = "find what is Supplier text and its whole document content"


@pytest.mark.parametrize(
    "query,phrase",
    [
        (QUERY, "what is Supplier"),
        ('find "what is Supplier" and its whole document content', "what is Supplier"),
        ("show whole document containing Supplier", "Supplier"),
        ("Supplier", "Supplier"),
        ("find podman build command and its whole document content", "podman build"),
    ],
)
def test_document_phrase_extraction(query, phrase):
    assert document_phrase(query) == phrase
    assert wants_document(query) == ("document" in query)


def test_whole_document_uses_original_text_without_overlap_or_other_sources(services, monkeypatch):
    original = (
        b"List example\r\n\r\n1. what is Supplier ?\r\n"
        + b"Original note material. " * 15
        + b"\r\nFinal line\r\n"
    )
    expected = services.ingestion.ingest("supplier-note.txt", original)
    services.ingestion.ingest(
        "unrelated.txt", b"Other Java lessons and an irrelevant supplier invoice."
    )
    monkeypatch.setattr(
        services.embedder,
        "query",
        lambda query: pytest.fail("Whole-file lookup must not invoke query embeddings"),
    )
    with TestClient(create_app(services.settings, services)) as client:
        result = client.post("/search", json={"query": QUERY}).json()
        assert result["result_type"] == "documents"
        assert len(result["documents"]) == 1
        assert result["documents"][0]["document_id"] == expected["document_id"]
        full = client.get(f"/documents/{expected['document_id']}/content").json()
        assert full["content"] == original.decode()
        assert full["content"].count("what is Supplier") == 1
        assert "irrelevant supplier invoice" not in full["content"]
        assert full["content_format"] == "original_text"


def test_missing_document_phrase_does_not_fall_back_to_nearest_chunks(services):
    services.ingestion.ingest("other.txt", b"A supplier sells goods. Java interfaces are useful.")
    result = services.document_finder.find(QUERY)
    assert not result["documents"] and result["total_matches"] == 0
    assert "No imported document contains" in result["message"]


def test_whole_document_filter_and_multiple_candidates(services):
    first = services.ingestion.ingest("first.txt", b"what is Supplier? First complete note")
    second = services.ingestion.ingest("second.txt", b"what is Supplier? Second complete note")
    result = services.document_finder.find(QUERY)
    assert result["total_matches"] == 2
    assert "Choose a document" in result["message"]
    filtered = services.document_finder.find(QUERY, [second["document_id"]])
    assert [d["document_id"] for d in filtered["documents"]] == [second["document_id"]]
    services.ingestion.delete(first["document_id"])
    with TestClient(create_app(services.settings, services)) as client:
        assert client.get(f"/documents/{first['document_id']}/content").status_code == 404
        assert client.get("/documents/missing/content").status_code == 404


def test_explicit_passage_mode_keeps_passage_contract(services):
    services.ingestion.ingest("note.txt", b"what is Supplier? A question in a learning note")
    with TestClient(create_app(services.settings, services)) as client:
        result = client.post("/search", json={"query": QUERY, "mode": "passages"}).json()
        assert "chunks" in result and "documents" not in result
