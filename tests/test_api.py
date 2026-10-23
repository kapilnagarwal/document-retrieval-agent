from fastapi.testclient import TestClient

from backend.main import create_app
from backend.rag.generator import GenerationError


def test_api_lifecycle_and_validation(services):
    with TestClient(create_app(services.settings, services)) as client:
        assert client.get("/health").json()["chunks"] == 0
        assert "gemini_api_key" not in client.get("/config").json()
        result = client.post(
            "/documents/upload",
            files=[
                ("files", ("guide.txt", b"RAG retrieves passages.")),
                ("files", ("bad.exe", b"bad")),
            ],
        ).json()
        assert result[0]["status"] == "ready" and result[1]["status"] == "error"
        document_id = result[0]["document_id"]
        assert len(client.get("/documents").json()) == 1
        assert client.post("/search", json={"query": "RAG"}).json()["chunks"]
        assert client.post("/search", json={"query": " ", "top_k": 0}).status_code == 422
        assert client.post("/chat", json={"question": "RAG?"}).json()["debug"] is None
        assert client.delete("/documents/" + document_id).status_code == 200
        assert client.delete("/documents/" + document_id).status_code == 404


def test_size_and_empty(services):
    services.settings.max_upload_size_mb = 1
    with TestClient(create_app(services.settings, services)) as client:
        for content in (b"", b"x" * (1024 * 1024 + 1)):
            result = client.post(
                "/documents/upload", files={"files": ("guide.txt", content)}
            ).json()
            assert result[0]["status"] == "error"


def test_generator_failure_is_actionable(services, monkeypatch):
    services.ingestion.ingest("guide.txt", b"RAG reference")

    def unavailable(prompt):
        raise GenerationError("Gemini unavailable")

    monkeypatch.setattr(services.pipeline.generator, "generate", unavailable)
    with TestClient(create_app(services.settings, services)) as client:
        assert client.post("/chat", json={"question": "RAG?"}).status_code == 503
