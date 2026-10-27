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


def test_prefaced_question_uses_semantic_fallback_only_in_automatic_mode(services):
    document = services.ingestion.ingest(
        "handshake.txt", b"N32-c negotiation rejects invalid security capabilities."
    )
    body = {
        "query": "check what is the handshake status code in failure scenarios",
        "document_ids": [document["document_id"]],
    }
    with TestClient(create_app(services.settings, services)) as client:
        automatic = client.post("/search", json=body).json()
        assert automatic["embedding_dimension"] > 0
        assert automatic["chunks"][0]["document_id"] == document["document_id"]
        assert automatic["chunks"][0]["match_type"] == "semantic"
        strict = client.post("/search", json={**body, "mode": "passages"}).json()
        assert strict["chunks"] == []
        assert strict["embedding_dimension"] == 0
        # Recognizing a question must not remove exact version constraints.
        missing_version = client.post(
            "/search", json={**body, "query": body["query"] + " for version 99.9.9"}
        ).json()
        assert missing_version["chunks"] == []


def test_generator_failure_is_actionable(services, monkeypatch):
    services.ingestion.ingest("guide.txt", b"RAG reference")

    def unavailable(prompt):
        raise GenerationError("Gemini unavailable")

    monkeypatch.setattr(services.pipeline.generator, "generate", unavailable)
    with TestClient(create_app(services.settings, services)) as client:
        assert client.post("/chat", json={"question": "RAG?"}).status_code == 503


def test_chat_scope_filters_evidence_and_never_falls_back_to_all(services):
    note = services.ingestion.ingest("note.txt", b"RAG retrieves note evidence.")
    with TestClient(create_app(services.settings, services)) as client:
        notes = client.post("/chat", json={"question": "What is RAG?", "scope": "notes"}).json()
        assert notes["sources"][0]["document_id"] == note["document_id"]
        pdfs = client.post("/chat", json={"question": "What is RAG?", "scope": "pdf"}).json()
        assert pdfs["retrieved_chunks"] == []
        assert pdfs["answer_status"] == "insufficient_evidence"
        conflict = client.post(
            "/chat",
            json={
                "question": "What is RAG?",
                "scope": "pdf",
                "document_ids": [note["document_id"]],
            },
        ).json()
        assert conflict["retrieved_chunks"] == []


def test_follow_up_retrieves_with_resolved_question_and_passes_history(services, monkeypatch):
    note = services.ingestion.ingest("errors.txt", b"Handshake 403 Forbidden means disabled peer.")
    history = [{"question": "Explain handshake errors", "answer": "403 means disabled peer [S1]"}]

    def rewrite(question, turns):
        assert question == "Explain that one"
        assert turns == history
        return "What does handshake 403 Forbidden mean?", {"total_tokens": 3}

    def generate(prompt):
        assert "Conversation history" in prompt and "Explain that one" in prompt
        assert "Handshake 403 Forbidden means disabled peer" in prompt
        return "403 means disabled peer [S1]", {"total_tokens": 7}

    monkeypatch.setattr(services.pipeline.generator, "rewrite_question", rewrite)
    monkeypatch.setattr(services.pipeline.generator, "generate", generate)
    with TestClient(create_app(services.settings, services)) as client:
        response = client.post(
            "/chat", json={"question": "Explain that one", "history": history, "debug": True}
        )
        assert response.status_code == 200
        result = response.json()
        assert result["answer_status"] == "answered"
        assert result["debug"]["retrieval_query"] == "What does handshake 403 Forbidden mean?"
        assert result["sources"][0]["document_id"] == note["document_id"]
        assert result["token_usage"]["total_tokens"] == 10


def test_follow_up_keeps_exact_constraints_and_empty_scope_skips_model(services, monkeypatch):
    services.ingestion.ingest("errors.txt", b"Handshake error reference for version 1.2.3.")
    history = [{"question": "Explain errors", "answer": "Here are the errors [S1]"}]

    def rewrite(question, turns):
        return "What are the handshake errors?", None

    def fail(prompt):
        raise AssertionError("No evidence: must not call answer generation")

    monkeypatch.setattr(services.pipeline.generator, "rewrite_question", rewrite)
    monkeypatch.setattr(services.pipeline.generator, "generate", fail)
    with TestClient(create_app(services.settings, services)) as client:
        result = client.post(
            "/chat", json={"question": "And for 99.9.9?", "history": history}
        ).json()
        assert result["retrieved_chunks"] == []
        monkeypatch.setattr(services.pipeline.generator, "rewrite_question", fail)
        result = client.post(
            "/chat", json={"question": "Explain those", "history": history, "scope": "pdf"}
        ).json()
        assert result["answer_status"] == "insufficient_evidence"


def test_chat_rejects_unbounded_history(services):
    with TestClient(create_app(services.settings, services)) as client:
        turn = {"question": "Question?", "answer": "Answer"}
        assert (
            client.post("/chat", json={"question": "Next", "history": [turn] * 5}).status_code
            == 422
        )
        assert (
            client.post(
                "/chat", json={"question": "Next", "history": [{**turn, "answer": "x" * 32001}]}
            ).status_code
            == 422
        )
