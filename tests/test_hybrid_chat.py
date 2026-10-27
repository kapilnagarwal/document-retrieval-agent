import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from google.genai import types
from pydantic import SecretStr

from backend.config import Settings
from backend.main import create_app
from backend.models.schemas import ChatRequest
from backend.rag.chat_routing import route_turn
from backend.rag.generator import GeminiGenerator
from backend.rag.web_grounding import ground_answer


def no_retrieval(*args, **kwargs):
    raise AssertionError("This request does not need document retrieval")


def test_online_comparison_routes_to_web_with_context_and_no_rag(services, monkeypatch):
    monkeypatch.setattr(services.retriever, "search", no_retrieval)
    calls = []

    def web(prompt, emit=None):
        calls.append(json.loads(prompt))
        return (
            "The specification differs [W1](https://example.org/spec).",
            {"total_tokens": 5},
            [{"title": "Official spec", "url": "https://example.org/spec"}],
            True,
            None,
        )

    monkeypatch.setattr(services.pipeline.generator, "web_chat", web, raising=False)
    result = services.pipeline.answer(
        ChatRequest(
            answer_mode="auto",
            question="compare it with public sepp specs available online",
            history=[{"question": "Handshake codes?", "answer": "400 and 403 [S1]"}],
        )
    )
    assert result["answer_origin"] == "web"
    assert result["sources"] == [] and result["retrieved_chunks"] == []
    assert result["web_sources"][0]["title"] == "Official spec"
    assert calls[0]["conversation"][0]["answer"] == "400 and 403 "
    assert result["answer_status"] == "answered"


def test_reformatting_uses_conversation_without_citation_warnings(services, monkeypatch):
    monkeypatch.setattr(services.retriever, "search", no_retrieval)

    def chat(prompt, emit=None):
        assert "400" in prompt and "Make it shorter" in prompt
        return "400 and 403. [S1]", None

    monkeypatch.setattr(services.pipeline.generator, "chat", chat, raising=False)
    result = services.pipeline.answer(
        ChatRequest(
            answer_mode="auto",
            question="Make it shorter",
            history=[{"question": "Handshake codes?", "answer": "400 and 403 [S1]"}],
        )
    )
    assert result["answer_origin"] == "conversation"
    assert result["warnings"] == []
    assert "[S1]" not in result["answer"]
    assert result["sources"] == []


def test_auto_general_fallback_does_not_change_strict_document_mode(services, monkeypatch):
    monkeypatch.setattr(
        services.pipeline.generator,
        "chat",
        lambda *a, **k: ("A general explanation.", None),
        raising=False,
    )
    automatic = services.pipeline.answer(
        ChatRequest(answer_mode="auto", question="What is a binary search algorithm?")
    )
    assert automatic["answer_origin"] == "general"
    strict = services.pipeline.answer(
        ChatRequest(answer_mode="documents", question="What is a binary search algorithm?")
    )
    assert strict["answer_status"] == "insufficient_evidence"
    assert "answer_origin" not in strict  # Response schema supplies the documents default.
    explicit = services.pipeline.answer(
        ChatRequest(
            answer_mode="auto",
            question="According to my uploaded documents, what is a binary search algorithm?",
        )
    )
    assert explicit["answer_status"] == "insufficient_evidence"


def test_web_without_grounding_cannot_masquerade_as_verified_comparison(services, monkeypatch):
    monkeypatch.setattr(
        services.pipeline.generator,
        "web_chat",
        lambda *a, **k: ("I searched online and proved everything.", None, [], False, None),
        raising=False,
    )
    result = services.pipeline.answer(
        ChatRequest(answer_mode="web", question="Compare with online standards")
    )
    assert result["answer_status"] == "unverified"
    assert "proved everything" not in result["answer"]
    assert "could not obtain verifiable" in result["answer"]
    assert result["warnings"]


def test_general_and_web_turns_persist_and_stream_with_same_conversation(services, monkeypatch):
    seen = []

    def chat(prompt, emit=None):
        seen.append(json.loads(prompt))
        if emit:
            emit("delta", {"text": "General answer."})
        return "General answer.", None

    monkeypatch.setattr(services.pipeline.generator, "chat", chat, raising=False)
    monkeypatch.setattr(services.retriever, "search", no_retrieval)
    with TestClient(create_app(services.settings, services)) as client:
        identifier = client.post("/conversations", json={"title": "General chat"}).json()["id"]
        for number, question in enumerate(["Write a short poem", "Make it shorter"]):
            response = client.post(
                "/chat/stream",
                json={
                    "conversation_id": identifier,
                    "request_id": str(number),
                    "answer_mode": "auto",
                    "question": question,
                },
            )
            events = [json.loads(line) for line in response.text.splitlines()]
            assert events[-1]["event"] == "final"
            assert events[-1]["data"]["answer_origin"] == (
                "general" if number == 0 else "conversation"
            )
        saved = client.get("/conversations/" + identifier).json()
        assert len(saved["turns"]) == 2
        assert saved["turns"][1]["response"]["answer_origin"] == "conversation"
        assert seen[1]["conversation"][0]["answer"] == "General answer."


def test_sdk_web_grounding_uses_search_tool_and_metadata(monkeypatch):
    from backend.rag import generator

    client = MagicMock()
    client.__enter__.return_value = client
    metadata = types.GroundingMetadata(
        grounding_chunks=[
            types.GroundingChunk(web={"uri": "https://example.org/spec", "title": "Official spec"})
        ],
        grounding_supports=[
            types.GroundingSupport(
                segment={"start_index": 0, "end_index": 12, "text": "Public fact."},
                grounding_chunk_indices=[0],
            )
        ],
        search_entry_point={"rendered_content": "<div>Search suggestions</div>"},
    )
    client.models.generate_content.return_value = SimpleNamespace(
        text="Public fact.",
        usage_metadata=None,
        candidates=[SimpleNamespace(finish_reason="STOP", grounding_metadata=metadata)],
    )
    monkeypatch.setattr(generator.genai, "Client", lambda **k: client)
    provider = GeminiGenerator(
        Settings(
            _env_file=None, gemini_api_key=SecretStr("fake"), allow_gemini_document_context=True
        )
    )
    text, usage, sources, cited, suggestions = provider.web_chat("Public topic only")
    config = client.models.generate_content.call_args.kwargs["config"]
    assert config.tools[0].google_search is not None
    assert sources[0]["url"] == "https://example.org/spec"
    assert "[W1]" in text and cited
    assert suggestions == "<div>Search suggestions</div>"


def test_grounding_rejects_unsafe_urls_or_invalid_segment_offsets():
    text, sources, cited = ground_answer(
        "Example",
        {
            "grounding_chunks": [{"web": {"uri": "javascript:alert(1)"}}],
            "grounding_supports": [{"segment": {"end_index": 7}, "grounding_chunk_indices": [0]}],
        },
    )
    assert sources == [] and not cited
    text, sources, cited = ground_answer(
        "Example",
        {
            "grounding_chunks": [{"web": {"uri": "https://example.org"}}],
            "grounding_supports": [
                {
                    "segment": {"start_index": 0, "end_index": 7, "text": "Invented"},
                    "grounding_chunk_indices": [0],
                }
            ],
        },
    )
    assert sources and not cited and text == "Example"


def test_document_mode_remains_default_for_api_and_web_is_explicit():
    assert ChatRequest(question="hello").answer_mode == "documents"
    assert (
        route_turn(
            ChatRequest(
                question="Summarize that",
                answer_mode="auto",
                history=[{"question": "q", "answer": "a"}],
            )
        )
        == "conversation"
    )
    assert (
        route_turn(
            ChatRequest(
                question="compare it with public sepp specs available online", answer_mode="auto"
            )
        )
        == "web"
    )
    assert (
        route_turn(ChatRequest(question="compare it online", answer_mode="documents"))
        == "documents"
    )


def test_web_comparison_retains_the_end_of_a_long_previous_answer(services, monkeypatch):
    long_answer = "Earlier table rows. " * 400 + "Final row: 403 REQUESTED_PURPOSE_NOT_ALLOWED"
    monkeypatch.setattr(
        services.pipeline.generator, "chat", lambda *a, **k: (long_answer, None), raising=False
    )

    def web(prompt, emit=None):
        assert "Final row: 403 REQUESTED_PURPOSE_NOT_ALLOWED" in prompt
        return (
            "Comparison [W1](https://example.org/spec)",
            None,
            [{"url": "https://example.org/spec", "title": "Spec"}],
            True,
            None,
        )

    monkeypatch.setattr(services.pipeline.generator, "web_chat", web, raising=False)
    with TestClient(create_app(services.settings, services)) as client:
        identifier = client.post("/conversations", json={"title": "Long table"}).json()["id"]
        client.post(
            "/chat",
            json={
                "conversation_id": identifier,
                "question": "Write a detailed table",
                "answer_mode": "general",
            },
        ).raise_for_status()
        result = client.post(
            "/chat",
            json={
                "conversation_id": identifier,
                "question": "Compare it online",
                "answer_mode": "auto",
            },
        )
        assert result.status_code == 200
        assert result.json()["answer_origin"] == "web"


def test_routing_respects_web_negation_and_document_version_questions():
    assert (
        route_turn(
            ChatRequest(question="Explain TLS without using the internet", answer_mode="auto")
        )
        != "web"
    )
    assert (
        route_turn(
            ChatRequest(
                question="According to NG.146, what is the latest version listed?",
                answer_mode="auto",
            )
        )
        == "documents"
    )
