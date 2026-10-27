import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from backend.config import Settings
from backend.models.schemas import ChatRequest
from backend.rag.generator import GeminiGenerator, GenerationError
from backend.rag.query_focus import correction_target, error_code_focus
from backend.rag.verification import missing_error_codes, parse_audit


def test_broad_code_question_collects_http_and_tls_and_repairs_omissions(services, monkeypatch):
    services.ingestion.ingest(
        "NG.146-example.txt",
        b"Handshake error examples. TLS handshake alert cause bad_certificate.\n\nHandshake response 400 NOK R16 MANDATORY_QUERY_PARAM_INCORRECT.\nHandshake response 403 NOK R17+ REQUESTED_PURPOSE_NOT_ALLOWED.",
    )
    calls = []

    def generate(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return "TLS alert bad_certificate [S1]", None
        return (
            "400 NOK R16 MANDATORY_QUERY_PARAM_INCORRECT; 403 NOK R17+ REQUESTED_PURPOSE_NOT_ALLOWED; TLS alert bad_certificate [S1][S2][S3]",
            None,
        )

    monkeypatch.setattr(services.pipeline.generator, "generate", generate)
    result = services.pipeline.answer(
        ChatRequest(question="What are the handshake error codes based on NG.146?")
    )
    context = "\n".join(c["text"] for c in result["retrieved_chunks"])
    assert "400 NOK" in context and "403 NOK" in context and "bad_certificate" in context
    assert "concise table" in calls[0]
    assert len(calls) == 2
    assert "Required error codes omitted" in calls[1]
    assert "400" in result["answer"] and "403" in result["answer"]


def test_correction_rechecks_original_question_and_retains_named_source(services, monkeypatch):
    target = services.ingestion.ingest(
        "NG.146-example.txt",
        b"Handshake response 400 NOK R16 MANDATORY_QUERY_PARAM_INCORRECT. Handshake response 403 NOK R17+ REQUESTED_PURPOSE_NOT_ALLOWED.",
    )
    services.ingestion.ingest(
        "unrelated.txt", b"Handshake HTTP response 500 Internal Server Error in another product."
    )

    def rewrite(*args):
        raise AssertionError("Correction must not be searched as a standalone complaint")

    def generate(prompt):
        assert "Re-check the original question" in prompt
        assert "400 NOK" in prompt and "403 NOK" in prompt
        assert "500 Internal" not in prompt
        return "400 and 403 [S1]", None

    monkeypatch.setattr(services.pipeline.generator, "rewrite_question", rewrite)
    monkeypatch.setattr(services.pipeline.generator, "generate", generate)
    result = services.pipeline.answer(
        ChatRequest(
            question="this is not the correct info",
            history=[
                {
                    "question": "What are the handshake error codes based on NG.146?",
                    "answer": "TLS only",
                },
                {"question": "What are the handshake error codes?", "answer": "TLS only"},
            ],
        )
    )
    assert result["question"] == "this is not the correct info"
    assert {s["document_id"] for s in result["sources"]} == {target["document_id"]}
    assert "400 and 403" in result["answer"]


def test_focus_and_quote_normalization_do_not_merge_different_codes():
    assert error_code_focus("what are handshake error codes") == "both"
    assert error_code_focus("TLS handshake alert codes") == "tls"
    assert error_code_focus("HTTP handshake error codes") == "http"
    assert (
        correction_target("this is incorrect", [{"question": "original", "answer": "draft"}])
        == "original"
    )
    chunks = [{"text": "R16 400 NOK\nMANDATORY_QUERY_PARAM_INCORRECT"}]
    assert not parse_audit(
        json.dumps(
            {
                "supported": True,
                "checks": [
                    {"source": "S1", "quote": "R16 400 NOK MANDATORY_QUERY_PARAM_INCORRECT"}
                ],
            }
        ),
        chunks,
    )
    assert parse_audit(
        json.dumps(
            {
                "supported": True,
                "checks": [
                    {"source": "S1", "quote": "R16 403 NOK MANDATORY_QUERY_PARAM_INCORRECT"}
                ],
            }
        ),
        chunks,
    )
    assert missing_error_codes("handshake error codes", "TLS only", chunks, "both") == ["400"]
    assert not missing_error_codes("Explain error code 403", "403", chunks, "both")


def test_truncated_provider_output_is_never_returned_as_complete(monkeypatch):
    from backend.rag import generator

    client = MagicMock()
    client.__enter__.return_value = client
    client.models.generate_content.return_value = SimpleNamespace(
        text="An incomplete sentence",
        usage_metadata=None,
        candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
    )
    monkeypatch.setattr(generator.genai, "Client", lambda **kwargs: client)
    provider = GeminiGenerator(
        Settings(
            _env_file=None, gemini_api_key=SecretStr("fake"), allow_gemini_document_context=True
        )
    )
    with pytest.raises(GenerationError, match="output limit"):
        provider.generate("synthetic")
