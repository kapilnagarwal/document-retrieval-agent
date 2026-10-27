from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from google.genai import errors
from pydantic import SecretStr

from backend.config import Settings
from backend.rag.generator import GeminiGenerator, GenerationError


def test_missing_key_and_context_consent():
    settings = Settings(_env_file=None, gemini_api_key=SecretStr(""))
    with pytest.raises(GenerationError, match="GEMINI_API_KEY"):
        GeminiGenerator(settings).generate("reference")
    settings.gemini_api_key = SecretStr("fake")
    with pytest.raises(GenerationError, match="ALLOW_GEMINI"):
        GeminiGenerator(settings).generate("reference")


def test_sdk_retry_and_tokens(monkeypatch):
    from backend.rag import generator

    client = MagicMock()
    client.__enter__.return_value = client
    client.models.generate_content.side_effect = [
        errors.ServerError(503, {"error": {"message": "busy"}}),
        SimpleNamespace(
            text="Answer [S1]",
            usage_metadata=SimpleNamespace(
                prompt_token_count=3, candidates_token_count=2, total_token_count=5
            ),
        ),
    ]
    monkeypatch.setattr(generator.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(generator.time, "sleep", lambda seconds: None)
    settings = Settings(
        _env_file=None, gemini_api_key=SecretStr("fake"), allow_gemini_document_context=True
    )
    text, usage = GeminiGenerator(settings).generate("test")
    assert text == "Answer [S1]" and usage["total_tokens"] == 5
    assert client.models.generate_content.call_count == 2


def test_follow_up_rewrite_uses_separate_instruction_and_same_consent(monkeypatch):
    settings = Settings(
        _env_file=None, gemini_api_key=SecretStr("fake"), allow_gemini_document_context=True
    )
    generator = GeminiGenerator(settings)
    calls = []

    def generate(prompt, system_instruction, max_output_tokens=2048):
        calls.append((prompt, system_instruction, max_output_tokens))
        return "What is handshake error 403?", {"total_tokens": 5}

    monkeypatch.setattr(generator, "_generate", generate)
    question, usage = generator.rewrite_question(
        "Explain that", [{"question": "Handshake errors?", "answer": "403 [S1]"}]
    )
    assert question == "What is handshake error 403?"
    assert "Explain that" in calls[0][0] and "Handshake errors?" in calls[0][0]
    assert "Do not answer" in calls[0][1] and calls[0][2] == 512
    assert usage["total_tokens"] == 5
    denied = GeminiGenerator(Settings(_env_file=None, gemini_api_key=SecretStr("fake")))
    with pytest.raises(GenerationError, match="ALLOW_GEMINI"):
        denied.rewrite_question("Explain that", [{"question": "Test", "answer": "Test"}])


@pytest.mark.parametrize(
    "code,match",
    [(403, "authentication"), (404, "model unavailable"), (429, "quota"), (503, "temporarily")],
)
def test_sdk_errors(monkeypatch, code, match):
    from backend.rag import generator

    client = MagicMock()
    client.__enter__.return_value = client
    client.models.generate_content.side_effect = errors.APIError(
        code, {"error": {"message": "failure"}}
    )
    monkeypatch.setattr(generator.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(generator.time, "sleep", lambda seconds: None)
    settings = Settings(
        _env_file=None, gemini_api_key=SecretStr("fake"), allow_gemini_document_context=True
    )
    with pytest.raises(GenerationError, match=match):
        GeminiGenerator(settings).generate("test")
    assert client.models.generate_content.call_count <= 3


def test_real_sdk_stream_events_and_final_usage(monkeypatch):
    from backend.rag import generator

    client = MagicMock()
    client.__enter__.return_value = client
    client.models.generate_content_stream.return_value = iter(
        [
            SimpleNamespace(text="TLS ", usage_metadata=None),
            SimpleNamespace(
                text="encrypts [S1]",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=3, candidates_token_count=2, total_token_count=5
                ),
            ),
        ]
    )
    monkeypatch.setattr(generator.genai, "Client", lambda **kwargs: client)
    settings = Settings(
        _env_file=None, gemini_api_key=SecretStr("fake"), allow_gemini_document_context=True
    )
    events = []
    text, usage = GeminiGenerator(settings).generate_stream(
        "synthetic", lambda kind, payload: events.append((kind, payload))
    )
    assert text == "TLS encrypts [S1]"
    assert usage["total_tokens"] == 5
    assert [e[1]["text"] for e in events] == ["TLS ", "encrypts [S1]"]
    assert not client.models.generate_content.called


def test_provider_wall_clock_deadline_bounds_stalled_dns(monkeypatch):
    import time
    from threading import Event

    settings = Settings(_env_file=None, generation_timeout_seconds=1)
    provider = GeminiGenerator(settings)
    release = Event()
    monkeypatch.setattr(provider, "_generate_request", lambda *args: release.wait(10))
    started = time.monotonic()
    try:
        with pytest.raises(GenerationError, match="deadline"):
            provider.generate("synthetic")
        assert time.monotonic() - started < 3
    finally:
        release.set()
