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
