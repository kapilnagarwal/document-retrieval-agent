import time
from typing import Protocol

import httpx
from google import genai
from google.genai import errors, types

from backend.rag.prompt import SYSTEM


class GenerationError(Exception):
    pass


class LLMProvider(Protocol):
    def generate(self, prompt: str) -> tuple[str, dict | None]: ...


class GeminiGenerator:
    def __init__(self, settings):
        self.settings = settings

    def generate(self, prompt):
        if not self.settings.gemini_api_key.get_secret_value():
            raise GenerationError("Set GEMINI_API_KEY in the backend .env file and restart.")
        if not self.settings.allow_gemini_document_context:
            raise GenerationError(
                "To send retrieved passages to Google, set ALLOW_GEMINI_DOCUMENT_CONTEXT=true after confirming permission."
            )
        # Disable SDK retries so the three attempts below remain the only retry budget.
        with genai.Client(
            api_key=self.settings.gemini_api_key.get_secret_value(),
            http_options=types.HttpOptions(
                timeout=30000, retry_options=types.HttpRetryOptions(attempts=1)
            ),
        ) as client:
            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model=self.settings.gemini_model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM, temperature=0.2, max_output_tokens=2048
                        ),
                    )
                    if not response.text:
                        raise GenerationError(
                            "Gemini returned no answer, possibly due to content filtering."
                        )
                    usage = response.usage_metadata
                    return response.text, (
                        {
                            "prompt_tokens": usage.prompt_token_count,
                            "output_tokens": usage.candidates_token_count,
                            "total_tokens": usage.total_token_count,
                        }
                        if usage
                        else None
                    )
                except errors.APIError as exc:
                    code = int(exc.code or 0)
                    if code in (401, 403):
                        raise GenerationError(
                            "Gemini authentication or access failed. Check the backend API key and project permissions."
                        ) from None
                    if code == 404:
                        raise GenerationError(
                            "Gemini model unavailable. Run python -m scripts.check_gemini and change GEMINI_MODEL."
                        ) from None
                    if code not in (429, 500, 502, 503, 504):
                        raise GenerationError(
                            "Gemini rejected the request. Check model access and request configuration."
                        ) from None
                    if attempt == 2:
                        message = (
                            "Gemini rate limit or quota exhausted; try later or inspect AI Studio quotas."
                            if code == 429
                            else "Gemini is temporarily unavailable. Try later."
                        )
                        raise GenerationError(message) from None
                except (httpx.TransportError, TimeoutError):
                    if attempt == 2:
                        raise GenerationError(
                            "Network error contacting Gemini. Check connectivity and try again."
                        ) from None
                time.sleep(2**attempt)
        raise GenerationError("Gemini generation failed")
