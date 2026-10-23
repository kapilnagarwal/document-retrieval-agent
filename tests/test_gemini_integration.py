import os

import pytest

from backend.config import Settings
from backend.rag.generator import GeminiGenerator


@pytest.mark.integration
def test_optional_live_gemini():
    settings = Settings()
    if os.getenv("RUN_GEMINI_INTEGRATION") != "1" or not settings.gemini_api_key.get_secret_value():
        pytest.skip("Requires RUN_GEMINI_INTEGRATION=1 and GEMINI_API_KEY")
    if not settings.allow_gemini_document_context:
        pytest.skip("Requires explicit Gemini context opt-in")
    # Synthetic reference only; never send the SEPP guides in this test.
    answer, _ = GeminiGenerator(settings).generate(
        "Reference [S1]: The test color is blue. Question: What is the test color?"
    )
    assert "blue" in answer.lower()
