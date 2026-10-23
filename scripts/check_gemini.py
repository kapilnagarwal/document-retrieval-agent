"""Lists accessible Gemini models; does not send any document text."""

from google import genai

from backend.config import Settings


def main():
    settings = Settings()
    if not settings.gemini_api_key.get_secret_value():
        raise SystemExit("Set GEMINI_API_KEY in .env first; do not paste it into chat.")
    with genai.Client(api_key=settings.gemini_api_key.get_secret_value()) as client:
        for model in client.models.list():
            if "generateContent" in (model.supported_actions or []):
                print(model.name)


if __name__ == "__main__":
    main()
