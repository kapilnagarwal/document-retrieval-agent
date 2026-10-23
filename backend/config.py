from pathlib import Path

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")
    gemini_api_key: SecretStr = SecretStr("")
    data_profile: str = "custom"
    gemini_model: str = "gemini-2.5-flash-lite"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_persist_directory: Path = ROOT / "data/chroma_db"
    upload_directory: Path = ROOT / "data/uploads"
    chunk_size: int = 800
    chunk_overlap: int = 150
    retrieval_top_k: int = 5
    retrieval_distance_threshold: float | None = None
    chat_min_keyword_coverage: float = 0.5
    notes_source_directory: Path = Path.home() / "Downloads/notepadd++ docs"
    max_upload_size_mb: int = 20
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    backend_url: str = "http://127.0.0.1:8000"
    log_level: str = "INFO"
    allow_gemini_document_context: bool = False

    @field_validator("retrieval_distance_threshold", mode="before")
    @classmethod
    def empty_threshold(cls, value):
        return None if value == "" else value

    @field_validator("chroma_persist_directory", "upload_directory")
    @classmethod
    def absolute_paths(cls, value):
        return value if value.is_absolute() else ROOT / value

    @model_validator(mode="after")
    def valid_limits(self):
        if not 0 < self.chat_min_keyword_coverage <= 1:
            raise ValueError("CHAT_MIN_KEYWORD_COVERAGE must be greater than 0 and at most 1")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("Require 0 <= CHUNK_OVERLAP < CHUNK_SIZE")
        if not 1 <= self.retrieval_top_k <= 30 or self.max_upload_size_mb <= 0:
            raise ValueError("Invalid retrieval or upload limits")
        if (
            self.retrieval_distance_threshold is not None
            and not 0 <= self.retrieval_distance_threshold <= 2
        ):
            raise ValueError("Cosine distance threshold must be between 0 and 2")
        return self

    def public(self):
        # Explicit allowlist: never serialize the complete Settings object.
        return {
            key: getattr(self, key)
            for key in (
                "data_profile",
                "gemini_model",
                "embedding_model",
                "chunk_size",
                "chunk_overlap",
                "retrieval_top_k",
                "retrieval_distance_threshold",
                "max_upload_size_mb",
                "allow_gemini_document_context",
                "chat_min_keyword_coverage",
            )
        }
