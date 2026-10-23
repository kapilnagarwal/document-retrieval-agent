import hashlib

import numpy as np
import pytest

from backend.config import Settings
from backend.services import Services


class TinyModel:
    def get_sentence_embedding_dimension(self):
        return 8

    def encode(self, texts, **kwargs):
        # Deterministic fake vectors isolate persistence/API tests from downloads.
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vector = np.array(list(digest[:8]), dtype=float) + 1
            vectors.append(vector / np.linalg.norm(vector))
        return np.array(vectors)


class FakeGenerator:
    def generate(self, prompt):
        return "A grounded example [S1]", {"total_tokens": 12}


@pytest.fixture
def services(tmp_path):
    from backend.ingestion.embedder import Embedder

    config = Settings(
        _env_file=None,
        chroma_persist_directory=tmp_path / "chroma",
        upload_directory=tmp_path / "uploads",
        chunk_size=120,
        chunk_overlap=20,
    )
    return Services(config, Embedder("fake", model=TinyModel()), FakeGenerator())
