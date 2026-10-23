from functools import lru_cache
from threading import Lock

import numpy as np


@lru_cache(maxsize=4)
def load_model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device="cpu", trust_remote_code=False)


class Embedder:
    def __init__(self, model_name, model=None):
        self.model_name = model_name
        self._model = model
        self._lock = Lock()

    @property
    def model(self):
        # Serialize the first load, then reuse the same CPU model for this service.
        with self._lock:
            if self._model is None:
                self._model = load_model(self.model_name)
        return self._model

    @property
    def dimension(self):
        dimension_method = getattr(self.model, "get_embedding_dimension", None)
        return (
            dimension_method()
            if dimension_method
            else self.model.get_sentence_embedding_dimension()
        )

    def documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("Cannot embed blank text")
        vectors = np.asarray(
            self.model.encode(
                texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
            )
        )
        if vectors.shape != (len(texts), self.dimension) or not np.isfinite(vectors).all():
            raise ValueError("Unexpected embedding dimensions or non-finite vectors")
        return vectors.tolist()

    def query(self, text: str) -> list[float]:
        return self.documents([text])[0]
