import pytest
from conftest import TinyModel

from backend.ingestion.embedder import Embedder


def test_embedding_contract():
    service = Embedder("fake", TinyModel())
    assert service.dimension == 8
    assert len(service.documents(["hello", "world"])) == 2
    assert len(service.query("hello")) == 8
    assert service.documents([]) == []
    with pytest.raises(ValueError):
        service.query(" ")


def test_wrong_dimensions():
    class BadModel(TinyModel):
        def get_sentence_embedding_dimension(self):
            return 4

    with pytest.raises(ValueError):
        Embedder("fake", BadModel()).documents(["hello"])
