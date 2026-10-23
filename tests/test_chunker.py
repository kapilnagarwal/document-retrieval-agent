import pytest

from backend.ingestion.chunker import Chunker
from backend.ingestion.loader import TextSection


def test_chunk_boundaries_and_metadata():
    chunks = Chunker(80, 15).split([TextSection("hello world " * 50, 7)], "doc", "guide.pdf")
    assert len(chunks) > 2
    assert all(len(c.text) <= 80 and c.page_number == 7 and c.document_id == "doc" for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert chunks == Chunker(80, 15).split(
        [TextSection("hello world " * 50, 7)], "doc", "guide.pdf"
    )


def test_bad_overlap():
    with pytest.raises(ValueError):
        Chunker(10, 10)
