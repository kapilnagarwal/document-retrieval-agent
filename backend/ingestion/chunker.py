from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.ingestion.loader import TextSection


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    page_number: int | None
    text: str


class Chunker:
    def __init__(self, size=800, overlap=150):
        if not 0 <= overlap < size:
            raise ValueError("Overlap must be smaller than chunk size")
        # Small chunks offer focus but lose context. Large chunks dilute retrieval and cost tokens.
        # Overlap preserves sentences near boundaries. Size here is characters, not model tokens.
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)

    def split(self, sections: list[TextSection], document_id: str, filename: str) -> list[Chunk]:
        chunks = []
        # Split each page separately so citations retain a real PDF page number.
        for section in sections:
            for text in self.splitter.split_text(section.text):
                index = len(chunks)
                chunks.append(
                    Chunk(
                        f"{document_id}:{index}",
                        document_id,
                        filename,
                        index,
                        section.page_number,
                        text,
                    )
                )
        return chunks
