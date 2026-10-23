from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    mode: Literal["auto", "passages", "documents", "links", "semantic"] = "auto"
    scope: Literal["all", "notes", "pdf"] = "all"
    query: str = Field(min_length=1, max_length=8000)
    top_k: int | None = Field(default=None, ge=1, le=30)
    document_ids: list[str] = Field(default_factory=list, max_length=200)

    @field_validator("query")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Query cannot be blank")
        return value.strip()


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    top_k: int | None = Field(default=None, ge=1, le=30)
    document_ids: list[str] = Field(default_factory=list, max_length=200)
    debug: bool = False

    @field_validator("question")
    @classmethod
    def nonblank(cls, value):
        return SearchRequest.nonblank(value)


class DocumentInfo(BaseModel):
    document_id: str
    source_id: str | None = None
    version: int = 1
    supersedes: str | None = None
    source_name: str | None = None
    filename: str
    sha256: str
    size_bytes: int
    character_count: int
    chunk_count: int
    duration_seconds: float
    status: str
    created_at: str


class IngestionResult(BaseModel):
    filename: str
    status: str
    document_id: str | None = None
    source_id: str | None = None
    version: int = 1
    supersedes: str | None = None
    size_bytes: int = 0
    character_count: int = 0
    chunk_count: int = 0
    duration_seconds: float = 0
    error: str | None = None


class ChunkHit(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    page_number: int | None = None
    text: str
    distance: float | None = None
    line_number: int | None = None
    command_count: int | None = None
    source_name: str | None = None
    match_type: str = "semantic"
    matched_phrase: str | None = None
    matched_text: str | None = None
    lexical_score: float = 0
    hybrid_score: float = 0


class SearchResponse(BaseModel):
    total_matches: int | None = None
    result_type: Literal["passages"] = "passages"
    plan: dict | None = None
    query: str
    embedding_dimension: int
    chunks: list[ChunkHit]
    retrieval_seconds: float
    required_terms: list[str] = Field(default_factory=list)
    retrieval_method: str = "semantic"
    message: str | None = None


class Source(BaseModel):
    source_id: str
    chunk_id: str
    filename: str
    page_number: int | None = None
    cited: bool
    document_id: str | None = None
    source_document_id: str | None = None
    version: int = 1
    sha256: str | None = None


class ChatResponse(BaseModel):
    question: str
    answer: str
    answer_status: Literal["answered", "insufficient_evidence", "unverified"] = "answered"
    evidence: dict = Field(default_factory=dict)
    sources: list[Source]
    retrieved_chunks: list[ChunkHit]
    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float
    token_usage: dict[str, int | None] | None = None
    warnings: list[str] = Field(default_factory=list)
    debug: dict | None = None


class DocumentContent(BaseModel):
    document_id: str
    filename: str
    source_name: str | None = None
    content: str
    character_count: int
    content_format: Literal["original_text", "extracted_text"]


class DocumentMatch(BaseModel):
    document_id: str
    filename: str
    source_name: str | None = None
    character_count: int
    match_count: int


class DocumentSearchResponse(BaseModel):
    plan: dict | None = None
    result_type: Literal["documents"] = "documents"
    query: str
    search_phrase: str
    documents: list[DocumentMatch]
    total_matches: int
    retrieval_seconds: float
    message: str


class LinkEvidence(BaseModel):
    document_id: str
    filename: str
    source_name: str | None = None
    line_number: int
    text: str


class SavedLink(BaseModel):
    url: str
    title: str
    match_reason: str
    sources: list[LinkEvidence]


class LinkSearchResponse(BaseModel):
    result_type: Literal["links"] = "links"
    query: str
    plan: dict
    links: list[SavedLink]
    total_matches: int
    retrieval_seconds: float
    message: str
