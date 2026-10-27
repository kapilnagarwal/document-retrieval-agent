import pytest

from backend.models.schemas import ChatRequest
from backend.rag.query_focus import (
    field_content_query,
    field_evidence_score,
    field_question,
    needs_reference_resolution,
)


@pytest.mark.parametrize(
    "entity", ["SecNegotiateRspData", "CreateSessionResponse", "customerProfile"]
)
def test_contents_question_identifies_entity_without_inheriting_prior_topic(entity):
    assert field_question(f"what does {entity} contains?") == [entity]
    assert not needs_reference_resolution(f"what does {entity} contains?")
    assert needs_reference_resolution("Which of those are errors?")
    assert field_question(f"what are {entity} error codes?") == []


def test_fields_rank_above_bare_error_mentions_and_stay_on_entity(services, monkeypatch):
    errors = services.ingestion.ingest("errors.txt", b"SecNegotiateRspData errors 400 and 403.")
    fields = services.ingestion.ingest(
        "fields.txt", b"SecNegotiateRspData\nsender: r-SEPP\nselectedSecCapability: TLS or PRINS"
    )
    services.ingestion.ingest(
        "request.txt", b"SecNegotiateReqData\nsender: i-SEPP\ncapability: TLS"
    )
    for record in services.store.retrieval_records():
        record["embedding"] = (
            [1.0, 0.0] if record["document_id"] == errors["document_id"] else [0.0, 1.0]
        )
    monkeypatch.setattr(services.embedder, "query", lambda query: [1.0, 0.0])
    result = services.retriever.search(
        "what does SecNegotiateRspData contains?", field_entities=["SecNegotiateRspData"]
    )
    assert result["chunks"]
    assert {hit["document_id"] for hit in result["chunks"]} == {fields["document_id"]}


def test_explicit_contents_question_discards_error_history_for_retrieval_and_generation(
    services, monkeypatch
):
    services.ingestion.ingest("errors.txt", b"SecNegotiateRspData error responses 400 and 403.")
    source = services.ingestion.ingest(
        "fields.txt", b"SecNegotiateRspData\nsender: r-SEPP\nselectedSecCapability: TLS or PRINS"
    )

    def rewrite(*args):
        raise AssertionError("Explicit entity question must not inherit an error-only scope")

    def generate(prompt):
        assert "Current task: explain the contents/fields of SecNegotiateRspData" in prompt
        assert "sender: r-SEPP" in prompt
        assert "Conversation history" not in prompt
        assert "400 and 403" not in prompt
        return "Fields: sender and selectedSecCapability [S1]", None

    monkeypatch.setattr(services.pipeline.generator, "rewrite_question", rewrite)
    monkeypatch.setattr(services.pipeline.generator, "generate", generate)
    result = services.pipeline.answer(
        ChatRequest(
            question="what does SecNegotiateRspData contains?",
            history=[{"question": "Handshake error codes?", "answer": "400 and 403"}],
            debug=True,
        )
    )
    assert result["answer_status"] == "answered"
    assert result["debug"]["retrieval_query"] == "what does SecNegotiateRspData contains"
    assert result["sources"][0]["document_id"] == source["document_id"]


def test_request_fields_before_response_mention_are_not_response_definition():
    assert (
        field_evidence_score("sender: i-SEPP\n403 SecNegotiateRspData", ["SecNegotiateRspData"])
        == 0
    )


@pytest.mark.parametrize(
    "question",
    [
        "SecNegotiateReqData give details for this",
        "give me more details about SecNegotiateReqData",
        "describe SecNegotiateReqData in detail",
        "SecNegotiateReqData provide information for this",
    ],
)
def test_entity_detail_request_is_not_mistaken_for_missing_evidence(services, question):
    services.ingestion.ingest(
        "request.txt", b"SecNegotiateReqData\nsender: i-SEPP\nsupportedSecCapability: TLS or PRINS"
    )
    result = services.pipeline.answer(ChatRequest(question=question, debug=True))
    assert result["answer_status"] == "answered"
    assert result["sources"]
    assert field_question(question) == ["SecNegotiateReqData"]


def test_query_cleaning_preserves_technical_qualifiers_and_exact_phrases(services):
    question = 'SecNegotiateReqData give details for this with encryptedToken and "details for this" version 99.9.9'
    cleaned = field_content_query(question)
    assert 'encryptedToken and "details for this" version 99.9.9' in cleaned
    services.ingestion.ingest("request.txt", b"SecNegotiateReqData\nsender: i-SEPP")
    result = services.pipeline.answer(ChatRequest(question=question))
    assert result["answer_status"] == "insufficient_evidence"
