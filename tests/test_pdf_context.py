import pymupdf

from backend.models.schemas import ChatRequest
from backend.rag.pdf_context import PDFContext
from backend.rag.source_scope import named_source_scope


def procedure_pdf():
    document = pymupdf.open()
    document.new_page().insert_text((40, 60), "Handshake error codes: wrong PLMNid procedure")
    page = document.new_page()
    columns, rows = [35, 120, 170, 430, 575], [60, 90, 170, 250]
    for x in columns:
        page.draw_line((x, rows[0]), (x, rows[-1]))
    for y in rows:
        page.draw_line((columns[0], y), (columns[-1], y))
    cells = [
        ["Step", "Direction", "Message", "Expected behavior"],
        [
            "2a (If R16)",
            "<--",
            "400 NOK (SecNegotiateRspData)\nMANDATORY_QUERY_PARAM_INCORRECT",
            "Request rejected",
        ],
        [
            "2b (If R17+)",
            "<--",
            "403 NOK (SecNegotiateRspData)\nREQUESTED_PURPOSE_NOT_ALLOWED",
            "Request rejected",
        ],
    ]
    for row, values in enumerate(cells):
        for col, value in enumerate(values):
            page.insert_textbox(
                (columns[col] + 3, rows[row] + 3, columns[col + 1] - 3, rows[row + 1] - 3),
                value,
                fontsize=8,
            )
    content = document.tobytes()
    document.close()
    return content


def test_pdf_answer_recovers_response_page_and_table_rows(services, monkeypatch):
    source = services.ingestion.ingest("NG.146-v2.0-5.pdf", procedure_pdf())
    services.ingestion.ingest("generic.txt", b"403 means disabled peer in another product.")
    seed = next(
        r
        for r in services.store.retrieval_records()
        if r["document_id"] == source["document_id"] and r["page_number"] == 1
    )

    def search(query, top_k, identifiers, **kwargs):
        assert identifiers == [source["document_id"]]
        assert "NG.146" not in query
        return {
            "chunks": [{**seed, "distance": 0.1}],
            "embedding_dimension": 8,
            "retrieval_seconds": 0.01,
        }

    monkeypatch.setattr(services.retriever, "search", search)
    monkeypatch.setattr(
        services.pipeline.generator,
        "generate",
        lambda prompt: (
            "R16: 400 NOK MANDATORY_QUERY_PARAM_INCORRECT [S2]. R17+: 403 NOK REQUESTED_PURPOSE_NOT_ALLOWED [S2].",
            None,
        ),
    )
    monkeypatch.setattr(
        services.pipeline.generator,
        "rewrite_question",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("Standalone source question must not be rewritten")
        ),
    )
    result = services.pipeline.answer(
        ChatRequest(
            question="NG.146 handshake error codes",
            history=[{"question": "Other product errors?", "answer": "Other product behavior."}],
            debug=True,
        )
    )
    assert result["answer_status"] == "answered"
    context = result["debug"]["context"]
    assert "REQUESTED_PURPOSE_NOT_ALLOWED" in context and "R17+" in context
    assert "MANDATORY_QUERY_PARAM_INCORRECT" in context and "R16" in context
    assert "disabled peer" not in context
    assert {s["page_number"] for s in result["sources"]} == {1, 2}
    table = next(c["text"] for c in result["retrieved_chunks"] if c["page_number"] == 2)
    assert "Structured tables:" in table
    row = next(line for line in table.splitlines() if line.startswith("|2b"))
    assert all(
        value in row
        for value in ["R17+", "403 NOK", "REQUESTED_PURPOSE_NOT_ALLOWED", "Request rejected"]
    )
    assert all(c["distance"] is None for c in result["retrieved_chunks"])


def test_pdf_context_budget_preserves_original_hit_without_partial_rows(services):
    source = services.ingestion.ingest("procedure.pdf", procedure_pdf())
    seed = services.store.retrieval_records()[0]
    expanded = PDFContext(services.ingestion, max_characters=1).expand(
        [seed], {source["document_id"]: source}
    )
    assert expanded == [seed]


def test_explicit_source_name_intersects_filter_and_prefers_full_version():
    records = [
        {"document_id": "new", "filename": "NG.146-v2.0-5.pdf"},
        {"document_id": "old", "filename": "NG.146-v1.0.pdf"},
        {"document_id": "guide", "filename": "generic-guide.pdf"},
    ]
    ids, query = named_source_scope(
        "According to NG.146, explain handshake errors", records, ["new", "old", "guide"]
    )
    assert ids == ["new", "old"] and "NG.146" not in query
    ids, _ = named_source_scope("NG.146-v2.0-5.pdf handshake errors", records, ["new", "old"])
    assert ids == ["new"]
    ids, _ = named_source_scope("NG.146 handshake errors", records, ["guide"])
    assert ids == ["__no_selected_imported_documents__"]
