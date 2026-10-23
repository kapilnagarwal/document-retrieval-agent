from backend.rag.retriever import Retriever


def test_exact_match_and_filter(services):
    first = services.ingestion.ingest("first.txt", b"The network encrypts traffic.")
    second = services.ingestion.ingest("second.txt", b"The cake is baked.")
    result = services.retriever.search("The network encrypts traffic.", 2)
    assert result["chunks"][0]["document_id"] == first["document_id"]
    assert abs(result["chunks"][0]["distance"]) < 0.00001
    assert result["retrieval_seconds"] >= 0
    filtered = services.retriever.search(
        "The network encrypts traffic.", 2, [second["document_id"]]
    )
    assert all(c["document_id"] == second["document_id"] for c in filtered["chunks"])
    assert not services.retriever.search("hello", 5, ["missing"])["chunks"]
    strict = Retriever(services.embedder, services.store, threshold=0.000001)
    assert not strict.search("completely different")["chunks"]
