import pytest

from backend.rag.lexical import contains_literal, required_literals


@pytest.mark.parametrize(
    "text,expected",
    [
        ("podman tag repo/image:1.0.4-3 repo/copy:1.0.4-3", True),
        ("podman load --input image-1.0.4-3.tar", True),
        ("podman tag repo/image:1.0.4-30 repo/copy:1.0.4-30", False),
        ("podman tag repo/image:1.0.4-3.1 repo/copy:1.0.4-3.1", False),
        ("podman tag repo/image:24.2.0 repo/copy:24.2.0", False),
    ],
)
def test_literal_tag_boundaries(text, expected):
    assert contains_literal(text, "1.0.4-3") is expected


def test_extract_command_version_and_quoted_phrase():
    assert required_literals("find podman cmds with 1.0.4-3 tag") == ["1.0.4-3"]
    assert required_literals('find "podman load" with 1.0.4-3 tag') == ["podman load", "1.0.4-3"]


def test_absent_exact_tag_does_not_substitute_other_version(services):
    services.ingestion.ingest(
        "images.txt", b"podman tag localhost/image:24.2.0 registry/image:24.2.0"
    )
    result = services.retriever.search("find podman cmds with 1.0.4-3 tag")
    assert result["chunks"] == []
    assert result["required_terms"] == ["1.0.4-3"]
    assert "not been substituted" in result["message"]
    assert result["embedding_dimension"] == 0


def test_existing_tag_wins_over_similar_wrong_tag(services):
    wrong = services.ingestion.ingest(
        "wrong.txt", b"podman tag image:1.0.4-30 registry/image:1.0.4-30"
    )
    expected = services.ingestion.ingest(
        "correct.txt", b"podman tag image:1.0.4-3 registry/image:1.0.4-3"
    )
    result = services.retriever.search("find podman cmds with 1.0.4-3 tag")
    assert result["chunks"] and all(
        hit["document_id"] == expected["document_id"] for hit in result["chunks"]
    )
    assert all(hit["document_id"] != wrong["document_id"] for hit in result["chunks"])


def test_filter_cannot_leak_exact_match_from_other_document(services):
    allowed = services.ingestion.ingest(
        "allowed.txt", b"podman tag image:24.2.0 registry/image:24.2.0"
    )
    services.ingestion.ingest("excluded.txt", b"podman tag image:1.0.4-3 registry/image:1.0.4-3")
    assert (
        services.retriever.search("podman 1.0.4-3", document_ids=[allowed["document_id"]])["chunks"]
        == []
    )


def test_keyword_match_rescues_misleading_dense_vector(services, monkeypatch):
    expected = services.ingestion.ingest("correct.txt", b"podman load --input target.tar")
    irrelevant = services.ingestion.ingest("irrelevant.txt", b"flower garden trees")
    # Force a deliberately wrong semantic winner. Exact command keywords must rescue the result.
    for record in services.store.retrieval_records():
        record["embedding"] = (
            [1.0, 0.0] if record["document_id"] == irrelevant["document_id"] else [0.0, 1.0]
        )
    monkeypatch.setattr(services.embedder, "query", lambda query: [1.0, 0.0])
    result = services.retriever.search("find podman load commands", top_k=1)
    assert result["chunks"][0]["document_id"] == expected["document_id"]
    assert result["chunks"][0]["lexical_score"] > 0


def test_cached_index_invalidates_after_deletion(services):
    first = services.ingestion.ingest("first.txt", b"podman load image:1.0.4-3")
    assert services.retriever.search("podman 1.0.4-3")["chunks"]
    services.ingestion.delete(first["document_id"])
    assert not services.retriever.search("podman 1.0.4-3")["chunks"]
    services.ingestion.ingest("second.txt", b"podman tag image:1.0.4-3")
    assert services.retriever.search("podman 1.0.4-3")["chunks"]


def test_quoted_command_is_required(services):
    services.ingestion.ingest("wrong.txt", b"podman tag image:24.2.0")
    assert not services.retriever.search('find "podman load"')["chunks"]
