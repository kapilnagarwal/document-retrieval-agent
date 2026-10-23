import json

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app

QUERY = "find confluence pages related to cncc"


def search(services, query=QUERY, **kwargs):
    with TestClient(create_app(services.settings, services)) as client:
        response = client.post("/search", json={"query": query, **kwargs})
        assert response.status_code == 200, response.text
        return response.json()


def test_links_are_actual_source_urls_not_version_snippets(services, monkeypatch):
    url = "https://confluence.example/display/SPACE/Supported+CNCC+Operations"
    source = services.ingestion.ingest("links.txt", url.encode())
    services.ingestion.ingest("versions.txt", b"cncc 24.2.0-beta.40 -> beta.67")
    monkeypatch.setattr(
        services.embedder,
        "query",
        lambda query: pytest.fail("Reference lookup must not invoke embeddings or Gemini"),
    )
    result = search(services)
    assert result["result_type"] == "links" and result["total_matches"] == 1
    assert result["links"][0]["url"] == url
    assert result["links"][0]["sources"][0]["document_id"] == source["document_id"]
    assert "versions.txt" not in str(result)


def test_aliases_encoded_titles_and_nearby_topic_evidence(services):
    urls = [
        "https://confluence.example/display/SPACE/CNCConsole+Guide",
        "https://confluence.example/display/SPACE/%43%4E%43%43+Operations",
        "https://confluence.example/pages/viewpage.action?pageId=12",
    ]
    services.ingestion.ingest(
        "links.txt", ("\n\n".join(urls[:2]) + "\n\nCNCC setup notes:\n" + urls[2]).encode()
    )
    result = search(services)
    assert {link["url"] for link in result["links"]} == set(urls)
    assert {
        link["url"] for link in search(services, "show CNCConsole Confluence URLs")["links"]
    } == set(urls)


def test_faraway_topic_does_not_attribute_unrelated_link(services):
    services.ingestion.ingest(
        "unrelated.txt",
        b"CNCC release notes\n\nNRF reference\nhttps://confluence.example/display/SPACE/NRF+Guide",
    )
    result = search(services)
    assert result["links"] == []
    assert "No saved link" in result["message"]


def test_deduplicates_links_and_keeps_all_provenance(services):
    url = "https://confluence.example/display/SPACE/CNCC+Guide"
    services.ingestion.ingest("first.txt", ("first\n" + url).encode())
    services.ingestion.ingest("second.txt", ("second\n" + url).encode())
    result = search(services)
    assert result["total_matches"] == 1
    assert len(result["links"][0]["sources"]) == 2
    assert all(source["line_number"] == 2 for source in result["links"][0]["sources"])


def test_full_source_links_survive_chunk_boundaries_and_scope(services):
    url = "https://confluence.example/display/SPACE/" + "LongTitle" * 40 + "+CNCC"
    allowed = services.ingestion.ingest("allowed.txt", url.encode())
    other = services.ingestion.ingest("other.txt", b"https://confluence.example/display/SPACE/NRF")
    assert search(services, document_ids=[allowed["document_id"]])["links"][0]["url"] == url
    assert search(services, document_ids=[other["document_id"]])["links"] == []
    assert search(services, document_ids=["missing"])["links"] == []


def test_provider_filter_malformed_urls_and_no_invented_url(services):
    services.ingestion.ingest(
        "notes.txt",
        b"CNCC\nhttp://[broken\nhttps://jira.example/browse/CNCC-12\nhttps://user:password@confluence.example/display/CNCC",
    )
    assert search(services)["links"] == []
    result = search(services, "find jira links about cncc")
    assert [link["url"] for link in result["links"]] == ["https://jira.example/browse/CNCC-12"]


def test_explicit_all_returns_more_than_top_k_and_catalog_updates(services):
    source = services.ingestion.ingest(
        "links.txt",
        (
            "\n".join(
                "https://confluence.example/display/SPACE/CNCC+Guide" + str(i) for i in range(8)
            )
        ).encode(),
    )
    result = search(services, "list all confluence links about cncc", top_k=2)
    assert len(result["links"]) == result["total_matches"] == 8
    services.ingestion.delete(source["document_id"])
    assert search(services)["links"] == []
    services.ingestion.ingest("new.txt", b"https://confluence.example/display/SPACE/CNCC+New")
    assert search(services)["total_matches"] == 1


def test_filename_lookup_and_original_source_name(services):
    record = services.ingestion.ingest(
        "local-note-opaque.txt", b"Complete note with podman build command"
    )
    (services.settings.upload_directory / "notes_filename_mapping.json").write_text(
        json.dumps({record["document_id"]: "new 30.txt"})
    )
    result = search(services, "find file new 30.txt")
    assert result["result_type"] == "documents"
    assert [document["document_id"] for document in result["documents"]] == [record["document_id"]]


def test_strict_lookup_does_not_silently_return_semantic_noise(services):
    services.ingestion.ingest("other.txt", b"flowers grow in a garden")
    result = search(services, "find notes related to quantum encryption")
    assert result["chunks"] == [] and result["total_matches"] == 0
    approximate = search(services, "quantum encryption", mode="semantic")
    assert approximate["chunks"] and "semantic suggestions" in approximate["message"]


def test_scope_restriction_applies_to_links(services):
    services.ingestion.ingest("private-note.txt", b"https://confluence.example/display/SPACE/CNCC")
    assert search(services, scope="notes")["links"]
    assert search(services, scope="pdf")["links"] == []
