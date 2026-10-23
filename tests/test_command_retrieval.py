import json

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.rag.lexical import matching_excerpt


@pytest.mark.parametrize(
    "command",
    [
        "podman build",
        "podman load",
        "podman tag",
        "helm list",
        "helm install",
        "kubectl create secret",
        "git cherry-pick",
    ],
)
def test_exact_command_is_not_outranked_or_padded_by_partial_matches(
    services, monkeypatch, command
):
    target = services.ingestion.ingest("target.txt", (command + " --example target").encode())
    for index in range(12):
        services.ingestion.ingest(
            f"noise-{index}.txt",
            (command.split()[0] + f" unrelated-command example {index}").encode(),
        )
    # Make every partial semantic match closer than the exact command to reproduce the original rank-fusion failure.
    for record in services.store.retrieval_records():
        record["embedding"] = (
            [0.0, 1.0] if record["document_id"] == target["document_id"] else [1.0, 0.0]
        )
    monkeypatch.setattr(services.embedder, "query", lambda query: [1.0, 0.0])
    result = services.retriever.search("find " + command + " cmd", top_k=5)
    assert len(result["chunks"]) == 1
    assert result["chunks"][0]["document_id"] == target["document_id"]
    assert result["chunks"][0]["match_type"] == "exact_phrase"
    assert command in result["chunks"][0]["matched_text"]


def test_build_does_not_match_builder_and_excerpt_excludes_unrelated_commands(services):
    expected = services.ingestion.ingest(
        "build.txt", b"kubectl cp before\npodman build --tag example .\npodman push example"
    )
    services.ingestion.ingest("builder.txt", b"podman builder is a different token")
    result = services.retriever.search("find podman build command")
    assert len(result["chunks"]) == 1
    assert result["chunks"][0]["document_id"] == expected["document_id"]
    assert result["chunks"][0]["matched_text"] == "podman build --tag example ."
    assert "kubectl cp before" in result["chunks"][0]["text"]


def test_matching_excerpt_preserves_multiline_shell_command():
    source = "before\npodman build " + chr(92) + "\n  --tag example .\nafter"
    assert (
        matching_excerpt(source, "podman build")
        == "podman build " + chr(92) + "\n  --tag example ."
    )


def test_original_source_name_is_returned_by_document_and_search_apis(services):
    record = services.ingestion.ingest("local-note-hash.txt", b"podman build --tag example .")
    path = services.settings.upload_directory / "notes_filename_mapping.json"
    path.write_text(json.dumps({record["document_id"]: "new 30.txt"}))
    with TestClient(create_app(services.settings, services)) as client:
        assert client.get("/documents").json()[0]["source_name"] == "new 30.txt"
        result = client.post("/search", json={"query": "find podman build cmd"}).json()
        assert result["chunks"][0]["source_name"] == "new 30.txt"
        assert result["chunks"][0]["matched_text"] == "podman build --tag example ."


def test_no_exact_keyword_match_is_labelled_as_semantic_suggestion(services):
    services.ingestion.ingest("other.txt", b"A flower grows in a garden")
    result = services.retriever.search("network encryption")
    assert result["chunks"][0]["match_type"] == "semantic"
    assert "semantic suggestions" in result["message"]
