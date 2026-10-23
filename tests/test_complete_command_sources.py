from tests.test_link_retrieval import search


def test_command_lines_survive_chunks_and_exclude_other_versions(services, monkeypatch):
    commands = [
        f"podman {action} " + "long-registry/" * 30 + "service:1.0.4-3"
        for action in ("pull", "tag", "push")
    ]
    services.ingestion.ingest(
        "commands.txt",
        (
            "unrelated surrounding material\n"
            + "\n".join(commands)
            + "\npodman push service:1.0.4-30\npodman build ."
        ).encode(),
    )
    monkeypatch.setattr(
        services.embedder,
        "query",
        lambda query: (_ for _ in ()).throw(AssertionError("No embedding needed")),
    )
    result = search(services, "find podman cmds with 1.0.4-3 tag")
    assert len(result["chunks"]) == 1
    hit = result["chunks"][0]
    assert hit["matched_text"] == "\n".join(commands)
    assert hit["command_count"] == 3 and hit["distance"] is None and hit["line_number"] == 2
    assert search(services, "find podman cmds with 9.9.9-1 tag")["chunks"] == []


def test_subcommand_numbered_prefix_and_shell_continuations(services):
    source = (
        "2.\tpodman build -f Dockerfile "
        + chr(92)
        + "\n  -t example:1.0.4-3 .\npodman tag example:1.0.4-3 dest:1.0.4-3\n# unrelated"
    )
    services.ingestion.ingest("commands.txt", source.encode())
    build = search(services, "find podman build cmd")["chunks"][0]
    assert build["matched_text"] == source.split("\npodman tag")[0]
    tag = search(services, "find podman tag 1.0.4-3 command")["chunks"][0]
    assert tag["command_count"] == 1 and tag["matched_text"].startswith("podman tag")


def test_different_command_lines_cannot_satisfy_combined_terms(services):
    services.ingestion.ingest(
        "split.txt", b"podman build example:1.0.4-30\npodman push example:1.0.4-3"
    )
    assert search(services, "find podman build 1.0.4-3 cmd")["chunks"] == []
