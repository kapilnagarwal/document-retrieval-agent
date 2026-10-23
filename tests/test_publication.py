"""Publication checks use constructed synthetic examples, never real credentials."""

from scripts.check_publication import private_path, scan_text


def test_private_artifacts_and_allowed_template():
    assert private_path(".env")
    assert private_path("nested/.env.production")
    assert private_path("data/private-note.txt")
    assert private_path(".ssh/id_rsa")
    assert private_path("docs/python-for-java/README.md")
    assert not private_path(".env.example")
    assert not private_path("backend/config.py")


def test_keys_paths_and_real_emails_are_blocked_without_echoing_values():
    assert "Google API key" in scan_text("AIza" + "A" * 35)
    assert "personal home path" in scan_text("/Users/" + "example/private.txt")
    assert "personal or non-placeholder email" in scan_text("person" + "@gmail.com")
    assert "URL with embedded credentials" in scan_text(
        "https://" + "name:secret@real-host.invalid/path"
    )


def test_synthetic_fixtures_and_noreply_metadata_are_allowed():
    assert not scan_text("git@github.com:owner/repository.git")
    assert not scan_text("https://user:password@confluence.example/display/TEST")
    assert not scan_text("contributor@users.noreply.github.com", metadata=True)
    assert scan_text("person@example.com", metadata=True)
