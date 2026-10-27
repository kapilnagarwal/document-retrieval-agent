from scripts.run_profile import profile_environment


def test_profile_isolation(monkeypatch):
    monkeypatch.delenv("LOCAL_ALLOW_GEMINI_DOCUMENT_CONTEXT", raising=False)
    learning = profile_environment("learning")
    local = profile_environment("local")
    assert learning["CHROMA_PERSIST_DIRECTORY"] != local["CHROMA_PERSIST_DIRECTORY"]
    assert learning["UPLOAD_DIRECTORY"] != local["UPLOAD_DIRECTORY"]
    assert local["ALLOW_GEMINI_DOCUMENT_CONTEXT"] == "false"
    assert learning["ALLOW_GEMINI_DOCUMENT_CONTEXT"] == "true"
    assert learning["BACKEND_PORT"] != local["BACKEND_PORT"]


def test_local_generation_requires_separate_opt_in(monkeypatch):
    monkeypatch.setenv("ALLOW_GEMINI_DOCUMENT_CONTEXT", "true")
    monkeypatch.delenv("LOCAL_ALLOW_GEMINI_DOCUMENT_CONTEXT", raising=False)
    assert profile_environment("local")["ALLOW_GEMINI_DOCUMENT_CONTEXT"] == "false"
    monkeypatch.setenv("LOCAL_ALLOW_GEMINI_DOCUMENT_CONTEXT", "true")
    assert profile_environment("local")["ALLOW_GEMINI_DOCUMENT_CONTEXT"] == "true"


def test_launcher_preserves_proxy_bypass_hosts(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "generativelanguage.googleapis.com,localhost")
    assert profile_environment("local")["NO_PROXY"].split(",") == [
        "127.0.0.1",
        "localhost",
        "generativelanguage.googleapis.com",
    ]
