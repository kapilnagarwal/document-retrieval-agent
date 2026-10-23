from scripts.run_profile import profile_environment


def test_profile_isolation():
    learning = profile_environment("learning")
    local = profile_environment("local")
    assert learning["CHROMA_PERSIST_DIRECTORY"] != local["CHROMA_PERSIST_DIRECTORY"]
    assert learning["UPLOAD_DIRECTORY"] != local["UPLOAD_DIRECTORY"]
    assert local["ALLOW_GEMINI_DOCUMENT_CONTEXT"] == "false"
    assert learning["ALLOW_GEMINI_DOCUMENT_CONTEXT"] == "true"
    assert learning["BACKEND_PORT"] != local["BACKEND_PORT"]
