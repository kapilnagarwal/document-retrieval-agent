"""Regression checks for choosing the correct corpus in the main UI."""

from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "frontend/app.py"


def fake_api(monkeypatch, note_name="local-note-test.txt"):
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        private = ":8001/" in url
        if url.endswith("/health"):
            body = {"status": "ok", "documents": 1, "chunks": 1, "vector_database": "ready"}
        elif url.endswith("/config"):
            body = {
                "data_profile": "local" if private else "learning",
                "retrieval_top_k": 5,
                "allow_gemini_document_context": not private,
                "max_upload_size_mb": 20,
                "embedding_model": "test-model",
                "gemini_model": "test-gemini",
            }
        elif url.endswith("/documents"):
            body = [
                {
                    "document_id": "private-note" if private else "sample",
                    "filename": note_name if private else "sample.txt",
                    "chunk_count": 1,
                }
            ]
        elif url.endswith("/search"):
            body = {
                "query": kwargs["json"]["query"],
                "embedding_dimension": 384,
                "retrieval_seconds": 0.01,
                "chunks": [
                    {
                        "document_id": "private-note",
                        "filename": "local-note-test.txt",
                        "chunk_index": 0,
                        "distance": 0.2,
                        "text": "podman load --input sample.tar",
                    }
                ],
            }
        else:
            raise AssertionError("Unexpected endpoint: " + url)
        return httpx.Response(200, json=body, request=httpx.Request(method, url))

    monkeypatch.setattr(httpx, "request", request)
    return calls


def test_main_ui_defaults_to_private_notes_and_searches_locally(monkeypatch):
    calls = fake_api(monkeypatch)
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert app.radio(key="document_collection_v2").value == "My notes and SEPP (local only)"
    assert all(":8001/" in url for _, url, _ in calls)
    app.text_input[0].set_value("does it contain podman load command?")
    app.button[0].click().run()
    assert not app.exception
    searches = [call for call in calls if call[1].endswith("/search")]
    assert searches[-1][2]["json"]["document_ids"] == []
    assert searches[-1][2]["json"]["scope"] == "notes"
    assert not any(url.endswith("/chat") for _, url, _ in calls)
    assert any("podman load" in item.value for item in app.text)


def test_learning_collection_requires_explicit_selection(monkeypatch):
    calls = fake_api(monkeypatch)
    app = AppTest.from_file(str(APP)).run()
    calls.clear()
    app.radio(key="document_collection_v2").set_value("Learning samples (Gemini chat)").run()
    assert not app.exception
    assert calls and all(":8000/" in url for _, url, _ in calls)


def test_uploaded_text_note_is_included_without_special_filename(monkeypatch):
    calls = fake_api(monkeypatch, note_name="my-working-notes.txt")
    app = AppTest.from_file(str(APP)).run()
    app.text_input[0].set_value("podman load")
    app.button[0].click().run()
    searches = [call for call in calls if call[1].endswith("/search")]
    assert not app.exception
    assert searches[-1][2]["json"]["document_ids"] == []
    assert searches[-1][2]["json"]["scope"] == "notes"


def test_imported_note_invalidates_stale_search_results(monkeypatch):
    fake_api(monkeypatch)
    original = httpx.request
    changed = {"value": False}

    def request(method, url, **kwargs):
        result = original(method, url, **kwargs)
        if changed["value"] and url.endswith("/documents"):
            body = result.json() + [
                {"document_id": "new-import", "filename": "new-note.txt", "chunk_count": 2}
            ]
            return httpx.Response(200, json=body, request=httpx.Request(method, url))
        return result

    monkeypatch.setattr(httpx, "request", request)
    app = AppTest.from_file(str(APP)).run()
    app.text_input[0].set_value("podman load")
    app.button[0].click().run()
    assert any("podman load --input sample.tar" in item.value for item in app.text)
    changed["value"] = True
    app.run()
    assert not app.exception
    assert any("Indexed documents changed" in item.value for item in app.info)
    assert not any("podman load --input sample.tar" in item.value for item in app.text)


def test_whole_note_request_displays_only_complete_source(monkeypatch):
    fake_api(monkeypatch)
    original = httpx.request
    source_text = "List example\n\n1. what is Supplier ?\nOther material in this same source.\n"

    def request(method, url, **kwargs):
        if url.endswith("/search"):
            body = {
                "result_type": "documents",
                "query": kwargs["json"]["query"],
                "search_phrase": "what is Supplier",
                "documents": [
                    {
                        "document_id": "private-note",
                        "filename": "note.txt",
                        "source_name": "Original Supplier note",
                        "character_count": len(source_text),
                        "match_count": 1,
                    }
                ],
                "total_matches": 1,
                "retrieval_seconds": 0.01,
                "message": "Complete matching source",
            }
            return httpx.Response(200, json=body, request=httpx.Request(method, url))
        if url.endswith("/documents/private-note/content"):
            body = {
                "document_id": "private-note",
                "filename": "note.txt",
                "source_name": "Original Supplier note",
                "content": source_text,
                "character_count": len(source_text),
                "content_format": "original_text",
            }
            return httpx.Response(200, json=body, request=httpx.Request(method, url))
        return original(method, url, **kwargs)

    monkeypatch.setattr(httpx, "request", request)
    app = AppTest.from_file(str(APP)).run()
    app.text_input[0].set_value("find what is Supplier text and its whole document content")
    app.button[0].click().run()
    assert not app.exception
    assert len(app.text_area) == 1 and app.text_area[0].value == source_text
    assert not any("podman load --input sample.tar" in item.value for item in app.text)


def test_changing_result_mode_does_not_show_old_passage_results(monkeypatch):
    fake_api(monkeypatch)
    app = AppTest.from_file(str(APP)).run()
    app.text_input[0].set_value("Supplier")
    app.button[0].click().run()
    assert any("podman load --input sample.tar" in item.value for item in app.text)
    app.radio(key="result_mode_local").set_value("Whole matching document").run()
    assert not app.exception
    assert any("Search settings changed" in item.value for item in app.info)
    assert not any("podman load --input sample.tar" in item.value for item in app.text)


def test_exact_command_is_primary_display_with_original_filename(monkeypatch):
    fake_api(monkeypatch)
    original = httpx.request

    def request(method, url, **kwargs):
        response = original(method, url, **kwargs)
        if url.endswith("/search"):
            body = response.json()
            body["chunks"][0].update(
                source_name="new 30.txt",
                match_type="exact_phrase",
                matched_phrase="podman build",
                matched_text="podman build --tag example .",
                text="kubectl cp before\npodman build --tag example .\npodman push example",
            )
            return httpx.Response(200, json=body, request=httpx.Request(method, url))
        return response

    monkeypatch.setattr(httpx, "request", request)
    app = AppTest.from_file(str(APP)).run()
    app.text_input[0].set_value("find podman build cmd")
    app.button[0].click().run()
    assert not app.exception
    assert app.text[0].value == "podman build --tag example ."
    assert any("new 30.txt" in item.value for item in app.caption)
    assert any("Surrounding source passage" == item.label for item in app.expander)


def test_links_render_actual_urls_and_source_evidence(monkeypatch):
    fake_api(monkeypatch)
    original = httpx.request
    url = "https://example.test/display/OC5G/Supported+CNCC+Operations"

    def request(method, endpoint, **kwargs):
        if endpoint.endswith("/search"):
            body = {
                "result_type": "links",
                "query": kwargs["json"]["query"],
                "links": [
                    {
                        "url": url,
                        "title": "Supported CNCC Operations",
                        "match_reason": "topic_in_url",
                        "sources": [
                            {
                                "document_id": "private-note",
                                "filename": "note.txt",
                                "source_name": "new 13.txt",
                                "line_number": 28,
                                "text": url,
                            }
                        ],
                    }
                ],
                "total_matches": 1,
                "retrieval_seconds": 0.01,
                "message": "1 saved reference",
                "plan": {"intent": "links", "subject": "cncc", "scope": "imported_sources_only"},
            }
            return httpx.Response(200, json=body, request=httpx.Request(method, endpoint))
        return original(method, endpoint, **kwargs)

    monkeypatch.setattr(httpx, "request", request)
    app = AppTest.from_file(str(APP)).run()
    app.text_input[0].set_value("find confluence pages related to cncc")
    app.button[0].click().run()
    assert not app.exception
    assert any(item.value == url for item in app.text)
    assert any("new 13.txt" in item.value and "28" in item.value for item in app.caption)
    assert not any("podman load" in item.value for item in app.text)


def test_failed_new_search_clears_old_results(monkeypatch):
    fake_api(monkeypatch)
    original = httpx.request
    broken = {"value": False}

    def request(method, url, **kwargs):
        if broken["value"] and url.endswith("/search"):
            return httpx.Response(
                500, json={"detail": "unavailable"}, request=httpx.Request(method, url)
            )
        return original(method, url, **kwargs)

    monkeypatch.setattr(httpx, "request", request)
    app = AppTest.from_file(str(APP)).run()
    app.text_input[0].set_value("podman load")
    app.button[0].click().run()
    assert any("sample.tar" in item.value for item in app.text)
    broken["value"] = True
    app.text_input[0].set_value("different question")
    app.button[0].click().run()
    assert not app.exception and app.error
    assert not any("sample.tar" in item.value for item in app.text)


def test_engine_upgrade_invalidates_old_results(monkeypatch):
    fake_api(monkeypatch)
    original = httpx.request
    version = {"value": "first"}

    def request(method, url, **kwargs):
        response = original(method, url, **kwargs)
        if url.endswith("/health"):
            return httpx.Response(
                200,
                json=dict(response.json(), retrieval_engine_version=version["value"]),
                request=httpx.Request(method, url),
            )
        return response

    monkeypatch.setattr(httpx, "request", request)
    app = AppTest.from_file(str(APP)).run()
    app.text_input[0].set_value("podman load")
    app.button[0].click().run()
    version["value"] = "second"
    app.run()
    assert not app.exception
    assert not any("sample.tar" in item.value for item in app.text)
