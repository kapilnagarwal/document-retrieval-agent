"""Regression checks for choosing the correct corpus in the main UI."""

import json
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import httpx
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "frontend/app.py"


def fake_api(monkeypatch, note_name="local-note-test.txt", allow_local_chat=False):
    calls = []
    conversations = {}

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        private = ":8001/" in url
        if url.endswith("/health"):
            body = {"status": "ok", "documents": 1, "chunks": 1, "vector_database": "ready"}
        elif url.endswith("/config"):
            body = {
                "data_profile": "local" if private else "learning",
                "retrieval_top_k": 5,
                "allow_gemini_document_context": not private or allow_local_chat,
                "max_upload_size_mb": 20,
                "embedding_model": "test-model",
                "gemini_model": "test-gemini",
            }
        elif url.endswith("/conversations"):
            if method == "GET":
                body = list(conversations.values())
            else:
                identifier = str(uuid4())
                body = {
                    "id": identifier,
                    "title": kwargs["json"]["title"],
                    "settings": {k: kwargs["json"][k] for k in ("scope", "document_ids")},
                    "turns": [],
                }
                conversations[identifier] = body
        elif "/conversations/" in url:
            identifier = url.rsplit("/", 1)[-1]
            if method == "DELETE":
                conversations.pop(identifier)
                body = {"deleted": identifier}
            else:
                body = conversations[identifier]
        elif url.endswith("/documents"):
            body = [
                {
                    "document_id": "private-note" if private else "sample",
                    "filename": note_name if private else "sample.txt",
                    "chunk_count": 1,
                }
            ]
        elif url.endswith("/chat"):
            body = {
                "question": kwargs["json"]["question"],
                "answer": "Generated explanation from the document [S1]",
                "answer_status": "answered",
                "sources": [],
                "retrieved_chunks": [],
            }
            identifier = kwargs["json"].get("conversation_id")
            if identifier:
                conversations[identifier]["turns"].append({"status": "complete", "response": body})
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

    @contextmanager
    def stream(method, url, **kwargs):
        # Exercise the real frontend NDJSON reader, including draft/reset/final.
        response = httpx.request(method, url.removesuffix("/stream"), **kwargs)
        events = [
            {"event": "delta", "data": {"text": "Draft answer"}},
            {"event": "reset", "data": {}},
            {"event": "final", "data": response.json()},
        ]
        yield httpx.Response(
            200, text="\n".join(json.dumps(e) for e in events), request=httpx.Request(method, url)
        )

    monkeypatch.setattr(httpx, "stream", stream)
    monkeypatch.setattr(httpx, "request", request)
    return calls


def test_main_ui_defaults_to_private_notes_and_searches_locally(monkeypatch):
    calls = fake_api(monkeypatch)
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert not any(radio.label == "Document collection" for radio in app.radio)
    assert all(":8001/" in url for _, url, _ in calls)
    app.text_input[0].set_value("does it contain podman load command?")
    app.button[0].click().run()
    assert not app.exception
    searches = [call for call in calls if call[1].endswith("/search")]
    assert searches[-1][2]["json"]["document_ids"] == []
    assert searches[-1][2]["json"]["scope"] == "notes"
    assert not any(url.endswith("/chat") for _, url, _ in calls)
    assert any("podman load" in item.value for item in app.text)


def test_main_prompt_generates_answer_by_default_when_enabled(monkeypatch):
    calls = fake_api(monkeypatch, note_name="NG.146.pdf", allow_local_chat=True)
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert app.radio(key="result_mode_local_answers_v1").value == "AI answer"
    assert app.chat_input[0].placeholder == "Ask a question or follow up on the answer…"
    assert len(app.text_input) == 0
    app.multiselect[0].set_value(["private-note"]).run()
    app.chat_input[0].set_value("Explain the handshake failure response codes").run()
    assert not app.exception
    chats = [call for call in calls if call[1].endswith("/chat")]
    assert chats[-1][2]["json"]["document_ids"] == ["private-note"]
    assert chats[-1][2]["json"]["scope"] == "all"
    assert any("Generated explanation" in item.value for item in app.markdown)
    assert not any(url.endswith("/search") for _, url, _ in calls)
    assert not any("not generated answers" in item.value for item in app.caption)


def test_answer_scope_and_switch_to_local_retrieval(monkeypatch):
    calls = fake_api(monkeypatch, allow_local_chat=True)
    app = AppTest.from_file(str(APP)).run()
    next(radio for radio in app.radio if radio.label == "Search in").set_value("My notes").run()
    app.chat_input[0].set_value("Explain podman load").run()
    assert not app.exception
    chats = [call for call in calls if call[1].endswith("/chat")]
    assert chats[-1][2]["json"]["scope"] == "notes"
    app.radio(key="result_mode_local_answers_v1").set_value("Matching passages").run()
    assert not any("Generated explanation" in item.value for item in app.markdown)
    app.text_input[0].set_value("podman load")
    app.button[0].click().run()
    assert not app.exception
    assert any(url.endswith("/search") for _, url, _ in calls)


def test_main_answer_keeps_follow_up_history_and_can_start_over(monkeypatch):
    calls = fake_api(monkeypatch, allow_local_chat=True)
    app = AppTest.from_file(str(APP)).run()
    for question in ["Explain handshake errors", "Which of those are server errors?"]:
        app.chat_input[0].set_value(question).run()
        assert not app.exception
    chats = [call for call in calls if call[1].endswith("/chat")]
    assert chats[0][2]["json"]["history"] == []
    assert chats[1][2]["json"]["history"] == [
        {
            "question": "Explain handshake errors",
            "answer": "Generated explanation from the document [S1]",
        }
    ]
    assert sum("Generated explanation" in item.value for item in app.markdown) == 2
    next(button for button in app.button if button.label == "New conversation").click().run()
    assert not any("Generated explanation" in item.value for item in app.markdown)
    app.chat_input[0].set_value("New topic").run()
    assert [call for call in calls if call[1].endswith("/chat")][-1][2]["json"]["history"] == []


def test_changing_documents_resets_conversation_context(monkeypatch):
    calls = fake_api(monkeypatch, allow_local_chat=True)
    app = AppTest.from_file(str(APP)).run()
    app.chat_input[0].set_value("Explain handshake errors").run()
    app.multiselect[0].set_value(["private-note"]).run()
    assert not any("Generated explanation" in item.value for item in app.markdown)
    app.chat_input[0].set_value("Explain that").run()
    assert not app.exception
    chats = [call for call in calls if call[1].endswith("/chat")]
    assert chats[-1][2]["json"]["history"] == []


def test_one_workspace_combines_chat_and_search_modes(monkeypatch):
    fake_api(monkeypatch, allow_local_chat=True)
    app = AppTest.from_file(str(APP)).run()
    assert app.radio(key="explore_local_v2").value == "Start Chat"
    assert app.radio(key="explore_local_v2").options == [
        "Start Chat",
        "Dashboard",
        "Document Management",
        "RAG Debugger",
    ]
    assert app.radio(key="result_mode_local_answers_v1").options == [
        "AI answer",
        "Automatic",
        "Matching passages",
        "Whole matching document",
        "Links / references",
        "Semantic suggestions",
    ]
    assert len(app.chat_input) == 1


def test_old_chat_selection_migrates_to_unified_workspace_with_history(monkeypatch):
    calls = fake_api(monkeypatch, allow_local_chat=True)
    app = AppTest.from_file(str(APP))
    app.session_state["explore_local_v2"] = "Chat With Documents"
    app.session_state["chat_conversation_local"] = {
        "signature": ((), (("private-note", 1),)),
        "turns": [{"question": "Earlier question", "answer": "Earlier answer"}],
    }
    app.run()
    assert not app.exception
    assert app.radio(key="explore_local_v2").value == "Start Chat"
    assert any(item.value == "Earlier answer" for item in app.markdown)
    app.chat_input[0].set_value("Explain that").run()
    chats = [call for call in calls if call[1].endswith("/chat")]
    assert chats[-1][2]["json"]["history"][0]["question"] == "Earlier question"


def test_old_collection_selection_does_not_route_away_from_main_backend(monkeypatch):
    calls = fake_api(monkeypatch)
    app = AppTest.from_file(str(APP))
    app.session_state["document_collection_v2"] = "Learning samples (Gemini chat)"
    app.run()
    assert not app.exception
    assert not any(radio.label == "Document collection" for radio in app.radio)
    assert all(":8001/" in url for _, url, _ in calls)


def test_uploaded_text_note_is_included_without_special_filename(monkeypatch):
    calls = fake_api(monkeypatch, note_name="my-working-notes.txt")
    app = AppTest.from_file(str(APP)).run()
    app.text_input[0].set_value("podman load")
    app.button[0].click().run()
    searches = [call for call in calls if call[1].endswith("/search")]
    assert not app.exception
    assert searches[-1][2]["json"]["document_ids"] == []
    assert searches[-1][2]["json"]["scope"] == "notes"


def test_local_chat_requires_opt_in_and_passes_document_filter(monkeypatch):
    calls = fake_api(monkeypatch)
    original = httpx.request

    def request(method, url, **kwargs):
        if url.endswith("/chat"):
            calls.append((method, url, kwargs))
            return httpx.Response(
                200,
                json={"question": kwargs["json"]["question"], "answer": "Generated answer [S1]"},
                request=httpx.Request(method, url),
            )
        response = original(method, url, **kwargs)
        if url.endswith("/config"):
            return httpx.Response(
                200,
                json={**response.json(), "allow_gemini_document_context": True},
                request=httpx.Request(method, url),
            )
        return response

    app = AppTest.from_file(str(APP)).run()
    assert "AI answer" not in app.radio(key="result_mode_local").options
    monkeypatch.setattr(httpx, "request", request)
    app.run()
    assert any("Google Gemini" in item.value for item in app.info)
    assert app.radio(key="explore_local_v2").value == "Start Chat"
    app.multiselect[0].set_value(["private-note"]).run()
    app.chat_input[0].set_value("What is the handshake response?").run()
    assert not app.exception
    chats = [call for call in calls if call[1].endswith("/chat")]
    assert chats[-1][2]["json"]["document_ids"] == ["private-note"]
    assert any("Generated answer" in item.value for item in app.markdown)


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


def test_saved_conversation_reopens_after_browser_session_with_filters(monkeypatch):
    fake_api(monkeypatch, allow_local_chat=True)
    first = AppTest.from_file(str(APP)).run()
    first.multiselect[0].set_value(["private-note"]).run()
    first.chat_input[0].set_value("Explain TLS").run()
    identifier = first.session_state["active_saved_chat"]
    reopened = AppTest.from_file(str(APP)).run()
    next(box for box in reopened.selectbox if box.label == "Saved conversations").set_value(
        identifier
    ).run()
    next(button for button in reopened.button if button.label == "Open conversation").click().run()
    assert not reopened.exception
    assert reopened.multiselect[0].value == ["private-note"]
    assert reopened.session_state["active_saved_chat"] == identifier
    assert any("Generated explanation" in item.value for item in reopened.markdown)
    assert not any("Draft answer" in item.value for item in reopened.markdown)


def test_auto_general_and_web_modes_are_sent_without_changing_conversation(monkeypatch):
    calls = fake_api(monkeypatch, allow_local_chat=True)
    app = AppTest.from_file(str(APP)).run()
    assert app.radio(key="chat_answer_source").value == "Auto"
    app.chat_input[0].set_value("Write a short explanation").run()
    first = [c for c in calls if c[1].endswith("/chat")][-1][2]["json"]
    assert first["answer_mode"] == "auto"
    app.radio(key="chat_answer_source").set_value("Web search").run()
    app.chat_input[0].set_value("Compare it with public specs online").run()
    second = [c for c in calls if c[1].endswith("/chat")][-1][2]["json"]
    assert second["answer_mode"] == "web"
    assert second["conversation_id"] == first["conversation_id"]


def test_web_answers_show_web_sources_without_document_abstention(monkeypatch):
    fake_api(monkeypatch, allow_local_chat=True)
    original = httpx.request

    def request(method, url, **kwargs):
        if url.endswith("/chat"):
            return httpx.Response(
                200,
                json={
                    "question": kwargs["json"]["question"],
                    "answer": "Public-source comparison.",
                    "answer_origin": "web",
                    "answer_status": "answered",
                    "sources": [],
                    "retrieved_chunks": [],
                    "web_sources": [{"title": "Official spec", "url": "https://example.org/spec"}],
                    "search_suggestions_html": "<div>Google Search suggestions</div>",
                },
                request=httpx.Request(method, url),
            )
        return original(method, url, **kwargs)

    monkeypatch.setattr(httpx, "request", request)
    app = AppTest.from_file(str(APP)).run()
    app.chat_input[0].set_value("Compare it online").run()
    assert not app.exception
    assert any("Public web search" in item.value for item in app.caption)
    assert not any("retrieved passages did not" in item.value for item in app.info)
    assert not app.expander


def test_general_chat_can_start_without_indexed_documents(monkeypatch):
    calls = fake_api(monkeypatch, allow_local_chat=True)
    original = httpx.request

    def request(method, url, **kwargs):
        if url.endswith("/documents"):
            return httpx.Response(200, json=[], request=httpx.Request(method, url))
        return original(method, url, **kwargs)

    monkeypatch.setattr(httpx, "request", request)
    app = AppTest.from_file(str(APP)).run()
    app.radio(key="chat_answer_source").set_value("General chat").run()
    app.chat_input[0].set_value("Write a short poem").run()
    assert not app.exception
    assert [c for c in calls if c[1].endswith("/chat")][-1][2]["json"]["answer_mode"] == "general"
