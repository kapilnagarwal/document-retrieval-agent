"""A pure HTTP client: no imports from backend, no direct database access."""

import json
import os
from pathlib import Path
from uuid import uuid4

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
st.set_page_config(page_title="Document Retrieval AI Agent", page_icon="📚", layout="wide")
st.sidebar.title("Explore")
BASE_URL = os.getenv("LOCAL_BACKEND_URL", "http://127.0.0.1:8001").rstrip("/")


def api(method, route, **kwargs):
    try:
        response = httpx.request(method, BASE_URL + route, timeout=600, trust_env=False, **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", "Request failed")
        except ValueError:
            detail = "Backend returned an invalid response"
        st.error(str(detail))
    except httpx.RequestError:
        st.error(
            "Cannot reach the backend. Start it with: python -m scripts.run_profile local backend (see RUNNING.md)."
        )
    return None


def stream_answer_content(payload):
    status = st.empty()
    preview = st.empty()
    draft = ""
    try:
        with httpx.stream(
            "POST", BASE_URL + "/chat/stream", json=payload, timeout=600, trust_env=False
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                kind, data = event["event"], event["data"]
                if kind == "delta":
                    draft += data["text"]
                    preview.markdown(draft)
                elif kind == "reset":
                    draft = ""
                    preview.empty()
                elif kind == "status":
                    status.caption(data["message"])
                elif kind == "final":
                    preview.markdown(data["answer"])
                    status.empty()
                    show_sources(data)
                    return data
                elif kind == "error":
                    st.error(data["message"])
                    break
    except (httpx.HTTPError, ValueError, KeyError):
        st.error(
            "Connection interrupted. Reopen the saved conversation to recover its answer, or retry."
        )
    preview.empty()
    status.empty()
    return None


def stream_answer(payload):
    with st.chat_message("user"):
        st.markdown(payload["question"])
    with st.chat_message("assistant"):
        return stream_answer_content(payload)


def show_sources(result):
    origin = result.get("answer_origin", "documents")
    if origin != "documents":
        labels = {
            "general": "General knowledge — not verified against your documents or the web",
            "conversation": "Conversation follow-up — no new document search",
            "web": "Public web search — earlier answers are comparison inputs",
        }
        st.caption(labels.get(origin, origin))
        for warning in result.get("warnings", []):
            st.warning(warning)
        for index, source in enumerate(result.get("web_sources", []), 1):
            st.link_button(f"W{index}: {source['title']}", source["url"])
        if result.get("search_suggestions_html"):
            st.html(result["search_suggestions_html"])
        return
    if result.get("answer_status") == "insufficient_evidence":
        st.info("The retrieved passages did not provide enough evidence for an answer.")
    elif result.get("answer_status") == "unverified":
        st.warning("The answer needs verification. Check the warnings and supporting passages.")
    for warning in result.get("warnings", []):
        st.warning(warning)
    with st.expander("Sources and retrieved passages"):
        for source in result.get("sources", []):
            page = f" · PDF page {source['page_number']}" if source.get("page_number") else ""
            st.write(
                f"[{source['source_id']}] {source['filename']}{page} · "
                + ("cited" if source["cited"] else "retrieved, uncited")
            )
        for chunk in result.get("retrieved_chunks", []):
            evidence_label = (
                f"cosine distance {chunk['distance']:.4f} (lower is closer)"
                if chunk.get("distance") is not None
                else f"complete PDF page {chunk.get('page_number', '')}"
            )
            st.caption(f"{chunk['filename']} · {evidence_label}")
            st.text(chunk["text"])


def conversation_turns(name, signature):
    state = st.session_state.get(name)
    if state and state["signature"] != signature:
        if state["signature"][:2] == signature[:2]:
            state["signature"] = (
                signature  # Preserve chat while retrieving current document versions.
            )
        else:
            st.session_state.pop("active_saved_chat", None)
    if not state or state["signature"] != signature:
        state = {"signature": signature, "turns": []}
        st.session_state[name] = state
    return state["turns"]


def recent_history(turns):
    return [
        {"question": turn["question"][:8000], "answer": turn["answer"][:32000]}
        for turn in turns[-4:]
    ]


def show_conversation(turns):
    for turn in turns:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            show_sources(turn)


def show_whole_document(result):
    st.caption(
        f"Source lookup: {result['search_phrase']} · {result['total_matches']} matching document(s)"
    )
    matches = result["documents"]
    if not matches:
        st.info(result["message"])
        return
    if len(matches) == 1:
        selected_document = matches[0]
    else:
        st.info(result["message"])
        by_id = {document["document_id"]: document for document in matches}
        document_id = st.selectbox(
            "Choose the matching source document",
            list(by_id),
            index=None,
            format_func=lambda value: by_id[value].get("source_name") or by_id[value]["filename"],
            key=f"full_source_{config.get('data_profile')}_{result['search_phrase']}",
        )
        if not document_id:
            return
        selected_document = by_id[document_id]
    full = api("GET", f"/documents/{selected_document['document_id']}/content")
    if full is None:
        return
    st.subheader("Complete source document")
    st.text(full.get("source_name") or full["filename"])
    st.caption(
        f"{full['character_count']:,} characters · {len(full['content'].splitlines())} lines · "
        + (
            "original imported text"
            if full["content_format"] == "original_text"
            else "extracted source text"
        )
    )
    st.text_area(
        "Complete imported text (read only)",
        value=full["content"],
        height=600,
        disabled=True,
        key=f"original_text_{full['document_id']}",
    )
    st.download_button(
        "Download complete text",
        full["content"],
        file_name=Path(full["filename"]).stem + ".txt",
        mime="text/plain",
    )
    st.caption(
        "This is the full imported source, including its own surrounding topics. Text from other documents has not been added. No Gemini call was made."
    )


def show_saved_links(result):
    st.info(result["message"])
    for index, item in enumerate(result["links"], 1):
        with st.container(border=True):
            st.link_button(f"{index}. {item['title']}", item["url"])
            st.text(item["url"])
            st.caption(
                "Topic matched in the saved URL/title"
                if item["match_reason"] == "topic_in_url"
                else "Topic matched in nearby source text"
            )
            with st.expander("Where this link was found"):
                for evidence in item["sources"]:
                    st.caption(
                        f"{evidence.get('source_name') or evidence['filename']} · line {evidence['line_number']}"
                    )
                    st.text(evidence["text"])


st.title("Start Chat")
st.caption("Local embeddings and retrieval · Gemini answers · an inspectable RAG learning project")
health = api("GET", "/health")
if health is None:
    st.stop()
if health.get("recovery_issues"):
    st.warning(
        "Storage integrity issues were detected. Preserve the data directory and restore a verified backup; see RUNNING.md."
    )
config = api("GET", "/config")
documents = api("GET", "/documents")
if config is None or documents is None:
    st.stop()
if config["allow_gemini_document_context"]:
    st.info(
        "Your documents are stored locally. AI chat sends your question and recent conversation to Google Gemini, with document passages when needed. Online requests use Google Search. Passage, document, and link searches run locally."
    )
else:
    st.info(
        "Your documents stay local. Local search is available; Gemini answer generation is disabled."
    )
sections = ["Start Chat", "Dashboard", "Document Management", "RAG Debugger"]
explore_key = f"explore_{config.get('data_profile')}_v2"
previous_section = st.session_state.get(explore_key)
if previous_section in {"Search Documents", "Chat With Documents", "Ask Documents"}:
    st.session_state[explore_key] = "Start Chat"
st.sidebar.caption(f"{health['documents']} documents · {health['chunks']:,} chunks")
section = st.sidebar.radio("Explore", sections, key=explore_key, label_visibility="collapsed")
lookup = {d["document_id"]: d.get("source_name") or d["filename"] for d in documents}
corpus_signature = tuple(sorted((d["document_id"], d["chunk_count"]) for d in documents))

if section == "Dashboard":
    columns = st.columns(3)
    columns[0].metric("Documents", health["documents"])
    columns[1].metric("Indexed chunks", health["chunks"])
    columns[2].metric("Backend", health["status"])
    st.write("Vector database:", health["vector_database"])
    st.write("Embedding model:", config["embedding_model"])
    st.write("Gemini model:", config["gemini_model"])
    st.write("Gemini document context enabled:", config["allow_gemini_document_context"])
    st.info(
        "Search stays local. Chat sends retrieved passages to Google when enabled in .env. Confirm permission before using internal documents."
    )
elif section == "Start Chat":
    restore = st.session_state.pop("restore_chat", None)
    if restore:
        st.session_state["chat_scope"] = {
            "all": "All documents",
            "notes": "My notes",
            "pdf": "SEPP guides",
        }[restore["settings"]["scope"]]
        st.session_state["chat_documents"] = restore["settings"]["document_ids"]
        st.session_state["active_saved_chat"] = restore["id"]
        st.session_state["saved_chat_settings"] = restore["settings"]
    answers_enabled = config["allow_gemini_document_context"]
    st.subheader("Ask a question" if answers_enabled else "Search your indexed documents")
    st.caption(
        "Ask about your documents, follow up on an answer, or use general chat and public web sources."
        if answers_enabled
        else "Ask a natural-language question to retrieve matching passages. This search runs locally and does not call Gemini."
    )
    st.caption(
        "Search covers imported snapshots. Unsaved editor tabs and later edits are not indexed automatically; upload or re-import them first."
    )
    if config.get("data_profile") == "local":
        scope = st.radio(
            "Search in",
            ["My notes", "SEPP guides", "All documents"],
            index=2 if answers_enabled else 0,
            key="chat_scope",
            horizontal=True,
        )
        if scope == "My notes":
            candidates = [d for d in documents if not d["filename"].lower().endswith(".pdf")]
        elif scope == "SEPP guides":
            candidates = [d for d in documents if d["filename"].lower().endswith(".pdf")]
        else:
            candidates = documents
    else:
        scope, candidates = "Learning documents", documents
    st.caption(
        f"{len(candidates)} documents · {sum(d['chunk_count'] for d in candidates):,} indexed chunks"
    )
    missing_saved_ids = [
        identifier
        for identifier in st.session_state.get("chat_documents", [])
        if identifier not in lookup
    ]
    if missing_saved_ids:
        st.warning(
            "Some saved sources were deleted. They remain in the filter so your question cannot silently search other documents."
        )
        lookup.update(
            {identifier: "Unavailable source: " + identifier for identifier in missing_saved_ids}
        )
    selected = st.sidebar.multiselect(
        "Restrict to specific documents (optional)",
        [d["document_id"] for d in candidates] + missing_saved_ids,
        key="chat_documents",
        format_func=lookup.get,
        max_selections=200,
    )
    top_k = st.sidebar.slider("Maximum displayed results", 1, 30, config["retrieval_top_k"])
    result_mode = st.radio(
        "Return",
        (["AI answer"] if answers_enabled else [])
        + [
            "Automatic",
            "Matching passages",
            "Whole matching document",
            "Links / references",
            "Semantic suggestions",
        ],
        horizontal=True,
        key=f"result_mode_{config.get('data_profile')}"
        + ("_answers_v1" if answers_enabled else ""),
    )
    mode_value = {
        "AI answer": "answer",
        "Automatic": "auto",
        "Matching passages": "passages",
        "Whole matching document": "documents",
        "Links / references": "links",
        "Semantic suggestions": "semantic",
    }[result_mode]
    chat_mode = "documents"
    if mode_value == "answer":
        chat_source = st.radio(
            "Answer using",
            ["Auto", "Documents only", "General chat", "Web search"],
            horizontal=True,
            key="chat_answer_source",
        )
        chat_mode = {
            "Auto": "auto",
            "Documents only": "documents",
            "General chat": "general",
            "Web search": "web",
        }[chat_source]
        st.caption(
            "Auto uses documents when relevant, handles follow-ups directly, and searches the public web when you ask for online information. General answers are labeled. Web search sends your prompt and recent conversation to Google and may incur search usage."
        )
    scope_value = {"My notes": "notes", "SEPP guides": "pdf"}.get(scope, "all")
    request_signature = (
        scope_value,
        health.get("retrieval_engine_version", "legacy"),
        mode_value,
        top_k,
        tuple(sorted(selected or [d["document_id"] for d in candidates])),
    )
    conversation_key = f"document_conversation_{config.get('data_profile')}"
    legacy_chat = st.session_state.get(f"chat_conversation_{config.get('data_profile')}")
    if (
        previous_section == "Chat With Documents"
        and legacy_chat
        and scope_value == "all"
        and legacy_chat["signature"] == (tuple(sorted(selected)), corpus_signature)
    ):
        st.session_state[conversation_key] = {
            "signature": (scope_value, tuple(sorted(selected)), corpus_signature),
            "turns": legacy_chat["turns"],
        }
    if restore:
        st.session_state[conversation_key] = {
            "signature": (scope_value, tuple(sorted(selected)), corpus_signature),
            "turns": [t["response"] for t in restore["turns"] if t["response"]],
        }
        for turn in restore["turns"]:
            if turn["status"] in {"failed", "interrupted"}:
                st.warning("An unfinished question can be retried: " + turn["question"])
    initial_conversation = conversation_key not in st.session_state
    conversation = conversation_turns(
        conversation_key,
        (scope_value, tuple(sorted(selected)), corpus_signature),
    )
    previous = st.session_state.get(f"document_search_result_{config.get('data_profile')}")
    if (
        initial_conversation
        and mode_value == "answer"
        and previous
        and previous.get("request_signature") == request_signature
        and previous.get("corpus_signature") == corpus_signature
        and previous["result"].get("answer")
    ):
        conversation.append(previous["result"])
    if mode_value == "answer":
        saved_chats = api("GET", "/conversations") or []
        chosen = st.sidebar.selectbox(
            "Saved conversations",
            [None] + [c["id"] for c in saved_chats],
            format_func=lambda value: next(
                (c["title"] for c in saved_chats if c["id"] == value), "Current conversation"
            ),
        )
        if chosen and st.sidebar.button("Open conversation"):
            saved_chat = api("GET", "/conversations/" + chosen)
            if saved_chat:
                st.session_state["restore_chat"] = saved_chat
                st.rerun()
        if chosen:
            saved_chat = api("GET", "/conversations/" + chosen)
            if saved_chat:
                st.sidebar.download_button(
                    "Export conversation",
                    json.dumps(saved_chat, indent=2),
                    file_name="conversation.json",
                    mime="application/json",
                )
            if st.sidebar.button("Delete saved conversation"):
                if api("DELETE", "/conversations/" + chosen):
                    st.session_state.pop("active_saved_chat", None)
                    conversation.clear()
                    st.rerun()
    if mode_value == "answer":
        st.caption(
            "Ask follow-ups in the message box pinned below. Conversations are saved locally. Recent messages are sent to Gemini. Document answers are checked against retrieved evidence; general chat and web answers are labeled separately."
        )
        question, submitted = None, False
    else:
        with st.form("document_search"):
            question = st.text_input(
                "What would you like to find?",
                placeholder="Ask about a topic in your documents",
                key=f"document_prompt_{config.get('data_profile')}",
            )
            submitted = st.form_submit_button("Search documents", type="primary")
    if mode_value == "answer" and st.sidebar.button("New conversation"):
        conversation.clear()
        st.session_state.pop("active_saved_chat", None)
        st.session_state.pop(f"document_search_result_{config.get('data_profile')}", None)
    if mode_value == "answer":
        show_conversation(conversation)
        question = st.chat_input(
            "Ask a question or follow up on the answer…",
            key=f"document_chat_prompt_{config.get('data_profile')}",
        )
        submitted = question is not None
    if submitted:
        st.session_state.pop(f"document_search_result_{config.get('data_profile')}", None)
        if not question.strip():
            st.warning("Enter a question or a few search words.")
        elif not candidates and (mode_value != "answer" or chat_mode == "documents"):
            st.warning("No indexed documents in this category.")
        else:
            with st.spinner(
                "Retrieving evidence and generating an answer…"
                if mode_value == "answer"
                else "Searching imported sources…"
            ):
                if mode_value == "answer":
                    chat_settings = dict(scope=scope_value, document_ids=selected)
                    active_id = st.session_state.get("active_saved_chat")
                    if st.session_state.get("saved_chat_settings") != chat_settings:
                        active_id = None
                    if not active_id:
                        saved_chat = api(
                            "POST",
                            "/conversations",
                            json=dict(title=question.strip()[:120], **chat_settings),
                        )
                        if saved_chat:
                            active_id = saved_chat["id"]
                            st.session_state["active_saved_chat"] = active_id
                            st.session_state["saved_chat_settings"] = chat_settings
                    if active_id:
                        result = stream_answer(
                            dict(
                                question=question.strip(),
                                answer_mode=chat_mode,
                                top_k=top_k,
                                history=recent_history(conversation),
                                conversation_id=active_id,
                                request_id=str(uuid4()),
                                **chat_settings,
                            )
                        )
                    else:
                        result = None
                else:
                    result = api(
                        "POST",
                        "/search",
                        json=dict(
                            query=question.strip(),
                            top_k=top_k,
                            mode=mode_value,
                            document_ids=selected,
                            scope=scope_value,
                        ),
                    )
            if result is not None:
                if mode_value == "answer":
                    conversation.append(result)
                st.session_state[f"document_search_result_{config.get('data_profile')}"] = dict(
                    scope=scope,
                    result=result,
                    corpus_signature=corpus_signature,
                    request_signature=request_signature,
                )
    saved = st.session_state.get(f"document_search_result_{config.get('data_profile')}")
    if saved and saved.get("corpus_signature") != corpus_signature:
        st.info("Indexed documents changed. Search again to include the latest imports.")
    elif saved and saved.get("request_signature") != request_signature:
        st.info(
            "Search settings changed. Search again to apply the selected result type and filters."
        )
    if (
        saved
        and mode_value != "answer"
        and saved["scope"] == scope
        and saved.get("corpus_signature") == corpus_signature
        and saved.get("request_signature") == request_signature
    ):
        result = saved["result"]
        st.write(
            "Prompt:" if mode_value == "answer" else "Search:",
            result.get("question") or result.get("query"),
        )
        plan = result.get("plan") or {}
        if plan:
            output = {
                "links": "saved links",
                "commands": "command evidence",
                "documents": "complete source documents",
                "passages": "source passages",
            }.get(plan["intent"], "sources")
            st.caption(
                f"Looking for {output}: {plan['subject'] or 'all matching sources'} · imported documents only"
            )
        if result.get("result_type") == "links":
            show_saved_links(result)
        elif result.get("result_type") == "documents":
            show_whole_document(result)
        else:
            dimension_label = (
                f"{result['embedding_dimension']}-dimensional query"
                if result["embedding_dimension"]
                else "query embedding skipped"
            )
            st.caption(
                f"{len(result['chunks'])} passages · {result['retrieval_seconds']:.3f} seconds · {dimension_label}"
            )
            if result.get("required_terms"):
                st.caption("Required exact matches: " + ", ".join(result["required_terms"]))
            if not result["chunks"]:
                st.info(
                    result.get("message")
                    or "No matching passages found. Try different words or a broader document selection."
                )
            if result.get("message") and result["chunks"]:
                st.info(result["message"])
            for rank, chunk in enumerate(result["chunks"], 1):
                with st.container(border=True):
                    page = f" · PDF page {chunk['page_number']}" if chunk.get("page_number") else ""
                    st.markdown(f"**Result {rank}**")
                    source_name = chunk.get("source_name") or chunk["filename"]
                    if chunk.get("command_count"):
                        st.caption(
                            f"{source_name}{page} · {chunk['command_count']} matching commands · first at line {chunk['line_number']}"
                        )
                    else:
                        distance_label = (
                            f" · cosine distance {chunk['distance']:.4f}"
                            if chunk.get("distance") is not None
                            else ""
                        )
                        st.caption(
                            f"{source_name}{page} · chunk {chunk['chunk_index']}{distance_label}"
                        )
                    if chunk.get("match_type") == "exact_phrase":
                        st.caption("Exact phrase match: " + chunk["matched_phrase"])
                    st.text(chunk.get("matched_text") or chunk["text"])
                    if chunk.get("matched_text") and chunk["matched_text"] != chunk["text"]:
                        with st.expander("Surrounding source passage"):
                            st.text(chunk["text"])
    if mode_value != "answer" and (
        not saved or saved["result"].get("result_type") not in {"documents", "links"}
    ):
        st.caption(
            "These are retrieved source passages, not generated answers. For the entire matching note, ask for its whole document content or select Whole matching document."
        )
elif section == "Document Management":
    uploads = st.file_uploader(
        "PDF, DOCX, UTF-8 TXT or Markdown",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    st.caption(
        f"Maximum {config['max_upload_size_mb']} MB per file. Scanned PDFs require OCR; OCR is not included."
    )
    if st.button("Index uploaded documents", disabled=not uploads):
        for file in uploads:
            api("POST", "/index-jobs", files={"file": (file.name, file.getvalue())})
        st.rerun()
    jobs = api("GET", "/index-jobs") or []
    if jobs:
        st.subheader("Indexing jobs")
        st.dataframe(
            [{k: j.get(k) for k in ("id", "filename", "status", "error")} for j in jobs],
            use_container_width=True,
        )
        st.button("Refresh indexing status")
        failed = [j["id"] for j in jobs if j["status"] == "failed"]
        if failed:
            retry_id = st.selectbox("Failed job to retry", failed)
            if st.button("Retry indexing"):
                api("POST", "/index-jobs/" + retry_id + "/retry")
                st.rerun()
    if "ingestion_results" in st.session_state:
        st.dataframe(st.session_state["ingestion_results"], use_container_width=True)
    st.dataframe(documents, use_container_width=True)
    if config.get("data_profile") == "local":
        with st.expander("Check saved-note coverage"):
            st.caption(
                "Check which saved files in your Downloads notes folder are indexed, changed, empty or missing from the index. Unsaved editor tabs are not monitored."
            )
            if st.button("Check notes folder"):
                audit = api("GET", "/sources/notes/audit")
                if audit:
                    st.json(audit["counts"])
                    st.dataframe(audit["files"], use_container_width=True)
                    st.caption(audit["message"])
    selected = st.selectbox("Document to delete", list(lookup), format_func=lookup.get, index=None)
    confirm = st.checkbox("Delete this document and all its indexed chunks")
    if st.button("Delete selected document", disabled=not selected or not confirm):
        if api("DELETE", f"/documents/{selected}"):
            st.rerun()
    with st.expander("Source versions and replacement"):
        st.caption(
            "Choose a document above. A replacement becomes active only after indexing succeeds; earlier versions remain available for inspection."
        )
        if st.button("Show source versions", disabled=not selected):
            versions = api("GET", f"/documents/{selected}/versions")
            if versions:
                st.dataframe(versions, use_container_width=True)
        replacement = st.file_uploader(
            "Replacement document", type=["pdf", "docx", "txt", "md"], key="replacement"
        )
        if st.button("Index replacement version", disabled=not selected or replacement is None):
            result = api(
                "PUT",
                f"/documents/{selected}",
                files={"file": (replacement.name, replacement.getvalue())},
            )
            if result:
                st.session_state["ingestion_results"] = [result]
                st.rerun()
elif section == "RAG Debugger":
    selected = st.sidebar.multiselect(
        "Search these documents (empty = all)", list(lookup), format_func=lookup.get
    )
    top_k = st.sidebar.slider("Top-k passages", 1, 30, config["retrieval_top_k"])
    st.warning(
        "The debugger displays uploaded document content, including the exact context sent to Gemini."
    )
    question = st.text_input("Question to inspect")
    modes = (
        ["Local semantic search", "Full RAG with Gemini"]
        if config["allow_gemini_document_context"]
        else ["Local semantic search"]
    )
    action = st.radio("Mode", modes, horizontal=True)
    if st.button("Inspect pipeline", disabled=not question.strip()):
        with st.spinner("Running pipeline…"):
            if action == "Local semantic search":
                result = api(
                    "POST",
                    "/search",
                    json=dict(query=question, top_k=top_k, document_ids=selected, mode="semantic"),
                )
            else:
                result = api(
                    "POST",
                    "/chat",
                    json=dict(question=question, top_k=top_k, document_ids=selected, debug=True),
                )
        if result:
            st.session_state["debug_result"] = result
    result = st.session_state.get("debug_result")
    if result:
        st.json(result)
        if result.get("debug"):
            st.subheader("Context")
            st.code(result["debug"]["context"], language="json")
            st.subheader("System instruction")
            st.text(result["debug"]["system_prompt"])
            st.subheader("User prompt")
            st.text(result["debug"]["user_prompt"])
