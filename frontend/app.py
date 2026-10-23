"""A pure HTTP client: no imports from backend, no direct database access."""

import os
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
st.set_page_config(page_title="Document Retrieval AI Agent", page_icon="📚", layout="wide")
st.sidebar.title("Choose your documents")
workspace = st.sidebar.radio(
    "Document collection",
    ["My notes and SEPP (local only)", "Learning samples (Gemini chat)"],
    key="document_collection_v2",
)
private_workspace = workspace == "My notes and SEPP (local only)"
BASE_URL = (
    os.getenv("LOCAL_BACKEND_URL", "http://127.0.0.1:8001")
    if private_workspace
    else os.getenv("LEARNING_BACKEND_URL", "http://127.0.0.1:8000")
).rstrip("/")


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
            "Cannot reach the backend. Start the matching profile backend using scripts.run_profile (see RUNNING.md)."
        )
    return None


def show_sources(result):
    if result.get("answer_status") == "insufficient_evidence":
        st.info("The retrieved passages did not provide enough evidence for an answer.")
    elif result.get("answer_status") == "unverified":
        st.warning("The answer has citation warnings. Check its supporting passages.")
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
            st.caption(
                f"{chunk['filename']} · cosine distance {chunk['distance']:.4f} (lower is closer)"
            )
            st.text(chunk["text"])


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


st.title("Search My Notes" if private_workspace else "Learning RAG Chat")
st.caption("Local embeddings and retrieval · Gemini answers · an inspectable RAG learning project")
health = api("GET", "/health")
if health is None:
    st.stop()
config = api("GET", "/config")
documents = api("GET", "/documents")
if config is None or documents is None:
    st.stop()
if config.get("data_profile") == "learning":
    st.info(
        "Learning samples are selected. Your private notes are in the other collection in the sidebar. Chat sends sample passages to Google."
    )
elif config.get("data_profile") == "local":
    st.info(
        "Private local profile: notes and SEPP guides stay local. Use Search Documents to query your notes; Gemini answer generation is disabled."
    )
sections = (
    ["Search Documents", "Dashboard", "Document Management", "RAG Debugger"]
    if config.get("data_profile") == "local"
    else [
        "Dashboard",
        "Document Management",
        "Search Documents",
        "Chat With Documents",
        "RAG Debugger",
    ]
)
st.sidebar.caption(f"{health['documents']} documents · {health['chunks']:,} chunks")
section = st.sidebar.radio("Explore", sections, key=f"explore_{config.get('data_profile')}_v2")
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
elif section == "Search Documents":
    st.subheader("Search your indexed documents")
    st.caption(
        "Ask a natural-language question to retrieve matching passages. This search runs locally and does not call Gemini."
    )
    st.caption(
        "Search covers imported snapshots. Unsaved editor tabs and later edits are not indexed automatically; upload or re-import them first."
    )
    if config.get("data_profile") == "local":
        scope = st.radio("Search in", ["My notes", "SEPP guides", "All documents"], horizontal=True)
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
    selected = st.sidebar.multiselect(
        "Restrict to specific documents (optional)",
        [d["document_id"] for d in candidates],
        format_func=lookup.get,
        max_selections=200,
    )
    top_k = st.sidebar.slider("Maximum displayed results", 1, 30, config["retrieval_top_k"])
    result_mode = st.radio(
        "Return",
        [
            "Automatic",
            "Matching passages",
            "Whole matching document",
            "Links / references",
            "Semantic suggestions",
        ],
        horizontal=True,
        key=f"result_mode_{config.get('data_profile')}",
    )
    mode_value = {
        "Automatic": "auto",
        "Matching passages": "passages",
        "Whole matching document": "documents",
        "Links / references": "links",
        "Semantic suggestions": "semantic",
    }[result_mode]
    scope_value = {"My notes": "notes", "SEPP guides": "pdf"}.get(scope, "all")
    request_signature = (
        scope_value,
        health.get("retrieval_engine_version", "legacy"),
        mode_value,
        top_k,
        tuple(sorted(selected or [d["document_id"] for d in candidates])),
    )
    with st.form("document_search"):
        question = st.text_input(
            "What would you like to find?", placeholder="Ask about a topic in your notes"
        )
        submitted = st.form_submit_button("Search documents", type="primary")
    if submitted:
        st.session_state.pop(f"document_search_result_{config.get('data_profile')}", None)
        if not question.strip():
            st.warning("Enter a question or a few search words.")
        elif not candidates:
            st.warning("No indexed documents in this category.")
        else:
            with st.spinner("Searching imported sources…"):
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
        and saved["scope"] == scope
        and saved.get("corpus_signature") == corpus_signature
        and saved.get("request_signature") == request_signature
    ):
        result = saved["result"]
        st.write("Search:", result["query"])
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
    if not saved or saved["result"].get("result_type") not in {"documents", "links"}:
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
        progress = st.progress(0)
        results = []
        for i, file in enumerate(uploads):
            with st.spinner(f"Indexing {file.name}"):
                result = api(
                    "POST", "/documents/upload", files=[("files", (file.name, file.getvalue()))]
                )
                if result:
                    results.extend(result)
            progress.progress((i + 1) / len(uploads))
        st.session_state["ingestion_results"] = results
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
else:
    selected = st.sidebar.multiselect(
        "Search these documents (empty = all)", list(lookup), format_func=lookup.get
    )
    top_k = st.sidebar.slider("Top-k passages", 1, 30, config["retrieval_top_k"])
    if section == "Chat With Documents":
        st.info(
            "Each question is independent. Displayed history is not sent as conversation context."
        )
        if not config["allow_gemini_document_context"]:
            st.warning(
                "Gemini context sharing is disabled. Local search is available in Search Documents."
            )
        history = st.session_state.setdefault("history", [])
        if st.sidebar.button("Clear chat history"):
            history.clear()
            st.rerun()
        for item in history:
            with st.chat_message("user"):
                st.write(item["question"])
            with st.chat_message("assistant"):
                st.write(item["answer"])
                show_sources(item)
        question = st.chat_input("Ask about your documents")
        if question:
            with st.spinner("Retrieving passages and asking Gemini…"):
                result = api(
                    "POST",
                    "/chat",
                    json=dict(question=question, top_k=top_k, document_ids=selected),
                )
            if result:
                history.append(result)
                st.rerun()
    else:
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
                        json=dict(
                            query=question, top_k=top_k, document_ids=selected, mode="semantic"
                        ),
                    )
                else:
                    result = api(
                        "POST",
                        "/chat",
                        json=dict(
                            question=question, top_k=top_k, document_ids=selected, debug=True
                        ),
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
