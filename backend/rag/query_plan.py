"""Explicit local query planning. No LLM call, remote search, or hidden document instructions."""

import re

from backend.rag.document_finder import document_phrase, wants_document
from backend.rag.lexical import query_tokens, required_literals

ENGINE_VERSION = "planned-evidence-v2"
NATURAL_QUESTION = re.compile(
    r"^\s*(?:please\s+)?(?:(?:check|tell\s+me|show\s+me|find\s+out)\s+)?"
    r"(?:how|why|what|when|where|which|who|explain)\b",
    re.I,
)
PROVIDERS = {"confluence": r"\bconfluence\b", "jira": r"\bjira\b", "gitlab": r"\bgitlab\b"}
REFERENCE = re.compile(
    r"\b(?:links?|urls?|references?|webpages?)\b|\b(?:confluence|jira|gitlab)\s+pages?\b", re.I
)
TOOLS = re.compile(r"\b(?:podman|docker|kubectl|helm|git|curl|gradle|mvn)\b", re.I)
FILLER = {
    "related",
    "relevant",
    "associated",
    "regarding",
    "pages",
    "page",
    "links",
    "link",
    "urls",
    "url",
    "references",
    "reference",
    "webpages",
    "webpage",
    "information",
    "details",
    "notes",
    "note",
    "files",
    "file",
    "documents",
    "document",
    "text",
    "content",
}


def plan_query(query, mode="auto"):
    provider = next(
        (name for name, pattern in PROVIDERS.items() if re.search(pattern, query, re.I)), None
    )
    filename_match = re.search(
        r'\b(?:file|note|document)\s+(?:(?:named|called)\s+)?["`]?([^\n"`/]{1,120}?\.(?:txt|md|pdf|docx))\b',
        query,
        re.I,
    )
    filename = filename_match.group(1).strip() if filename_match else None
    if filename is None:
        bare = re.sub(r"^\s*(?:find|show|read|open)\s+", "", query, flags=re.I).strip(' "`')
        if re.fullmatch(r"[^/\n]{1,120}\.(?:txt|md|pdf|docx)", bare, re.I) and not re.match(
            r"(?:podman|docker|kubectl|helm|git|curl|gradle|mvn)\s+", bare, re.I
        ):
            filename = bare
    if mode == "documents" or (mode == "auto" and (wants_document(query) or filename)):
        intent = "documents"
        subject = filename or document_phrase(query)
    elif mode == "links" or (
        mode == "auto"
        and (REFERENCE.search(query) or (provider and re.search(r"\bpages?\b", query, re.I)))
    ):
        intent = "links"
        subject = " ".join(
            word for word in query_tokens(query) if word not in FILLER and word not in PROVIDERS
        )
    else:
        intent = (
            "commands"
            if TOOLS.search(query) or re.search(r"\b(?:cmds?|commands?)\b", query, re.I)
            else "passages"
        )
        cleaned = document_phrase(query) if wants_document(query) else query
        subject = " ".join(word for word in query_tokens(cleaned) if word not in FILLER)
    if intent == "links":
        subject = re.sub(r"\bcnc\s*console\b", "cncc", subject, flags=re.I)
    return dict(
        intent=intent,
        subject=subject,
        provider=provider,
        filename=filename,
        required_terms=required_literals(query),
        all_requested=bool(re.search(r"\ball\b", query, re.I)),
        natural_question=bool(NATURAL_QUESTION.match(query)),
        scope="imported_sources_only",
        engine_version=ENGINE_VERSION,
    )


def scoped_document_ids(records, requested_ids, scope):
    allowed = set(requested_ids or [])
    identifiers = [
        record["document_id"]
        for record in records
        if (not allowed or record["document_id"] in allowed)
        and (
            scope == "all"
            or (scope == "notes" and not record["filename"].lower().endswith(".pdf"))
            or (scope == "pdf" and record["filename"].lower().endswith(".pdf"))
        )
    ]
    # An empty selection must never be reinterpreted by lower layers as 'all documents'.
    return identifiers or ["__no_selected_imported_documents__"]
