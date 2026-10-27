# Document retrieval architecture

The application separates HTTP presentation, document lifecycle, local retrieval, and optional external generation. Components are explicitly connected in `backend/services.py`; tests inject fake embeddings and generation.

```text
Streamlit HTTP client
  → FastAPI routes and validated Pydantic contracts
    → Ingestion → extraction → chunking → local embeddings → Chroma
    → Search planning → complete-source finders or hybrid passage retrieval
    → Chat → active evidence snapshot → evidence gate → Gemini → citation checks
```

## Document storage and lifecycle

The filesystem retains imported bytes. SQLite stores document metadata and lifecycle state. Chroma stores chunk text, vectors, and metadata. A collection namespace includes model and chunk settings. An RLock serializes storage work within one backend process. Backend lifespan and supported CLI import/backup entry points acquire an exclusive OS lease before opening the profile.

Imports pass through `processing` to `ready`. An explicit replacement keeps stable `source_id`, creates a new immutable document ID/version, indexes new data first, and atomically commits new-ready/old-superseded metadata in SQLite. Failed indexing removes only the incomplete version. Superseded original bytes and records remain inspectable; old vectors are removed after publication. Startup cleans interrupted work and retries stale vector cleanup. The metadata migration removes the legacy unique-hash constraint so historical bytes can legitimately reappear in a later version.

No transaction spans filesystem, SQLite, and Chroma. SQLite connections have explicit transaction and close handling. Key/hash lookups use SQL indexes. Source-name mappings remain local and are inherited by replacement versions.

## Search paths

`POST /search` plans intent using explicit local rules. Source scope intersects requested document IDs with ready records. An empty scope uses a nonexistent sentinel rather than falling back to all documents.

Document, command, and saved-link finders read complete imported sources. Commands preserve shell continuations and exact versions. Links retain source-line provenance and are never fetched. Whole-document results identify candidates; their content is requested separately.

Passage retrieval prioritizes exact phrases and complete keyword matches, then combines BM25 and cosine ranks with reciprocal-rank fusion. The current implementation scans cached Chroma records in Python, not Chroma ANN for each query. Results carry evidence type and distance; distances are not probabilities.

## Answer path

`POST /chat` calls the passage pipeline directly. It snapshots active-document filtering, passages, content hashes, source identity, and versions under the ingestion lock. It releases the lock before generation, so a slow provider does not block local storage operations.

The evidence gate requires configured keyword coverage and exact literals within a passage. Only supported passages enter the prompt. Weak evidence yields `answer_status=insufficient_evidence` without a provider call. The gate is a transparent heuristic, not an entailment verifier; it can reject valid paraphrases.

Explicit source names in chat questions (full filenames, stems, or document identifiers present in filenames) narrow the allowed document scope; a conflicting document filter produces an empty scope. For supported PDF hits, answer context recovers the complete original page and its immediate neighbors, including table rows reconstructed with strict line detection. This preserves scenario headings, release conditions, response bodies, and table continuations without reindexing. Context is limited to 12 pages and 24,000 characters; whole pages are skipped rather than truncated, with the original hit retained if expansion is unavailable. Page citations identify the actual expanded page; seed chunk similarity is not presented as a page similarity score. This improves context coverage but does not guarantee exhaustive retrieval or perfect reconstruction of every PDF table.

For follow-ups, `/chat` accepts up to four recent question/answer turns, with up to 8,000 characters per question and 32,000 per answer. Gemini resolves a standalone retrieval question outside the storage lock. Current exact literals are retained even if the rewrite omits them. Retrieval and the evidence gate use the resolved question; generation receives the original follow-up, history as conversational reference data, and newly retrieved supported passages. Earlier assistant text is not factual evidence, and citations refer only to current sources. Follow-ups may therefore use one provider call for question resolution even if the subsequent evidence gate abstains; empty document scopes skip both calls. Token usage and generation time include question resolution.

Explicit source or technical-entity questions without references such as "those" skip conversational rewriting and omit earlier turns from the generation prompt. This prevents a new contents question from inheriting a previous error-only topic. Contents/schema intent for named entities restricts retrieval to exact entity mentions and prioritizes nearby field declarations or definition text over bare mentions in error examples. This is a local heuristic, not a general schema parser; generation describes fields supported by the retrieved examples and must not present them as an exhaustive schema.

Entity detail requests such as "give details for this" are recognized as contents intent. Presentation wording is removed from the retrieval/evidence query while the original question is retained for generation. Quoted phrases and technical qualifiers remain constraints, so request wording does not create a false evidence failure or silently remove a requested version or field.

Gemini generation checks local sharing consent and the key, limits application attempts, and returns token usage. Uppercase underscore-separated identifiers in a draft (for example, protocol cause names) are checked against current reference text. Unsupported identifiers trigger one correction call; persistent mismatches produce an `unverified` warning. Correction usage and time are included in totals. Per-paragraph/table-row checks also compare CamelCase identifiers, HTTP error codes and release labels with the actually cited passages. A separate Gemini audit checks answer focus and factual associations and must supply exact quotations found in current evidence. Unavailable, malformed or negative audits mark the result unverified. These are conservative checks, not proof of semantic entailment. Citation-ID warnings also yield `unverified`; a normally cited answer yields `answered`; model abstention yields `insufficient_evidence`. Sources retain the captured document hash/version, so citations remain attributable if a source changes during the remote request.

## Profiles and verification

Learning and local profiles use separate storage. Local disables Gemini context by default; `LOCAL_ALLOW_GEMINI_DOCUMENT_CONTEXT=true` explicitly enables Gemini chat for that collection. The UI displays the sharing state and exposes chat only when enabled for the local profile. The UI exposes one main document collection without a collection selector. Its Explore menu contains Start Chat, Dashboard, Document Management, and RAG Debugger. Start Chat combines conversation and retrieval modes; LOCAL_BACKEND_URL is an explicit deployment override. The isolated learning profile remains available for CLI/API experiments. SQLite conversations retain scope/document filters, ordered turns, request IDs, status, and complete answer/evidence snapshots. Scope changes start another chat; corpus updates use fresh retrieval without erasing history. Stream requests return NDJSON status/delta/reset/final/error events. Drafts are unverified until final; browser disconnection does not cancel persistence. Idempotent request IDs recover completed responses; incomplete turns become interrupted at startup. Neither profile has authentication or a multi-user authorization model. There is no implemented autonomous agent loop, OCR, multi-user authorization or distributed locking.

Tests use temporary storage and deterministic doubles. The independent synthetic benchmark uses real MiniLM embeddings and measures recall, reciprocal rank, evidence acceptance/false support, and latency. It makes no Gemini requests and does not claim production-corpus accuracy.


Non-PDF evidence expansion includes up to two adjacent chunks on either side within a 12,000-character budget, retaining contributing chunk IDs. This recovers field lists split by character chunking while preserving document boundaries and source version metadata. PDF expansion keeps its existing page/table budget.

`IndexJobs` persists queued upload bytes and SQLite state before returning HTTP 202. A serial worker publishes through the existing transactional manifest lifecycle. Running jobs are requeued at startup; a published duplicate is harmless on replay. Failed jobs retain payloads and can retry. Lifespan shutdown waits for the current indexing operation and stream workers before releasing the lease. This is a local queue, not a distributed task system.

The representative 34-case evaluation reports results per requested category. Offline mode measures labeled source recall, required-value evidence coverage, and out-of-domain evidence rejection; follow-ups use labeled standalone queries. `--live` exercises the actual rewrite, generation, audit and abstention paths with lexical answer assertions. Failed provider connectivity produces an explicit incomplete report. Neither synthetic score nor a model audit establishes production-corpus accuracy.


## Hybrid chat routing

`ChatRequest.answer_mode` defaults to `documents` for existing API clients. The UI defaults to `auto`; explicit `general` and `web` modes also exist. `chat_routing.py` detects public-online intent and conversational transformations before retrieval. Auto otherwise follows the document pipeline; an insufficient evidence gate can fall back to general knowledge only when the current question does not explicitly require document evidence. Strict Documents only behavior is unchanged. Routing is a transparent local heuristic, not a general-purpose autonomous planner.

General/conversation turns use a separate conversational system prompt and no vector search or document citation audit. Old S identifiers are removed from the conversational reference data. Responses carry `answer_origin`, persist in the same conversation, and stream through the existing NDJSON protocol. These answers are labeled rather than presented as document-verified.

Web turns enable the SDK Google Search tool. Provider grounding metadata supplies source URLs, supported text segments and Search Suggestions. Citation insertion validates segment offsets/text; invalid or missing grounding leaves the answer unverified, and no-source responses are replaced with a clear limitation. An unsupported tool produces an actionable provider error. These checks establish citation provenance, not correctness of every web claim. UI Search Suggestions use Streamlit's sanitized HTML renderer. Web mode sends recent conversation to Google as comparison input; users see the transmission/usage notice next to the source selector.
