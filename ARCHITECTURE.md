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

The filesystem retains imported bytes. SQLite stores document metadata and lifecycle state. Chroma stores chunk text, vectors, and metadata. A collection namespace includes model and chunk settings. An RLock serializes storage work within one backend process.

Imports pass through `processing` to `ready`. An explicit replacement keeps stable `source_id`, creates a new immutable document ID/version, indexes new data first, and atomically commits new-ready/old-superseded metadata in SQLite. Failed indexing removes only the incomplete version. Superseded original bytes and records remain inspectable; old vectors are removed after publication. Startup cleans interrupted work and retries stale vector cleanup. The metadata migration removes the legacy unique-hash constraint so historical bytes can legitimately reappear in a later version.

No transaction spans filesystem, SQLite, and Chroma. SQLite connections have explicit transaction and close handling. Key/hash lookups use SQL indexes. Source-name mappings remain local and are inherited by replacement versions.

## Search paths

`POST /search` plans intent using explicit local rules. Source scope intersects requested document IDs with ready records. An empty scope uses a nonexistent sentinel rather than falling back to all documents.

Document, command, and saved-link finders read complete imported sources. Commands preserve shell continuations and exact versions. Links retain source-line provenance and are never fetched. Whole-document results identify candidates; their content is requested separately.

Passage retrieval prioritizes exact phrases and complete keyword matches, then combines BM25 and cosine ranks with reciprocal-rank fusion. The current implementation scans cached Chroma records in Python, not Chroma ANN for each query. Results carry evidence type and distance; distances are not probabilities.

## Answer path

`POST /chat` calls the passage pipeline directly. It snapshots active-document filtering, passages, content hashes, source identity, and versions under the ingestion lock. It releases the lock before generation, so a slow provider does not block local storage operations.

The evidence gate requires configured keyword coverage and exact literals within a passage. Only supported passages enter the prompt. Weak evidence yields `answer_status=insufficient_evidence` without a provider call. The gate is a transparent heuristic, not an entailment verifier; it can reject valid paraphrases.

Gemini generation checks local sharing consent and the key, limits application attempts, and returns token usage. Citation-ID warnings yield `unverified`; a normally cited answer yields `answered`; model abstention yields `insufficient_evidence`. Sources retain the captured document hash/version, so citations remain attributable if a source changes during the remote request.

## Profiles and verification

Learning and local profiles use separate storage. Local disables Gemini context. Neither has authentication or a multi-user authorization model. There is no implemented autonomous agent loop, OCR, conversation memory, or distributed locking.

Tests use temporary storage and deterministic doubles. The independent synthetic benchmark uses real MiniLM embeddings and measures recall, reciprocal rank, evidence acceptance/false support, and latency. It makes no Gemini requests and does not claim production-corpus accuracy.
