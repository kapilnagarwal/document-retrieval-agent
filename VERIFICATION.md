# Verification

## Automated checks

```bash
ruff check .
ruff format --check .
python -m pytest -q
python -m scripts.validate_offline
```

The tests cover extraction, chunk metadata, persistence, rollback/restart recovery, strict literal versions, command/link/document retrieval, request validation, UI result invalidation, provider errors, and citation IDs. Added regressions cover connection closure, legacy manifest migration, replacement rollback/history, stale-vector cleanup, concurrent generation/deletion with an evidence snapshot, negative-evidence abstention, and evaluation metrics.

Most tests use hash-based fake embeddings and a fake generator; these validate behavior, not learned semantic accuracy. The optional live Gemini test uses synthetic context, requires a configured key and explicit opt-in, and is skipped in ordinary CI. No private corpus or credentials are shipped.

## Real embeddings

```bash
python -m scripts.evaluate_retrieval
```

This uses an isolated temporary index and synthetic labeled inputs. Reports are written under ignored `data/evaluation/`. See [evaluation/README.md](evaluation/README.md) for metrics and the measured baseline. A passing tiny benchmark does not establish general document understanding or answer factuality.

GitHub Actions runs Python 3.11 and 3.13 checks when pushed. Local verification does not establish that a remote workflow has finished; inspect GitHub Actions for the remote result.

## Reliability and evaluation implementation — 2026-10-27

Implemented a 32-case representative synthetic suite across normal responses,
errors, fields, release differences, follow-ups, unsupported questions, and a
chunk-boundary field list. Real cached MiniLM evaluation passes all source and
required-evidence checks. The original 20-query benchmark also passes (recall/MRR
1.0, false-support rate 0, positive support 13/14). Offline follow-ups use labeled
standalone questions; this does not establish LLM rewrite or answer accuracy.

Added bounded adjacent context with contributing chunk IDs, per-citation technical
value checks, and a Gemini answer audit requiring exact source quotations. Added
SQLite saved conversations, request idempotency/concurrency controls, NDJSON draft
streaming and checked final responses. Recovery tests cover interrupted turns,
replayed indexing, duplicate publication, failed-job retry, corrupt queued payloads,
missing originals, exclusive profile ownership, and backup checksum failures.

Validation: `pytest -q`: **173 passed, 1 skipped**. Ruff checks/format, six standalone
offline checks, Git whitespace checks and publication scan pass. The combined
launcher was exercised on localhost 8001/8502. The real API smoke test saved a
streamed unsupported-query response, recovered it by request ID, reopened its
stored turn, and deleted only the temporary test conversation. Original corpus
counts remained **180 documents / 10,033 chunks**, with no startup integrity issues.

Live Gemini evaluation is **incomplete**: direct credential-free HTTPS connectivity
to Google's endpoint timed out, and a bounded synthetic generation request hit its
application deadline. `data/evaluation/quality-live-report.json` records that failure;
no live generated-answer score is claimed. Provider streaming is covered with an
SDK test double; local HTTP stream/persistence behavior was verified against the
running backend without sending private content to Google.


## Handshake answer correction regression — 2026-10-27

The reported question was answered with TLS certificate alerts while HTTP
capability-exchange failures were omitted. Broad code questions now retrieve
separate HTTP and TLS evidence within the same source scope. Code-list prompts
request compact tables, omitted numeric response codes trigger one correction,
and feedback such as "this is not the correct info" rechecks the original question
while preserving the user's named document. Incomplete provider output is rejected
rather than published as a complete answer. Audit quotes normalize PDF whitespace
without changing codes or identifiers. Streaming renders the new user/assistant
pair once in the transcript rather than rendering a second final pair.

The 34-case real-MiniLM offline evaluation passes. Full tests: 177 passed, 1 skipped.
A separate local-only test ingested the actual uploaded NG.146 PDF into temporary
storage and verified that both the original question and correction include 400
NOK, 403 NOK, R16, R17+, and TLS bad_certificate evidence, exclusively from NG.146.
The live Gemini recheck was blocked by automatic approval review because it would
transmit NG.146 passages externally; no live corrected-answer claim is made.
Existing saved answers remain historical snapshots and are not overwritten.

## Conversational/general/web routing — 2026-10-27

The UI now sends `answer_mode=auto`; API clients keep strict document mode by
default. General chat bypasses retrieval, conversation transformations use recent
answers directly, and explicit online comparisons invoke the configured Gemini
Google Search grounding tool. Answer origin is stored and displayed. General turns
have no document-citation warnings; web turns use provider metadata for W links and
Search Suggestions. Missing grounding or tool availability is reported explicitly.
The recent-history answer bound increased from 4,000 to 32,000 characters to retain
long tables for comparison; at most four exchanges are still used.

Validation: **190 tests passed, 1 skipped**; six standalone offline checks, Ruff and
Git whitespace checks pass. Regression tests cover the reported online SEPP
comparison wording, no-retrieval follow-ups, strict-mode abstention, general chat
without indexed documents, source labeling, web metadata/citations, no-grounding
failure, saved/streamed non-document turns, and long prior answers.

A live web smoke test sent only a public RFC 9110 question with empty history and
no document context. It returned HTTP 503 after the Gemini application deadline;
no live web-grounding success is claimed. No private corpus or saved conversation
was transmitted by that test. The existing index remains 180 documents and 10,033
chunks. Browser verification confirmed the Auto/Documents only/General chat/Web
search selector and a single bottom composer on the updated running app.
