# Document Retrieval AI Agent

A Python learning project for local document search and retrieval-augmented generation (RAG). Import PDF, DOCX, TXT, or Markdown files, inspect matching evidence, and optionally ask Google Gemini to answer from retrieved passages.

The backend uses FastAPI, local CPU MiniLM embeddings, Chroma persistence, and a SQLite document manifest. Streamlit is a separate HTTP client. No OpenAI API, Docker, or cloud vector database is required.

## Features

- Hybrid passage retrieval with exact literals, BM25 keywords, cosine distance, and reciprocal-rank fusion.
- Complete-document, shell-command, and saved-URL lookup with original source evidence.
- Conservative evidence checks before generation, structured answer status, and citation-ID validation.
- Conversational follow-ups with bounded recent history, question resolution, and fresh document evidence for each answer.
- Explicit source-name filtering and PDF page/table context recovery for release-specific procedure details.
- Explicit document replacement with stable source identity and inspectable version history.
- Separate learning and private-local profiles. Private-local generation requires a separate opt-in.
- A RAG debugger showing context, prompts, distances, timings, and token usage.
- Deterministic regression tests and a separate real-embedding retrieval benchmark.
- Public architecture and learning documentation in [ARCHITECTURE.md](ARCHITECTURE.md) and [LEARNING_GUIDE.md](LEARNING_GUIDE.md).

## Install

Use Python 3.11 or newer; Python 3.13 is the locally verified runtime.

```bash
git clone git@github.com:kapilnagarwal/document-retrieval-agent.git
cd document-retrieval-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
# Create local settings only if they do not already exist:
test -f .env || cp .env.example .env
```

`requirements.txt` declares bounded runtime dependencies; `requirements.lock.txt` is the historical installed snapshot, not a cross-platform resolution guarantee. Configure proxy/certificate settings locally if required; retain TLS verification. The first embedding use may download MiniLM from Hugging Face.

## Run the app

Run these in separate terminals with the virtual environment activated:

```bash
python -m scripts.run_profile local backend
python -m scripts.run_profile local frontend
```

Open http://127.0.0.1:8502. The sidebar has one **Explore** menu: **Start Chat**, **Dashboard**, **Document Management**, and **RAG Debugger**. All documents use the main backend on port 8001; API docs are at http://127.0.0.1:8001/docs. There is no collection selector.

Import the synthetic samples while the corresponding backend is stopped:

```bash
python -m scripts.run_profile local index --folder sample_documents
```

Local search needs no Gemini key. For chat, put your own `GEMINI_API_KEY` and an available `GEMINI_MODEL` in ignored `.env`. The main launcher requires `LOCAL_ALLOW_GEMINI_DOCUMENT_CONTEXT=true` to enable generated answers. Chat sends the question and selected passages to Google; use documents you are permitted to share. Model availability and quota depend on your account. `python -m scripts.check_gemini` lists accessible generation models.

## Private local collection

```bash
python -m scripts.run_profile local backend
python -m scripts.run_profile local frontend
# Stop the local backend before CLI import:
python -m scripts.run_profile local index --folder /path/to/documents
```

The private backend uses port 8001, its own storage, and Gemini context disabled by default. Its frontend launches on 8502 and connects to the main backend by default; `LOCAL_BACKEND_URL` provides an explicit deployment override. Set `NOTES_SOURCE_DIRECTORY` in local `.env` for saved-note audits/imports, or pass `--folder` to note imports. No private corpus is included in Git.

To enable generated answers for the local collection, set `LOCAL_ALLOW_GEMINI_DOCUMENT_CONTEXT=true` in ignored `.env`, configure `GEMINI_API_KEY` and `GEMINI_MODEL`, and restart the local backend. This permits sending questions and retrieved passages to Google. **Start Chat** then defaults to **AI answer** with **Answer using → Auto**: enter a question or instruction in the bottom message box and press Enter or click its send arrow. Restrict the document selection as needed; supporting passages appear in an expander. Other return modes perform local retrieval. Remove the opt-in or set it to `false` and restart to disable generation.

## Replace a source safely

In **Document Management**, select an active document and expand **Source versions and replacement**. A successful replacement keeps its `source_id`, increments `version`, and marks the previous document `superseded`. Failed indexing preserves the previous active document. Old content remains inspectable; normal search uses active versions only.

```bash
curl -X PUT -F 'file=@updated-guide.pdf' http://127.0.0.1:8000/documents/DOCUMENT_ID
curl http://127.0.0.1:8000/documents/DOCUMENT_ID/versions
```

Folder imports support `--replace-existing` to replace a uniquely named active import. Ambiguous same-filename imports are rejected; select an explicit document ID through the API/UI instead. Historical duplicate imports are not silently merged. Retained versions use disk until explicitly deleted.

## Check quality

```bash
ruff check .
ruff format --check .
python -m pytest -q
python -m scripts.validate_offline
python -m scripts.evaluate_retrieval
```

Ordinary tests use temporary data and fake AI implementations, require no API key/model download, and skip the live Gemini test unless explicitly enabled. The separate evaluation uses real local MiniLM embeddings and an isolated synthetic corpus; it never calls Gemini. See [evaluation/README.md](evaluation/README.md) for metrics, thresholds, and limitations. GitHub Actions runs lint/format checks and tests on Python 3.11/3.13; manual workflow dispatch additionally runs real retrieval evaluation.

## Publication privacy

Enable the versioned pre-commit check once per clone:

```bash
git config core.hooksPath .githooks
python -m scripts.check_publication --staged --compare-local-secrets
python -m scripts.check_publication
```

The local check scans the Git index and can compare it to configured secret values without printing them. CI scans reachable file history and commit email metadata. It rejects common token/key formats, private artifacts, personal home paths, non-placeholder emails, and credentials in non-example URLs. These are preventive pattern checks, not a guarantee that every possible secret or personal detail will be recognized. Use GitHub noreply addresses for commit authorship and keep private source collections outside Git.

## Architecture and limits

See [ARCHITECTURE.md](ARCHITECTURE.md), [RUNNING.md](RUNNING.md), and [LEARNING_GUIDE.md](LEARNING_GUIDE.md).

This is a local, single-user application. It has an exclusive process lease per profile, durable SQLite conversations, a recoverable indexing queue, and streamed Gemini drafts followed by a checked final answer. Saved conversations can be reopened, exported, or deleted from Start Chat. Document questions retrieve current active document versions; stored answers retain their original evidence and version/hash snapshots. Up to four recent exchanges are used for follow-ups.

Answer verification checks citations and exact technical values against their cited passages, then asks Gemini to audit factual support and release/condition associations with verbatim evidence quotations. Failed or unavailable checks mark the answer `unverified`. A passing audit is not a guarantee of correctness. This remains a single-user tool without authentication, OCR, a distributed job queue, or an autonomous agent loop. The hybrid retriever scans cached records in Python; the lexical evidence gate can reject paraphrases with little shared vocabulary.

The chat pipeline snapshots evidence/version metadata under the storage lock and releases it before remote generation. A source deleted during generation can therefore still be cited as the captured historical snapshot.


## Quality and dependable operation

Start both services with readiness checks and coordinated shutdown:

```bash
python -m scripts.run_profile local app
```

Uploads in Document Management are queued durably; refresh their status and retry failed jobs there. Unfinished jobs replay after restart, with content deduplication preventing duplicate published documents. CLI imports and backups refuse to run while the same profile is open.

The representative evaluation contains 34 synthetic cases across normal responses, errors, field definitions, release differences, follow-ups, and unsupported requests. It also checks definitions split across chunks. Run:

```bash
HF_HOME="$PWD/data/model_cache" HF_HUB_OFFLINE=1 python -m scripts.evaluate_quality
# Optional: sends only synthetic fixtures to Gemini and incurs provider usage.
HF_HOME="$PWD/data/model_cache" HF_HUB_OFFLINE=1 python -m scripts.evaluate_quality --live
```

Offline results measure evidence retrieval and required-value coverage, not generated-answer quality. See [evaluation/README.md](evaluation/README.md) and [RUNNING.md](RUNNING.md) for live evaluation, backup, and recovery details.


## Conversation, general chat, and public web questions

**Answer using** has four choices within Start Chat:

- **Auto:** route explicit online requests to public web search, handle rewriting/summarizing follow-ups from conversation, and use document retrieval for other questions. If no document evidence matches a question that does not explicitly require documents, answer from general knowledge with a visible label.
- **Documents only:** retain strict RAG behavior; unsupported document questions abstain.
- **General chat:** use conversation and model knowledge without reading the document index. This works before any files are uploaded.
- **Web search:** use Gemini Google Search grounding with clickable public sources. An online comparison uses earlier conversation as comparison input, not as independently verified evidence. Missing grounding produces a warning rather than a claimed verified comparison.

You can ask "make it shorter", "put that in a table", or "compare it with public SEPP specs available online" in the same conversation. Document, general-knowledge, conversation and web answers are labeled separately. Web requests send the prompt and recent conversation to Google and can incur provider search usage. Keep **Documents only** selected when you want every factual answer restricted to your indexed documents.
