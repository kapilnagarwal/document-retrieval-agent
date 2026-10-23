# Document Retrieval AI Agent

A Python learning project for local document search and retrieval-augmented generation (RAG). Import PDF, DOCX, TXT, or Markdown files, inspect matching evidence, and optionally ask Google Gemini to answer from retrieved passages.

The backend uses FastAPI, local CPU MiniLM embeddings, Chroma persistence, and a SQLite document manifest. Streamlit is a separate HTTP client. No OpenAI API, Docker, or cloud vector database is required.

## Features

- Hybrid passage retrieval with exact literals, BM25 keywords, cosine distance, and reciprocal-rank fusion.
- Complete-document, shell-command, and saved-URL lookup with original source evidence.
- Conservative evidence checks before generation, structured answer status, and citation-ID validation.
- Explicit document replacement with stable source identity and inspectable version history.
- Separate learning and private-local profiles. Private-local generation is disabled.
- A RAG debugger showing context, prompts, distances, timings, and token usage.
- Deterministic regression tests and a separate real-embedding retrieval benchmark.
- A file-by-file Python walkthrough for Java developers under [docs/python-for-java](docs/python-for-java/README.md).

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

## Run the learning profile

Run these in separate terminals with the virtual environment activated:

```bash
python -m scripts.run_profile learning backend
python -m scripts.run_profile learning frontend
```

Open http://127.0.0.1:8501 and explicitly select **Learning samples (Gemini chat)** in the collection selector. The shared UI initially selects the private collection. The learning backend is on port 8000; API docs are at http://127.0.0.1:8000/docs.

Import the synthetic samples while the corresponding backend is stopped:

```bash
python -m scripts.run_profile learning index
```

Local search needs no Gemini key. For chat, put your own `GEMINI_API_KEY` and an available `GEMINI_MODEL` in ignored `.env`. The learning launcher permits context sharing for that isolated collection. Chat sends the question and selected passages to Google; use documents you are permitted to share. Model availability and quota depend on your account. `python -m scripts.check_gemini` lists accessible generation models.

## Private local collection

```bash
python -m scripts.run_profile local backend
python -m scripts.run_profile local frontend
# Stop the local backend before CLI import:
python -m scripts.run_profile local index --folder /path/to/documents
```

The private backend uses port 8001, its own storage, and Gemini context disabled. Its frontend launches on 8502; either frontend can select either running backend. Set `NOTES_SOURCE_DIRECTORY` in local `.env` for saved-note audits/imports, or pass `--folder` to note imports. No private corpus is included in Git.

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

## Architecture and limits

See [ARCHITECTURE.md](ARCHITECTURE.md), [RUNNING.md](RUNNING.md), and [LEARNING_GUIDE.md](LEARNING_GUIDE.md).

This is a local, single-user learning application with one database owner per profile. It has no authentication, OCR, multi-process locking, conversation memory, or autonomous agent loop. Chroma persists embeddings; the current hybrid retriever scans cached records in Python. Chunk sizes are characters, not model tokens. Citation checks validate identifiers, not whether every claim follows from evidence. The lexical evidence gate is deliberately conservative and can reject paraphrases with little shared vocabulary; it is not a calibrated probability of correctness.

The chat pipeline snapshots evidence/version metadata under the storage lock and releases it before remote generation. A source deleted during generation can therefore still be cited as the captured historical snapshot.
