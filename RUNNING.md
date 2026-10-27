# Running the project

Activate `.venv` in each terminal. See [README.md](README.md) for installation.

## Optional isolated learning backend

```bash
python -m scripts.run_profile learning backend
```

The optional learning backend is available on port 8000 for isolated CLI/API experiments. It is not a selectable UI collection. The normal UI uses the main backend on port 8001. To deliberately point a development UI at another backend, set `LOCAL_BACKEND_URL` before launching it.

Stop the backend with Ctrl-C before CLI indexing:

```bash
python -m scripts.run_profile learning index
```

## Private collection

```bash
python -m scripts.run_profile local app
# Or run the backend and frontend separately in two terminals:
# python -m scripts.run_profile local backend
# python -m scripts.run_profile local frontend
```

API: http://127.0.0.1:8001. UI: http://127.0.0.1:8502. Gemini context is disabled by default. Each collection has a separate database and upload directory under ignored `data/`.

For generated answers, explicitly set `LOCAL_ALLOW_GEMINI_DOCUMENT_CONTEXT=true` in `.env` with a valid Gemini key/model, then restart the backend. **Start Chat** defaults to **AI answer** with a message box pinned to the bottom of the screen. Select the documents to use and enter a question or instruction. Questions and retrieved passages are sent to Google; supporting evidence is expandable below the answer. Other return modes use local retrieval in the same workspace. Setting the opt-in to `false` and restarting disables generation again. CLI indexing always disables generation for the import process.

Stop the private backend before importing:

```bash
python -m scripts.run_profile local index --folder /path/to/documents
python -m scripts.run_profile local index-notes --folder /path/to/notes
# Explicitly replace unique existing source names:
python -m scripts.run_profile local index --folder /path/to/documents --replace-existing
```

`NOTES_SOURCE_DIRECTORY` can configure the saved-note folder in `.env`. Folder replacement rejects ambiguous basenames. Notes replacement uses original relative names. Plain imports still deduplicate identical bytes and do not silently overwrite same-named sources.

## Check a running private backend

```bash
python -m scripts.verify_local_notes
```

This checks health, private sharing configuration, saved-note audit results, and strict missing-source behavior. It contains no private filenames or expected corpus counts and does not call Gemini. Pass `--backend-url` for another local endpoint.

## Follow-up questions

In **AI answer** mode, keep entering prompts in the bottom message box; press Enter or use the send arrow; earlier questions and answers remain visible. Up to four recent turns are sent to Gemini to resolve references before fresh document retrieval. Use **New conversation** to start over. Changing document selection or scope starts a separate conversation; importing a new document does not erase saved history. Use **Saved conversations → Open conversation** to restore a chat and its document filters after closing the browser. Export saves JSON including answer/evidence snapshots; deleting a chat does not delete its source documents. New questions use current active versions. Deleted source IDs remain restricted rather than silently widening the search.

Draft text streams immediately, with a visible verification-pending status. The final event replaces the draft after citation, technical-value, and model-audit checks. A correction may change the final wording. On a disconnected stream, reopen the saved conversation: the backend finishes and saves the answer. Failed/interrupted questions remain identifiable for retry. The API accepts an idempotent request ID to recover a committed answer without generating it twice.

## Model cache and network

Launchers use `data/model_cache`. Reuse it for an offline evaluation after the model has been downloaded:

```bash
HF_HOME="$PWD/data/model_cache" HF_HUB_OFFLINE=1 python -m scripts.evaluate_retrieval
```

Set `GEMINI_API_KEY` and a model available to your account in `.env` before chat. Launchers load local proxy/certificate settings. Never disable TLS verification to work around network errors. Restart backends after configuration changes; the launcher does not install background services or automatically restart existing processes.

## Reproducible checks

```bash
ruff check .
ruff format --check .
python -m pytest -q
python -m scripts.validate_offline
```

The real-embedding evaluation and live Gemini integration are separate opt-in workloads. Public Git content excludes `.env`, private imports, generated databases/reports, editor state, and local historical notes in `.local/`.


## Indexing, ownership, and recovery

Document Management uploads go through `POST /index-jobs`. Jobs are stored on disk before submission succeeds and run serially. The UI displays queued/running/complete/failed states and offers refresh/retry. Completed payloads are removed; failed payloads remain available for retry. On restart, running jobs return to queued; partial ingestion is cleaned up and replayed. A crash after publication is safe to replay because ingestion deduplicates content. Existing synchronous `/documents/upload` and replacement APIs remain available, but only queued uploads automatically replay after a crash.

A nonblocking OS file lease protects the entire data profile during backend operation, CLI indexing, and backup (macOS/Linux). Do not launch additional writers outside these entry points. The combined launcher waits up to 180 seconds for backend readiness, refuses occupied ports, and stops its own companion process if either service exits. Ctrl-C stops both. It does not install a system service or automatically restart a failed process.

Provider requests have an application wall-clock deadline (`GENERATION_TIMEOUT_SECONDS=100`, range 1–300), in addition to SDK socket timeouts and bounded retries. At most four provider operations may remain in flight; a stalled DNS/proxy connection cannot create unlimited workers. A failed support audit leaves the answer visibly unverified. Repair network configuration before retrying; local retrieval stays available without Gemini.

For a local-only run with an already cached embedding model:

```bash
HF_HUB_OFFLINE=1 python -m scripts.run_profile local app
```

## Offline backup and restore

Stop the profile first. Create a new backup directory outside its data folder:

```bash
python -m scripts.backup_profile local --output /path/to/new-backup
python -m scripts.backup_profile local --verify /path/to/new-backup
```

The backup includes originals, Chroma, manifests, indexing jobs and saved conversations; it excludes the shared model cache and `.env`. SHA-256 checks detect missing, changed, or extra files. Backups contain the same private content as the profile. A lease prevents copying while this backend or the supported CLI importer is writing.

To recover, verify the backup, stop the app, preserve the current `data/local` directory under a different name, and copy the verified backup into a new `data/local` directory. Keep the same embedding/chunk configuration and model cache. Restart and check health, counts, saved chats, and a known query before removing the preserved directory. Restore never requires deleting the existing copy first; this project does not perform an automatic destructive restore.

Startup health also reports missing original files and chunk-count mismatches as
`degraded`, preserving the affected records. The combined launcher refuses to
present a healthy UI for that state. Preserve the current data and restore from a
verified backup; do not delete a damaged profile before retaining a recoverable copy.


## Answer source and online comparison

In **AI answer**, leave **Answer using** on **Auto** for a conversational assistant. Reformatting, summarizing, translating or explaining a previous answer does not require another retrieval pass. New questions try the indexed documents; explicit document/source requests still abstain when unsupported. Other unsupported topics may use model knowledge and are labeled **General knowledge**. Choose **General chat** to bypass retrieval entirely, including when no documents exist. Choose **Documents only** to preserve the previous strict behavior.

Requests such as "compare it with public SEPP specs available online" route to **Web search** in Auto, or you can select Web search directly. This uses the configured Gemini model's [Google Search grounding](https://ai.google.dev/gemini-api/docs/generate-content/google-search), including provider-supplied source links, inline W citations and Search Suggestions. It requires model/project access to that tool and can incur search usage. Requests and recent conversation go to Google. No extra search API key is required by this implementation. If the tool is unavailable or returns no grounding, the app reports that limitation; it does not present model memory as a verified web comparison.

Saved chats retain each turn's answer origin and web sources. Up to four recent turns are included, with questions bounded to 8,000 characters and answers to 32,000 characters, so follow-ups can retain the end of long answer tables. Older turns remain saved and visible but are outside the provider's recent-history window.

Direct API clients can set `answer_mode` to `auto`, `documents`, `general`, or `web` on `/chat` and `/chat/stream`. The API default remains `documents` for backward compatibility; the UI explicitly sends `auto`. General/conversation turns bypass document citation auditing and do not display missing-document warnings. Online turns use provider grounding metadata and never borrow old S citation identifiers.
