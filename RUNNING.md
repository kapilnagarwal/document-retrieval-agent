# Running the project

Activate `.venv` in each terminal. See [README.md](README.md) for installation.

## Learning collection

```bash
python -m scripts.run_profile learning backend
python -m scripts.run_profile learning frontend
```

API: http://127.0.0.1:8000. UI: http://127.0.0.1:8501. Select Learning samples in the UI collection selector; the initial selection is private-local.

Stop the backend with Ctrl-C before CLI indexing:

```bash
python -m scripts.run_profile learning index
```

## Private collection

```bash
python -m scripts.run_profile local backend
python -m scripts.run_profile local frontend
```

API: http://127.0.0.1:8001. UI: http://127.0.0.1:8502. Gemini context remains disabled. Each collection has a separate database and upload directory under ignored `data/`.

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
