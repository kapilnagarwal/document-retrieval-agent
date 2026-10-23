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
