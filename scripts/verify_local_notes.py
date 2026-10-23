"""Read-only smoke checks for a running private-local backend."""

import argparse
import json
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-url", default="http://127.0.0.1:8001")
    parser.add_argument("--output", type=Path, default=Path("data/local-smoke-report.json"))
    args = parser.parse_args()
    with httpx.Client(base_url=args.backend_url, trust_env=False, timeout=30) as client:
        health = client.get("/health")
        health.raise_for_status()
        config = client.get("/config")
        config.raise_for_status()
        audit = client.get("/sources/notes/audit")
        audit.raise_for_status()
        empty = client.post("/search", json={"query": "find file missing-smoke-source-85931.txt"})
        empty.raise_for_status()
    counts = audit.json()["counts"]
    checks = {
        "backend_healthy": health.json()["status"] == "ok",
        "private_profile": config.json()["data_profile"] == "local",
        "gemini_context_disabled": config.json()["allow_gemini_document_context"] is False,
        "saved_folder_exists": audit.json()["folder_exists"],
        "saved_imports_current": counts["changed_since_import"] == counts["not_imported"] == 0,
        "missing_source_not_substituted": empty.json().get("documents") == [],
    }
    report = {
        "checks": checks,
        "passed": all(checks.values()),
        "scope": "Read-only localhost checks; no Gemini calls or private source content in this report.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
