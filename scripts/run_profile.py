"""Run isolated learning or private-local data profiles."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def profile_environment(profile):
    if profile not in {"learning", "local"}:
        raise ValueError("Unknown profile")
    port = "8000" if profile == "learning" else "8001"
    return {
        "DATA_PROFILE": profile,
        "CHROMA_PERSIST_DIRECTORY": str(ROOT / "data" / profile / "chroma_db"),
        "UPLOAD_DIRECTORY": str(ROOT / "data" / profile / "uploads"),
        "ALLOW_GEMINI_DOCUMENT_CONTEXT": (
            "true"
            if profile == "learning"
            else os.getenv("LOCAL_ALLOW_GEMINI_DOCUMENT_CONTEXT", "false")
        ),
        "BACKEND_PORT": port,
        "BACKEND_URL": "http://127.0.0.1:" + port,
        "HF_HOME": str(ROOT / "data/model_cache"),
        "HF_HUB_DISABLE_XET": "1",
        "NO_PROXY": ",".join(
            dict.fromkeys(
                ["127.0.0.1", "localhost"]
                + [host.strip() for host in os.getenv("NO_PROXY", "").split(",") if host.strip()]
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=["learning", "local"])
    parser.add_argument("action", choices=["app", "backend", "frontend", "index", "index-notes"])
    parser.add_argument("--folder", type=Path, help="Source folder for an index action")
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    os.environ.update(profile_environment(args.profile))
    if args.action in {"index", "index-notes"}:
        os.environ["ALLOW_GEMINI_DOCUMENT_CONTEXT"] = "false"
    os.chdir(ROOT)
    if args.action == "app":
        command = [sys.executable, "-m", "scripts.run_app", args.profile]
    elif args.action == "backend":
        command = [sys.executable, "-m", "backend.main"]
    elif args.action == "frontend":
        command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "frontend/app.py",
            "--server.address",
            "127.0.0.1",
            "--server.port",
            "8501" if args.profile == "learning" else "8502",
            "--browser.gatherUsageStats",
            "false",
            "--server.headless",
            "true",
        ]
    elif args.action == "index-notes":
        if args.profile != "local":
            parser.error("Private notes may only be imported into the local profile")
        command = [sys.executable, "-m", "scripts.import_notes"]
        if args.folder:
            command.append(str(args.folder))
    else:
        folder = args.folder or (ROOT / "sample_documents" if args.profile == "learning" else None)
        if folder is None:
            parser.error("Private folder imports require --folder /path/to/documents")
        command = [sys.executable, "-m", "scripts.ingest_folder", str(folder)]
    if args.action in {"index", "index-notes"} and args.replace_existing:
        command.append("--replace-existing")
    # Replace the launcher so Ctrl-C and process termination reach the server directly.
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
