"""Scan Git content without printing secrets or private field values."""

import argparse
import re
import subprocess
from pathlib import PurePosixPath

PATTERNS = {
    "Google API key": re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    "GitHub token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "personal home path": re.compile(r"(?:/Users/|/home/|[A-Za-z]:\\Users\\)[A-Za-z0-9_.-]+[/\\]"),
    "internal domain": re.compile(r"\b[\w.-]*oraclecorp\.com\b", re.I),
}
EMAIL = re.compile(r"[A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,})", re.I)
URL_CREDENTIALS = re.compile(r"https?://[^\s/]+:[^\s/]+@([^\s/\"']+)", re.I)
PRIVATE_DIRECTORIES = {"data", ".local", ".venv", ".idea", ".oca", ".aws", ".ssh", "__pycache__"}


def git(*arguments: str) -> bytes:
    return subprocess.check_output(["git", *arguments])


def safe_email(address: str, metadata: bool = False) -> bool:
    if address.endswith("@users.noreply.github.com") or address == "noreply@github.com":
        return True
    if metadata:
        return False
    domain = address.rsplit("@", 1)[-1].lower()
    return (
        address == "git@github.com"
        or domain.endswith((".example", ".test", ".invalid"))
        or domain
        in {
            "example.com",
            "example.org",
            "example.net",
        }
    )


def scan_text(text: str, metadata: bool = False) -> list[str]:
    findings = [label for label, pattern in PATTERNS.items() if pattern.search(text)]
    if any(not safe_email(match.group(), metadata) for match in EMAIL.finditer(text)):
        findings.append("personal or non-placeholder email")
    for match in URL_CREDENTIALS.finditer(text):
        host = match.group(1).split(":", 1)[0].lower()
        if not host.endswith((".example", ".test")):
            findings.append("URL with embedded credentials")
    return findings


def private_path(name: str) -> bool:
    path = PurePosixPath(name)
    if path.parts[:2] == ("docs", "python-for-java"):
        return True
    if PRIVATE_DIRECTORIES.intersection(path.parts):
        return True
    return (
        (path.name.startswith(".env") and path.name != ".env.example")
        or path.name == "gemini-api-key.env"
        or path.suffix in {".pem", ".key", ".sqlite3", ".pyc", ".p12", ".pfx", ".jks"}
    )


def staged_blobs() -> list[tuple[str, str]]:
    entries = git("ls-files", "--stage", "-z").split(b"\0")
    return [
        (entry.split(b"\t", 1)[1].decode(), entry.split()[1].decode()) for entry in entries if entry
    ]


def history_blobs(revision: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    blobs = set()
    metadata = []
    for commit in git("rev-list", revision).decode().splitlines():
        metadata.append(
            (commit[:12], git("show", "-s", "--format=%an%n%ae%n%cn%n%ce", commit).decode())
        )
        for entry in git("ls-tree", "-r", "-z", commit).split(b"\0"):
            if entry:
                header, name = entry.split(b"\t", 1)
                _, kind, identifier = header.split()
                if kind == b"blob":
                    blobs.add((name.decode(), identifier.decode()))
    return sorted(blobs), metadata


def local_secrets() -> list[str]:
    # Optional exact-value comparison stays on the developer machine, never in logs.
    from dotenv import dotenv_values

    values = []
    for name in (".env", "gemini-api-key.env"):
        for key, value in dotenv_values(name).items():
            if key.upper() == "NO_PROXY":
                continue
            if (
                value
                and len(value) >= 8
                and any(
                    term in key.upper()
                    for term in ("KEY", "TOKEN", "SECRET", "PASSWORD", "PROXY", "CERT")
                )
            ):
                values.append(value)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staged", action="store_true", help="Scan the exact Git index before commit"
    )
    parser.add_argument(
        "--revision",
        default="HEAD",
        help="Scan all content and metadata reachable from this revision",
    )
    parser.add_argument("--compare-local-secrets", action="store_true")
    args = parser.parse_args()
    blobs, metadata = (staged_blobs(), []) if args.staged else history_blobs(args.revision)
    secrets = local_secrets() if args.compare_local_secrets else []
    findings = []
    for name, identifier in blobs:
        if private_path(name):
            findings.append((name, "private artifact"))
        text = git("cat-file", "blob", identifier).decode("utf-8", errors="replace")
        findings.extend((name, category) for category in scan_text(text))
        if any(secret in text for secret in secrets):
            findings.append((name, "exact match to a configured local secret/network setting"))
    for commit, text in metadata:
        findings.extend(
            ("commit " + commit, category) for category in scan_text(text, metadata=True)
        )
    for location, category in findings:
        print(f"BLOCKED: {location}: {category}")
    print(
        f"Publication scan: {len(blobs)} file snapshots, {len(metadata)} commits, {len(findings)} findings."
    )
    raise SystemExit(bool(findings))


if __name__ == "__main__":
    main()
