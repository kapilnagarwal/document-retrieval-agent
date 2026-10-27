"""Create or verify an offline profile backup; never overwrite an existing backup."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from uuid import uuid4

from backend.state import ProcessLease
from scripts.run_profile import ROOT


def hashes(directory):
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and path.name != "backup-manifest.json"
        and not path.name.endswith(".lock")
    }


def backup(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if not source.is_dir() or destination.exists() or destination.is_relative_to(source):
        raise ValueError("Require an existing profile and a new backup directory outside it")
    temporary = destination.with_name(destination.name + ".partial-" + str(uuid4()))
    with ProcessLease(source / "uploads" / ".owner.lock"):
        try:
            shutil.copytree(source, temporary, ignore=shutil.ignore_patterns("*.lock"))
            (temporary / "backup-manifest.json").write_text(
                json.dumps(hashes(temporary), indent=2) + "\n"
            )
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return destination


def verify(directory):
    expected = json.loads((directory / "backup-manifest.json").read_text())
    if hashes(directory) != expected:
        raise ValueError("Backup verification failed: files are missing, changed, or unexpected")
    return len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=["local", "learning"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    try:
        if args.verify:
            print(f"Verified {verify(args.verify)} files")
        elif args.output:
            backup(ROOT / "data" / args.profile, args.output)
            print(f"Backup created and verified: {verify(args.output)} files")
        else:
            parser.error("Use --output NEW_DIRECTORY or --verify BACKUP_DIRECTORY")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
