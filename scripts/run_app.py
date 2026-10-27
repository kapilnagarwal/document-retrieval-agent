"""Start one profile's backend and UI; stop owned children together on exit."""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time

import httpx
from dotenv import load_dotenv

from scripts.run_profile import ROOT, profile_environment


def port_open(port):
    with socket.socket() as connection:
        return connection.connect_ex(("127.0.0.1", port)) == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=["local", "learning"], nargs="?", default="local")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    environment = {**os.environ, **profile_environment(args.profile)}
    backend_port = int(environment["BACKEND_PORT"])
    frontend_port = 8502 if args.profile == "local" else 8501
    if any(port_open(port) for port in (backend_port, frontend_port)):
        parser.error(
            "A configured port is already in use. Stop the existing app first; this launcher never kills unrelated processes."
        )
    children = []

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        backend = subprocess.Popen(
            [sys.executable, "-m", "backend.main"], cwd=ROOT, env=environment
        )
        children.append(backend)
        deadline = time.monotonic() + 180
        with httpx.Client(trust_env=False, timeout=2) as client:
            while True:
                if backend.poll() is not None:
                    raise RuntimeError("Backend exited during startup; inspect the error above.")
                try:
                    response = client.get(environment["BACKEND_URL"] + "/health")
                    if response.status_code == 200:
                        if response.json().get("status") == "degraded":
                            raise RuntimeError(
                                "Storage integrity check failed. Preserve current data and restore a verified backup before starting the UI."
                            )
                        if response.json().get("status") == "ok":
                            break
                except (httpx.HTTPError, ValueError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        "Backend did not become ready in 180 seconds. Check model cache and disk space."
                    )
                time.sleep(0.5)
        # The unified UI connects to this explicitly selected backend.
        environment["LOCAL_BACKEND_URL"] = environment["BACKEND_URL"]
        children.append(
            subprocess.Popen(
                [sys.executable, "-m", "scripts.run_profile", args.profile, "frontend"],
                cwd=ROOT,
                env=environment,
            )
        )
        print(f"Ready: http://127.0.0.1:{frontend_port}/ (Ctrl-C stops both services)", flush=True)
        while all(child.poll() is None for child in children):
            time.sleep(0.5)
        raise RuntimeError(
            "A service exited. Stopping its companion; restart after correcting the error."
        )
    except KeyboardInterrupt:
        pass
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
        for child in reversed(children):
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    main()
