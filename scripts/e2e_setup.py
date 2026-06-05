"""Workload e2e for setup + streaming against a live API server + local ollama.

Exercises: /api/setup readiness, /api/ollama/start (idempotent), token-streaming
/api/chat/stream, and /api/model/pull progress (on an already-present model).

    python scripts/e2e_setup.py --model gemma4:e4b
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agentica_core  # noqa: F401
from agentica_core import apiserver

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:22s} {detail}")


def post_stream(url, body, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="POST" if body is not None else "GET")
    events = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass
    return events


def get_json(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4:e4b")
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    ap.add_argument("--port", type=int, default=8779)
    args = ap.parse_args()

    state = apiserver.State(ollama_host=args.ollama_host, model=args.model,
                            workspace="sample_workspace", db_path=tempfile.mktemp(suffix=".db"))
    from http.server import ThreadingHTTPServer
    server = ThreadingHTTPServer(("127.0.0.1", args.port), apiserver.make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.4)
    base = f"http://127.0.0.1:{args.port}"

    try:
        s = get_json(f"{base}/api/setup")
        check("setup ready", s.get("ready") is True, f"running={s.get('ollama_running')} model_present={s.get('model_present')}")

        start = post_stream(f"{base}/api/ollama/start", {}, timeout=40) if False else None  # POST json endpoint
        # /api/ollama/start returns JSON, not SSE:
        st = json.loads(urllib.request.urlopen(
            urllib.request.Request(f"{base}/api/ollama/start", data=b"{}",
                                   headers={"Content-Type": "application/json"}, method="POST")).read())
        check("ollama start idempotent", st.get("ok") is True, st.get("message", ""))

        deltas = [e["delta"] for e in post_stream(f"{base}/api/chat/stream",
                  {"message": "Reply with exactly: pong", "mode": "plain"}, timeout=120) if "delta" in e]
        text = "".join(deltas)
        check("chat stream tokens", len(deltas) >= 1 and bool(text.strip()), f"{len(deltas)} chunks -> {text[:40]!r}")

        pull = post_stream(f"{base}/api/model/pull?model={args.model}", None, timeout=120)
        check("model pull progress", any(e.get("done") or e.get("status") == "success" for e in pull),
              f"{len(pull)} events")
    finally:
        server.shutdown()

    npass = sum(1 for r in results if r[1])
    print(f"\n=== {npass}/{len(results)} setup-workload checks passed ===")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
