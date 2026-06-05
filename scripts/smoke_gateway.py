"""Gateway HTTP smoke test against a local Ollama (no SLURM, no tunnel).

Starts the gateway server in a background thread pointed at a local ollama host,
then exercises: web chat HTML, /health, /v1/models, Bearer auth (401/200),
/v1/chat/completions passthrough, and a real /chat agent round-trip.

Usage:
    python scripts/smoke_gateway.py [--model gemma4:e4b] [--ollama-host http://127.0.0.1:11434]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agentica_core  # noqa: F401
from agentica_core import gateway


def _req(url, method="GET", token=None, body=None, timeout=120):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4:e4b")
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--token", default="sk-smoke-123")
    ap.add_argument("--chat", action="store_true", help="Also do a real /chat agent round-trip (slower).")
    args = ap.parse_args()

    host = "127.0.0.1"
    base = f"http://{host}:{args.port}"
    app = gateway.build_app(ollama_host=args.ollama_host, model_name=args.model,
                            workspace="sample_workspace", db_path="/tmp/smoke-gateway.db",
                            auth_token=args.token)
    handler = gateway.make_gateway_handler(
        app, auth_token=args.token, remote_v1_base=args.ollama_host.rstrip("/") + "/v1",
        v1_mode="passthrough", model_label=args.model, public_api_base=base + "/v1")
    server = gateway.run_gateway_server(handler, host, args.port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.4)

    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")

    try:
        s, b = _req(f"{base}/health")
        check("/health 200", s == 200 and '"ok": true' in b)

        s, b = _req(f"{base}/")
        check("/ serves chat HTML", s == 200 and "slurm-open-agentic" in b)

        s, b = _req(f"{base}/v1/models")  # no token -> 401 (authed route)
        check("/v1/models requires auth", s == 401, f"got {s}")

        s, b = _req(f"{base}/v1/models", token=args.token)
        check("/v1/models 200 with token", s == 200 and args.model in b)

        s, b = _req(f"{base}/v1/chat/completions", method="POST", token=args.token,
                    body={"model": args.model, "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
                          "stream": False, "max_tokens": 8})
        passthrough_ok = s == 200 and '"choices"' in b
        check("/v1/chat/completions passthrough", passthrough_ok, f"status {s}")
        if passthrough_ok:
            print("    upstream reply:", json.loads(b)["choices"][0]["message"]["content"][:60].replace("\n", " "))

        if args.chat:
            s, b = _req(f"{base}/chat", method="POST", token=args.token,
                        body={"message": "Say hello in one short sentence."}, timeout=180)
            data = json.loads(b) if s == 200 else {}
            check("/chat agent round-trip", s == 200 and bool(data.get("final_answer")),
                  f"status {s}")
            if data.get("final_answer"):
                print("    agent:", data["final_answer"][:80].replace("\n", " "))
    finally:
        server.shutdown()

    print("RESULT:", "OK" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
