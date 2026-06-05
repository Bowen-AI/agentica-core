"""Online end-to-end harness: run the 10 example tasks against a local Ollama.

Mode 2 (agentic JOB: Planner->Executor->Auditor + deterministic backstops) and
Mode 1 (gateway chat / OpenAI /v1), all with real inference. Each task has a
deterministic verifier. Exposes JOBS + run_single_job() for the benchmark driver.

    python scripts/e2e.py --model gemma4:e4b
    python scripts/e2e.py --model qwen3.5:9b --only jobs
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
from agentic_loop.ollama_model import OllamaChatModel
from agentica_core import gateway
from agentica_core.config import PlanConfig, SuccessCriteria
from agentica_core.on_node_runner import run_job


def _w(ws: Path, rel: str, content: str) -> None:
    p = ws / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


# --------------------------------------------------------------------------- #
# The 10 example tasks (Mode 2 agentic jobs; deterministic test = the gate)
# --------------------------------------------------------------------------- #
JOBS = [
    dict(id="J1-hello", title="write a file with exact content",
         goal="Use the write_file tool to create a file named hello.txt whose entire content is exactly: hello",
         setup=None, tests="grep -qx hello hello.txt", artifacts=["hello.txt"]),
    dict(id="J2-add", title="implement add(a,b)",
         goal="Create src/add.py defining a function add(a, b) that returns a + b. Use the write_file tool.",
         setup=None,
         tests="python3 -c \"import sys;sys.path.insert(0,'src');from add import add;assert add(2,3)==5;print('ok')\"",
         artifacts=["src/add.py"]),
    dict(id="J3-greet", title="implement greet(name)",
         goal=("Create src/greet.py defining greet(name) that returns the string 'Hello, ' + name + '!'. "
               "Use the write_file tool."),
         setup=None,
         tests="python3 -c \"import sys;sys.path.insert(0,'src');from greet import greet;assert greet('Bob')=='Hello, Bob!';print('ok')\"",
         artifacts=["src/greet.py"]),
    dict(id="J4-json", title="write a JSON config",
         goal=("Use write_file to create data.json containing a JSON object with a single key "
               "\"version\" whose value is the number 1."),
         setup=None,
         tests="python3 -c \"import json;assert json.load(open('data.json'))['version']==1;print('ok')\"",
         artifacts=["data.json"]),
    dict(id="J5-reverse", title="implement reverse(s)",
         goal="Create src/reverse.py defining reverse(s) that returns the reversed string. Use the write_file tool.",
         setup=None,
         tests="python3 -c \"import sys;sys.path.insert(0,'src');from reverse import reverse;assert reverse('abc')=='cba';print('ok')\"",
         artifacts=["src/reverse.py"]),
    dict(id="J6-fixbug", title="fix a bug so the test passes",
         goal=("There is a bug in src/buggy.py: inc(x) must return x + 1 but currently returns x - 1. "
               "Read the file and fix it (write_file or run_shell)."),
         setup=lambda ws: _w(ws, "src/buggy.py", "def inc(x):\n    return x - 1\n"),
         tests="python3 -c \"import sys;sys.path.insert(0,'src');from buggy import inc;assert inc(5)==6;print('ok')\"",
         artifacts=["src/buggy.py"]),
    dict(id="J7-readme", title="create a markdown README",
         goal="Use write_file to create README.md whose first line is exactly: # Project",
         setup=None, tests="head -1 README.md | grep -qx '# Project'", artifacts=["README.md"]),
    dict(id="J8-count", title="count lines and write the number",
         goal=("Count the number of lines in input.txt and write ONLY that number (nothing else) into out.txt. "
               "You may use run_shell."),
         setup=lambda ws: _w(ws, "input.txt", "alpha\nbeta\ngamma\n"),
         tests="grep -qx 3 out.txt", artifacts=["out.txt"]),
    dict(id="J9-fizzbuzz", title="implement fizzbuzz(n)",
         goal=("Create src/fb.py defining fizzbuzz(n): 'Fizz' if divisible by 3 only, 'Buzz' if by 5 only, "
               "'FizzBuzz' if by both, else str(n). Use write_file."),
         setup=None,
         tests=("python3 -c \"import sys;sys.path.insert(0,'src');from fb import fizzbuzz as f;"
                "assert f(3)=='Fizz' and f(5)=='Buzz' and f(15)=='FizzBuzz' and f(7)=='7';print('ok')\""),
         artifacts=["src/fb.py"]),
    dict(id="J10-video", title="generate a video artifact (stub backend)",
         goal=("Use the generate_video tool with prompt 'clouds over mountains' and out='output/clip.mp4' "
               "to produce the artifact."),
         setup=None, tests="test -s output/clip.mp4", artifacts=["output/clip.mp4"]),
]


def _run_with_timeout(fn, timeout: float):
    box: dict = {}

    def target():
        try:
            box["result"] = fn()
        except Exception as exc:  # noqa: BLE001
            box["error"] = exc

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        raise TimeoutError(f"task exceeded {timeout}s")
    if "error" in box:
        raise box["error"]
    return box["result"]


def run_single_job(spec: dict, model: str, host: str, timeout: float = 240.0,
                   max_iterations: int = 2, max_steps: int = 6) -> dict:
    """Run one job task end-to-end and return {passed, secs, detail}."""
    ws = Path(tempfile.mkdtemp(prefix=f"e2e-{spec['id']}-"))
    if spec["setup"]:
        spec["setup"](ws)
    plan = PlanConfig(
        title=spec["title"], goal=spec["goal"], workspace=str(ws), checklist=[spec["title"]],
        success_criteria=SuccessCriteria(tests=spec["tests"], artifacts=spec["artifacts"]),
        max_iterations=max_iterations, max_steps_per_iteration=max_steps,
    )
    t0 = time.monotonic()
    passed, detail = False, ""
    try:
        mdl = OllamaChatModel(model=model, host=host, timeout_s=90)
        out = _run_with_timeout(
            lambda: run_job(plan, workspace=str(ws), db_path=str(ws / "job.db"),
                            provider="ollama", model_name=model, ollama_host=host,
                            model=mdl, _print=lambda *a: None),
            timeout=timeout)
        passed = bool(out.passed)
        detail = f"verdict={out.verdict} tests_ok={out.tests_ok} arts_ok={out.artifacts_ok} iters={out.iterations}"
    except TimeoutError:
        detail = "TIMEOUT"
    except Exception as exc:  # noqa: BLE001
        detail = f"ERROR {type(exc).__name__}: {exc}"
    return {"passed": passed, "secs": time.monotonic() - t0, "detail": detail}


def run_jobs(model: str, host: str, results: list) -> None:
    for spec in JOBS:
        r = run_single_job(spec, model, host)
        results.append(("job", spec["id"], spec["title"], r["passed"], r["secs"], r["detail"]))
        print(f"  [{'PASS' if r['passed'] else 'FAIL'}] {spec['id']:12s} {r['secs']:5.1f}s  {r['detail']}")


# --------------------------------------------------------------------------- #
# Mode 1 chat / API
# --------------------------------------------------------------------------- #
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


def run_chats(model: str, host: str, token: str, port: int, results: list) -> None:
    app = gateway.build_app(ollama_host=host, model_name=model, workspace="sample_workspace",
                            db_path=tempfile.mktemp(suffix=".db"), auth_token=token)
    base = f"http://127.0.0.1:{port}"
    handler = gateway.make_gateway_handler(
        app, auth_token=token, remote_v1_base=host.rstrip("/") + "/v1",
        v1_mode="passthrough", model_label=model, public_api_base=base + "/v1")
    server = gateway.run_gateway_server(handler, "127.0.0.1", port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.4)

    def record(cid, title, passed, t0, detail):
        results.append(("chat", cid, title, passed, time.monotonic() - t0, detail))
        print(f"  [{'PASS' if passed else 'FAIL'}] {cid:12s} {time.monotonic()-t0:5.1f}s  {detail}")

    try:
        t0 = time.monotonic()
        s, b = _req(f"{base}/chat", "POST", token, {"message": "What is 2+2? Reply with just the number."}, 180)
        ans = json.loads(b).get("final_answer", "") if s == 200 else ""
        record("C1-math", "agent /chat arithmetic", "4" in ans, t0, f"ans={ans[:40]!r}")

        t0 = time.monotonic()
        s, b = _req(f"{base}/v1/chat/completions", "POST", token,
                    {"model": model, "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
                     "stream": False}, 120)
        txt = json.loads(b)["choices"][0]["message"]["content"] if s == 200 else ""
        record("C2-v1pass", "/v1 passthrough", "pong" in txt.lower(), t0, f"reply={txt[:40]!r}")

        t0 = time.monotonic()
        s, b = _req(f"{base}/v1/chat/completions", "POST", token,
                    {"model": "agentic", "messages": [{"role": "user", "content": "Capital of France? One word."}],
                     "stream": False}, 180)
        txt = json.loads(b)["choices"][0]["message"]["content"] if s == 200 else ""
        record("C3-v1agent", "/v1 agentic mode", "paris" in txt.lower(), t0, f"reply={txt[:40]!r}")

        t0 = time.monotonic()
        s, b = _req(f"{base}/chat", "POST", token, {"message": "Remember my favorite color is teal."}, 180)
        sid = json.loads(b).get("session_id") if s == 200 else None
        s, b = _req(f"{base}/chat", "POST", token,
                    {"message": "What is my favorite color? One word.", "session_id": sid}, 180)
        ans = json.loads(b).get("final_answer", "") if s == 200 else ""
        record("C4-memory", "session memory recall", "teal" in ans.lower(), t0, f"ans={ans[:40]!r}")
    finally:
        server.shutdown()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4:e4b")
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    ap.add_argument("--only", choices=["jobs", "chat"], default=None)
    ap.add_argument("--port", type=int, default=8801)
    ap.add_argument("--token", default="sk-e2e")
    args = ap.parse_args()

    print(f"=== e2e model={args.model} host={args.ollama_host} ===\n")
    results: list = []
    if args.only != "chat":
        print("MODE 2 -- agentic jobs:")
        run_jobs(args.model, args.ollama_host, results)
        print()
    if args.only != "jobs":
        print("MODE 1 -- gateway chat / OpenAI /v1:")
        run_chats(args.model, args.ollama_host, args.token, args.port, results)
        print()

    npass = sum(1 for r in results if r[3])
    print("=" * 60)
    print(f"SUMMARY: {npass}/{len(results)} passed")
    for mode, cid, title, passed, secs, detail in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {mode:4s} {cid:12s} {secs:5.1f}s  {title}")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
