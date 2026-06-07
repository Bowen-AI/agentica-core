"""agentica-core JSON API -- the backend the Agentica (React) UI talks to.

Endpoints (all JSON, CORS-enabled for the Vite dev server):

  GET  /api/health
  GET  /api/hosts                      -> ssh-config hosts + a "local" target
  POST /api/chat       {message, mode: agentic|plain, session_id?, workspace?}
  GET  /api/history?session_id=&workspace=
  POST /api/plan/draft {goal, workspace?}                 -> {title, lines[], tests, artifacts}
  POST /api/plan/refine{plan, comments[]}                 -> updated plan
  POST /api/job/submit {plan, target}                     -> {job_id, jobdir, target, local}
  GET  /api/job/status?target=&job=&jobdir=&local_id=
  GET  /api/job/logs?...

Chat reuses the gateway's AgentServerApp (agentic loop, tools, memory). Plain chat
+ planning are direct ollama /v1 completions. Jobs go local (background thread) or
remote (job.submit over ssh/SLURM, any ~/.ssh/config target).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import gateway, job, sshconfig
from .config import ClusterConfig, PlanConfig, SuccessCriteria
from .on_node_runner import run_job

__version__ = "0.1.0"


# --------------------------------------------------------------------------- #
# cluster targets (so the UI can drive SLURM, not just bare ssh aliases)
# --------------------------------------------------------------------------- #
def load_clusters(clusters_dir: str | None) -> dict[str, dict]:
    """Scan a folder of cluster.yaml files into {name: {path, host, scheduler}} so
    the UI can offer SLURM/ssh clusters (which carry account/partition/setup) as job
    targets. Files that don't parse as a ClusterConfig (e.g. plan.yaml) are skipped."""
    d = (clusters_dir or os.environ.get("AGENTICA_CLUSTERS_DIR")
         or os.path.expanduser("~/.config/agentica/clusters"))
    out: dict[str, dict] = {}
    root = Path(d)
    if not root.is_dir():
        return out
    for f in sorted([*root.glob("*.yaml"), *root.glob("*.yml")]):
        try:
            cfg = ClusterConfig.load(f)
        except Exception:  # noqa: BLE001 - skip non-cluster yaml / unreadable files
            continue
        out[cfg.name] = {"path": str(f), "host": cfg.ssh.host, "scheduler": cfg.scheduler}
    return out


# --------------------------------------------------------------------------- #
# in-memory state
# --------------------------------------------------------------------------- #
class State:
    def __init__(self, *, ollama_host: str, model: str, workspace: str, db_path: str,
                 clusters_dir: str | None = None):
        self.ollama_host = ollama_host.rstrip("/")
        self.model = model
        self.workspace = workspace
        self.db_path = db_path
        self._apps: dict[str, object] = {}          # workspace -> AgentServerApp (agentic chat)
        self.local_jobs: dict[str, dict] = {}        # local_id -> {status, outcome, ...}
        self._model_resolved = False
        self.clusters = load_clusters(clusters_dir)  # name -> {path, host, scheduler}

    def cluster_path(self, target: str) -> str:
        """Map a target name to its cluster.yaml path if it's a known cluster, else
        return the target unchanged (a bare ~/.ssh/config alias, or 'local')."""
        c = self.clusters.get(target)
        return c["path"] if c else target

    def resolve_model(self) -> str:
        """Effective model for inference. If the configured model isn't installed
        but Ollama has others, fall back to a present one (preferring the same
        family, then a qwen*, then the first installed) and memoize it -- so chat
        and jobs work with whatever the user actually has instead of dead-ending
        on a 404 'model not found'. When Ollama is down/empty we keep the
        configured model untouched (the Setup flow guides the user to it)."""
        if self._model_resolved:
            return self.model
        models = ollama_models(self.ollama_host)
        if not models:
            return self.model
        if not model_present(self.model, models):
            family = self.model.split(":")[0]
            self.model = (next((m for m in models if m.split(":")[0] == family), None)
                          or next((m for m in models if m.startswith("qwen")), None)
                          or models[0])
        self._model_resolved = True
        return self.model

    def app_for(self, workspace: str | None):
        ws = workspace or self.workspace
        if ws not in self._apps:
            self._apps[ws] = gateway.build_app(
                ollama_host=self.ollama_host, model_name=self.resolve_model(),
                workspace=ws, db_path=self.db_path, auth_token=None)
        return self._apps[ws]

    def complete(self, messages: list[dict], temperature: float = 0.2, timeout: float = 180) -> str:
        """Plain (non-agentic) chat completion via the engine's OpenAI /v1."""
        url = self.ollama_host + "/v1/chat/completions"
        body = json.dumps({"model": self.resolve_model(), "messages": messages,
                           "temperature": temperature, "stream": False}).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except (urllib.error.URLError, TimeoutError) as exc:
            reason = getattr(exc, "reason", None) or str(exc) or "request timed out"
            raise RuntimeError(f"model unreachable at {self.ollama_host}: {reason}")


# --------------------------------------------------------------------------- #
# workspace context + planning
# --------------------------------------------------------------------------- #
def workspace_summary(ws: str, max_files: int = 40, max_bytes: int = 1500) -> str:
    root = Path(ws)
    if not root.exists():
        return ""
    lines = [f"Workspace: {root}"]
    files = [p for p in sorted(root.rglob("*")) if p.is_file()][:max_files]
    for p in files:
        rel = p.relative_to(root)
        lines.append(f"- {rel}")
    # include a small preview of a few text files
    for p in files[:6]:
        try:
            txt = p.read_text(encoding="utf-8")[:max_bytes]
        except Exception:
            continue
        lines.append(f"\n### {p.relative_to(root)}\n{txt}")
    return "\n".join(lines)


def draft_plan(state: State, goal: str, workspace: str | None) -> dict:
    ctx = workspace_summary(workspace) if workspace else ""
    sys = ("You are a planning assistant. Given a goal, produce a concrete, ordered, checkable "
           "plan. The plan is carried out by an AUTONOMOUS agent that has file read/write and "
           "shell tools -- NOT a human at a GUI. Steps must be concrete agent actions (write a "
           "file, run a command); never 'open a text editor', 'save the file', or manual GUI steps. "
           "Reply STRICTLY as JSON: {\"title\": str, \"steps\": [str, ...], "
           "\"tests\": str (a shell command that genuinely verifies success via exit code -- e.g. "
           "grep/diff/an assertion script, NOT a bare echo/print that always succeeds, or \"\"), "
           "\"artifacts\": [str, ...]}. Steps are short imperative lines. No prose outside JSON.")
    user = (f"Goal:\n{goal}\n\n" + (f"Workspace context:\n{ctx}\n\n" if ctx else "")
            + "Return the JSON plan.")
    raw = state.complete([{"role": "system", "content": sys}, {"role": "user", "content": user}])
    parsed = _extract_json(raw) or {}
    steps = parsed.get("steps") or _fallback_steps(raw)
    # The model authored the tests string -> NOT a vetted gate (see _plan_payload).
    return _plan_payload(parsed.get("title") or goal[:60], goal, steps,
                         parsed.get("tests", ""), parsed.get("artifacts", []),
                         tests_authoritative=False)


def refine_plan(state: State, plan: dict, comments: list[dict]) -> dict:
    steps = [ln["text"] for ln in plan.get("lines", [])]
    rendered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    cmts = "\n".join(
        f"- on '{c.get('anchor') or ('line ' + str(c.get('line', '?')))}': {c.get('text', '')}"
        for c in comments) or "(no comments)"
    sys = ("Revise the plan to incorporate the user's comments. Reply STRICTLY as JSON: "
           "{\"title\": str, \"steps\": [str], \"tests\": str, \"artifacts\": [str]}.")
    user = (f"Goal:\n{plan.get('goal', '')}\n\nCurrent plan:\n{rendered}\n\n"
            f"Tests: {plan.get('tests', '')}\nArtifacts: {plan.get('artifacts', [])}\n\n"
            f"User comments:\n{cmts}\n\nReturn the revised JSON plan.")
    raw = state.complete([{"role": "system", "content": sys}, {"role": "user", "content": user}])
    parsed = _extract_json(raw) or {}
    new_steps = parsed.get("steps") or steps
    return _plan_payload(parsed.get("title") or plan.get("title", ""), plan.get("goal", ""),
                         new_steps, parsed.get("tests", plan.get("tests", "")),
                         parsed.get("artifacts", plan.get("artifacts", [])),
                         tests_authoritative=False)


# Shell fragments that indicate a test actually *verifies* something (can fail on bad work).
_VERIFY_TOKENS = ("test ", "[ ", "[[", "grep", "diff", "cmp", "assert", "pytest",
                  "unittest", "python", "node", "exit 1", "|| exit", "return 1")


def _test_is_trivial(cmd: str) -> bool:
    """A model-drafted 'tests' string that cannot meaningfully fail (e.g. `true`, a bare
    `echo ok`) would rubber-stamp wrong work if used as the gate -- treat it as no test."""
    c = (cmd or "").strip()
    if not c or c in ("true", ":", "exit 0", "/bin/true"):
        return True
    return not any(tok in c.lower() for tok in _VERIFY_TOKENS)


def _plan_payload(title, goal, steps, tests, artifacts, *, tests_authoritative: bool = True) -> dict:
    test_cmd = (tests or "").strip()
    # Drop an obviously vacuous model-drafted test so it can't masquerade as a real gate.
    if test_cmd and not tests_authoritative and _test_is_trivial(test_cmd):
        test_cmd = ""
    return {
        "title": title, "goal": goal,
        "lines": [{"id": f"L{i+1}", "text": str(s)} for i, s in enumerate(steps) if str(s).strip()],
        "tests": test_cmd,
        # Whether the tests command is a vetted gate. Model-drafted tests are NOT vetted;
        # the UI should let the user review/replace them before relying on the auto-pass.
        "tests_authoritative": bool(tests_authoritative and test_cmd),
        "artifacts": list(artifacts or []),
    }


def _extract_json(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _fallback_steps(text: str) -> list[str]:
    out = []
    for ln in text.splitlines():
        s = ln.strip().lstrip("-*0123456789.) ").strip()
        if s:
            out.append(s)
    return out[:12] or ["Accomplish the goal."]


# --------------------------------------------------------------------------- #
# jobs
# --------------------------------------------------------------------------- #
def plan_to_yaml(plan: dict, workspace: str) -> str:
    import yaml
    doc = {
        "title": plan.get("title", "agentica job"),
        "goal": plan.get("goal", ""),
        "workspace": workspace,
        "checklist": [ln["text"] for ln in plan.get("lines", [])],
        "success_criteria": {"tests": plan.get("tests", "") or None,
                              "artifacts": plan.get("artifacts", [])},
        "max_iterations": 3,
        "max_steps_per_iteration": 12,
    }
    return yaml.safe_dump(doc, sort_keys=False)


def submit_local(state: State, plan: dict, workspace: str) -> dict:
    """Run the agentic job in a background thread against local ollama."""
    local_id = "local-" + uuid.uuid4().hex[:8]
    ws = workspace or tempfile.mkdtemp(prefix="agentica-job-")
    Path(ws).mkdir(parents=True, exist_ok=True)
    pc = PlanConfig(
        title=plan.get("title", "job"), goal=plan.get("goal", ""), workspace=ws,
        checklist=[ln["text"] for ln in plan.get("lines", [])],
        success_criteria=SuccessCriteria(tests=plan.get("tests") or None,
                                         artifacts=plan.get("artifacts", [])),
        max_iterations=3, max_steps_per_iteration=12)
    rec = {"local_id": local_id, "target": "local", "status": "running",
           "workspace": ws, "log": [], "outcome": None}
    state.local_jobs[local_id] = rec

    def runner():
        def note(msg):
            rec["log"].append(msg)
        try:
            outcome = run_job(pc, workspace=ws, db_path=str(Path(ws) / "job.db"),
                              provider="ollama", model_name=state.resolve_model(),
                              ollama_host=state.ollama_host, model_timeout=300, _print=note)
            rec["outcome"] = outcome.to_dict()
            rec["status"] = "passed" if outcome.passed else "failed"
        except Exception as exc:  # noqa: BLE001
            rec["status"] = "error"
            rec["log"].append(f"ERROR {type(exc).__name__}: {exc}")

    threading.Thread(target=runner, daemon=True).start()
    return {"local": True, "local_id": local_id, "target": "local", "workspace": ws}


def submit_remote(plan: dict, target: str, cluster_path: str, workspace: str) -> dict:
    """Write a plan.yaml and submit to a ssh/SLURM target via job.submit. ``target``
    is the display name shown/polled by the UI; ``cluster_path`` is the cluster.yaml
    (full SLURM config) or a bare ssh alias to actually connect with."""
    tmp = Path(tempfile.mkdtemp(prefix="agentica-plan-"))
    plan_path = tmp / "plan.yaml"
    plan_path.write_text(plan_to_yaml(plan, workspace or "./workspace"), encoding="utf-8")
    captured: list[str] = []
    rc = job.submit(cluster_path, str(plan_path), sync_code=True, _print=captured.append)
    jobid = jobdir = None
    for line in captured:
        if "job_id=" in line:
            for tok in line.split():
                if tok.startswith("job_id="):
                    jobid = tok.split("=", 1)[1]
                if tok.startswith("jobdir="):
                    jobdir = tok.split("=", 1)[1]
    return {"local": False, "target": target, "rc": rc, "job_id": jobid, "jobdir": jobdir,
            "output": captured}


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# ollama setup / install / models  (defensive: these never raise to the caller)
# --------------------------------------------------------------------------- #
OLLAMA_HOME = Path(os.path.expanduser("~/.local/share/agentica/ollama"))


def ollama_reachable(host: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(host.rstrip("/") + "/api/tags", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def ollama_models(host: str, timeout: float = 4.0) -> list[str]:
    try:
        with urllib.request.urlopen(host.rstrip("/") + "/api/tags", timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception:
        return []
    names = {m.get("name") or m.get("model") for m in data.get("models", [])}
    return sorted(n for n in names if n)


def find_ollama_bin() -> str | None:
    rootless = OLLAMA_HOME / "bin" / "ollama"
    return shutil.which("ollama") or (str(rootless) if rootless.exists() else None)


def model_present(model: str, models: list[str]) -> bool:
    return (model in models or f"{model}:latest" in models
            or (":" not in model and any(m.split(":")[0] == model for m in models)))


def setup_status(state: "State") -> dict:
    running = ollama_reachable(state.ollama_host)
    models = ollama_models(state.ollama_host) if running else []
    present = model_present(state.model, models)
    return {
        "ollama_installed": bool(find_ollama_bin()) or running,
        "ollama_running": running,
        "models": models,
        "model": state.model,
        "model_present": present,
        "ready": running and present,
    }


def start_ollama(host: str, timeout_s: float = 25.0) -> tuple[bool, str]:
    if ollama_reachable(host):
        return True, "already running"
    binp = find_ollama_bin()
    if not binp:
        return False, "ollama is not installed"
    p = urlparse(host)
    env = dict(os.environ)
    env["OLLAMA_HOST"] = f"{p.hostname or '127.0.0.1'}:{p.port or 11434}"
    lib = OLLAMA_HOME / "lib"
    if lib.exists():
        env["LD_LIBRARY_PATH"] = f"{lib}:{env.get('LD_LIBRARY_PATH', '')}"
    try:
        subprocess.Popen([binp, "serve"], env=env, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    except Exception as exc:  # noqa: BLE001
        return False, f"could not start ollama: {exc}"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if ollama_reachable(host):
            return True, "started"
        time.sleep(0.5)
    return False, "started but not reachable yet"


def _zstd_decompress(src: Path, dst: Path) -> bool:
    try:
        import zstandard  # type: ignore

        with open(src, "rb") as f, open(dst, "wb") as o:
            zstandard.ZstdDecompressor().copy_stream(f, o)
        return True
    except ImportError:
        pass
    if shutil.which("zstd"):
        return subprocess.run(["zstd", "-dqf", str(src), "-o", str(dst)]).returncode == 0
    return False


def install_ollama_rootless(progress) -> bool:
    """Rootless Ollama install into ~/.local (no sudo). Reports progress via the
    callback; designed to NEVER raise -- failures are reported and return False."""
    try:
        if find_ollama_bin():
            progress("Ollama is already available.")
            return True
        arch = "amd64" if os.uname().machine in ("x86_64", "amd64") else "arm64"
        url = f"https://github.com/ollama/ollama/releases/latest/download/ollama-linux-{arch}.tar.zst"
        OLLAMA_HOME.mkdir(parents=True, exist_ok=True)
        tarzst, tar = OLLAMA_HOME / "ollama.tar.zst", OLLAMA_HOME / "ollama.tar"
        progress(f"Downloading rootless Ollama ({arch})...")
        urllib.request.urlretrieve(url, tarzst)
        progress("Decompressing...")
        if not _zstd_decompress(tarzst, tar):
            progress("ERROR: need `zstd` or python `zstandard` to unpack Ollama — "
                     "install one, or get Ollama from https://ollama.com/download")
            return False
        progress("Extracting...")
        subprocess.run(["tar", "-xf", str(tar), "-C", str(OLLAMA_HOME)], check=True)
        for f in (tarzst, tar):
            try:
                f.unlink()
            except OSError:
                pass
        if not (OLLAMA_HOME / "bin" / "ollama").exists():
            progress("ERROR: extraction did not produce bin/ollama")
            return False
        progress("Installed rootless Ollama into ~/.local/share/agentica/ollama")
        return True
    except Exception as exc:  # noqa: BLE001
        progress(f"ERROR: install failed: {exc}")
        return False


def make_handler(state: State):
    class H(BaseHTTPRequestHandler):
        server_version = f"agentica-core/{__version__}"

        def log_message(self, *a):
            return

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _json(self, payload, status=200):
            raw = json.dumps(payload, default=str).encode()
            self.send_response(status)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.end_headers()

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

        # -- SSE streaming --
        def _sse_start(self):
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

        def _sse(self, obj):
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode("utf-8"))
            self.wfile.flush()

        def _stream_pull(self, model):
            self._sse_start()
            try:
                req = urllib.request.Request(
                    state.ollama_host.rstrip("/") + "/api/pull",
                    data=json.dumps({"name": model, "stream": True}).encode(),
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=3600) as resp:
                    for raw in resp:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line:
                            continue
                        try:
                            evt = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        self._sse(evt)
                        if evt.get("status") == "success" or evt.get("error"):
                            break
                self._sse({"done": True})
            except Exception as exc:  # noqa: BLE001
                self._safe_sse({"error": str(exc), "done": True})

        def _stream_install(self):
            self._sse_start()
            ok = install_ollama_rootless(lambda m: self._safe_sse({"status": m}))
            if ok:
                started, msg = start_ollama(state.ollama_host)
                self._safe_sse({"status": msg, "ok": started})
            self._safe_sse({"done": True, "ok": ok, **setup_status(state)})

        def _stream_chat(self, body):
            self._sse_start()
            message = (body.get("message") or "").strip()
            mode = body.get("mode", "agentic")
            ws = body.get("workspace")
            try:
                if mode == "plain":
                    msgs = []
                    if ws:
                        ctx = workspace_summary(ws)
                        if ctx:
                            msgs.append({"role": "system", "content": "Workspace context:\n" + ctx})
                    msgs.append({"role": "user", "content": message})
                    req = urllib.request.Request(
                        state.ollama_host.rstrip("/") + "/v1/chat/completions",
                        data=json.dumps({"model": state.resolve_model(), "messages": msgs, "stream": True}).encode(),
                        headers={"Content-Type": "application/json"}, method="POST")
                    with urllib.request.urlopen(req, timeout=600) as resp:
                        for raw in resp:
                            line = raw.decode("utf-8", "replace").strip()
                            if not line.startswith("data: "):
                                continue
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                            except json.JSONDecodeError:
                                continue
                            d = (chunk.get("choices") or [{}])[0].get("delta", {})
                            # Reasoning models (e.g. qwen3.5) stream a long `reasoning`
                            # trace with empty `content` while thinking -- forward it as a
                            # distinct event so the UI shows live "thinking" instead of a
                            # frozen empty bubble, then stream the answer as `content` lands.
                            if d.get("reasoning"):
                                self._sse({"reasoning": d["reasoning"]})
                            if d.get("content"):
                                self._sse({"delta": d["content"]})
                    self._sse({"done": True})
                else:
                    app = state.app_for(ws)
                    res = app.chat(message, body.get("session_id"))
                    self._sse({"final": res.get("final_answer"), "steps": res.get("steps", []),
                               "session_id": res.get("session_id"), "done": True})
            except Exception as exc:  # noqa: BLE001
                self._safe_sse({"error": str(exc), "done": True})

        def _safe_sse(self, obj):
            try:
                self._sse(obj)
            except Exception:  # noqa: BLE001 - client disconnected
                pass

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path == "/api/health":
                    return self._json({"ok": True, "version": __version__, "model": state.model})
                if u.path == "/api/setup":
                    return self._json(setup_status(state))
                if u.path == "/api/model/pull":
                    return self._stream_pull((q.get("model") or [state.model])[0])
                if u.path == "/api/hosts":
                    hosts = [{"alias": "local", "hostname": "this machine", "user": "",
                              "proxy_jump": "", "kind": "local"}]
                    hosts += [{**h, "kind": "ssh"} for h in sshconfig.list_hosts()]
                    # Configured clusters (carry SLURM account/partition/setup) as targets.
                    hosts += [{"alias": name, "hostname": c["host"], "user": "", "proxy_jump": "",
                               "kind": "cluster", "scheduler": c["scheduler"]}
                              for name, c in sorted(state.clusters.items())]
                    return self._json({"hosts": hosts, "default_model": state.model})
                if u.path == "/api/history":
                    sid = (q.get("session_id") or [""])[0]
                    ws = (q.get("workspace") or [None])[0]
                    app = state.app_for(ws)
                    msgs = app.storage.load_messages(sid) if sid else []
                    return self._json({"messages": [{"role": m.role, "content": m.content}
                                                    for m in msgs if m.role in ("user", "assistant")]})
                if u.path == "/api/job/status":
                    return self._json(self._job_status(q))
                if u.path == "/api/job/logs":
                    return self._json(self._job_logs(q))
                return self._json({"error": "not found"}, 404)
            except Exception as exc:  # noqa: BLE001
                return self._json({"error": type(exc).__name__, "message": str(exc)}, 500)

        def do_POST(self):
            u = urlparse(self.path)
            try:
                body = self._body()
                if u.path == "/api/chat":
                    return self._json(self._chat(body))
                if u.path == "/api/chat/stream":
                    return self._stream_chat(body)
                if u.path == "/api/ollama/start":
                    ok, msg = start_ollama(state.ollama_host)
                    return self._json({"ok": ok, "message": msg, **setup_status(state)})
                if u.path == "/api/ollama/install":
                    return self._stream_install()
                if u.path == "/api/plan/draft":
                    return self._json(draft_plan(state, body.get("goal", ""), body.get("workspace")))
                if u.path == "/api/plan/refine":
                    return self._json(refine_plan(state, body.get("plan", {}), body.get("comments", [])))
                if u.path == "/api/job/submit":
                    return self._json(self._submit(body))
                return self._json({"error": "not found"}, 404)
            except Exception as exc:  # noqa: BLE001
                return self._json({"error": type(exc).__name__, "message": str(exc)}, 500)

        # -- handlers --
        def _chat(self, body):
            message = (body.get("message") or "").strip()
            if not message:
                raise ValueError("message required")
            mode = body.get("mode", "agentic")
            ws = body.get("workspace")
            if mode == "plain":
                msgs = []
                if ws:
                    ctx = workspace_summary(ws)
                    if ctx:
                        msgs.append({"role": "system",
                                     "content": "Use this workspace as context:\n" + ctx})
                msgs.append({"role": "user", "content": message})
                return {"mode": "plain", "final_answer": state.complete(msgs), "steps": []}
            app = state.app_for(ws)
            res = app.chat(message, body.get("session_id"))
            return {"mode": "agentic", "session_id": res.get("session_id"),
                    "final_answer": res.get("final_answer"), "steps": res.get("steps", []),
                    "transcript": res.get("transcript", [])}

        def _submit(self, body):
            plan = body.get("plan") or {}
            target = body.get("target", "local")
            ws = body.get("workspace") or plan.get("workspace") or ""
            if target == "local":
                return submit_local(state, plan, ws)
            # Map a cluster-name target to its cluster.yaml (account/partition/setup);
            # a bare ssh alias passes through unchanged.
            return submit_remote(plan, target, state.cluster_path(target), ws)

        def _job_status(self, q):
            local_id = (q.get("local_id") or [None])[0]
            if local_id:
                rec = state.local_jobs.get(local_id)
                if not rec:
                    return {"error": "unknown local job"}
                return {"status": rec["status"], "outcome": rec["outcome"],
                        "log": rec["log"][-30:], "target": "local", "workspace": rec["workspace"]}
            target = (q.get("target") or [""])[0]
            jid = (q.get("job") or [""])[0]
            jobdir = (q.get("jobdir") or [None])[0]
            # Structured status (status/outcome/lines) so the UI's poll loop can
            # terminate on passed/failed/error for remote jobs just like local ones.
            # Map a cluster-name target back to its cluster.yaml to connect.
            return job.status_struct(state.cluster_path(target), jid, jobdir=jobdir)

        def _job_logs(self, q):
            local_id = (q.get("local_id") or [None])[0]
            if local_id:
                rec = state.local_jobs.get(local_id, {})
                return {"log": rec.get("log", [])}
            target = (q.get("target") or [""])[0]
            jid = (q.get("job") or [""])[0]
            jobdir = (q.get("jobdir") or [None])[0]
            out: list[str] = []
            job.logs(target, jid, jobdir=jobdir, _print=out.append)
            return {"log": out}

    return H


def serve(*, host: str = "127.0.0.1", port: int = 8770, workspace: str = "sample_workspace",
          db_path: str = ".agentic/agentica.db", ollama_host: str = "http://127.0.0.1:11434",
          model: str = "qwen3.5:4b-mlx", clusters_dir: str | None = None) -> int:
    state = State(ollama_host=ollama_host, model=model, workspace=workspace, db_path=db_path,
                  clusters_dir=clusters_dir)
    server = ThreadingHTTPServer((host, port), make_handler(state))
    print(f"agentica-core API on http://{host}:{port}  (model={model}, ollama={ollama_host})")
    if state.clusters:
        print(f"  cluster targets: {sorted(state.clusters)}")
    print(f"  Agentica UI dev server should call this base URL.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0
