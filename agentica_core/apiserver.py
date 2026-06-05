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
# in-memory state
# --------------------------------------------------------------------------- #
class State:
    def __init__(self, *, ollama_host: str, model: str, workspace: str, db_path: str):
        self.ollama_host = ollama_host.rstrip("/")
        self.model = model
        self.workspace = workspace
        self.db_path = db_path
        self._apps: dict[str, object] = {}          # workspace -> AgentServerApp (agentic chat)
        self.local_jobs: dict[str, dict] = {}        # local_id -> {status, outcome, ...}

    def app_for(self, workspace: str | None):
        ws = workspace or self.workspace
        if ws not in self._apps:
            self._apps[ws] = gateway.build_app(
                ollama_host=self.ollama_host, model_name=self.model,
                workspace=ws, db_path=self.db_path, auth_token=None)
        return self._apps[ws]

    def complete(self, messages: list[dict], temperature: float = 0.2, timeout: float = 180) -> str:
        """Plain (non-agentic) chat completion via the engine's OpenAI /v1."""
        url = self.ollama_host + "/v1/chat/completions"
        body = json.dumps({"model": self.model, "messages": messages,
                           "temperature": temperature, "stream": False}).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except urllib.error.URLError as exc:
            raise RuntimeError(f"model unreachable at {self.ollama_host}: {getattr(exc, 'reason', exc)}")


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
           "plan. Reply STRICTLY as JSON: {\"title\": str, \"steps\": [str, ...], "
           "\"tests\": str (a shell command that verifies success, or \"\"), "
           "\"artifacts\": [str, ...]}. Steps are short imperative lines. No prose outside JSON.")
    user = (f"Goal:\n{goal}\n\n" + (f"Workspace context:\n{ctx}\n\n" if ctx else "")
            + "Return the JSON plan.")
    raw = state.complete([{"role": "system", "content": sys}, {"role": "user", "content": user}])
    parsed = _extract_json(raw) or {}
    steps = parsed.get("steps") or _fallback_steps(raw)
    return _plan_payload(parsed.get("title") or goal[:60], goal, steps,
                         parsed.get("tests", ""), parsed.get("artifacts", []))


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
                         parsed.get("artifacts", plan.get("artifacts", [])))


def _plan_payload(title, goal, steps, tests, artifacts) -> dict:
    return {
        "title": title, "goal": goal,
        "lines": [{"id": f"L{i+1}", "text": str(s)} for i, s in enumerate(steps) if str(s).strip()],
        "tests": tests or "",
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
                              provider="ollama", model_name=state.model,
                              ollama_host=state.ollama_host, model_timeout=300, _print=note)
            rec["outcome"] = outcome.to_dict()
            rec["status"] = "passed" if outcome.passed else "failed"
        except Exception as exc:  # noqa: BLE001
            rec["status"] = "error"
            rec["log"].append(f"ERROR {type(exc).__name__}: {exc}")

    threading.Thread(target=runner, daemon=True).start()
    return {"local": True, "local_id": local_id, "target": "local", "workspace": ws}


def submit_remote(plan: dict, target: str, workspace: str) -> dict:
    """Write a plan.yaml and submit to a ssh/SLURM target via job.submit."""
    tmp = Path(tempfile.mkdtemp(prefix="agentica-plan-"))
    plan_path = tmp / "plan.yaml"
    plan_path.write_text(plan_to_yaml(plan, workspace or "./workspace"), encoding="utf-8")
    captured: list[str] = []
    rc = job.submit(target, str(plan_path), sync_code=True, _print=captured.append)
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

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path == "/api/health":
                    return self._json({"ok": True, "version": __version__, "model": state.model})
                if u.path == "/api/hosts":
                    hosts = [{"alias": "local", "hostname": "this machine", "user": "",
                              "proxy_jump": "", "kind": "local"}]
                    hosts += [{**h, "kind": "ssh"} for h in sshconfig.list_hosts()]
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
            return submit_remote(plan, target, ws)

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
            out: list[str] = []
            job.status(target, jid, jobdir=jobdir, _print=out.append)
            return {"target": target, "job": jid, "lines": out}

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
          model: str = "llama3.2:3b") -> int:
    state = State(ollama_host=ollama_host, model=model, workspace=workspace, db_path=db_path)
    server = ThreadingHTTPServer((host, port), make_handler(state))
    print(f"agentica-core API on http://{host}:{port}  (model={model}, ollama={ollama_host})")
    print(f"  Agentica UI dev server should call this base URL.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0
