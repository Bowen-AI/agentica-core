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

import datetime
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import catalog, gateway, job, serving, sshconfig
from .voice_stream import stream_agent_turn
from .voice_gateway import start_voice_gateway
from .config import (DEFAULT_OLLAMA_PORT, DEFAULT_VLLM_PORT, ClusterConfig, ModelConfig,
                     PlanConfig, SuccessCriteria)
from .on_node_runner import run_job
from .transport import Transport

__version__ = "0.2.3"


# --------------------------------------------------------------------------- #
# runtime context injected into the model (open models have a stale training
# cutoff and no idea of "now" / where they run -- give them a live preamble).
# --------------------------------------------------------------------------- #
AGENTICA_SYSTEM_PROMPT = (
    "You are Agentica, a helpful AI assistant for the user's Agentica workspace. "
    "Answer directly, or use your tools (read/write files, run shell commands, search the web) "
    "to get things done in the workspace -- the runtime executes tools and enforces policy. "
    "Be concise and accurate; if you don't know something, say so instead of guessing."
)


def _runtime_preamble(
    workspace: str | None = None,
    *,
    target: str | None = None,
    model: str | None = None,
    engine: str | None = None,
) -> str:
    """Live context for the model -- authoritative for any date/time/'today'/'now'
    question (the model's own training data is stale)."""
    now = datetime.datetime.now().astimezone()
    osname = "macOS" if os.uname().sysname == "Darwin" else os.uname().sysname
    t = (target or "local").strip() or "local"
    m = f" using {model}" if model else ""
    e = f" via {engine}" if engine else ""
    lines = [
        "Live context (authoritative -- your training data is stale, prefer THIS for any "
        "date, time, 'today', 'now', or 'current' question):",
        f"- Current date & time: {now:%A, %B %-d, %Y at %-I:%M %p} {now.tzname() or ''}".rstrip(),
    ]
    if t == "local":
        lines.append(f"- Model inference target: local on {osname} ({os.uname().machine}){m}{e}.")
    else:
        lines.append(f"- Model inference target: {t}{m}{e}.")
        lines.append(
            f"- Agentica's UI/backend and chat tools run on the local {osname} "
            f"({os.uname().machine}); submitted jobs run on the selected target."
        )
    if workspace:
        lines.append(f"- Working directory: {workspace}")
    return "\n".join(lines)


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
        out[cfg.name] = {
            "path": str(f),
            "host": cfg.ssh.host,
            "scheduler": cfg.scheduler,
            "model": cfg.model.name,
            "engine": cfg.model.engine,
            "gpu_type": cfg.slurm.gpu_type,
            "gpu_count": cfg.slurm.gpu_count,
        }
    return out


# --------------------------------------------------------------------------- #
# in-memory state
# --------------------------------------------------------------------------- #
@dataclass
class RuntimeBinding:
    target: str
    model: str
    engine: str
    base_url: str
    tunnel_cm: object | None = None
    tunnel: object | None = None
    job_id: str | None = None

    @property
    def api_base(self) -> str:
        return self.base_url.rstrip("/") + "/v1"

    def alive(self) -> bool:
        tun = self.tunnel
        return bool(tun is None or getattr(tun, "alive")())


class State:
    def __init__(self, *, ollama_host: str, model: str, workspace: str, db_path: str,
                 clusters_dir: str | None = None):
        self.ollama_host = ollama_host.rstrip("/")
        self.model = model
        self.workspace = workspace
        self.db_path = db_path
        self._apps: dict[tuple, object] = {}        # runtime/workspace -> AgentServerApp
        self._runtimes: dict[tuple[str, str, str], RuntimeBinding] = {}
        self._runtime_lock = threading.Lock()
        self.local_jobs: dict[str, dict] = {}        # local_id -> {status, outcome, ...}
        self._model_resolved = False
        self.clusters = load_clusters(clusters_dir)  # name -> {path, host, scheduler}

    def cluster_path(self, target: str) -> str:
        """Map a target name to its cluster.yaml path if it's a known cluster, else
        return the target unchanged (a bare ~/.ssh/config alias, or 'local')."""
        c = self.clusters.get(target)
        if c:
            return c["path"]
        # The UI may expose an SSH-config alias (e.g. "pinotage.usc.edu") while a
        # cluster YAML uses a friendlier name ("pinotage"). Treat matching hosts as
        # the configured cluster so chat/jobs inherit its model and scheduler knobs.
        for cfg in self.clusters.values():
            if target and target == cfg.get("host"):
                return cfg["path"]
        return target

    def target_cluster(self, target: str | None) -> ClusterConfig | None:
        t = (target or "local").strip() or "local"
        if t == "local":
            return None
        return ClusterConfig.resolve(self.cluster_path(t))

    def default_model_for_target(self, target: str | None) -> str:
        cluster = self.target_cluster(target)
        if cluster:
            return cluster.model.name
        return self.resolve_model()

    def resolve_model(self, model: str | None = None) -> str:
        """Effective model for inference. If the configured model isn't installed
        but Ollama has others, fall back to a present one (preferring the same
        family, then a qwen*, then the first installed) and memoize it -- so chat
        and jobs work with whatever the user actually has instead of dead-ending
        on a 404 'model not found'. When Ollama is down/empty we keep the
        configured model untouched (the Setup flow guides the user to it)."""
        if model:
            return model
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

    def _engine_for(self, target: str, model: str, engine: str | None) -> str:
        if engine in {"ollama", "vllm"}:
            return engine
        cluster = self.target_cluster(target)
        if cluster and model == cluster.model.name:
            return cluster.model.engine
        return "vllm" if "/" in model else "ollama"

    def runtime_for(self, target: str | None = None, model: str | None = None,
                    engine: str | None = None, notify=None) -> RuntimeBinding:
        t = (target or "local").strip() or "local"
        selected_model = model or self.default_model_for_target(t)
        selected_engine = self._engine_for(t, selected_model, engine)
        if t == "local":
            if selected_engine != "ollama":
                raise RuntimeError("local chat currently supports Ollama models only")
            return RuntimeBinding(t, self.resolve_model(selected_model), "ollama", self.ollama_host)

        key = (t, selected_model, selected_engine)
        with self._runtime_lock:
            cached = self._runtimes.get(key)
            if cached and cached.alive():
                return cached
            if notify:
                notify(f"Starting {selected_engine} model {selected_model} on {t}...")
            cluster = self.target_cluster(t) or ClusterConfig.resolve(t)
            model_cfg = _model_config_for_selection(cluster.model, selected_model, selected_engine)
            fit = serving.preflight(cluster, model_cfg)
            if fit.verdict == "won't fit":
                raise RuntimeError(f"model {selected_model} will not fit on {t}: {fit.message}")
            transport = Transport.from_cluster(cluster)
            remote_jobdir = transport.expand_home(
                f"{cluster.remote_workdir}/chat-{uuid.uuid4().hex[:8]}"
            )
            handle = serving.bring_up(
                transport, cluster, remote_jobdir, model=model_cfg,
                wait_timeout_s=max(60.0, model_cfg.timeout_s),
            )
            local_port = _free_port()
            readiness = f"http://127.0.0.1:{local_port}{serving.readiness_path(handle.engine)}"
            if notify:
                notify(f"Opening tunnel to {handle.node}:{handle.port}...")
            cm = transport.tunnel(
                local_port, handle.node, handle.port,
                readiness_url=readiness, readiness_timeout_s=max(60.0, model_cfg.timeout_s),
            )
            tunnel = cm.__enter__()
            runtime = RuntimeBinding(
                target=t, model=selected_model, engine=selected_engine,
                base_url=tunnel.base_url, tunnel_cm=cm, tunnel=tunnel, job_id=handle.job_id,
            )
            self._runtimes[key] = runtime
            return runtime

    def app_for(self, workspace: str | None, target: str | None = None,
                model: str | None = None, engine: str | None = None, notify=None):
        ws = workspace or self.workspace
        rt = self.runtime_for(target, model, engine, notify=notify)
        # Key by date too, so a long-running backend rebuilds with a fresh "today" in
        # the system preamble (the AgentServerApp caches its controller/system prompt).
        key = (ws, datetime.date.today().isoformat(), rt.target, rt.model, rt.engine, rt.base_url)
        if key not in self._apps:
            provider = "ollama" if rt.engine == "ollama" else "openai-compatible"
            self._apps[key] = gateway.build_app(
                ollama_host=rt.base_url, model_name=rt.model,
                workspace=ws, db_path=self.db_path, auth_token=None,
                provider=provider, api_base=rt.api_base,
                system_prompt=AGENTICA_SYSTEM_PROMPT + "\n\n" + _runtime_preamble(
                    ws, target=rt.target, model=rt.model, engine=rt.engine,
                ))
        return self._apps[key]

    def complete(self, messages: list[dict], temperature: float = 0.2, timeout: float = 180,
                 target: str | None = None, model: str | None = None,
                 engine: str | None = None) -> str:
        """Plain (non-agentic) chat completion via the engine's OpenAI /v1."""
        rt = self.runtime_for(target, model, engine)
        url = rt.api_base + "/chat/completions"
        body = json.dumps({"model": rt.model, "messages": messages,
                           "temperature": temperature, "stream": False}).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except (urllib.error.URLError, TimeoutError) as exc:
            reason = getattr(exc, "reason", None) or str(exc) or "request timed out"
            raise RuntimeError(f"model unreachable at {rt.base_url}: {reason}")

    def complete_stream(self, messages: list[dict], on_reasoning=None, on_content=None,
                        think: bool = True, fmt: str | None = None,
                        temperature: float = 0.2, timeout: float = 600,
                        target: str | None = None, model: str | None = None,
                        engine: str | None = None, on_status=None) -> str:
        """Streaming completion via Ollama's native /api/chat. Calls
        on_reasoning(text)/on_content(text) per delta and returns the full content.

        `think=False` disables the model's chain-of-thought (qwen3.5 et al.) -- the
        planner uses this so a simple plan returns in seconds instead of after a
        minute-plus of reasoning. `fmt="json"` constrains output to valid JSON
        (so plan parsing can't fail on malformed model output)."""
        rt = self.runtime_for(target, model, engine, notify=on_status)
        if rt.engine != "ollama":
            return self._complete_stream_openai(
                rt, messages, on_content=on_content, fmt=fmt,
                temperature=temperature, timeout=timeout,
            )
        url = rt.base_url + "/api/chat"
        payload: dict = {"model": rt.model, "messages": messages,
                         "think": think, "stream": True, "options": {"temperature": temperature}}
        if fmt:
            payload["format"] = fmt
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        content: list[str] = []
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = chunk.get("message") or {}
                    if msg.get("thinking") and on_reasoning:
                        on_reasoning(msg["thinking"])
                    if msg.get("content"):
                        content.append(msg["content"])
                        if on_content:
                            on_content(msg["content"])
                    if chunk.get("done"):
                        break
        except (urllib.error.URLError, TimeoutError) as exc:
            reason = getattr(exc, "reason", None) or str(exc) or "request timed out"
            raise RuntimeError(f"model unreachable at {rt.base_url}: {reason}")
        return "".join(content)

    def _complete_stream_openai(self, rt: RuntimeBinding, messages: list[dict], *,
                                on_content=None, fmt: str | None = None,
                                temperature: float = 0.2, timeout: float = 600) -> str:
        url = rt.api_base + "/chat/completions"
        payload: dict = {"model": rt.model, "messages": messages,
                         "temperature": temperature, "stream": True}
        if fmt == "json":
            payload["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        content: list[str] = []
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        text = delta.get("content")
                        if text:
                            content.append(text)
                            if on_content:
                                on_content(text)
        except (urllib.error.URLError, TimeoutError) as exc:
            reason = getattr(exc, "reason", None) or str(exc) or "request timed out"
            raise RuntimeError(f"model unreachable at {rt.base_url}: {reason}")
        return "".join(content)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _model_config_for_selection(base: ModelConfig, model: str, engine: str) -> ModelConfig:
    if engine == "ollama":
        port = base.serve_port if base.engine == "ollama" else DEFAULT_OLLAMA_PORT
    else:
        port = base.serve_port if base.engine == "vllm" else DEFAULT_VLLM_PORT
    return replace(base, engine=engine, name=model, serve_port=port)


def _checkpoint_for_engine(model_key: str, engine: str) -> tuple[str | None, str]:
    spec = catalog.get_model(model_key)
    checkpoints = spec.checkpoints if spec else ()
    if engine == "ollama":
        for ckpt in checkpoints:
            if ckpt.lower().startswith("ollama:"):
                return ckpt.split(":", 1)[1].strip(), "ollama"
        return None, "ollama"
    for ckpt in checkpoints:
        if ckpt.lower().startswith("ollama:"):
            continue
        if "/" in ckpt and "community" not in ckpt.lower() and "verify" not in ckpt.lower():
            return ckpt.strip(), "huggingface"
    return model_key, "huggingface"


def _catalog_label(model_key: str, model_id: str) -> str:
    spec = catalog.get_model(model_key)
    return spec.display if spec else model_id


def _add_model_option(options: list[dict], seen: set[tuple[str, str]], *,
                      model_id: str, engine: str, label: str | None = None,
                      source: str = "ollama", catalog_key: str | None = None,
                      installed: bool = False, recommended: bool = False,
                      configured: bool = False, fit: str | None = None,
                      note: str = "") -> None:
    key = (model_id, engine)
    if key in seen:
        for opt in options:
            if (opt["id"], opt["engine"]) == key:
                opt["installed"] = bool(opt.get("installed") or installed)
                opt["recommended"] = bool(opt.get("recommended") or recommended)
                opt["configured"] = bool(opt.get("configured") or configured)
                if note and note not in opt.get("note", ""):
                    opt["note"] = (opt.get("note", "") + " " + note).strip()
                return
    seen.add(key)
    options.append({
        "id": model_id,
        "label": label or model_id,
        "engine": engine,
        "source": source,
        "catalog_key": catalog_key,
        "installed": installed,
        "recommended": recommended,
        "configured": configured,
        "fit": fit,
        "note": note,
    })


# Cache the remote box's installed-model list briefly so /api/models doesn't SSH
# on every keystroke/target-switch. Short TTL: a fresh pull should show up soon.
_remote_models_cache: dict[str, tuple[float, list[str] | None]] = {}
_REMOTE_MODELS_TTL = 30.0


def remote_ollama_models(state: "State", target: str | None, timeout: float = 12.0) -> list[str] | None:
    """The models ACTUALLY installed on a bare-SSH target's Ollama, or None if we
    can't reach it (caller falls back to catalog presets). SLURM targets return
    None — their models live on compute nodes, not the login node."""
    key = (target or "").strip()
    now = time.time()
    hit = _remote_models_cache.get(key)
    if hit and now - hit[0] < _REMOTE_MODELS_TTL:
        return hit[1]
    result = _remote_ollama_models_uncached(state, target, timeout)
    _remote_models_cache[key] = (now, result)
    return result


def _remote_ollama_models_uncached(state: "State", target: str | None, timeout: float) -> list[str] | None:
    try:
        cluster = state.target_cluster(target)
        if cluster is None or cluster.scheduler == "slurm":
            return None
        transport = Transport.from_cluster(cluster)
        port = getattr(cluster.model, "serve_port", 11434) or 11434
        # One round-trip: prefer a running server's /api/tags (JSON), else `ollama list`.
        cmd = (f"curl -sf -m 5 http://127.0.0.1:{port}/api/tags 2>/dev/null "
               f"|| ollama list 2>/dev/null")
        res = transport.exec(cmd, timeout=timeout)
        out = (getattr(res, "out", "") or "").strip()
        if not out:
            return None
        if out.startswith("{"):  # JSON from /api/tags
            data = json.loads(out)
            models = (data or {}).get("models") or []
            names = [m.get("name") or m.get("model") for m in models if isinstance(m, dict)]
            return sorted({n for n in names if n})
        # else parse the `ollama list` table: skip the NAME header, take col 1.
        names = []
        for i, line in enumerate(out.splitlines()):
            parts = line.split()
            if not parts:
                continue
            if i == 0 and parts[0].upper() == "NAME":
                continue
            names.append(parts[0])
        return sorted(set(names))
    except Exception:  # noqa: BLE001 - any failure -> "couldn't reach remote ollama"
        return None


def model_catalog_for_target(state: State, target: str | None) -> dict:
    t = (target or "local").strip() or "local"
    options: list[dict] = []
    seen: set[tuple[str, str]] = set()
    warning = ""

    if t == "local":
        installed = ollama_models(state.ollama_host)
        selected = state.resolve_model()
        for name in installed:
            _add_model_option(options, seen, model_id=name, engine="ollama",
                              source="ollama", installed=True,
                              recommended=name == selected, fit="installed locally")
        for key in ("qwen3.5-9b", "gemma4-e4b", "gemma4-e2b", "llama3.2-3b"):
            model_id, source = _checkpoint_for_engine(key, "ollama")
            if model_id:
                _add_model_option(
                    options, seen, model_id=model_id, engine="ollama",
                    label=_catalog_label(key, model_id), source=source,
                    catalog_key=key, fit="small local Ollama option",
                    note="Pulls from Ollama if not installed.",
                )
        if selected and (selected, "ollama") not in seen:
            _add_model_option(options, seen, model_id=selected, engine="ollama",
                              source="ollama", configured=True)
        return {"target": t, "selected_model": selected, "selected_engine": "ollama",
                "options": options, "warning": warning}

    cluster = state.target_cluster(t)
    assert cluster is not None
    selected = cluster.model.name
    selected_engine = cluster.model.engine
    _add_model_option(options, seen, model_id=selected, engine=selected_engine,
                      label=f"{selected} (target default)",
                      source="ollama" if selected_engine == "ollama" else "huggingface",
                      configured=True, recommended=True, fit="configured for this target")

    gpu_types = []
    if cluster.slurm.gpu_type and cluster.slurm.gpu_type != "any":
        gpu_types.append(cluster.slurm.gpu_type)
    gpu_types.extend(g for g in cluster.resources.gpu_types if g not in gpu_types)
    for gpu in gpu_types:
        try:
            presets = catalog.presets_for_gpu(gpu, cluster.slurm.gpu_count)
        except KeyError:
            continue
        rec = catalog.recommend(gpu, cluster.slurm.gpu_count)
        for p in presets:
            spec = catalog.get_model(p.model)
            if not spec or spec.kind != "llm" or p.engine not in {"ollama", "vllm"}:
                continue
            model_id, source = _checkpoint_for_engine(p.model, p.engine)
            if not model_id:
                continue
            _add_model_option(
                options, seen, model_id=model_id, engine=p.engine,
                label=_catalog_label(p.model, model_id), source=source,
                catalog_key=p.model,
                recommended=bool(rec and rec.name == p.name),
                fit=f"{p.gpu_count}x {p.gpu} · {p.quant}",
                note=p.note or "Catalog preset that fits this target profile.",
            )
    if not gpu_types:
        warning = "No GPU profile for this target yet; showing target default plus small Ollama-safe options."
        for key in ("llama3.2-3b", "qwen3.5-9b", "gemma4-e4b"):
            model_id, source = _checkpoint_for_engine(key, "ollama")
            if model_id:
                _add_model_option(
                    options, seen, model_id=model_id, engine="ollama",
                    label=_catalog_label(key, model_id), source=source,
                    catalog_key=key, fit="unchecked target fit",
                    note="Run discovery or add a cluster YAML GPU type for exact fit filtering.",
                )

    # Surface what's ACTUALLY installed on the box (bare-SSH targets): mark matching
    # presets installed, and add any real models we didn't already list. This is the
    # difference between "the UI guesses" and "the UI shows what the server has".
    remote_installed = remote_ollama_models(state, t)
    if remote_installed is not None:
        for name in remote_installed:
            _add_model_option(options, seen, model_id=name, engine="ollama",
                              source="ollama", installed=True, fit="installed on this server")
        if not remote_installed:
            warning = ("Reached the server, but no Ollama models are installed yet — "
                       "pick one below and click Download.")
    else:
        warning = warning or ("Couldn't reach this server's Ollama to list installed "
                              "models; showing catalog presets you can download.")
    return {"target": t, "selected_model": selected, "selected_engine": selected_engine,
            "options": options, "warning": warning}


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


def draft_plan(state: State, goal: str, workspace: str | None, on_reasoning=None,
               target: str | None = None, model: str | None = None,
               engine: str | None = None) -> dict:
    ctx = workspace_summary(workspace) if workspace else ""
    sys = ("You are a planning assistant. Given a goal, produce a concrete, ordered, checkable "
           "plan. The plan is carried out by an AUTONOMOUS agent that has file read/write and "
           "shell tools -- NOT a human at a GUI. Steps must be concrete agent actions (write a "
           "file, run a command); never 'open a text editor', 'save the file', or manual GUI steps. "
           "Reply STRICTLY as JSON: {\"title\": str, \"steps\": [str, ...], "
           "\"tests\": str (a shell command that genuinely verifies success via exit code -- e.g. "
           "grep/diff/an assertion script, NOT a bare echo/print that always succeeds, or \"\"), "
           "\"artifacts\": [str, ...]}. Steps are short imperative lines. No prose outside JSON.")
    user = (_runtime_preamble(workspace, target=target, model=model, engine=engine)
            + "\n\n" + f"Goal:\n{goal}\n\n"
            + (f"Workspace context:\n{ctx}\n\n" if ctx else "") + "Return the JSON plan.")
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    # Planning is a simple structured task -> disable chain-of-thought (fast: seconds, not
    # minutes) and force valid JSON so parsing can't fail.
    raw = state.complete_stream(
        msgs, on_reasoning, think=False, fmt="json",
        target=target, model=model, engine=engine,
    )
    parsed = _extract_json(raw) or {}
    steps = parsed.get("steps") or _fallback_steps(raw)
    # The model authored the tests string -> NOT a vetted gate (see _plan_payload).
    return _plan_payload(parsed.get("title") or goal[:60], goal, steps,
                         parsed.get("tests", ""), parsed.get("artifacts", []),
                         tests_authoritative=False)


def refine_plan(state: State, plan: dict, comments: list[dict], on_reasoning=None,
                target: str | None = None, model: str | None = None,
                engine: str | None = None) -> dict:
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
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    raw = state.complete_stream(
        msgs, on_reasoning, think=False, fmt="json",
        target=target, model=model, engine=engine,
    )
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
def plan_to_yaml(plan: dict, workspace: str, model: str | None = None,
                 engine: str | None = None) -> str:
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
    if model:
        doc["model"] = {"engine": engine or ("vllm" if "/" in model else "ollama"),
                        "name": model}
    return yaml.safe_dump(doc, sort_keys=False)


def _derive_protect(test_cmd: str | None, ws: str) -> list[str]:
    """Best-effort: the file path(s) referenced by the test command that already
    exist in the workspace = the grader(s) to snapshot+restore (tamper-evidence)."""
    import shlex
    try:
        toks = shlex.split(test_cmd or "")
    except ValueError:
        toks = (test_cmd or "").split()
    wsp = Path(ws)
    out: list[str] = []
    for t in toks:
        t = t.strip()
        if not t or t.startswith("-"):
            continue
        if ("/" in t or "." in t) and (wsp / t).exists() and (wsp / t).is_file():
            out.append(t)
    return out


def submit_local(state: State, plan: dict, workspace: str, model: str | None = None,
                 engine: str | None = None) -> dict:
    """Run the agentic job in a background thread against local ollama."""
    local_id = "local-" + uuid.uuid4().hex[:8]
    ws = workspace or tempfile.mkdtemp(prefix="agentica-job-")
    Path(ws).mkdir(parents=True, exist_ok=True)
    tests_cmd = plan.get("tests") or None
    pc = PlanConfig(
        title=plan.get("title", "job"), goal=plan.get("goal", ""), workspace=ws,
        checklist=[ln["text"] for ln in plan.get("lines", [])],
        success_criteria=SuccessCriteria(
            tests=tests_cmd,
            artifacts=plan.get("artifacts", []),
            # Snapshot+restore the grader file(s) referenced by the test command so the
            # agent (which has shell + write access to the workspace) can't tamper the
            # gate it is judged by. Without this the tamper snapshot is empty (the C3 bug).
            protect=_derive_protect(tests_cmd, ws),
            # Model-drafted tests are not a vetted gate -> the backstop won't auto-PASS on
            # them alone (see on_node_runner).
            tests_authoritative=bool(plan.get("tests_authoritative", True)),
        ),
        max_iterations=3, max_steps_per_iteration=12)
    rec = {"local_id": local_id, "target": "local", "status": "running",
           "workspace": ws, "log": [], "outcome": None}
    state.local_jobs[local_id] = rec

    def runner():
        def note(msg):
            rec["log"].append(msg)
        try:
            selected_model = model or state.resolve_model()
            selected_engine = engine or ("vllm" if "/" in selected_model else "ollama")
            if selected_engine != "ollama":
                raise RuntimeError("local jobs currently support Ollama models only")
            outcome = run_job(pc, workspace=ws, db_path=str(Path(ws) / "job.db"),
                              provider="ollama", model_name=selected_model,
                              ollama_host=state.ollama_host, model_timeout=300, _print=note)
            rec["outcome"] = outcome.to_dict()
            rec["status"] = "passed" if outcome.passed else "failed"
        except Exception as exc:  # noqa: BLE001
            rec["status"] = "error"
            rec["log"].append(f"ERROR {type(exc).__name__}: {exc}")

    threading.Thread(target=runner, daemon=True).start()
    return {"local": True, "local_id": local_id, "target": "local", "workspace": ws}


def submit_remote(plan: dict, target: str, cluster_path: str, workspace: str,
                  model: str | None = None, engine: str | None = None) -> dict:
    """Write a plan.yaml and submit to a ssh/SLURM target via job.submit. ``target``
    is the display name shown/polled by the UI; ``cluster_path`` is the cluster.yaml
    (full SLURM config) or a bare ssh alias to actually connect with."""
    tmp = Path(tempfile.mkdtemp(prefix="agentica-plan-"))
    plan_path = tmp / "plan.yaml"
    plan_path.write_text(plan_to_yaml(plan, workspace or "./workspace", model, engine), encoding="utf-8")
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
_IS_MAC = os.uname().sysname == "Darwin"


def _provisioned_ollama_bin() -> Path:
    # mac (ollama-darwin.tgz) extracts FLAT: ollama + libs at the root.
    # linux (tar.zst) extracts to bin/ollama + lib/.
    return OLLAMA_HOME / ("ollama" if _IS_MAC else "bin/ollama")


def _provisioned_ollama_lib() -> Path:
    return OLLAMA_HOME if _IS_MAC else OLLAMA_HOME / "lib"


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
        # ollama usually returns {"models": [...]}, but {"models": null} when none are
        # pulled (and .get's default only applies to a MISSING key, not a null value) --
        # the parse must stay inside the try so this can't escape as a 500.
        models = (data or {}).get("models") or []
        names = {m.get("name") or m.get("model") for m in models if isinstance(m, dict)}
        return sorted(n for n in names if n)
    except Exception:  # noqa: BLE001 - any transport/parse failure -> "no models"
        return []


def find_ollama_bin() -> str | None:
    prov = _provisioned_ollama_bin()
    return shutil.which("ollama") or (str(prov) if prov.exists() else None)


def model_present(model: str, models: list[str]) -> bool:
    return (model in models or f"{model}:latest" in models
            or (":" not in model and any(m.split(":")[0] == model for m in models)))


def setup_status(state: "State") -> dict:
    # A status check must NEVER 500 -- degrade to "not ready" on any unexpected error.
    try:
        running = ollama_reachable(state.ollama_host)
        models = ollama_models(state.ollama_host) if running else []
        present = model_present(state.model or "", models)
        return {
            "ollama_installed": bool(find_ollama_bin()) or running,
            "ollama_running": running,
            "models": models,
            "model": state.model,
            "model_present": present,
            # Ready once Ollama is up and ANY model is installed -- resolve_model()
            # will use an existing model, so we only need to pull when there are none.
            "ready": running and bool(models),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ollama_installed": False, "ollama_running": False, "models": [],
                "model": getattr(state, "model", ""), "model_present": False,
                "ready": False, "error": str(exc)}


def start_ollama(host: str, timeout_s: float = 25.0) -> tuple[bool, str]:
    if ollama_reachable(host):
        return True, "already running"
    binp = find_ollama_bin()
    if not binp:
        return False, "ollama is not installed"
    p = urlparse(host)
    env = dict(os.environ)
    env["OLLAMA_HOST"] = f"{p.hostname or '127.0.0.1'}:{p.port or 11434}"
    # Only point the dynamic loader at our provisioned libs when we're starting our
    # provisioned binary (a system ollama brings its own).
    lib = _provisioned_ollama_lib()
    if str(binp) == str(_provisioned_ollama_bin()) and lib.exists():
        key = "DYLD_LIBRARY_PATH" if _IS_MAC else "LD_LIBRARY_PATH"
        env[key] = f"{lib}:{env.get(key, '')}"
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
    """Provision Ollama into ~/.local/share/agentica/ollama (no sudo, no GUI install)
    so the app is zero-setup. macOS uses the standalone ollama-darwin.tgz (flat); Linux
    uses ollama-linux-<arch>.tar.zst. Reports progress; NEVER raises (reports + returns False)."""
    try:
        if find_ollama_bin():
            progress("Ollama is already available.")
            return True
        OLLAMA_HOME.mkdir(parents=True, exist_ok=True)
        base = "https://github.com/ollama/ollama/releases/latest/download"
        if _IS_MAC:
            tgz = OLLAMA_HOME / "ollama-darwin.tgz"
            progress("Downloading Ollama for macOS (~143 MB, one time)...")
            urllib.request.urlretrieve(f"{base}/ollama-darwin.tgz", tgz)
            progress("Extracting...")
            subprocess.run(["tar", "-xzf", str(tgz), "-C", str(OLLAMA_HOME)], check=True)
            try:
                tgz.unlink()
            except OSError:
                pass
        else:
            arch = "amd64" if os.uname().machine in ("x86_64", "amd64") else "arm64"
            tarzst, tar = OLLAMA_HOME / "ollama.tar.zst", OLLAMA_HOME / "ollama.tar"
            progress(f"Downloading Ollama for Linux ({arch}, one time)...")
            urllib.request.urlretrieve(f"{base}/ollama-linux-{arch}.tar.zst", tarzst)
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
        if not _provisioned_ollama_bin().exists():
            progress("ERROR: extraction did not produce the ollama binary")
            return False
        progress("Installed Ollama into ~/.local/share/agentica/ollama")
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
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Agentica-Token")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _authorized(self) -> bool:
            # CORS:* lets any page SEND a request; the shared token is the actual
            # boundary that stops it from DRIVING the agent / reading the result.
            tok = getattr(state, "api_token", None)
            if not tok:
                return True  # auth disabled (manual serve-api / dev)
            got = self.headers.get("X-Agentica-Token") or ""
            if not got:
                auth = self.headers.get("Authorization") or ""
                if auth.lower().startswith("bearer "):
                    got = auth[7:].strip()
            if not got:
                got = (parse_qs(urlparse(self.path).query).get("token") or [""])[0]
            # constant-time compare
            import hmac
            return hmac.compare_digest(got, tok)

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
            # Serialize writers: the agentic branch emits step frames from a worker
            # thread while the handler thread emits keepalive pings — unguarded,
            # the two can interleave mid-frame and corrupt the stream.
            lock = getattr(self, "_sse_lock", None)
            if lock is None:
                lock = self._sse_lock = threading.Lock()
            with lock:
                self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode("utf-8"))
                self.wfile.flush()

        def _pull_loop(self, model, base_url):
            """Stream ollama /api/pull progress SSE from base_url until success/error."""
            req = urllib.request.Request(
                base_url.rstrip("/") + "/api/pull",
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

        def _stream_pull(self, model, target=None):
            self._sse_start()
            t = (target or "local").strip() or "local"
            try:
                if t == "local":
                    self._pull_loop(model, state.ollama_host)
                    self._sse({"done": True})
                    return
                # Remote target: pull on the box itself, over an ssh tunnel to its Ollama.
                cluster = state.target_cluster(t)
                if cluster is None:
                    self._safe_sse({"error": f"unknown target {t}", "done": True})
                    return
                transport = Transport.from_cluster(cluster)
                port = getattr(cluster.model, "serve_port", 11434) or 11434
                self._safe_sse({"status": f"ensuring Ollama is running on {t}…"})
                # Start ollama on the box if it isn't already serving (non-fatal).
                transport.exec(
                    f"curl -sf -m3 http://127.0.0.1:{port}/api/tags >/dev/null 2>&1 || "
                    f"(nohup ollama serve >~/.agentica-ollama.log 2>&1 & sleep 2)",
                    timeout=60)
                local_port = _free_port()
                self._safe_sse({"status": f"opening a secure tunnel to {t}…"})
                with transport.tunnel(local_port, "127.0.0.1", port,
                                      readiness_url=f"http://127.0.0.1:{local_port}/api/tags",
                                      readiness_timeout_s=30) as tun:
                    self._safe_sse({"status": f"downloading {model} on {t}…"})
                    self._pull_loop(model, tun.base_url)
                # Bust the installed-models cache so the UI flips to installed immediately.
                _remote_models_cache.pop(t, None)
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

        def _stream_voice_install(self):
            # Provision local STT/TTS (Whisper wheel + Piper binary/voice) into the
            # data dir, streaming progress — same UX as the Ollama installer.
            self._sse_start()
            from .voice_provision import install_voice, voice_status
            ok = install_voice(lambda m: self._safe_sse({"status": m}))
            self._safe_sse({"done": True, "ok": ok, **voice_status()})

        def _stream_chat(self, body):
            self._sse_start()
            message = (body.get("message") or "").strip()
            mode = body.get("mode", "agentic")
            ws = body.get("workspace")
            target = body.get("target", "local")
            model = body.get("model")
            engine = body.get("engine")
            try:
                if mode == "plain":
                    msgs = [{"role": "system", "content": _runtime_preamble(
                        ws, target=target, model=model, engine=engine,
                    )}]
                    if ws:
                        ctx = workspace_summary(ws)
                        if ctx:
                            msgs.append({"role": "system", "content": "Workspace context:\n" + ctx})
                    msgs.append({"role": "user", "content": message})
                    # think=False -> direct, fast answers (the reasoning model otherwise
                    # streams a long chain-of-thought before any content, on even trivial
                    # prompts). Reasoning is still forwarded if a model emits it.
                    state.complete_stream(
                        msgs,
                        on_reasoning=lambda t: self._safe_sse({"reasoning": t}),
                        on_content=lambda t: self._safe_sse({"delta": t}),
                        think=False, target=target, model=model, engine=engine,
                        on_status=lambda t: self._safe_sse({"status": t}))
                    self._sse({"done": True})
                else:
                    app = state.app_for(
                        ws, target=target, model=model, engine=engine,
                        notify=lambda t: self._safe_sse({"status": t}),
                    )
                    # Stream per-step {step}/{artifact} frames as tools fire (the
                    # loop used to run monolithically -> a blank wait until done),
                    # then the authoritative terminal frame replaces them.
                    # WATCHDOG: run the turn on a worker and heartbeat from here.
                    # A frozen-build race once wedged this path with the client
                    # seeing NOTHING forever — a turn may be slow, but the stream
                    # must always be live and always end.
                    box: dict = {}

                    def _turn():
                        try:
                            box["res"] = stream_agent_turn(
                                app, message, body.get("session_id"),
                                emit=lambda frame: self._safe_sse(frame),
                            )
                        except Exception as exc:  # noqa: BLE001
                            box["err"] = exc

                    worker = threading.Thread(target=_turn, daemon=True,
                                              name="agentica-http-turn")
                    worker.start()
                    deadline = time.time() + 180
                    while worker.is_alive() and time.time() < deadline:
                        worker.join(2.0)
                        if worker.is_alive():
                            self._safe_sse({"ping": True})  # liveness; UI ignores it
                    if worker.is_alive():
                        self._safe_sse({"error": "the agent took too long and was stopped",
                                        "done": True})
                    elif "err" in box:
                        self._safe_sse({"error": str(box["err"]), "done": True})
                    else:
                        res = box.get("res") or {}
                        self._sse({"final": res.get("final_answer"), "steps": res.get("steps", []),
                                   "session_id": res.get("session_id"), "done": True})
            except Exception as exc:  # noqa: BLE001
                self._safe_sse({"error": str(exc), "done": True})

        def _safe_sse(self, obj):
            try:
                self._sse(obj)
            except Exception:  # noqa: BLE001 - client disconnected
                pass

        def _stream_plan(self, body, kind):
            # Stream the planner's "thinking" while it works (reasoning models take
            # minutes), then emit the finished plan -- so "Draft plan" / "Send notes"
            # show live progress instead of a multi-minute blank spinner.
            self._sse_start()
            on_r = lambda t: self._safe_sse({"reasoning": t})  # noqa: E731
            try:
                if kind == "draft":
                    plan = draft_plan(
                        state, body.get("goal", ""), body.get("workspace"), on_reasoning=on_r,
                        target=body.get("target"), model=body.get("model"),
                        engine=body.get("engine"),
                    )
                else:
                    plan = refine_plan(
                        state, body.get("plan", {}), body.get("comments", []), on_reasoning=on_r,
                        target=body.get("target"), model=body.get("model"),
                        engine=body.get("engine"),
                    )
                self._sse({"plan": plan, "done": True})
            except Exception as exc:  # noqa: BLE001
                self._safe_sse({"error": str(exc), "done": True})

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if not self._authorized():
                return self._json({"error": "unauthorized"}, 401)
            try:
                if u.path == "/api/health":
                    return self._json({"ok": True, "version": __version__, "model": state.model})
                if u.path == "/api/setup":
                    return self._json(setup_status(state))
                if u.path == "/api/model/pull":
                    return self._stream_pull((q.get("model") or [state.model])[0],
                                             (q.get("target") or ["local"])[0])
                if u.path == "/api/models":
                    return self._json(model_catalog_for_target(
                        state, (q.get("target") or ["local"])[0],
                    ))
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
                    app = state.app_for(
                        ws,
                        target=(q.get("target") or ["local"])[0],
                        model=(q.get("model") or [None])[0],
                        engine=(q.get("engine") or [None])[0],
                    )
                    msgs = app.storage.load_messages(sid) if sid else []
                    return self._json({"messages": [{"role": m.role, "content": m.content}
                                                    for m in msgs if m.role in ("user", "assistant")]})
                if u.path == "/api/job/status":
                    return self._json(self._job_status(q))
                if u.path == "/api/job/logs":
                    return self._json(self._job_logs(q))
                if u.path == "/api/voice/status":
                    from .voice_provision import voice_status
                    st = voice_status()
                    th = getattr(state, "voice_thread", None)
                    st["gateway_running"] = bool(th and th.is_alive())
                    st["voice_ws_port"] = getattr(state, "voice_port", None)
                    if not st["gateway_running"]:
                        st["gateway_error"] = ("voice WebSocket gateway not running — "
                                               "install 'websockets' (pip install websockets)")
                    return self._json(st)
                if u.path == "/api/voice/selftest":
                    # End-to-end proof the local STT+TTS pipeline works, in-process.
                    from .voice_provision import selftest
                    st = selftest()
                    th = getattr(state, "voice_thread", None)
                    st["gateway_ok"] = bool(th and th.is_alive())
                    st["voice_ws_port"] = getattr(state, "voice_port", None)
                    return self._json(st)
                return self._json({"error": "not found"}, 404)
            except Exception as exc:  # noqa: BLE001
                return self._json({"error": type(exc).__name__, "message": str(exc)}, 500)

        def do_POST(self):
            u = urlparse(self.path)
            if not self._authorized():
                return self._json({"error": "unauthorized"}, 401)
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
                    return self._json(draft_plan(
                        state, body.get("goal", ""), body.get("workspace"),
                        target=body.get("target"), model=body.get("model"),
                        engine=body.get("engine"),
                    ))
                if u.path == "/api/plan/draft/stream":
                    return self._stream_plan(body, "draft")
                if u.path == "/api/plan/refine":
                    return self._json(refine_plan(
                        state, body.get("plan", {}), body.get("comments", []),
                        target=body.get("target"), model=body.get("model"),
                        engine=body.get("engine"),
                    ))
                if u.path == "/api/plan/refine/stream":
                    return self._stream_plan(body, "refine")
                if u.path == "/api/job/submit":
                    return self._json(self._submit(body))
                if u.path == "/api/job/cancel":
                    return self._json(self._cancel(body))
                if u.path == "/api/voice/install":
                    return self._stream_voice_install()
                if u.path == "/api/voice/gemini/token":
                    return self._json(self._gemini_token(body))
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
            target = body.get("target", "local")
            model = body.get("model")
            engine = body.get("engine")
            if mode == "plain":
                msgs = [{"role": "system", "content": _runtime_preamble(
                    ws, target=target, model=model, engine=engine,
                )}]
                if ws:
                    ctx = workspace_summary(ws)
                    if ctx:
                        msgs.append({"role": "system",
                                     "content": "Use this workspace as context:\n" + ctx})
                msgs.append({"role": "user", "content": message})
                return {"mode": "plain",
                        "final_answer": state.complete(
                            msgs, target=target, model=model, engine=engine,
                        ),
                        "steps": [], "target": target, "model": model, "engine": engine}
            app = state.app_for(ws, target=target, model=model, engine=engine)
            res = app.chat(message, body.get("session_id"))
            return {"mode": "agentic", "session_id": res.get("session_id"),
                    "final_answer": res.get("final_answer"), "steps": res.get("steps", []),
                    "transcript": res.get("transcript", []),
                    "target": target, "model": model, "engine": engine}

        def _submit(self, body):
            plan = body.get("plan") or {}
            target = body.get("target", "local")
            ws = body.get("workspace") or plan.get("workspace") or ""
            model = body.get("model")
            engine = body.get("engine")
            if target == "local":
                return submit_local(state, plan, ws, model=model, engine=engine)
            # Map a cluster-name target to its cluster.yaml (account/partition/setup);
            # a bare ssh alias passes through unchanged.
            return submit_remote(plan, target, state.cluster_path(target), ws,
                                 model=model, engine=engine)

        def _gemini_token(self, body):
            # Mint an ephemeral Gemini Live token so the renderer can open a
            # realtime audio WebSocket directly to Google (cloud voice engine).
            # The key is supplied by the user (request body) or env; we only mint
            # a short-lived token, we don't persist the key. The agent turn +
            # canvas stay local. Returns {error} (not a 500) when no key is given.
            key = (body.get("api_key") or os.environ.get("GEMINI_API_KEY")
                   or os.environ.get("GOOGLE_API_KEY"))
            if not key:
                return {"error": "Add your Gemini API key in Settings to use the cloud voice engine."}
            try:
                from agentic_loop.voice_adapters import GeminiLiveTokenClient
                client = GeminiLiveTokenClient(api_key=key)
                return client.create_token(body.get("purpose", "voice"), body.get("session_id"))
            except Exception as exc:  # noqa: BLE001
                return {"error": str(exc)}

        def _cancel(self, body):
            # Remote jobs: scancel (SLURM) / best-effort kill (ssh). Local jobs run in an
            # in-process thread that can't be interrupted -- the UI just dismisses the panel.
            jid = body.get("job") or ""
            if not jid:
                return {"ok": False, "error": "cancel needs a remote job id"}
            out: list[str] = []
            rc = job.cancel(state.cluster_path(body.get("target", "")), jid, _print=out.append)
            return {"ok": rc == 0, "status": "cancelled", "lines": out}

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


def serve(*, host: str = "127.0.0.1", port: int = 8770, workspace: str | None = None,
          db_path: str | None = None, ollama_host: str = "http://127.0.0.1:11434",
          model: str = "qwen3.5:4b-mlx", clusters_dir: str | None = None) -> int:
    # The desktop app launches us with CWD=/ (read-only), so resolve data paths to a
    # WRITABLE absolute dir (the app passes AGENTICA_DATA_DIR; fall back to ~/.local).
    # Relative defaults like ".agentic" would otherwise fail with EROFS on every chat.
    data_dir = os.environ.get("AGENTICA_DATA_DIR") or os.path.expanduser("~/.local/share/agentica")
    workspace = workspace or os.path.join(data_dir, "workspace")
    db_path = db_path or os.path.join(data_dir, "agentica.db")
    try:
        os.makedirs(workspace, exist_ok=True)
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    except OSError as exc:  # noqa: BLE001
        print(f"warning: could not create data dir {data_dir}: {exc}")
    state = State(ollama_host=ollama_host, model=model, workspace=workspace, db_path=db_path,
                  clusters_dir=clusters_dir)
    # Per-launch shared secret (the desktop app mints it and passes it to BOTH
    # the backend env and the renderer). When set, every API + WS request must
    # carry it — this is what stops a visited web page from driving the local
    # agent (loopback bind + CORS:* alone do not). Unset (manual `serve-api`) =
    # open localhost for dev convenience.
    state.api_token = os.environ.get("AGENTICA_API_TOKEN")
    if state.api_token:
        print("  API auth: enabled (token required on :%d and :%d)" % (port, port + 1))
    else:
        print("  API auth: DISABLED (no AGENTICA_API_TOKEN) — dev/localhost only")
    # Full-duplex voice transport for the "local" voice engine (mic up / TTS down
    # + barge-in). Daemon thread; degrades to None if `websockets` isn't installed.
    # Capture the thread + port so /api/voice/status can report gateway health and
    # the renderer can derive the WS port instead of hardcoding the +1 convention.
    state.voice_port = port + 1
    state.voice_thread = start_voice_gateway(state, host=host, port=port + 1)
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
