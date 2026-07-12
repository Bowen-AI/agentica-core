"""Mode 1 -- interactive localhost gateway.

Wraps AgenticLocal's ``AgentServerApp`` (reusing its sessions / chat / memory /
events) and adds, in our own request handler:

* a ChatGPT-like web chat page at ``/`` (history + copy-API button),
* an OpenAI-compatible ``/v1/chat/completions`` + ``/v1/models`` for VS Code
  (every request runs the agent loop; there is no plain model passthrough),
* optional Bearer-token auth.

AgenticLocal itself is left untouched -- the model adapter is simply pointed at
``127.0.0.1:<tunnel_port>`` so inference runs on the remote GPU node.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from agentic_loop.model_selection import ModelSelection
from agentic_loop.server import AgentServerApp
from agentic_loop.tools import Tool, ToolContext, ToolRegistry, create_default_tools, inspect_csv

from .config import ClusterConfig
from . import __version__
from . import serving
from .transport import Transport, TransportError
from .voice_tools import register_voice_tools
from .webchat import chat_page_html

# --------------------------------------------------------------------------- #
# Gateway HTTP server
# --------------------------------------------------------------------------- #
def make_gateway_handler(
    app: AgentServerApp,
    *,
    auth_token: str | None,
    remote_v1_base: str,
    v1_mode: str,
    model_label: str,
    public_api_base: str,
):
    """Build the agentic gateway handler.

    ``remote_v1_base`` and ``v1_mode`` remain accepted for callers from older
    releases.  They no longer select a plain proxy: even the legacy
    ``v1_mode='passthrough'`` value is an agentic compatibility alias.
    """

    class GatewayHandler(BaseHTTPRequestHandler):
        server_version = f"agentica-core/{__version__}"

        def log_message(self, *args):  # quiet
            return

        # -- auth --
        def _authed(self) -> bool:
            if not auth_token:
                return True
            header = self.headers.get("Authorization", "")
            scheme, _, cred = header.partition(" ")
            return scheme.lower() == "bearer" and cred.strip() == auth_token

        def _need_auth(self) -> bool:
            if self._authed():
                return False
            self._json({"error": "unauthorized"}, status=401)
            return True

        # -- GET --
        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            if path in {"/", "/chat-ui"}:
                self._html(chat_page_html(
                    version=__version__, api_base=public_api_base,
                    token=auth_token, model_label=model_label,
                ))
                return
            if path == "/health":
                self._json({"ok": True, "version": __version__, "model": model_label})
                return
            if path == "/v1/models":
                if self._need_auth():
                    return
                self._json(_openai_models(model_label))
                return
            # everything below is authed
            if self._need_auth():
                return
            if path == "/history":
                q = parse_qs(parsed.query)
                sid = (q.get("session_id") or [""])[0]
                self._json({"messages": _load_history(app, sid)})
                return
            if path == "/sessions":
                self._json({"sessions": app.list_sessions()})
                return
            if path == "/memory":
                self._json({"records": app.memory_records()})
                return
            self._json({"error": "not found"}, status=404)

        # -- POST --
        def do_POST(self):
            parsed = urlparse(self.path)
            path = parsed.path
            if self._need_auth():
                return
            try:
                body = self._read_json()
                if path == "/chat":
                    message = (body.get("message") or body.get("goal") or "").strip()
                    if not message:
                        raise ValueError("request requires non-empty message")
                    sel = app.model_selection_from_payload(body, body.get("session_id"))
                    self._json(app.chat(message, body.get("session_id"), model_selection=sel))
                    return
                if path == "/v1/chat/completions":
                    self._handle_v1(body)
                    return
                self._json({"error": "not found"}, status=404)
            except Exception as exc:  # noqa: BLE001
                self._json({"error": type(exc).__name__, "message": str(exc)}, status=400)

        # -- /v1 translation --
        def _handle_v1(self, body: dict):
            self._v1_agentic(body, bool(body.get("stream")))

        def _v1_agentic(self, body: dict, stream: bool):
            """Run the full agent loop and return its final answer in OpenAI shape."""
            messages = body.get("messages") or []
            goal = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    goal = m.get("content") or ""
                    break
            result = app.run_once(goal)
            text = result.get("final_answer", "")
            if stream:
                self._sse_openai_single(text)
            else:
                self._json(_openai_completion(text, model_label))

        def _sse_openai_single(self, text: str):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            cid = "chatcmpl-" + uuid.uuid4().hex[:12]
            first = {"id": cid, "object": "chat.completion.chunk", "model": model_label,
                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": text},
                                  "finish_reason": None}]}
            last = {"id": cid, "object": "chat.completion.chunk", "model": model_label,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            for evt in (first, last):
                self.wfile.write(f"data: {json.dumps(evt)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")

        # -- io helpers --
        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if not length:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _json(self, payload, status=200):
            raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _html(self, html, status=200):
            raw = html.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    return GatewayHandler


def _openai_models(model_label: str) -> dict:
    return {"object": "list", "data": [
        {"id": model_label, "object": "model", "owned_by": "agentica-core"},
        {"id": "agentic", "object": "model", "owned_by": "agentica-core"},
    ]}


def _openai_completion(text: str, model_label: str) -> dict:
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:12],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_label,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _load_history(app: AgentServerApp, session_id: str) -> list[dict]:
    if not session_id:
        return []
    try:
        msgs = app.storage.load_messages(session_id)
    except Exception:
        return []
    return [{"role": m.role, "content": m.content} for m in msgs if m.role in {"user", "assistant"}]


_SHELL_DENYLIST = ("rm -rf /", "shutdown", "mkfs", ":(){", "/etc/passwd")


def _validate_shell_command(command: str) -> None:
    if not command:
        raise ValueError("run_shell requires a non-empty 'command'")
    lower = command.lower()
    for token in _SHELL_DENYLIST:
        if token in lower:
            raise PermissionError(f"run_shell blocked: command contains disallowed token {token!r}")


def _register_run_shell(registry: ToolRegistry, handler, *, location: str) -> ToolRegistry:
    if "run_shell" not in registry.names():
        registry.register(Tool(
            name="run_shell",
            description=f"Run a shell command in the {location} workspace and return exit code + output.",
            parameters={"type": "object", "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "number", "default": 300.0},
            }, "required": ["command"]},
            handler=handler,
            risk_level="high",
            source="job",
        ))
    return registry


def create_local_tools(*, enable_network: bool = True) -> ToolRegistry:
    """Code-capable tools whose file and shell operations stay in ToolContext.workspace_root."""
    registry = create_default_tools(enable_network=enable_network)

    def local_run_shell(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        command = (arguments.get("command") or arguments.get("cmd") or "").strip()
        _validate_shell_command(command)
        timeout = float(arguments.get("timeout", 300.0))
        root = context.workspace_root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        try:
            proc = subprocess.run(
                ["bash", "-lc", command], cwd=str(root), capture_output=True, text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return {"command": command, "exit_code": 124, "stdout": exc.stdout or "",
                    "stderr": f"timeout after {timeout}s", "timed_out": True}
        return {"command": command, "exit_code": proc.returncode,
                "stdout": proc.stdout[-8000:], "stderr": proc.stderr[-4000:],
                "timed_out": False}

    return _register_run_shell(registry, local_run_shell, location="local")


_REMOTE_LIST = """
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]).expanduser().resolve()
root.mkdir(parents=True, exist_ok=True)
requested = (root / sys.argv[2]).resolve()
if requested != root and root not in requested.parents:
    raise PermissionError(f"path escapes workspace: {sys.argv[2]}")
if not requested.exists():
    raise FileNotFoundError(sys.argv[2])
if not requested.is_dir():
    raise NotADirectoryError(sys.argv[2])
files = [str(path.relative_to(root)) for path in sorted(requested.rglob("*")) if path.is_file()]
print(json.dumps({"files": files}))
"""

_REMOTE_READ = """
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]).expanduser().resolve()
target = (root / sys.argv[2]).resolve()
if target != root and root not in target.parents:
    raise PermissionError(f"path escapes workspace: {sys.argv[2]}")
if not target.is_file():
    raise FileNotFoundError(sys.argv[2])
limit = int(sys.argv[3])
content = target.read_text(encoding="utf-8")
print(json.dumps({"content": content[:limit], "truncated": len(content) > limit}))
"""

_REMOTE_RESOLVE = """
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]).expanduser().resolve()
root.mkdir(parents=True, exist_ok=True)
target = (root / sys.argv[2]).resolve()
if target != root and root not in target.parents:
    raise PermissionError(f"path escapes workspace: {sys.argv[2]}")
if sys.argv[3] == "write":
    target.parent.mkdir(parents=True, exist_ok=True)
elif not target.is_file():
    raise FileNotFoundError(sys.argv[2])
print(json.dumps({"root": str(root), "target": str(target)}))
"""


def _python_command(source: str, *args: object) -> str:
    """Build a shell-safe command while keeping user-controlled values out of Python source."""
    return " ".join(["python3", "-c", shlex.quote(source),
                     *(shlex.quote(str(arg)) for arg in args)])


def _remote_json(transport: Transport, what: str, source: str, *args: object) -> dict:
    result = transport.exec(_python_command(source, *args))
    if result.rc == 255:
        # rc=255 is ssh itself failing (a rate-limiting bastion dropping the
        # dial), not the remote script. These reads are idempotent — retry once
        # after a beat instead of failing the agent's whole tool call.
        time.sleep(2.0)
        result = transport.exec(_python_command(source, *args))
    result.check(what)
    try:
        value = json.loads(result.out)
    except json.JSONDecodeError as exc:
        raise TransportError(f"{what} returned invalid JSON: {result.out[:200]!r}") from exc
    if not isinstance(value, dict):
        raise TransportError(f"{what} returned a non-object response")
    return value


def create_remote_tools(
    target: str,
    remote_workspace: str,
    *,
    transport: Transport | None = None,
) -> ToolRegistry:
    """Code tools backed by an SSH transport, with an injectable transport for tests."""
    if transport is None:
        cluster = ClusterConfig.resolve(target)
        transport = Transport.from_cluster(cluster)

    registry = ToolRegistry()

    def remote_list_files(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        path = str(arguments.get("path", "."))
        payload = _remote_json(transport, "remote list_files", _REMOTE_LIST,
                               remote_workspace, path)
        return {"path": path, "files": payload.get("files", [])}

    def remote_read_file(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        path = str(arguments["path"])
        max_chars = max(0, int(arguments.get("max_chars", 12000)))
        payload = _remote_json(transport, "remote read_file", _REMOTE_READ,
                               remote_workspace, path, max_chars)
        return {"path": path, "content": str(payload.get("content", "")),
                "truncated": bool(payload.get("truncated"))}

    def _resolve(path: str, mode: str) -> dict:
        return _remote_json(transport, "resolve remote path", _REMOTE_RESOLVE,
                            remote_workspace, path, mode)

    def remote_write_file(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        path = str(arguments["path"])
        content = str(arguments.get("content", ""))
        remote_target = str(_resolve(path, "write")["target"])
        fd, temp_path = tempfile.mkstemp()
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
            transport._rsync(temp_path, remote_target, to_remote=True, delete=False).check("push file")
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass
        return {"path": path, "bytes_written": len(content.encode("utf-8"))}

    def remote_inspect_csv(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        path = str(arguments["path"])
        remote_target = str(_resolve(path, "read")["target"])
        fd, temp_path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            transport._rsync(remote_target, temp_path, to_remote=False, delete=False).check("pull csv file")
            dummy_ctx = ToolContext(workspace_root=Path(temp_path).parent)
            result = inspect_csv(dummy_ctx, {"path": Path(temp_path).name})
            result["path"] = path
            return result
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    def remote_run_shell(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        command = (arguments.get("command") or arguments.get("cmd") or "").strip()
        _validate_shell_command(command)
        timeout = float(arguments.get("timeout", 300.0))
        root = str(_resolve(".", "write")["root"])
        try:
            result = transport.exec(f"cd -- {shlex.quote(root)} && {command}", timeout=timeout)
        except TransportError as exc:
            return {"command": command, "exit_code": 124, "stdout": "",
                    "stderr": str(exc), "timed_out": True}
        return {"command": command, "exit_code": result.rc,
                "stdout": result.out[-8000:], "stderr": result.err[-4000:],
                "timed_out": False}

    registry.register(Tool(
        name="list_files", description="List files inside the remote workspace.",
        parameters={"type": "object", "properties": {"path": {"type": "string", "default": "."}}, "required": []},
        handler=remote_list_files, risk_level="low", ui_component_hint="file_browser"))
    registry.register(Tool(
        name="read_file", description="Read a text file inside the remote workspace.",
        parameters={"type": "object", "properties": {"path": {"type": "string"}, "max_chars": {"type": "integer", "default": 12000}}, "required": ["path"]},
        handler=remote_read_file, risk_level="low", ui_component_hint="text_preview"))
    registry.register(Tool(
        name="write_file", description="Write a text file under the configured remote workspace.",
        parameters={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
        handler=remote_write_file, risk_level="medium", ui_component_hint="file_writer"))
    registry.register(Tool(
        name="inspect_csv", description="Inspect a CSV file inside the remote workspace.",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        handler=remote_inspect_csv, risk_level="low", ui_component_hint="table_preview"))
    _register_run_shell(registry, remote_run_shell, location="remote")

    default_registry = create_default_tools(enable_network=True)
    for name in ("remember", "recall", "current_datetime", "search_web", "search_news", "fetch_url"):
        if name in default_registry._tools:
            registry.register(default_registry._tools[name])
    return registry


def build_app(*, ollama_host: str, model_name: str, workspace: str, db_path: str,
              auth_token: str | None, system_prompt: str | None = None,
              provider: str = "ollama", api_base: str | None = None,
              api_key: str | None = None, tools_target: str | None = None,
              target: str | None = None) -> AgentServerApp:
    """Construct an app with model hosting and tool execution independently selectable.

    ``tools_target`` controls only file/shell placement. ``target`` remains a
    compatibility alias for callers written before that split; new callers should
    never derive ``tools_target`` from the inference target implicitly.
    """
    effective_tools_target = tools_target if tools_target is not None else (target or "local")
    kwargs = dict(
        workspace=workspace,
        db_path=db_path,
        provider=provider,
        model_name=model_name,
        ollama_host=ollama_host,
        api_base=api_base,
        api_key=api_key,
        # Selecting a workspace explicitly authorizes code edits throughout it.
        # WorkspacePolicy still confines paths to this root.
        write_roots=["."],
        enable_network_tools=True,
    )

    # Canvas/visual tools (get_weather, ...) are layered onto the default tools
    # via the AgentServerApp tools_factory hook -- kept in agentica-core so the
    # release never needs them in AgenticLocal's git main.
    def _tools_factory():
        if effective_tools_target and effective_tools_target != "local":
            registry = create_remote_tools(effective_tools_target, workspace)
        else:
            registry = create_local_tools(enable_network=True)
        return register_voice_tools(registry)

    # Pass tools_factory + system_prompt, degrading on an older engine that
    # lacks either kwarg (a release that pip-installs an older AgenticLocal).
    # IMPORTANT: keep a tools_factory-ONLY rung before the bare {} so that an
    # engine which has tools_factory but not system_prompt (e.g. AgenticLocal
    # main) still gets the canvas tools (get_weather, ...) instead of silently
    # falling all the way through to {} and losing them.
    for extra in (
        {"tools_factory": _tools_factory, "system_prompt": system_prompt},
        {"tools_factory": _tools_factory},
        {"system_prompt": system_prompt},
        {},
    ):
        try:
            return AgentServerApp(**kwargs, **extra)
        except TypeError:
            continue
    return AgentServerApp(**kwargs)


def run_gateway_server(handler_cls, host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), handler_cls)
    return server


# --------------------------------------------------------------------------- #
# `slurm-agentic up`
# --------------------------------------------------------------------------- #
def up(
    cluster_path: str,
    *,
    ollama_host: str | None = None,
    model_override: str | None = None,
    workspace: str = "sample_workspace",
    db_path: str = ".agentic/agentic.db",
    v1_mode: str = "agentic",
    skip_preflight: bool = False,
    serve_wait_s: float = 600.0,
    _print=print,
) -> int:
    """Bring the gateway up.

    If ``ollama_host`` is given, skip SLURM and point the gateway straight at it
    (local Ollama, or a manually-created ssh tunnel) -- the Slice 0 path.
    Otherwise: preflight, sbatch the serve job on the cluster, open an SSH tunnel,
    then serve locally.
    """
    cluster = ClusterConfig.resolve(cluster_path)
    token = cluster.auth_token
    model_name = model_override or cluster.model.name

    from dataclasses import replace as _replace
    fit_model = _replace(cluster.model, name=model_name)
    fit = serving.preflight(cluster, fit_model)
    _print(f"[preflight] {fit.message}")
    for w in fit.warnings:
        _print(f"[preflight] ! {w}")
    if fit.verdict == "won't fit" and not skip_preflight and ollama_host is None:
        _print("[preflight] Model does not fit the configured GPUs. "
               "Adjust cluster.yaml (gpus/quant/max_model_len/model) or pass --skip-preflight.")
        return 2

    if ollama_host:
        # Slice 0 / manual mode: no SLURM, no tunnel.
        remote_v1_base = ollama_host.rstrip("/") + "/v1"
        return _serve_local(cluster, ollama_host, remote_v1_base, model_name,
                            workspace, db_path, token, v1_mode, _print)

    # Full path: bring up the model on the cluster, then tunnel.
    transport = Transport.from_cluster(cluster)
    remote_jobdir = transport.expand_home(f"{cluster.remote_workdir}/serve-{uuid.uuid4().hex[:8]}")
    _print(f"[serve] submitting serve job on {cluster.ssh.host} ({cluster.model.engine} {model_name})...")
    handle = serving.bring_up(transport, cluster, remote_jobdir, model=fit_model, wait_timeout_s=serve_wait_s)
    _print(f"[serve] job {handle.job_id} running on node {handle.node}:{handle.port}")

    tunnel_port = cluster.gateway.tunnel_port
    readiness = f"http://127.0.0.1:{tunnel_port}{serving.readiness_path(handle.engine)}"
    _print(f"[tunnel] opening ssh -L {tunnel_port} -> {handle.node}:{handle.port} ...")
    try:
        with transport.tunnel(tunnel_port, handle.node, handle.port,
                              readiness_url=readiness, readiness_timeout_s=serve_wait_s) as tun:
            local_ollama = tun.base_url
            remote_v1_base = local_ollama + "/v1"
            _print(f"[tunnel] ready at {local_ollama}")
            try:
                _serve_local(cluster, local_ollama, remote_v1_base, model_name,
                             workspace, db_path, token, v1_mode, _print,
                             on_exit_note=f"SLURM job {handle.job_id} left RUNNING "
                                          f"(cancel with: slurm-agentic down {cluster_path} --job {handle.job_id})")
            except KeyboardInterrupt:
                _print("\n[gateway] stopping; tunnel will close.")
    except TransportError as exc:
        _print(f"[tunnel] error: {exc}")
        return 1
    return 0


def _serve_local(cluster, ollama_host, remote_v1_base, model_name, workspace, db_path,
                 token, v1_mode, _print, on_exit_note: str | None = None) -> int:
    app = build_app(ollama_host=ollama_host, model_name=model_name,
                    workspace=workspace, db_path=db_path, auth_token=token)
    host = cluster.gateway.bind_host
    port = cluster.gateway.port
    public_api_base = f"http://{host}:{port}/v1"
    handler = make_gateway_handler(
        app, auth_token=token, remote_v1_base=remote_v1_base, v1_mode=v1_mode,
        model_label=model_name, public_api_base=public_api_base,
    )
    server = run_gateway_server(handler, host, port)
    _print("")
    _print(f"  web chat:     http://{host}:{port}/")
    _print(f"  OpenAI API:   {public_api_base}   (model: {model_name}, mode: agentic)")
    _print(f"  API token:    {token or '(none — open access)'}")
    _print(f"  VS Code:      set apiBase={public_api_base} apiKey={token or 'sk-none'} model={model_name}")
    if on_exit_note:
        _print(f"  note:         {on_exit_note}")
    _print("  Ctrl+C to stop.")
    _print("")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _print("\n[gateway] stopped.")
    finally:
        server.server_close()
    return 0


def down(cluster_path: str, job_id: str | None) -> int:
    cluster = ClusterConfig.resolve(cluster_path)
    transport = Transport.from_cluster(cluster)
    if job_id:
        res = transport.scancel(job_id)
        print(f"scancel {job_id}: rc={res.rc} {res.err.strip()}")
        return 0 if res.ok else 1
    print("No --job given. Find running serve jobs with: squeue on the cluster, "
          "or pass --job <id> from the `up` output.")
    return 0
