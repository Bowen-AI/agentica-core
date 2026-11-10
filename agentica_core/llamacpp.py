"""One owned llama.cpp server, shared by agent, voice and job requests.

Hold a lease for the whole turn, so switching models never kills another job's
inference. Requests queue; only one GGUF stays resident. No shell or pip runtime.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request


def find_server() -> str | None:
    explicit = os.environ.get("AGENTICA_LLAMA_SERVER")
    if explicit:
        return explicit if os.path.isfile(explicit) and os.access(explicit, os.X_OK) else None
    return shutil.which("llama-server") or next((str(p) for p in (
        Path("/opt/homebrew/bin/llama-server"), Path("/usr/local/bin/llama-server"),
        Path.home() / ".local/bin/llama-server",
    ) if p.is_file() and os.access(p, os.X_OK)), None)


def gguf_path(model: str) -> Path:
    path = Path(model).expanduser()
    if not path.is_absolute() or path.suffix.lower() != ".gguf":
        raise ValueError("llama.cpp needs an absolute path to a local .gguf file (~/ is supported).")
    if not path.is_file():
        raise ValueError(f"GGUF file not found: {path}")
    with path.open("rb") as source:
        if source.read(4) != b"GGUF":
            raise ValueError(f"Not a GGUF model: {path}")
    return path.resolve()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class LlamaServer:
    def __init__(self, data_dir: Path, *, timeout: float = 180):
        self.data_dir = data_dir
        self.timeout = timeout
        self._lock = threading.RLock()
        self._process = None
        self._model = None
        self._base_url = None
        self._log = None
        self._closing = False

    @contextmanager
    def use(self, model: str, notify=None):
        with self._lock:
            if self._closing:
                raise RuntimeError("The local model runtime is shutting down.")
            yield self._start(model, notify)

    def _start(self, model: str, notify=None) -> str:
        path = gguf_path(model)
        if self._model == (path, model) and self._process and self._process.poll() is None:
            return self._base_url
        binary = find_server()
        if not binary:
            raise RuntimeError("llama-server is missing. Install llama.cpp (on Mac: brew install llama.cpp), "
                               "or set AGENTICA_LLAMA_SERVER to its executable. Ollama is also available.")
        context = int(os.environ.get("AGENTICA_LLAMA_CONTEXT", "8192"))
        if not 512 <= context <= 131072:
            raise ValueError("AGENTICA_LLAMA_CONTEXT must be between 512 and 131072 tokens")
        port = free_port()
        self._stop()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.data_dir / "llama-server.log"
        self._log = log_path.open("wb")
        self._base_url = f"http://127.0.0.1:{port}"
        if notify:
            notify(f"Loading {path.name} with llama.cpp…")
        try:
            # The alias stays the selected path, matching the agent's model ID.
            # A bounded context/one slot avoids multiplying KV memory on laptops.
            self._process = subprocess.Popen([
                binary, "--model", str(path), "--alias", model,
                "--host", "127.0.0.1", "--port", str(port), "--jinja",
                "--ctx-size", str(context), "--parallel", "1",
                "--n-gpu-layers", "99", "--no-webui",
            ], stdin=subprocess.DEVNULL, stdout=self._log, stderr=subprocess.STDOUT)
            self._model = (path, model)
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    details = log_path.read_text(errors="replace")[-2000:]
                    raise RuntimeError(f"llama-server exited while loading the model. See {log_path}\n{details}")
                try:
                    with urllib.request.urlopen(self._base_url + "/health", timeout=1) as response:
                        if json.load(response).get("status") == "ok":
                            return self._base_url
                except (urllib.error.URLError, TimeoutError, ValueError):
                    pass
                time.sleep(0.1)
            raise RuntimeError(f"llama-server did not become ready in {self.timeout:g}s. See {log_path}")
        except BaseException:
            self._stop()
            raise

    def _stop(self):
        if self._process:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
            self._process = None
        if self._log:
            self._log.close()
            self._log = None
        self._model = self._base_url = None

    def close(self):
        self._closing = True
        # Interrupt active HTTP inference before waiting for the lease to unwind.
        process = self._process
        if process and process.poll() is None:
            process.terminate()
        with self._lock:
            self._stop()


class LeasedAgentApp:
    """Keep storage methods available while leasing the model only for inference."""
    def __init__(self, delegate, server, model, factory):
        self.delegate, self.server, self.model, self.factory = delegate, server, model, factory

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def chat(self, message, session_id=None, *, cancel_event=None, on_event=None,
             max_steps=None, approval_callback=None):
        with self.server.use(self.model) as base_url:
            if cancel_event is not None and cancel_event.is_set():
                return {"session_id": session_id, "final_answer": "(interrupted)", "steps": []}
            return self.factory(base_url).chat(
                message, session_id, cancel_event=cancel_event, on_event=on_event,
                max_steps=max_steps, approval_callback=approval_callback,
            )
