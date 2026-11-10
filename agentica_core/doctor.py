"""Offline release/runtime validation; never downloads model weights."""
import importlib
import importlib.util
import inspect
import platform
import tempfile
from pathlib import Path

from . import __version__


def check(*, require_voice=False):
    errors = []
    modules = ["yaml", "websockets", "agentic_loop.server"]
    if require_voice:
        modules += ["faster_whisper", "kokoro_onnx", "soundfile"]
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            # MLX's native initializer can abort without a Metal device. Hosted
            # macOS builders must still validate the bundled package and use the
            # required CPU fallback; only import MLX when Metal is available.
            from .voice_provision import _metal_available
            if importlib.util.find_spec("mlx_whisper") is None:
                errors.append("mlx_whisper: package is missing")
            elif _metal_available():
                modules += ["mlx_whisper"]
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as exc:
            errors.append(f"{module}: {exc}")
    try:
        from .gateway import build_app
        from agentic_loop.server import AgentServerApp
        # Approval/cancellation support is required, never silently removed.
        for parameter in ("on_event", "cancel_event", "approval_callback"):
            if parameter not in inspect.signature(AgentServerApp.chat).parameters:
                errors.append(f"AgenticLocal is incompatible: missing chat({parameter}=...)")
        with tempfile.TemporaryDirectory(prefix="agentica-doctor-") as folder:
            app = build_app(ollama_host="http://127.0.0.1:11434", model_name="rule",
                            workspace=folder, db_path=str(Path(folder) / "test.db"),
                            auth_token=None, provider="rule")
            tools = app._create_tools().names()
            for name in ("read_file", "write_file", "run_shell", "get_weather", "show_web"):
                if name not in tools:
                    errors.append(f"Missing agent tool: {name}")
    except Exception as exc:
        errors.append(f"Agent runtime: {exc}")
    return {"ok": not errors, "version": __version__, "voice_checked": require_voice, "errors": errors}
