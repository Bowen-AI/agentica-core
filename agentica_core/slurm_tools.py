"""Custom ToolRegistry for agentic JOB mode (Mode 2).

Adds, on top of AgenticLocal's default file/memory tools:

* ``run_shell``   -- run a shell command, confined to the job workspace cwd, with a
  subprocess timeout. (NOTE: this is not a true sandbox; on a shared cluster pair
  it with an allowlist / dedicated node. AgenticLocal's path policy does not
  sandbox shell tools.)
* ``run_tests``   -- run the project's test command; exit code is the deterministic
  success backstop the auditor cannot override.
* ``check_artifact`` -- assert an expected output file exists under the workspace.
* ``submit_for_audit`` -- the ONLY accepted auditor verdict channel (strict enum).
* ``generate_image`` / ``generate_video`` -- pluggable diffusion stubs (swap the
  backend for a real ComfyUI/diffusers worker later).
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable

from agentic_loop.tools import Tool, ToolContext, ToolRegistry, create_default_tools

# A backend takes (kind, args, workspace_root) -> dict and produces media.
DiffusionBackend = Callable[[str, dict[str, Any], Path], dict[str, Any]]

AUDIT_STATUSES = {"PASS", "FAIL"}


# --------------------------------------------------------------------------- #
# tool handlers
# --------------------------------------------------------------------------- #
def _run_shell_factory(timeout_default: float, denylist: tuple[str, ...]):
    def run_shell(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        command = (arguments.get("command") or arguments.get("cmd") or "").strip()
        if not command:
            raise ValueError("run_shell requires a non-empty 'command'")
        low = command.lower()
        for bad in denylist:
            if bad in low:
                raise PermissionError(f"run_shell blocked: command contains disallowed token {bad!r}")
        timeout = float(arguments.get("timeout", timeout_default))
        cwd = context.workspace_root.resolve()
        try:
            proc = subprocess.run(
                ["bash", "-lc", command], cwd=str(cwd), capture_output=True, text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return {"command": command, "exit_code": 124, "stdout": exc.stdout or "",
                    "stderr": f"timeout after {timeout}s", "timed_out": True}
        return {
            "command": command,
            "exit_code": proc.returncode,
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-4000:],
            "timed_out": False,
        }

    return run_shell


def _run_tests_factory(timeout_default: float):
    def run_tests(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        command = (arguments.get("command") or "").strip()
        if not command:
            raise ValueError("run_tests requires a 'command' (e.g. 'pytest -q')")
        timeout = float(arguments.get("timeout", timeout_default))
        cwd = context.workspace_root.resolve()
        try:
            proc = subprocess.run(["bash", "-lc", command], cwd=str(cwd),
                                  capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            return {"command": command, "exit_code": 124, "passed": False,
                    "output": (exc.stdout or "") + "\n[timeout]"}
        return {
            "command": command,
            "exit_code": proc.returncode,
            "passed": proc.returncode == 0,
            "output": (proc.stdout + "\n" + proc.stderr)[-8000:],
        }

    return run_tests


def check_artifact(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    rel = arguments.get("path")
    if not rel:
        raise ValueError("check_artifact requires 'path'")
    root = context.workspace_root.resolve()
    target = (root / rel).resolve()
    exists = target.exists() and (root == target or root in target.parents)
    return {"path": rel, "exists": bool(exists),
            "bytes": target.stat().st_size if exists and target.is_file() else 0}


def submit_for_audit(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    status = str(arguments.get("status", "")).strip().upper()
    if status not in AUDIT_STATUSES:
        raise ValueError(f"submit_for_audit status must be one of {sorted(AUDIT_STATUSES)}")
    gaps = arguments.get("gaps") or ""
    if isinstance(gaps, list):
        gaps = "; ".join(str(g) for g in gaps)
    return {"status": status, "gaps": str(gaps)}


def _stub_diffusion(kind: str, args: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
    out = args.get("out") or ("output.mp4" if kind == "video" else "output.png")
    target = (workspace_root / out)
    target.parent.mkdir(parents=True, exist_ok=True)
    intent = {"kind": kind, "prompt": args.get("prompt", ""), "image": args.get("image"),
              "out": out, "stub": True}
    # Placeholder bytes so downstream check_artifact passes; swap for a real backend.
    target.write_bytes(b"SLURM_AGENTIC_STUB_" + kind.upper().encode())
    (workspace_root / (out + ".intent.json")).write_text(
        __import__("json").dumps(intent, indent=2), encoding="utf-8")
    return {"generated": out, "kind": kind, "stub": True,
            "note": "stub output -- wire a ComfyUI/diffusers backend to produce real media"}


def _generate_factory(kind: str, backend: DiffusionBackend):
    def handler(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        return backend(kind, arguments, context.workspace_root.resolve())

    return handler


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def create_job_tools(
    *,
    enable_network: bool = False,
    allow_shell: bool = True,
    shell_timeout: float = 300.0,
    shell_denylist: tuple[str, ...] = ("rm -rf /", "shutdown", "mkfs", ":(){", "/etc/passwd"),
    diffusion_backend: DiffusionBackend | None = None,
) -> ToolRegistry:
    """Build the Mode-2 tool registry (default tools + job tools)."""
    registry = create_default_tools(enable_network=enable_network)
    backend = diffusion_backend or _stub_diffusion

    if allow_shell:
        registry.register(Tool(
            name="run_shell",
            description="Run a shell command in the job workspace and return exit code + output.",
            parameters={"type": "object", "properties": {
                "command": {"type": "string"}, "timeout": {"type": "number", "default": shell_timeout}},
                "required": ["command"]},
            handler=_run_shell_factory(shell_timeout, shell_denylist),
            source="job", risk_level="high",
        ))
    registry.register(Tool(
        name="run_tests",
        description="Run the project's test command (e.g. 'pytest -q'); exit code 0 means pass.",
        parameters={"type": "object", "properties": {
            "command": {"type": "string"}, "timeout": {"type": "number", "default": shell_timeout}},
            "required": ["command"]},
        handler=_run_tests_factory(shell_timeout), source="job", risk_level="high",
    ))
    registry.register(Tool(
        name="check_artifact",
        description="Check that an expected output file exists under the workspace.",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        handler=check_artifact, source="job", risk_level="low",
    ))
    registry.register(Tool(
        name="submit_for_audit",
        description="Auditor verdict. status must be exactly 'PASS' or 'FAIL'; gaps lists what is missing.",
        parameters={"type": "object", "properties": {
            "status": {"type": "string", "enum": ["PASS", "FAIL"]},
            "gaps": {"type": "string"}}, "required": ["status"]},
        handler=submit_for_audit, source="job", risk_level="low",
    ))
    registry.register(Tool(
        name="generate_image",
        description="Generate an image from a text prompt (pluggable diffusion backend).",
        parameters={"type": "object", "properties": {
            "prompt": {"type": "string"}, "out": {"type": "string", "default": "output.png"}},
            "required": ["prompt"]},
        handler=_generate_factory("image", backend), source="job", risk_level="medium",
    ))
    registry.register(Tool(
        name="generate_video",
        description="Generate a video from an image and/or text prompt (pluggable diffusion backend).",
        parameters={"type": "object", "properties": {
            "prompt": {"type": "string"}, "image": {"type": "string"},
            "out": {"type": "string", "default": "output.mp4"}}, "required": ["prompt"]},
        handler=_generate_factory("video", backend), source="job", risk_level="medium",
    ))
    return registry
