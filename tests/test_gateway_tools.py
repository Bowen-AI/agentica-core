"""Gateway code tools: real local execution and SSH-free remote command coverage."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from agentic_loop.tools import ToolContext
from agentic_loop.types import ToolCall
from agentica_core import gateway
from agentica_core.transport import ExecResult, TransportError


class FakeRemoteTransport:
    """Execute remote commands locally while preserving the Transport surface."""

    def __init__(self):
        self.commands: list[str] = []

    def exec(self, command: str, timeout: float | None = 120.0) -> ExecResult:
        self.commands.append(command)
        proc = subprocess.run(
            ["bash", "-lc", command], capture_output=True, text=True, timeout=timeout,
        )
        return ExecResult(proc.returncode, proc.stdout, proc.stderr)

    def _rsync(self, src: str, dst: str, to_remote: bool, delete: bool) -> ExecResult:
        del to_remote, delete
        try:
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            return ExecResult(0, "", "")
        except OSError as exc:
            return ExecResult(1, "", str(exc))


def test_local_code_tools_run_in_workspace_and_allow_code_writes(tmp_path):
    tools = gateway.create_local_tools(enable_network=False)
    context = ToolContext(workspace_root=tmp_path)

    result = tools.run(
        "run_shell", context,
        {"command": "pwd; printf local-ok > generated.txt"},
    )
    assert result["exit_code"] == 0
    assert result["stdout"].strip() == str(tmp_path.resolve())
    assert (tmp_path / "generated.txt").read_text() == "local-ok"

    app = gateway.build_app(
        ollama_host="http://127.0.0.1:11434",
        model_name="test-model",
        workspace=str(tmp_path),
        db_path=str(tmp_path / "agent.db"),
        auth_token=None,
        tools_target="local",
    )
    controller = app._make_controller()
    decision = controller.policy.check(ToolCall("write_file", {"path": "src/module.py"}))
    assert decision.allowed, decision.reason
    assert "run_shell" in controller.tools.names()
    assert {"get_weather", "show_web"}.issubset(controller.tools.names())


def test_explicit_tools_target_is_independent_of_legacy_target(tmp_path, monkeypatch):
    def unexpected_remote_tools(*args, **kwargs):
        pytest.fail("remote tools must not be selected when tools_target='local'")

    monkeypatch.setattr(gateway, "create_remote_tools", unexpected_remote_tools)
    app = gateway.build_app(
        ollama_host="http://remote-model.invalid:11434",
        model_name="large-model",
        workspace=str(tmp_path),
        db_path=str(tmp_path / "split-target.db"),
        auth_token=None,
        target="remote-inference-host",
        tools_target="local",
    )
    assert "run_shell" in app._create_tools().names()


def test_fake_remote_file_and_shell_handlers_execute_with_quoted_paths(tmp_path):
    remote_workspace = tmp_path / "remote workspace's files"
    fake = FakeRemoteTransport()
    tools = gateway.create_remote_tools(
        "unused-host", str(remote_workspace), transport=fake,  # type: ignore[arg-type]
    )
    context = ToolContext(workspace_root=tmp_path)
    tricky_path = "src/it's ready.txt"

    written = tools.run("write_file", context, {"path": tricky_path, "content": "hello remote"})
    assert written["bytes_written"] == len("hello remote")
    assert (remote_workspace / tricky_path).read_text() == "hello remote"

    read = tools.run("read_file", context, {"path": tricky_path, "max_chars": 5})
    assert read == {"path": tricky_path, "content": "hello", "truncated": True}

    listed = tools.run("list_files", context, {"path": "."})
    assert tricky_path in listed["files"]

    csv_path = remote_workspace / "data set.csv"
    csv_path.write_text("name,value\na,1\nb,2\n")
    inspected = tools.run("inspect_csv", context, {"path": csv_path.name})
    assert inspected["path"] == csv_path.name
    assert inspected["rows"] == 2
    assert inspected["columns"] == ["name", "value"]

    shell = tools.run("run_shell", context, {"command": "pwd; printf shell-ok"})
    assert shell["exit_code"] == 0
    assert shell["stdout"].splitlines() == [str(remote_workspace.resolve()), "shell-ok"]
    assert any("python3 -c" in command for command in fake.commands)


def test_fake_remote_handlers_reject_workspace_escape(tmp_path):
    fake = FakeRemoteTransport()
    tools = gateway.create_remote_tools(
        "unused-host", str(tmp_path / "remote"), transport=fake,  # type: ignore[arg-type]
    )
    context = ToolContext(workspace_root=tmp_path)

    with pytest.raises(TransportError, match="path escapes workspace"):
        tools.run("read_file", context, {"path": "../outside.txt"})
