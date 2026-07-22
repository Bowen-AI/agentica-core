"""Remote job workspace selection and plain-SSH lifecycle regression tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentica_core import job
from agentica_core.config import ClusterConfig, PlanConfig, SSHConfig
from agentica_core.transport import ExecResult, TransportError


def _cluster(scheduler: str = "ssh") -> ClusterConfig:
    return ClusterConfig(
        name="box",
        ssh=SSHConfig(host="box"),
        scheduler=scheduler,
    )


def _plan(workspace: str) -> PlanConfig:
    return PlanConfig(title="job", goal="do work", workspace=workspace)


class _WorkspaceTransport:
    def __init__(self, *, remote_exists: bool = True):
        self.remote_exists = remote_exists
        self.commands: list[str] = []
        self.pushes: list[tuple[str, str]] = []

    def expand_home(self, path: str) -> str:
        return "/home/test" + path[1:] if path.startswith("~") else path

    def exec(self, command, timeout=120.0):
        self.commands.append(command)
        if command.startswith("test -d ") and not self.remote_exists:
            return ExecResult(1, "", "missing")
        return ExecResult(0, "", "")

    def push_dir(self, source: str, destination: str):
        self.pushes.append((source, destination))
        return ExecResult(0, "", "")


def test_workspace_spec_supports_explicit_argument_and_plan_prefix():
    assert job._workspace_spec("/local/repo", "local") == (
        "local", "/local/repo", True,
    )
    assert job._workspace_spec("remote:~/repo", None) == (
        "remote", "~/repo", True,
    )
    assert job._workspace_spec("local:/tmp/repo", None) == (
        "local", "/tmp/repo", True,
    )
    with pytest.raises(ValueError, match="workspace_source"):
        job._workspace_spec("/tmp/repo", "somewhere")


def test_local_workspace_is_staged_into_per_job_dir(tmp_path):
    workspace = tmp_path / "repo with spaces"
    workspace.mkdir()
    transport = _WorkspaceTransport()
    lines: list[str] = []

    execution = job._prepare_execution_workspace(
        transport, _plan(str(workspace)), "/remote/job-1", "local", lines.append,
    )

    assert execution == "/remote/job-1/workspace"
    assert transport.pushes == [(str(workspace), "/remote/job-1/workspace")]
    assert any("rsync workspace" in line for line in lines)


def test_explicit_missing_local_workspace_fails_loudly(tmp_path):
    transport = _WorkspaceTransport()
    missing = tmp_path / "missing"
    with pytest.raises(TransportError, match="local workspace"):
        job._prepare_execution_workspace(
            transport, _plan(str(missing)), "/remote/job-1", "local", print,
        )


def test_remote_workspace_is_used_in_place_and_not_uploaded():
    transport = _WorkspaceTransport()
    lines: list[str] = []

    execution = job._prepare_execution_workspace(
        transport, _plan("~/repo with spaces"), "/remote/job-1", "remote", lines.append,
    )

    assert execution == "/home/test/repo with spaces"
    assert transport.pushes == []
    assert transport.commands == ["test -d '/home/test/repo with spaces'"]
    assert lines == ["[workspace] using remote workspace /home/test/repo with spaces"]


def test_missing_remote_workspace_fails_before_submit():
    transport = _WorkspaceTransport(remote_exists=False)
    with pytest.raises(TransportError, match="remote workspace"):
        job._prepare_execution_workspace(
            transport, _plan("/srv/missing"), "/remote/job-1", "remote", print,
        )


def test_ssh_and_slurm_scripts_execute_in_selected_remote_workspace():
    cluster = _cluster()
    plan = _plan("/unused")
    workspace = "/srv/shared/repo with spaces"

    ssh_script = job._ssh_runner_script(
        cluster, plan, "/remote/job-1", execution_workspace=workspace,
    )
    slurm_script = job.render_job_sbatch(
        cluster, plan, "/remote/job-1", execution_workspace=workspace,
    )

    assert "--workspace '/srv/shared/repo with spaces'" in ssh_script
    assert "--workspace '/srv/shared/repo with spaces'" in slurm_script
    assert '--workspace "/remote/job-1/workspace"' not in ssh_script
    assert '--workspace "/remote/job-1/workspace"' not in slurm_script


def test_submit_threads_remote_workspace_to_ssh_runner(monkeypatch, tmp_path):
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        "title: remote source\ngoal: do work\nworkspace: remote:/srv/source\n",
        encoding="utf-8",
    )
    cluster = _cluster("ssh")
    transport = _WorkspaceTransport()
    captured: dict[str, str] = {}

    monkeypatch.setattr(
        job.ClusterConfig, "resolve", classmethod(lambda cls, target: cluster),
    )
    monkeypatch.setattr(
        job.Transport, "from_cluster", classmethod(lambda cls, selected: transport),
    )
    monkeypatch.setattr(
        job.serving,
        "preflight",
        lambda *args: SimpleNamespace(
            message="fits", warnings=[], verdict="good",
        ),
    )
    monkeypatch.setattr(job.serving, "detect_scheduler", lambda *args: "ssh")

    def fake_submit(
        selected_transport,
        selected_cluster,
        plan,
        remote_jobdir,
        cluster_path,
        printer,
        execution_workspace=None,
    ):
        captured["workspace"] = execution_workspace
        return 0

    monkeypatch.setattr(job, "_submit_ssh", fake_submit)

    rc = job.submit(
        "box",
        str(plan_path),
        sync_code=False,
    )

    assert rc == 0
    assert captured["workspace"] == "/srv/source"
    assert transport.pushes == []


def test_plain_ssh_launch_records_runner_pid():
    transport = _WorkspaceTransport()
    lines: list[str] = []

    rc = job._submit_ssh(
        transport,
        _cluster(),
        _plan("/unused"),
        "/remote/job-abc",
        "box",
        lines.append,
        execution_workspace="/srv/repo",
    )

    assert rc == 0
    launch = transport.commands[-1]
    assert "setsid bash runner.sh" in launch
    assert "runner.pid" in launch
    assert "pid=$!" in launch
    assert any("job_id=job-abc" in line for line in lines)


class _CancelTransport:
    def __init__(self):
        self.commands: list[str] = []
        self.cancelled: list[str] = []

    def expand_home(self, path: str) -> str:
        return "/home/test" + path[1:] if path.startswith("~") else path

    def exec(self, command, timeout=120.0):
        self.commands.append(command)
        return ExecResult(0, "terminated pid=123", "")

    def scancel(self, job_id: str):
        self.cancelled.append(job_id)
        return ExecResult(0, "", "")


@pytest.mark.parametrize("scheduler", ["ssh", "slurm"])
def test_cancel_uses_runner_pid_for_ssh_and_scancel_for_slurm(monkeypatch, scheduler):
    cluster = _cluster(scheduler)
    transport = _CancelTransport()
    monkeypatch.setattr(
        job.ClusterConfig, "resolve", classmethod(lambda cls, target: cluster),
    )
    monkeypatch.setattr(
        job.Transport, "from_cluster", classmethod(lambda cls, selected: transport),
    )

    rc = job.cancel("box", "job-abc")

    assert rc == 0
    if scheduler == "ssh":
        assert transport.cancelled == []
        assert len(transport.commands) == 1
        command = transport.commands[0]
        assert "/home/test/.slurm-agentic/jobs/job-abc/runner.pid" in command
        assert "kill -TERM" in command
        assert "cancel.marker" in command
    else:
        assert transport.cancelled == ["job-abc"]
        assert transport.commands == []
