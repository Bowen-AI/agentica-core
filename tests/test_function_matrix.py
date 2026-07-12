"""Cross-product coverage for Agentica's public agentic execution surfaces.

All SSH/SLURM behavior is mocked: these tests verify target routing and
workspace placement without requiring a configured host or model server.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentica_core import apiserver, cli, gateway, job, voice_gateway
from agentica_core.config import ClusterConfig, SSHConfig
from agentica_core.transport import ExecResult


_MODEL_WORKSPACE_MATRIX = [
    ("local", "local", "ollama"),
    ("local", "workspace-box", "ollama"),
    ("gpu-box", "local", "vllm"),
    ("gpu-box", "workspace-box", "vllm"),
]


class _AgentApp:
    def chat(self, message, session_id):
        return {
            "session_id": session_id or "matrix-session",
            "final_answer": f"acted: {message}",
            "steps": [{"tool_name": "run_shell"}],
            "transcript": [],
        }

    def create_session(self):
        return "matrix-voice-session"


class _RoutingState:
    def __init__(self):
        self.calls = []
        self.app = _AgentApp()

    def app_for(self, workspace, **kwargs):
        self.calls.append((workspace, kwargs))
        return self.app


@pytest.mark.parametrize("model_target,workspace_target,engine", _MODEL_WORKSPACE_MATRIX)
def test_chat_routes_every_model_workspace_combination_agentically(
    model_target, workspace_target, engine,
):
    state = _RoutingState()
    handler = apiserver.make_handler(state)
    result = handler._chat(SimpleNamespace(), {
        "message": "inspect and fix the repository",
        "mode": "plain",  # old clients cannot opt out of the agent loop
        "session_id": "matrix-session",
        "workspace": "/repo",
        "target": model_target,
        "workspace_target": workspace_target,
        "model": "local-tag" if engine == "ollama" else "org/large-model",
        "engine": engine,
    })

    assert result["mode"] == "agentic"
    assert result["steps"] == [{"tool_name": "run_shell"}]
    assert state.calls == [("/repo", {
        "target": model_target,
        "model": "local-tag" if engine == "ollama" else "org/large-model",
        "engine": engine,
        "workspace_target": workspace_target,
    })]


@pytest.mark.parametrize("model_target,workspace_target,engine", _MODEL_WORKSPACE_MATRIX)
def test_plan_reads_context_from_workspace_target_not_model_target(
    monkeypatch, model_target, workspace_target, engine,
):
    context_calls = []

    class PlanState:
        def cluster_path(self, target):
            return target if target == "local" else f"/clusters/{target}.yaml"

        def complete_stream(self, messages, *args, **kwargs):
            self.messages = messages
            self.kwargs = kwargs
            return json.dumps({"title": "Matrix", "steps": ["Inspect files"]})

    state = PlanState()
    monkeypatch.setattr(
        apiserver,
        "workspace_summary",
        lambda workspace, workspace_target=None, **_kwargs: (
            context_calls.append((workspace, workspace_target)) or "workspace context"
        ),
    )

    plan = apiserver.draft_plan(
        state, "make it work", "/repo",
        target=model_target,
        model="local-tag" if engine == "ollama" else "org/large-model",
        engine=engine,
        workspace_target=workspace_target,
    )

    expected_workspace_host = (
        "local" if workspace_target == "local"
        else f"/clusters/{workspace_target}.yaml"
    )
    assert context_calls == [("/repo", expected_workspace_host)]
    assert state.kwargs["target"] == model_target
    assert state.kwargs["engine"] == engine
    assert plan["lines"][0]["text"] == "Inspect files"


class _VoiceSocket:
    def __init__(self):
        self.frames = []

    async def send(self, raw):
        self.frames.append(json.loads(raw))


@pytest.mark.parametrize("model_target,workspace_target,engine", _MODEL_WORKSPACE_MATRIX)
def test_voice_start_routes_every_model_workspace_combination(
    monkeypatch, model_target, workspace_target, engine,
):
    monkeypatch.delenv("AGENTICA_API_TOKEN", raising=False)

    async def scenario():
        state = _RoutingState()
        ws = _VoiceSocket()
        conn = voice_gateway._Conn(state, ws)
        await conn._dispatch({
            "type": "start",
            "target": model_target,
            "model": "local-tag" if engine == "ollama" else "org/large-model",
            "model_engine": engine,
            "workspace": "/repo",
            "workspace_target": workspace_target,
        })
        return state.calls, ws.frames

    calls, frames = asyncio.run(scenario())
    assert calls == [("/repo", {
        "target": model_target,
        "model": "local-tag" if engine == "ollama" else "org/large-model",
        "engine": engine,
        "workspace_target": workspace_target,
    })]
    assert frames == [{"type": "session", "session_id": "matrix-voice-session"}]


@pytest.mark.parametrize("legacy_mode", ["agentic", "passthrough"])
@pytest.mark.parametrize("requested_model", ["served-model", "agentic", "agent"])
def test_openai_compat_modes_and_model_aliases_all_run_agent_loop(
    legacy_mode, requested_model,
):
    calls = []

    class App:
        def run_once(self, goal):
            calls.append(goal)
            return {"final_answer": "agent completed it"}

    handler = gateway.make_gateway_handler(
        App(),
        auth_token=None,
        remote_v1_base="http://plain-upstream-must-not-be-used.invalid/v1",
        v1_mode=legacy_mode,
        model_label="served-model",
        public_api_base="http://127.0.0.1:8770/v1",
    )
    payloads = []
    stub = SimpleNamespace(
        _v1_agentic=lambda body, stream: handler._v1_agentic(
            SimpleNamespace(
                _json=lambda payload: payloads.append(payload),
                _sse_openai_single=lambda text: payloads.append({"stream": text}),
            ),
            body,
            stream,
        )
    )

    handler._handle_v1(stub, {
        "model": requested_model,
        "messages": [{"role": "user", "content": "perform the task"}],
        "stream": False,
    })

    assert calls == ["perform the task"]
    assert payloads[0]["choices"][0]["message"]["content"] == "agent completed it"
    assert not hasattr(handler, "_v1_passthrough")


class _SubmitTransport:
    def __init__(self):
        self.commands = []
        self.pushes = []
        self.submissions = []

    def expand_home(self, path):
        return "/home/test" + path[1:] if path.startswith("~") else path

    def exec(self, command, timeout=120.0):
        self.commands.append(command)
        return ExecResult(0, "", "")

    def push_dir(self, source, destination):
        self.pushes.append((source, destination))
        return ExecResult(0, "", "")

    def sbatch(self, script):
        self.submissions.append(script)
        return "12345"


@pytest.mark.parametrize("scheduler", ["ssh", "slurm"])
@pytest.mark.parametrize("workspace_source", ["local", "remote"])
def test_remote_job_worker_matrix_stages_or_runs_in_place(
    monkeypatch, tmp_path, scheduler, workspace_source,
):
    workspace = tmp_path / "local repo" if workspace_source == "local" else Path("/srv/remote-repo")
    if workspace_source == "local":
        workspace.mkdir()
    plan_path = tmp_path / f"{scheduler}-{workspace_source}.yaml"
    plan_path.write_text(
        f"title: matrix\ngoal: perform work\nworkspace: {workspace}\n",
        encoding="utf-8",
    )
    cluster = ClusterConfig(
        name="worker", ssh=SSHConfig(host="worker"), scheduler=scheduler,
    )
    transport = _SubmitTransport()
    captured = {}

    monkeypatch.setattr(
        job.ClusterConfig, "resolve", classmethod(lambda cls, _selected: cluster),
    )
    monkeypatch.setattr(
        job.Transport, "from_cluster", classmethod(lambda cls, _selected: transport),
    )
    monkeypatch.setattr(
        job.serving,
        "preflight",
        lambda *_args: SimpleNamespace(message="fits", warnings=[], verdict="good"),
    )
    monkeypatch.setattr(job.serving, "detect_scheduler", lambda *_args: scheduler)

    def fake_ssh(
        _transport, _cluster, _plan, _jobdir, _cluster_path, _print,
        execution_workspace=None,
    ):
        captured["execution_workspace"] = execution_workspace
        return 0

    def fake_sbatch(_cluster, _plan, _jobdir, execution_workspace=None):
        captured["execution_workspace"] = execution_workspace
        return "#!/bin/bash\ntrue\n"

    monkeypatch.setattr(job, "_submit_ssh", fake_ssh)
    monkeypatch.setattr(job, "render_job_sbatch", fake_sbatch)

    assert job.submit(
        "worker", str(plan_path), sync_code=False,
        workspace_source=workspace_source,
    ) == 0

    if workspace_source == "local":
        assert captured["execution_workspace"].endswith("/workspace")
        assert transport.pushes == [
            (str(workspace), captured["execution_workspace"]),
        ]
    else:
        assert captured["execution_workspace"] == "/srv/remote-repo"
        assert transport.pushes == []
    assert bool(transport.submissions) is (scheduler == "slurm")


@pytest.mark.parametrize("workspace_source", ["local", "remote"])
def test_cli_job_submit_exposes_workspace_placement(monkeypatch, workspace_source):
    captured = {}

    def fake_submit(cluster, plan, **kwargs):
        captured.update(cluster=cluster, plan=plan, **kwargs)
        return 0

    monkeypatch.setattr(job, "submit", fake_submit)
    args = SimpleNamespace(
        job_command="submit",
        cluster="worker",
        plan="plan.yaml",
        no_sync_code=False,
        workspace_source=workspace_source,
    )

    assert cli._job(args) == 0
    assert captured == {
        "cluster": "worker",
        "plan": "plan.yaml",
        "sync_code": True,
        "workspace_source": workspace_source,
    }


def test_cli_job_cancel_threads_explicit_job_directory(monkeypatch):
    captured = {}

    def fake_cancel(cluster, job_id, **kwargs):
        captured.update(cluster=cluster, job_id=job_id, **kwargs)
        return 0

    monkeypatch.setattr(job, "cancel", fake_cancel)
    args = SimpleNamespace(
        job_command="cancel",
        cluster="worker",
        job="job-123",
        jobdir="/custom/jobs/job-123",
    )

    assert cli._job(args) == 0
    assert captured == {
        "cluster": "worker",
        "job_id": "job-123",
        "jobdir": "/custom/jobs/job-123",
    }
