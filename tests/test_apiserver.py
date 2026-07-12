"""Unit tests for the agentica-core JSON API helpers (no live model needed)."""

import json
import sqlite3
import threading
import types
from dataclasses import replace
from pathlib import Path

import pytest

from agentica_core import apiserver
from agentica_core.config import ClusterConfig, ModelConfig, PlanConfig, SSHConfig


def _state(monkeypatch_reply: str):
    st = apiserver.State(ollama_host="http://127.0.0.1:11434", model="test-model",
                         workspace="sample_workspace", db_path="/tmp/agentica-test.db")
    st.complete = lambda messages, **k: monkeypatch_reply  # type: ignore[assignment]
    # draft/refine use the streaming planner path (think=False, fmt=json)
    st.complete_stream = lambda messages, *a, **k: monkeypatch_reply  # type: ignore[assignment]
    return st


def test_extract_json():
    assert apiserver._extract_json('{"a": 1}') == {"a": 1}
    assert apiserver._extract_json('prefix {"a": [1,2]} suffix') == {"a": [1, 2]}
    assert apiserver._extract_json("no json here") is None
    assert apiserver._extract_json("{broken") is None


def test_fallback_steps():
    steps = apiserver._fallback_steps("1. do a\n2) do b\n- do c\n\n")
    assert steps == ["do a", "do b", "do c"]


def test_draft_plan_parses_json():
    st = _state('{"title":"T","steps":["step one","step two"],"tests":"pytest -q","artifacts":["out.txt"]}')
    plan = apiserver.draft_plan(st, "do the thing", None)
    assert plan["title"] == "T"
    assert [l["text"] for l in plan["lines"]] == ["step one", "step two"]
    assert plan["lines"][0]["id"] == "L1"
    assert plan["tests"] == "pytest -q"
    assert plan["artifacts"] == ["out.txt"]


def test_draft_plan_falls_back_on_non_json():
    st = _state("1. first\n2. second")
    plan = apiserver.draft_plan(st, "goal", None)
    assert [l["text"] for l in plan["lines"]] == ["first", "second"]
    assert plan["goal"] == "goal"


def test_refine_plan_incorporates_and_reparses():
    st = _state('{"title":"T2","steps":["revised"],"tests":"true","artifacts":[]}')
    base = apiserver._plan_payload("T", "goal", ["a", "b"], "", [])
    refined = apiserver.refine_plan(st, base, [{"line": "L1", "text": "make it better"}])
    assert refined["title"] == "T2"
    assert [l["text"] for l in refined["lines"]] == ["revised"]


def test_plan_to_yaml_roundtrips_into_planconfig(tmp_path):
    plan = apiserver._plan_payload("My Job", "achieve X", ["a", "b"], "pytest -q", ["x.txt"])
    y = apiserver.plan_to_yaml(plan, str(tmp_path))
    p = tmp_path / "plan.yaml"
    p.write_text(y)
    pc = PlanConfig.load(p)
    assert pc.title == "My Job"
    assert pc.goal == "achieve X"
    assert pc.checklist == ["a", "b"]
    assert pc.success_criteria.tests == "pytest -q"
    assert pc.success_criteria.artifacts == ["x.txt"]


def test_plan_to_yaml_can_pin_selected_model(tmp_path):
    plan = apiserver._plan_payload("My Job", "achieve X", ["a"], "", [])
    y = apiserver.plan_to_yaml(plan, str(tmp_path), model="Qwen/Qwen3-32B", engine="vllm")
    p = tmp_path / "plan.yaml"
    p.write_text(y)
    pc = PlanConfig.load(p)
    assert pc.model is not None
    assert pc.model.engine == "vllm"
    assert pc.model.name == "Qwen/Qwen3-32B"


def test_plan_to_yaml_preserves_full_resolved_model_config(tmp_path):
    model = ModelConfig(
        engine="vllm", name="org/large-model", quantization="awq",
        tensor_parallel_size=4, pipeline_parallel_size=2,
        gpu_memory_utilization=0.82, max_model_len=32768,
        serve_port=9012, timeout_s=777,
    )
    plan = apiserver._plan_payload("My Job", "achieve X", ["a"], "", [])
    p = tmp_path / "plan.yaml"
    p.write_text(apiserver.plan_to_yaml(
        plan, str(tmp_path), model_config=model,
    ))

    loaded = PlanConfig.load(p).model
    assert loaded == model


def test_runtime_preamble_describes_remote_inference():
    text = apiserver._runtime_preamble(
        "/ws", target="pinotage.usc.edu", model="llama3.2:3b", engine="ollama"
    )
    assert "Model inference target: pinotage.usc.edu using llama3.2:3b via ollama" in text
    # workspace_target defaults to local: the tools stay on this machine.
    assert "file/shell tools operate on the local" in text
    assert "- You are running locally" not in text


def test_runtime_preamble_describes_remote_workspace():
    text = apiserver._runtime_preamble(
        "/ws", target="local", workspace_target="pinotage",
    )
    assert "file/shell tools operate on the remote host pinotage over SSH" in text


def test_workspace_summary_lists_files(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("print(1)")
    summary = apiserver.workspace_summary(str(tmp_path))
    assert "a.txt" in summary
    assert "sub/b.py" in summary
    assert "hello" in summary  # small file preview included


def test_workspace_summary_missing_dir():
    assert apiserver.workspace_summary("/nonexistent/path/xyz") == ""


def test_remote_workspace_summary(monkeypatch):
    from agentica_core.config import ClusterConfig
    from agentica_core.transport import Transport, ExecResult

    class FakeTransport:
        def __init__(self, *args, **kwargs):
            pass
        def exec(self, command, timeout=120):
            return ExecResult(0, "Workspace: /remote/ws\n- file1.py", "")

    monkeypatch.setattr(ClusterConfig, "resolve", lambda target: "fake-cluster")
    monkeypatch.setattr(Transport, "from_cluster", lambda cluster: FakeTransport())

    summary = apiserver.workspace_summary("/remote/ws", target="remote-host")
    assert "Workspace: /remote/ws" in summary
    assert "- file1.py" in summary


def test_load_clusters_and_cluster_path_mapping(tmp_path):
    (tmp_path / "disco.yaml").write_text(
        "name: disco\nssh:\n  host: discovery.usc.edu\nscheduler: slurm\n")
    (tmp_path / "plan.yaml").write_text("title: p\ngoal: do x\n")  # not a cluster -> skipped
    clusters = apiserver.load_clusters(str(tmp_path))
    assert set(clusters) == {"disco"}
    assert clusters["disco"]["scheduler"] == "slurm"
    assert clusters["disco"]["host"] == "discovery.usc.edu"

    st = apiserver.State(ollama_host="http://h", model="m", workspace="w",
                         db_path="/tmp/agentica-cl.db", clusters_dir=str(tmp_path))
    assert st.cluster_path("disco") == str(tmp_path / "disco.yaml")   # cluster -> yaml path
    assert st.cluster_path("pinotage.usc.edu") == "pinotage.usc.edu"  # bare alias passthrough
    assert st.cluster_path("local") == "local"


def test_model_catalog_uses_matching_cluster_host(tmp_path):
    (tmp_path / "pinotage.yaml").write_text(
        "\n".join([
            "name: pinotage",
            "ssh:",
            "  host: pinotage.usc.edu",
            "scheduler: ssh",
            "model:",
            "  engine: ollama",
            "  name: llama3.2:3b",
            "slurm:",
            "  gpu_type: l40s",
            "  gpu_count: 1",
            "",
        ])
    )
    st = apiserver.State(ollama_host="http://h", model="qwen3.5:4b-mlx", workspace="w",
                         db_path="/tmp/agentica-models.db", clusters_dir=str(tmp_path))
    assert st.cluster_path("pinotage.usc.edu") == str(tmp_path / "pinotage.yaml")
    cat = apiserver.model_catalog_for_target(st, "pinotage.usc.edu")
    assert cat["selected_model"] == "llama3.2:3b"
    assert cat["selected_engine"] == "ollama"
    ids = {o["id"] for o in cat["options"]}
    assert "llama3.2:3b" in ids
    assert "qwen3.5:4b-mlx" not in ids


def test_app_cache_separates_inference_and_workspace_targets(tmp_path, monkeypatch):
    state = apiserver.State(
        ollama_host="http://local:11434", model="m", workspace=str(tmp_path),
        db_path=str(tmp_path / "db.sqlite"),
    )
    runtime = apiserver.RuntimeBinding(
        target="gpu-box", model="org/large", engine="vllm",
        base_url="http://127.0.0.1:9000",
    )
    monkeypatch.setattr(state, "runtime_for", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(
        state, "cluster_path",
        lambda target: "/configs/tools.yaml" if target == "tools-box" else target,
    )
    calls = []

    def fake_build_app(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(apiserver.gateway, "build_app", fake_build_app)

    local_tools = state.app_for(
        str(tmp_path), target="gpu-box", workspace_target="local",
    )
    assert state.app_for(
        str(tmp_path), target="gpu-box", workspace_target="local",
    ) is local_tools
    remote_tools = state.app_for(
        str(tmp_path), target="gpu-box", workspace_target="tools-box",
    )
    state.app_for(None, target="gpu-box", workspace_target="tools-box")

    assert remote_tools is not local_tools
    assert [call["tools_target"] for call in calls] == [
        "local", "/configs/tools.yaml", "/configs/tools.yaml",
    ]
    assert calls[2]["workspace"] == "."
    assert "Model inference target: gpu-box" in calls[0]["system_prompt"]
    assert "file/shell tools operate on the local" in calls[0]["system_prompt"]
    assert "remote host tools-box" in calls[1]["system_prompt"]


class _ChatApp:
    def chat(self, message, session_id):
        return {"session_id": session_id or "new-session", "final_answer": f"acted: {message}",
                "steps": [{"tool_name": "run_shell"}], "transcript": []}


class _ChatState:
    def __init__(self):
        self.calls = []

    def app_for(self, workspace, **kwargs):
        self.calls.append((workspace, kwargs))
        return _ChatApp()


def test_direct_chat_forces_old_plain_clients_through_agent(monkeypatch):
    state = _ChatState()
    handler = apiserver.make_handler(state)
    result = handler._chat(types.SimpleNamespace(), {
        "message": "fix it", "mode": "plain", "session_id": "s1",
        "workspace": "/local/repo", "target": "gpu-box",
        "workspace_target": "local", "model": "org/large", "engine": "vllm",
    })

    assert result["mode"] == "agentic"
    assert result["final_answer"] == "acted: fix it"
    assert result["workspace_target"] == "local"
    assert state.calls[0] == ("/local/repo", {
        "target": "gpu-box", "model": "org/large", "engine": "vllm",
        "workspace_target": "local",
    })


def test_stream_chat_forces_agentic_and_threads_workspace_target(monkeypatch):
    state = _ChatState()
    handler = apiserver.make_handler(state)
    events = []
    streamed = {}

    def fake_turn(app, message, session_id, emit):
        streamed.update(message=message, session_id=session_id)
        emit({"step": {"tool_name": "read_file"}})
        return {"session_id": "stream-session", "final_answer": "done", "steps": []}

    monkeypatch.setattr(apiserver, "stream_agent_turn", fake_turn)
    self_stub = types.SimpleNamespace(
        _sse_start=lambda: None,
        _safe_sse=events.append,
        _sse=events.append,
    )
    handler._stream_chat(self_stub, {
        "message": "inspect", "mode": "plain", "session_id": "old",
        "workspace": "/remote/repo", "target": "gpu-box",
        "workspace_target": "tools-box",
    })

    assert streamed == {"message": "inspect", "session_id": "old"}
    assert state.calls[0][1]["target"] == "gpu-box"
    assert state.calls[0][1]["workspace_target"] == "tools-box"
    assert events[-1]["final"] == "done"
    assert events[-1]["workspace_target"] == "tools-box"


def test_gateway_build_app_remote_tools():
    # Verify that build_app with a remote target registers remote tools
    from agentica_core import gateway
    app = gateway.build_app(
        ollama_host="http://localhost:11434",
        model_name="qwen3.5:9b",
        workspace="/remote/workspace",
        db_path="/tmp/test-remote-tools.db",
        auth_token=None,
        target="pinotage" # will resolve Pinotage config from examples
    )
    tools = app._create_tools()
    names = tools.names()
    assert "list_files" in names
    assert "read_file" in names
    assert "write_file" in names
    assert "inspect_csv" in names
    assert "run_shell" in names
    assert "remember" in names
    assert "recall" in names
    assert "current_datetime" in names
    assert "search_web" in names


def test_history_handler_reads_sqlite_without_building_runtime(tmp_path, monkeypatch):
    db = tmp_path / "history.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT)")
        conn.executemany(
            "INSERT INTO messages VALUES (?, 's1', ?, ?)",
            [(1, "user", "hello"), (2, "assistant", "hi"), (3, "tool", "hidden")],
        )
    state = apiserver.State(
        ollama_host="http://unreachable.invalid", model="m", workspace=str(tmp_path),
        db_path=str(db),
    )
    monkeypatch.setattr(
        state, "app_for",
        lambda *args, **kwargs: pytest.fail("history must not construct an agent/model runtime"),
    )
    handler = apiserver.make_handler(state)
    self_stub = types.SimpleNamespace(
        path="/api/history?session_id=s1",
        _authorized=lambda: True,
        _json=lambda payload, status=200: (status, payload),
    )

    status, payload = handler.do_GET(self_stub)
    assert status == 200
    assert payload == {"messages": [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]}


def test_submit_remote_preserves_cluster_model_tuning(monkeypatch, tmp_path):
    configured = ModelConfig(
        engine="vllm", name="org/default", quantization="awq",
        tensor_parallel_size=4, pipeline_parallel_size=2,
        gpu_memory_utilization=0.81, max_model_len=16384,
        serve_port=8123, timeout_s=640,
    )
    cluster = ClusterConfig(
        name="gpu", ssh=SSHConfig(host="gpu"), model=configured,
    )
    captured = {}
    monkeypatch.setattr(
        apiserver.ClusterConfig, "resolve", classmethod(lambda cls, path: cluster),
    )

    def fake_submit(cluster_path, plan_path, **kwargs):
        captured["model"] = PlanConfig.load(plan_path).model
        kwargs["_print"]("[submitted] job_id=42 jobdir=/jobs/42")
        return 0

    monkeypatch.setattr(apiserver.job, "submit", fake_submit)
    apiserver.submit_remote(
        {"goal": "do work"}, "gpu", "/configs/gpu.yaml", str(tmp_path),
        model="org/selected", engine="vllm", workspace_source="local",
    )

    assert captured["model"] == replace(configured, name="org/selected")


@pytest.mark.parametrize("rc, line, match", [
    (2, "preflight refused", "submission failed"),
    (0, "submitted but identifiers unavailable", "without a complete job id"),
])
def test_submit_remote_rejects_failed_or_identifierless_submit(
    monkeypatch, tmp_path, rc, line, match,
):
    def fake_submit(cluster_path, plan_path, **kwargs):
        kwargs["_print"](line)
        return rc

    monkeypatch.setattr(apiserver.job, "submit", fake_submit)
    with pytest.raises(RuntimeError, match=match):
        apiserver.submit_remote(
            {}, "gpu", "gpu", str(tmp_path), workspace_source="local",
        )


def test_cancel_and_logs_map_cluster_path_and_report_failures(tmp_path, monkeypatch):
    state = apiserver.State(
        ollama_host="http://local", model="m", workspace=str(tmp_path),
        db_path=str(tmp_path / "db.sqlite"),
    )
    monkeypatch.setattr(
        state, "cluster_path", lambda target: f"/configs/{target}.yaml",
    )
    calls = []

    def fake_cancel(cluster_path, job_id, jobdir=None, _print=print):
        calls.append(("cancel", cluster_path, job_id, jobdir))
        _print("scheduler refused cancellation")
        return 1

    def fake_logs(cluster_path, job_id, jobdir=None, _print=print):
        calls.append(("logs", cluster_path, job_id, jobdir))
        _print("line")
        return 0

    monkeypatch.setattr(apiserver.job, "cancel", fake_cancel)
    monkeypatch.setattr(apiserver.job, "logs", fake_logs)
    handler = apiserver.make_handler(state)

    cancelled = handler._cancel(types.SimpleNamespace(), {
        "target": "friendly", "job": "42", "jobdir": "/jobs/42",
    })
    logged = handler._job_logs(types.SimpleNamespace(), {
        "target": ["friendly"], "job": ["42"], "jobdir": ["/jobs/42"],
    })

    assert cancelled["ok"] is False and cancelled["status"] == "error"
    assert logged == {"log": ["line"]}
    assert calls == [
        ("cancel", "/configs/friendly.yaml", "42", "/jobs/42"),
        ("logs", "/configs/friendly.yaml", "42", "/jobs/42"),
    ]


def test_local_cancel_is_truthful_for_running_and_finished_jobs(tmp_path):
    state = apiserver.State(
        ollama_host="http://local", model="m", workspace=str(tmp_path),
        db_path=str(tmp_path / "db.sqlite"),
    )
    handler = apiserver.make_handler(state)
    cancel = threading.Event()
    state.local_jobs["running"] = {
        "status": "running", "cancel": cancel, "log": [], "outcome": None,
    }
    state.local_jobs["passed"] = {
        "status": "passed", "cancel": threading.Event(), "log": [], "outcome": {},
    }

    first = handler._cancel(types.SimpleNamespace(), {"local_id": "running"})
    again = handler._cancel(types.SimpleNamespace(), {"local_id": "running"})
    finished = handler._cancel(types.SimpleNamespace(), {"local_id": "passed"})

    assert first["ok"] and first["status"] == "cancelling" and cancel.is_set()
    assert again["ok"] and again["status"] == "cancelling"
    assert finished["ok"] is False and finished["status"] == "passed"
