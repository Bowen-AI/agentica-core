"""Session-history storage helpers + job-submit workspace routing.

History reads/deletes go straight to SQLite (never through app_for, which can
bring up a remote model runtime), and /api/job/submit maps the UI's
workspace_target onto job.submit's workspace_source.
"""

import sqlite3
import subprocess
import time
import types

from agentica_core import apiserver


# --------------------------------------------------------------------------- #
# session storage (direct SQLite)
# --------------------------------------------------------------------------- #
def _seed_db(path, sessions=2, msgs_per=2):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE sessions (session_id TEXT PRIMARY KEY,
                               created_at_unix REAL NOT NULL,
                               updated_at_unix REAL NOT NULL);
        CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT,
                               session_id TEXT NOT NULL, role TEXT NOT NULL,
                               content TEXT NOT NULL, name TEXT, tool_call_id TEXT,
                               created_at_unix REAL NOT NULL);
        CREATE TABLE runs (run_id TEXT PRIMARY KEY, session_id TEXT,
                           goal TEXT NOT NULL, final_answer TEXT,
                           evaluation_json TEXT, created_at_unix REAL NOT NULL,
                           completed_at_unix REAL);
        CREATE TABLE steps (id INTEGER PRIMARY KEY AUTOINCREMENT,
                            run_id TEXT NOT NULL, step_index INTEGER NOT NULL,
                            action TEXT NOT NULL, tool_name TEXT,
                            arguments_json TEXT NOT NULL, observation_json TEXT,
                            allowed INTEGER NOT NULL, error TEXT,
                            created_at_unix REAL NOT NULL);
        CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT,
                             event_type TEXT NOT NULL, payload_json TEXT NOT NULL,
                             session_id TEXT, run_id TEXT,
                             created_at_unix REAL NOT NULL);
        """
    )
    now = time.time()
    for i in range(sessions):
        sid = f"s{i}"
        conn.execute("INSERT INTO sessions VALUES (?, ?, ?)", (sid, now + i, now + i))
        for j in range(msgs_per):
            conn.execute(
                "INSERT INTO messages(session_id, role, content, created_at_unix) "
                "VALUES (?, 'user', ?, ?)", (sid, f"question {i}-{j}", now))
            conn.execute(
                "INSERT INTO messages(session_id, role, content, created_at_unix) "
                "VALUES (?, 'assistant', ?, ?)", (sid, f"answer {i}-{j}", now))
        conn.execute(
            "INSERT INTO runs VALUES (?, ?, 'goal', NULL, NULL, ?, NULL)",
            (f"run-{i}", sid, now))
        conn.execute(
            "INSERT INTO steps(run_id, step_index, action, arguments_json, allowed, "
            "created_at_unix) VALUES (?, 0, 'tool', '{}', 1, ?)", (f"run-{i}", now))
        conn.execute(
            "INSERT INTO events(event_type, payload_json, session_id, created_at_unix) "
            "VALUES ('x', '{}', ?, ?)", (sid, now))
    conn.commit()
    conn.close()


def test_list_sessions_titles_and_counts(tmp_path):
    db = str(tmp_path / "a.db")
    _seed_db(db)
    sessions = apiserver._list_sessions(db)
    assert [s["session_id"] for s in sessions] == ["s1", "s0"]  # newest first
    assert sessions[0]["title"] == "question 1-0"
    assert sessions[0]["messages"] == 4


def test_load_session_messages_roles_only(tmp_path):
    db = str(tmp_path / "a.db")
    _seed_db(db, sessions=1, msgs_per=1)
    msgs = apiserver._load_session_messages(db, "s0")
    assert msgs == [{"role": "user", "content": "question 0-0"},
                    {"role": "assistant", "content": "answer 0-0"}]
    assert apiserver._load_session_messages(db, "nope") == []
    assert apiserver._load_session_messages(str(tmp_path / "missing.db"), "s0") == []


def test_delete_sessions_by_id_and_all(tmp_path):
    db = str(tmp_path / "a.db")
    _seed_db(db, sessions=3)
    res = apiserver._delete_sessions(db, ["s1"], False)
    assert res == {"ok": True, "deleted": 1}
    left = apiserver._list_sessions(db)
    assert {s["session_id"] for s in left} == {"s0", "s2"}
    # the session's runs/steps/events went with it
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM runs WHERE session_id='s1'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM steps WHERE run_id='run-1'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM events WHERE session_id='s1'").fetchone()[0] == 0
    conn.close()

    res = apiserver._delete_sessions(db, None, True)
    assert res["ok"] and res["deleted"] == 2
    assert apiserver._list_sessions(db) == []
    # missing args is a client error, not a purge
    assert apiserver._delete_sessions(db, None, False)["ok"] is False
    # a missing db is fine (nothing stored yet)
    assert apiserver._delete_sessions(str(tmp_path / "missing.db"), None, True)["ok"]


# --------------------------------------------------------------------------- #
# remote workspace summary actually parses/runs (it was a silent SyntaxError)
# --------------------------------------------------------------------------- #
def test_remote_workspace_summary_command_is_valid_python(tmp_path, monkeypatch):
    from agentica_core.config import ClusterConfig
    from agentica_core.transport import Transport, ExecResult

    (tmp_path / "hello.py").write_text("print('hi')\n")
    (tmp_path / "notes.txt").write_text("remember the milk\n")

    class ShellTransport:
        """Runs the generated command in a real local shell — proving the
        embedded python is syntactically valid, not just that we built a string."""

        def exec(self, command, timeout=120):
            proc = subprocess.run(command, shell=True, capture_output=True, text=True)
            return ExecResult(proc.returncode, proc.stdout, proc.stderr)

    monkeypatch.setattr(ClusterConfig, "resolve", lambda target: "fake-cluster")
    monkeypatch.setattr(Transport, "from_cluster", lambda cluster: ShellTransport())

    summary = apiserver.workspace_summary(str(tmp_path), target="somehost")
    assert f"Workspace: {tmp_path}" in summary
    assert "- hello.py" in summary
    assert "remember the milk" in summary  # preview section made it through


# --------------------------------------------------------------------------- #
# /api/job/submit workspace_target routing
# --------------------------------------------------------------------------- #
def _handler_submit(monkeypatch, tmp_path, body):
    state = apiserver.State(ollama_host="http://127.0.0.1:11434", model="m",
                            workspace=str(tmp_path), db_path=str(tmp_path / "db.db"))
    H = apiserver.make_handler(state)
    calls = {}

    def fake_local(state_, plan, ws, model=None, engine=None):
        calls["local"] = {"ws": ws}
        return {"local": True, "local_id": "local-x"}

    def fake_remote(plan, target, cluster_path, ws, model=None, engine=None,
                    workspace_source=None):
        calls["remote"] = {"target": target, "ws": ws,
                           "workspace_source": workspace_source}
        return {"local": False, "target": target, "workspace_target": workspace_source}

    monkeypatch.setattr(apiserver, "submit_local", fake_local)
    monkeypatch.setattr(apiserver, "submit_remote", fake_remote)
    self_stub = types.SimpleNamespace()
    return H._submit(self_stub, body), calls


def test_submit_local_worker_local_workspace(monkeypatch, tmp_path):
    res, calls = _handler_submit(monkeypatch, tmp_path,
                                 {"plan": {}, "target": "local", "workspace": "/w"})
    assert res["local"] is True and "local" in calls


def test_submit_local_worker_rejects_remote_workspace(monkeypatch, tmp_path):
    import pytest
    with pytest.raises(ValueError, match="remote workspace"):
        _handler_submit(monkeypatch, tmp_path,
                        {"plan": {}, "target": "local", "workspace": "/w",
                         "workspace_target": "pinotage"})


def test_submit_remote_worker_local_workspace_stages(monkeypatch, tmp_path):
    res, calls = _handler_submit(monkeypatch, tmp_path,
                                 {"plan": {}, "target": "pinotage", "workspace": "/w",
                                  "workspace_target": "local"})
    assert calls["remote"]["workspace_source"] == "local"
    assert res["workspace_target"] == "local"


def test_submit_remote_worker_own_workspace_runs_in_place(monkeypatch, tmp_path):
    res, calls = _handler_submit(monkeypatch, tmp_path,
                                 {"plan": {}, "target": "pinotage",
                                  "workspace": "/home/u/proj",
                                  "workspace_target": "pinotage"})
    assert calls["remote"]["workspace_source"] == "remote"


def test_submit_third_machine_workspace_rejected(monkeypatch, tmp_path):
    import pytest
    with pytest.raises(ValueError, match="workspace machine"):
        _handler_submit(monkeypatch, tmp_path,
                        {"plan": {}, "target": "pinotage", "workspace": "/w",
                         "workspace_target": "champagne"})


def test_submit_remote_returns_sync_to_for_local_dir(monkeypatch, tmp_path):
    captured = {}

    def fake_job_submit(cluster_path, plan_path, sync_code=True,
                        workspace_source=None, _print=print):
        captured["workspace_source"] = workspace_source
        _print("[submitted] job_id=job-abc  jobdir=/remote/job-abc   (ssh: host)")
        return 0

    monkeypatch.setattr(apiserver.job, "submit", fake_job_submit)
    res = apiserver.submit_remote({}, "pinotage", "pinotage", str(tmp_path),
                                  workspace_source="local")
    assert captured["workspace_source"] == "local"
    assert res["job_id"] == "job-abc" and res["jobdir"] == "/remote/job-abc"
    assert res["sync_to"] == str(tmp_path)
    # a remote-source submission has nothing to sync back
    res = apiserver.submit_remote({}, "pinotage", "pinotage", "/on/remote",
                                  workspace_source="remote")
    assert res["sync_to"] is None


# --------------------------------------------------------------------------- #
# local job cancellation (cooperative)
# --------------------------------------------------------------------------- #
def test_local_job_cancel_flags_event(monkeypatch, tmp_path):
    state = apiserver.State(ollama_host="http://127.0.0.1:11434", model="m",
                            workspace=str(tmp_path), db_path=str(tmp_path / "db.db"))
    H = apiserver.make_handler(state)
    import threading
    cancel = threading.Event()
    state.local_jobs["local-1"] = {"local_id": "local-1", "status": "running",
                                   "log": [], "outcome": None, "cancel": cancel}
    res = H._cancel(types.SimpleNamespace(), {"local_id": "local-1"})
    assert res["ok"] is True
    assert cancel.is_set()
    res = H._cancel(types.SimpleNamespace(), {"local_id": "nope"})
    assert res["ok"] is False


def test_run_job_cancel_stops_before_planner(tmp_path):
    import threading
    from agentica_core.config import PlanConfig, SuccessCriteria
    from agentica_core.on_node_runner import run_job

    cancel = threading.Event()
    cancel.set()
    pc = PlanConfig(title="t", goal="g", workspace=str(tmp_path), checklist=["x"],
                    success_criteria=SuccessCriteria(tests=None, artifacts=[]),
                    max_iterations=1, max_steps_per_iteration=1)
    outcome = run_job(pc, workspace=str(tmp_path), db_path=str(tmp_path / "j.db"),
                      provider="ollama", model_name="m",
                      ollama_host="http://127.0.0.1:1", cancel_event=cancel,
                      _print=lambda *_: None)
    assert outcome.passed is False
    assert outcome.gaps == "cancelled by user"
    assert outcome.iterations == 0
