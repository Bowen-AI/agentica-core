"""job.status_struct: structured remote-job status for the JSON API / UI.

The UI poll loop terminates on status in {passed,failed,error}; before this it
got only log `lines` for remote jobs and polled forever. These tests fake the
transport so no cluster is needed."""
import json

from agentica_core import job
from agentica_core.transport import ExecResult


class _FakeTransport:
    def __init__(self, *, result="", marker="", cancelled="", logs="", pid_state="UNKNOWN",
                 scheduler="slurm", squeue=None, sacct="UNKNOWN"):
        self._result, self._marker, self._squeue, self._sacct = result, marker, squeue, sacct
        self._cancelled, self._logs = cancelled, logs
        self._pid_state, self._scheduler = pid_state, scheduler

    def exec(self, command, timeout=120.0):
        if "command -v sbatch" in command:
            return ExecResult(0, self._scheduler, "")
        if job._PROBE_SEP in command:
            # The consolidated one-round-trip probe: result / cancel / done /
            # pid-state / log-tail, in section order (see job._probe_jobdir).
            sep = f"\n{job._PROBE_SEP}\n"
            return ExecResult(0, sep.join(
                [self._result, self._cancelled, self._marker, self._pid_state, self._logs]
            ), "")
        if "tail -n" in command:
            return ExecResult(0, self._logs, "")
        return ExecResult(0, "", "")

    def squeue_job(self, job_id):
        return self._squeue or {}

    def sacct_state(self, job_id):
        return self._sacct


def _patch(monkeypatch, fake):
    monkeypatch.setattr(job.Transport, "from_cluster", classmethod(lambda cls, cluster: fake))


def test_status_struct_passed(monkeypatch):
    _patch(monkeypatch, _FakeTransport(result=json.dumps(
        {"passed": True, "iterations": 1, "verdict": "PASS", "tests_ok": True, "log": ["a", "b"]})))
    s = job.status_struct("host", "job-1", jobdir="/d")
    assert s["status"] == "passed"
    assert s["outcome"]["tests_ok"] is True
    assert "a" in s["lines"] and any("COMPLETED" in l for l in s["lines"])


def test_status_struct_failed(monkeypatch):
    _patch(monkeypatch, _FakeTransport(result=json.dumps({"passed": False, "tests_ok": False})))
    assert job.status_struct("host", "job-1", jobdir="/d")["status"] == "failed"


def test_status_struct_running_when_in_queue(monkeypatch):
    _patch(monkeypatch, _FakeTransport(
        squeue={"state": "RUNNING", "nodelist": "b11-09"},
        logs="iteration 1\nrunning tests",
    ))
    s = job.status_struct("host", "job-1", jobdir="/d")
    assert s["status"] == "running" and s["outcome"] is None
    assert "iteration 1" in s["lines"]
    assert "running tests" in s["lines"]


def test_status_struct_queued_when_pending(monkeypatch):
    # SLURM PENDING should read as "queued", not "running", so the UI can say so.
    _patch(monkeypatch, _FakeTransport(squeue={"state": "PENDING", "reason": "(Resources)"}))
    assert job.status_struct("host", "job-1", jobdir="/d")["status"] == "queued"


def test_status_struct_error_when_ended_without_result(monkeypatch):
    # left the queue, sacct shows a terminal state, but no result.json -> error (don't poll forever)
    _patch(monkeypatch, _FakeTransport(squeue=None, sacct="FAILED"))
    assert job.status_struct("host", "job-1", jobdir="/d")["status"] == "error"


def test_status_struct_ssh_done_marker_without_result(monkeypatch):
    _patch(monkeypatch, _FakeTransport(result="", marker="JOB_DONE rc=1"))
    assert job.status_struct("host", "job-1", jobdir="/d")["status"] == "error"


def test_status_struct_ssh_running_uses_pid_and_live_log(monkeypatch):
    _patch(monkeypatch, _FakeTransport(
        scheduler="ssh", pid_state="RUNNING", logs="model ready\nagent step 2",
    ))
    s = job.status_struct("host", "job-1", jobdir="/d")
    assert s["status"] == "running"
    assert any("PID is alive" in line for line in s["lines"])
    assert "model ready" in s["lines"]
    assert "agent step 2" in s["lines"]


def test_status_struct_ssh_stopped_without_marker_is_error(monkeypatch):
    _patch(monkeypatch, _FakeTransport(
        scheduler="ssh", pid_state="STOPPED", logs="python: command not found",
    ))
    s = job.status_struct("host", "job-1", jobdir="/d")
    assert s["status"] == "error"
    assert "python: command not found" in s["lines"]


def test_status_struct_cancelled_markers(monkeypatch):
    _patch(monkeypatch, _FakeTransport(
        scheduler="ssh", cancelled="CANCELLED by user", logs="stopping",
    ))
    s = job.status_struct("host", "job-1", jobdir="/d")
    assert s["status"] == "cancelled"
    assert "stopping" in s["lines"]


def test_status_struct_slurm_cancelled_state(monkeypatch):
    _patch(monkeypatch, _FakeTransport(sacct="CANCELLED+"))
    assert job.status_struct("host", "job-1", jobdir="/d")["status"] == "cancelled"
