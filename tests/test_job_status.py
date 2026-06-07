"""job.status_struct: structured remote-job status for the JSON API / UI.

The UI poll loop terminates on status in {passed,failed,error}; before this it
got only log `lines` for remote jobs and polled forever. These tests fake the
transport so no cluster is needed."""
import json

from agentica_core import job
from agentica_core.transport import ExecResult


class _FakeTransport:
    def __init__(self, *, result="", marker="", squeue=None, sacct="UNKNOWN"):
        self._result, self._marker, self._squeue, self._sacct = result, marker, squeue, sacct

    def exec(self, command, timeout=120.0):
        if "result.json" in command:
            return ExecResult(0, self._result, "")
        if "done.marker" in command:
            return ExecResult(0, self._marker, "")
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
    _patch(monkeypatch, _FakeTransport(squeue={"state": "RUNNING", "nodelist": "b11-09"}))
    s = job.status_struct("host", "job-1", jobdir="/d")
    assert s["status"] == "running" and s["outcome"] is None


def test_status_struct_error_when_ended_without_result(monkeypatch):
    # left the queue, sacct shows a terminal state, but no result.json -> error (don't poll forever)
    _patch(monkeypatch, _FakeTransport(squeue=None, sacct="FAILED"))
    assert job.status_struct("host", "job-1", jobdir="/d")["status"] == "error"


def test_status_struct_ssh_done_marker_without_result(monkeypatch):
    _patch(monkeypatch, _FakeTransport(result="", marker="JOB_DONE rc=1"))
    assert job.status_struct("host", "job-1", jobdir="/d")["status"] == "error"
