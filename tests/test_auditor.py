"""Auditor robustness + the Planner->Executor->Auditor outer loop (offline, no GPU).

Uses a deterministic RoleModel (detects role from the prompt, phase from the state
summary) instead of a real LLM, so the loop + deterministic backstops are tested
without Ollama.
"""

from types import SimpleNamespace

import pytest

from agentic_loop.types import AgentState, AgentStep, ModelResponse

from agentica_core.config import PlanConfig, SuccessCriteria
from agentica_core.on_node_runner import _extract_audit_verdict, run_job
from agentica_core.slurm_tools import submit_for_audit


EXEC_SHELL = r"""mkdir -p src tests
cat > src/client.py <<'PYEOF'
import time
def retry(fn, attempts=3):
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(0)
def fetch_with_retry(fn):
    return retry(fn)
PYEOF
cat > tests/test_client.py <<'PYEOF'
import sys
sys.path.insert(0, "src")
from client import fetch_with_retry
calls = {"n": 0}
def flaky():
    calls["n"] += 1
    if calls["n"] < 2:
        raise ValueError("boom")
    return "ok"
assert fetch_with_retry(flaky) == "ok"
print("ok")
PYEOF
"""


class RoleModel:
    """Deterministic stand-in for an LLM keyed off role (prompt) + phase (state)."""

    def __init__(self, executor_writes=True, audit_status="PASS"):
        self.executor_writes = executor_writes
        self.audit_status = audit_status

    def respond(self, messages, tools, state_summary):
        text = "\n".join(m.content for m in messages)
        first = "No steps have run yet" in state_summary
        if "You are the PLANNER" in text:
            return ModelResponse.final("Plan: implement retry() + tests.")
        if "You are the EXECUTOR" in text:
            if first and self.executor_writes:
                return ModelResponse.call("run_shell", {"command": EXEC_SHELL}, "e1")
            return ModelResponse.final("Implemented retry + tests.")
        if "You are the AUDITOR" in text:
            if first:
                gaps = "" if self.audit_status == "PASS" else "missing artifacts"
                return ModelResponse.call(
                    "submit_for_audit", {"status": self.audit_status, "gaps": gaps}, "a1")
            return ModelResponse.final("Audit complete.")
        return ModelResponse.final("done")


def _plan(workspace):
    return PlanConfig(
        title="retry", goal="Implement retry() and fetch_with_retry() with tests.",
        workspace=str(workspace), checklist=["retry exists", "tests cover failure then success"],
        success_criteria=SuccessCriteria(
            tests="python3 tests/test_client.py",
            artifacts=["src/client.py", "tests/test_client.py"]),
        max_iterations=2, max_steps_per_iteration=6,
    )


# --------------------------- unit: verdict channel --------------------------- #
def test_submit_for_audit_rejects_bad_status():
    ctx = SimpleNamespace(workspace_root=None)
    with pytest.raises(ValueError):
        submit_for_audit(ctx, {"status": "maybe"})
    assert submit_for_audit(ctx, {"status": "pass"})["status"] == "PASS"


def test_extract_verdict_ignores_prose_uses_tool_call():
    # final_answer says PASS but the structured tool call says FAIL -> FAIL wins.
    state = AgentState(goal="g")
    state.steps.append(AgentStep(index=1, action="tool_call", tool_name="submit_for_audit",
                                 observation={"status": "FAIL", "gaps": "no tests"}))
    state.final_answer = "Everything looks great, all tests PASS!"
    result = SimpleNamespace(state=state)
    status, gaps = _extract_audit_verdict(result)
    assert status == "FAIL"
    assert gaps == "no tests"


# --------------------------- end-to-end loop --------------------------------- #
def test_job_passes_when_executor_succeeds(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    outcome = run_job(_plan(ws), workspace=str(ws), db_path=str(tmp_path / "job.db"),
                      provider="rule", model=RoleModel(executor_writes=True, audit_status="PASS"))
    assert outcome.passed
    assert outcome.iterations == 1
    assert outcome.tests_ok
    assert outcome.artifacts_ok
    assert outcome.verdict == "PASS"
    assert (ws / "src" / "client.py").exists()


def test_backstops_win_over_model_verdict(tmp_path):
    # Executor does nothing; auditor lies "PASS". Deterministic backstops must FAIL it.
    ws = tmp_path / "ws"
    ws.mkdir()
    outcome = run_job(_plan(ws), workspace=str(ws), db_path=str(tmp_path / "job.db"),
                      provider="rule", model=RoleModel(executor_writes=False, audit_status="PASS"))
    assert not outcome.passed
    assert not (outcome.tests_ok and outcome.artifacts_ok)
