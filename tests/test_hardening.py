"""Tests for the deterministic-backstop hardening (judge-panel findings):
tamper guard, stdout success token, artifact symbol (AST) checks, and
model-drafted-test validation."""
from agentica_core import on_node_runner as onr
from agentica_core import apiserver
from agentica_core.config import PlanConfig


def _notes():
    log: list[str] = []
    return log, log.append


# ---- stdout success token (closes the exit-0-only gaming gap) ----
def test_tests_command_requires_success_token(tmp_path):
    _, note = _notes()
    assert onr._run_tests_command("echo nope", tmp_path, note, success_token="DONE") is False
    assert onr._run_tests_command("echo DONE", tmp_path, note, success_token="DONE") is True
    assert onr._run_tests_command("true", tmp_path, note) is True            # no token -> exit 0 suffices
    assert onr._run_tests_command("false", tmp_path, note) is False          # nonzero always fails


# ---- artifact symbol (AST) checks (a stub file no longer passes) ----
def test_artifacts_ok_requires_defined_symbols(tmp_path):
    (tmp_path / "m.py").write_text("def a():\n    pass\n")
    assert onr._artifacts_ok(["m.py"], tmp_path) is True
    assert onr._artifacts_ok(["m.py"], tmp_path, symbols={"m.py": ["a"]}) is True
    assert onr._artifacts_ok(["m.py"], tmp_path, symbols={"m.py": ["missing"]}) is False
    (tmp_path / "stub.py").write_text("")  # empty stub satisfies existence but not symbols
    assert onr._artifacts_ok(["stub.py"], tmp_path) is True
    assert onr._artifacts_ok(["stub.py"], tmp_path, symbols={"stub.py": ["a"]}) is False


# ---- tamper guard: seeded grader is restored before scoring ----
def test_protected_files_are_restored_after_tampering(tmp_path):
    grader = tmp_path / "test_spec.py"
    grader.write_text("ORIGINAL")
    log, note = _notes()
    snap = onr._snapshot_protected(tmp_path, ["test_spec.py"], note)
    grader.write_text("def cheat():\n    pass  # the agent gamed the test\n")
    onr._restore_protected(tmp_path, snap, note)
    assert grader.read_text() == "ORIGINAL"
    assert any("TAMPER" in m for m in log)
    # deleting the grader is also restored
    grader.unlink()
    onr._restore_protected(tmp_path, snap, note)
    assert grader.read_text() == "ORIGINAL"


# ---- model-drafted test validation ----
def test_trivial_drafted_tests_are_dropped_and_flagged():
    assert apiserver._test_is_trivial("true")
    assert apiserver._test_is_trivial("echo ok")
    assert apiserver._test_is_trivial("")
    assert not apiserver._test_is_trivial("grep -qx hi f")
    assert not apiserver._test_is_trivial("python3 test.py")

    # drafted (non-authoritative) trivial test -> dropped + flagged unvetted
    p = apiserver._plan_payload("t", "g", ["s"], "echo ok", ["a.txt"], tests_authoritative=False)
    assert p["tests"] == "" and p["tests_authoritative"] is False
    # drafted real test -> kept but still flagged unvetted (user should review)
    p2 = apiserver._plan_payload("t", "g", ["s"], "grep -qx hi f", [], tests_authoritative=False)
    assert p2["tests"] == "grep -qx hi f" and p2["tests_authoritative"] is False
    # user-supplied test -> authoritative
    p3 = apiserver._plan_payload("t", "g", ["s"], "pytest -q", [], tests_authoritative=True)
    assert p3["tests_authoritative"] is True


# ---- config parsing of the new hardening knobs ----
def test_success_criteria_parses_hardening_fields():
    sc = PlanConfig.from_dict({
        "goal": "x",
        "success_criteria": {
            "tests": "python3 t.py", "tests_success_token": "OK",
            "protect": ["t.py"], "artifact_symbols": {"m.py": ["f", "g"]}, "artifacts": ["m.py"],
        },
    }).success_criteria
    assert sc.tests_success_token == "OK"
    assert sc.protect == ["t.py"]
    assert sc.artifact_symbols == {"m.py": ["f", "g"]}
    bare = PlanConfig.from_dict({"goal": "x"}).success_criteria
    assert bare.tests_success_token is None and bare.protect == [] and bare.artifact_symbols == {}
