"""Unit tests for the agentica-core JSON API helpers (no live model needed)."""

import json
from pathlib import Path

from agentica_core import apiserver
from agentica_core.config import PlanConfig


def _state(monkeypatch_reply: str):
    st = apiserver.State(ollama_host="http://127.0.0.1:11434", model="test-model",
                         workspace="sample_workspace", db_path="/tmp/agentica-test.db")
    st.complete = lambda messages, **k: monkeypatch_reply  # type: ignore[assignment]
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
