"""Workflow-resolution regression -- the biggest wiring trap.

Custom workflows must be SEEDED into the SQLite registry, and their required_tools
must be present in the controller's tool registry, or `_workflow_error` hard-fails.
"""

from agentic_loop.factory import create_controller
from agentic_loop.rules import RuleResolver
from agentic_loop.storage import SQLiteStore

from agentica_core import workflows as wf
from agentica_core.config import PlanConfig, SuccessCriteria
from agentica_core.slurm_tools import create_job_tools


def _plan():
    return PlanConfig(
        title="t", goal="do the thing",
        checklist=["item one"],
        success_criteria=SuccessCriteria(tests="true", artifacts=["a.txt"]),
    )


def test_auditor_unresolvable_without_seeding(tmp_path):
    store = SQLiteStore(str(tmp_path / "a.db"))
    store.seed_default_workflow_registry()  # only loop/search/release
    assert RuleResolver(store).workflow("auditor") is None


def test_auditor_resolvable_after_seeding(tmp_path):
    store = SQLiteStore(str(tmp_path / "b.db"))
    store.seed_default_workflow_registry()
    wf.seed(store, wf.build_workflows(_plan()))

    resolver = RuleResolver(store)
    for key in ("planner", "executor", "auditor"):
        assert resolver.workflow(key) is not None, key
    # required_tools all present in the job registry
    names = create_job_tools().names()
    for key in ("planner", "executor", "auditor"):
        missing = set(resolver.workflow(key).required_tools) - names
        assert not missing, (key, missing)


def test_controller_workflow_error_none_with_job_tools(tmp_path):
    store = SQLiteStore(str(tmp_path / "c.db"))
    registry = create_job_tools()
    controller = create_controller(
        workspace=str(tmp_path / "ws"), db_path=str(tmp_path / "c.db"),
        storage=store, tools=registry, provider="rule",
    )
    wf.seed(store, wf.build_workflows(_plan()))
    auditor = controller.rule_resolver.workflow("auditor")
    assert controller._workflow_error("auditor", auditor) is None


def test_controller_workflow_error_when_tools_missing(tmp_path):
    from agentic_loop.tools import create_default_tools

    store = SQLiteStore(str(tmp_path / "d.db"))
    # default tools lack run_tests/check_artifact/submit_for_audit
    controller = create_controller(
        workspace=str(tmp_path / "ws"), db_path=str(tmp_path / "d.db"),
        storage=store, tools=create_default_tools(), provider="rule",
    )
    wf.seed(store, wf.build_workflows(_plan()))
    auditor = controller.rule_resolver.workflow("auditor")
    err = controller._workflow_error("auditor", auditor)
    assert err is not None and "submit_for_audit" in err
