"""Planner / Executor / Auditor workflows for agentic JOB mode.

These are AgenticLocal ``WorkflowDefinition`` objects, built per-job (the prompt
prefixes embed the plan's checklist + success criteria) and SEEDED into the job's
SQLite ``workflow_registry`` so ``RuleResolver`` can resolve them and the
controller's ``_workflow_error`` check passes. Without this seeding the workflows
are unreachable -- this is the wiring step the design review flagged.
"""

from __future__ import annotations

from agentic_loop.rules import WorkflowDefinition
from agentic_loop.storage import SQLiteStore

from .config import PlanConfig


def build_workflows(plan: PlanConfig) -> list[WorkflowDefinition]:
    checklist = "\n".join(f"  - {c}" for c in plan.checklist) or "  (derive a checklist from the goal)"
    tests = plan.success_criteria.tests or "(no test command configured)"
    artifacts = ", ".join(plan.success_criteria.artifacts) or "(none specified)"
    steps = plan.max_steps_per_iteration

    planner = WorkflowDefinition(
        key="planner", command="/planner",
        description="Turn the goal into a concrete, checkable plan.",
        rule_keys=("max_effort",), max_steps_override=steps,
        required_tools=("list_files", "read_file"),
        prompt_prefix=(
            "You are the PLANNER. Inspect the workspace (list_files / read_file) and produce a "
            "concrete, ordered plan to achieve the goal. Restate the acceptance checklist:\n"
            f"{checklist}\n"
            f"Success is verified by running tests: `{tests}` (exit 0) and these artifacts: {artifacts}. "
            "Finish with the plan as your final answer. Do not edit files yet."
        ),
    )
    executor = WorkflowDefinition(
        key="executor", command="/executor",
        description="Implement the plan and make the tests pass.",
        rule_keys=("max_effort",), max_steps_override=steps,
        required_tools=("run_shell", "write_file", "read_file", "run_tests"),
        prompt_prefix=(
            "You are the EXECUTOR. Implement the work to satisfy every checklist item, using "
            "write_file and run_shell inside the workspace. Run the tests with run_tests "
            f"(`{tests}`) and iterate until they pass. Address any auditor gaps provided. "
            "Finish with a short summary of what you changed."
        ),
    )
    auditor = WorkflowDefinition(
        key="auditor", command="/auditor",
        description="Verify completion against the checklist and tests; emit a strict verdict.",
        rule_keys=("max_effort",), max_steps_override=steps,
        required_tools=("run_tests", "check_artifact", "read_file", "submit_for_audit"),
        prompt_prefix=(
            "You are the AUDITOR. Independently verify each checklist item:\n"
            f"{checklist}\n"
            f"Run the tests (`{tests}`) with run_tests and confirm each artifact ({artifacts}) "
            "with check_artifact. You MUST finish by calling the submit_for_audit tool with "
            "status='PASS' only if every item is satisfied AND tests pass, otherwise status='FAIL' "
            "with a 'gaps' string describing exactly what is missing. Prose is ignored; only the "
            "submit_for_audit call counts."
        ),
    )
    return [planner, executor, auditor]


def seed(storage: SQLiteStore, workflows: list[WorkflowDefinition]) -> None:
    """Upsert the custom workflows into the registry (keeps AgenticLocal defaults)."""
    storage.seed_workflow_registry([wf.to_dict() for wf in workflows])
