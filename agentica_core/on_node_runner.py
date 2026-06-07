"""On-node agentic job runner: Planner -> Executor -> Auditor outer loop.

Runs INSIDE the SLURM allocation. The auditor is an explicit Python loop (NOT a
policy hook -- AgenticLocal's requires_approval does not pause the loop). The
deterministic backstops (test exit code + artifact existence) WIN over the model
verdict; the auditor's structured ``submit_for_audit`` call is the only model
signal that can additionally fail an otherwise-green run for incompleteness.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentic_loop.factory import create_controller
from agentic_loop.ollama_model import OllamaChatModel
from agentic_loop.storage import SQLiteStore

from . import workflows as wf
from .config import PlanConfig
from .slurm_tools import create_job_tools


@dataclass
class JobOutcome:
    passed: bool
    iterations: int
    verdict: str               # PASS | FAIL | NONE
    tests_ok: bool
    artifacts_ok: bool
    gaps: str = ""
    plan: str = ""
    log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def run_job(
    plan: PlanConfig,
    *,
    workspace: str,
    db_path: str,
    provider: str = "ollama",
    model_name: str | None = None,
    ollama_host: str = "http://127.0.0.1:11434",
    api_base: str | None = None,
    api_key: str | None = None,
    model=None,
    model_timeout: float = 120.0,
    checkpoint_dir: str | None = None,
    _print=print,
) -> JobOutcome:
    workspace_path = Path(workspace).resolve()
    workspace_path.mkdir(parents=True, exist_ok=True)
    store = SQLiteStore(db_path)
    sc = plan.success_criteria
    # Tamper-evidence: capture pristine bytes of seeded grader/fixture files BEFORE the
    # agent can touch them, so the backstop always scores against the original (the
    # executor has write_roots=["."] + shell, so "don't edit the test" can't be honor-system).
    protected = _snapshot_protected(workspace_path, sc.protect, _print)
    # Build the ollama adapter with a configurable timeout (slow GPUs need more).
    if model is None and provider == "ollama" and model_name:
        model = OllamaChatModel(model=model_name, host=ollama_host, timeout_s=model_timeout)

    registry = create_job_tools(allow_shell=True)
    # Seed planner/executor/auditor so RuleResolver can resolve them (critical wiring).
    wf.seed(store, wf.build_workflows(plan))

    controller = create_controller(
        workspace=workspace_path,
        db_path=db_path,
        storage=store,
        tools=registry,
        model=model,
        provider=provider,
        model_name=model_name,
        ollama_host=ollama_host,
        api_base=api_base,
        api_key=api_key,
        write_roots=["."],
        max_steps=plan.max_steps_per_iteration,
    )

    log: list[str] = []

    def note(msg: str) -> None:
        log.append(msg)
        _print(msg)

    # --- Planner ---
    note("[planner] producing plan...")
    plan_result = controller.run(plan.goal, workflow_key="planner")
    plan_text = plan_result.final_answer or ""
    _checkpoint(checkpoint_dir, "planner", {"plan": plan_text})

    verdict = "NONE"
    gaps = ""
    tests_ok = artifacts_ok = False
    executor_goal = plan.goal

    for i in range(1, plan.max_iterations + 1):
        note(f"[executor] iteration {i}/{plan.max_iterations}")
        controller.run(executor_goal, workflow_key="executor")

        # Deterministic backstops (authoritative). Restore any tampered grader/fixture
        # files to their pristine copy first, then score against them.
        _restore_protected(workspace_path, protected, note)
        tests_ok = _run_tests_command(sc.tests, workspace_path, note,
                                      success_token=sc.tests_success_token)
        artifacts_ok = _artifacts_ok(sc.artifacts, workspace_path, symbols=sc.artifact_symbols)
        note(f"[backstop] tests_ok={tests_ok} artifacts_ok={artifacts_ok}")

        # Auditor (model verdict; structured channel only).
        note("[auditor] verifying...")
        audit_result = controller.run(
            "Audit the work against the checklist and finish with submit_for_audit.",
            workflow_key="auditor",
        )
        verdict, gaps = _extract_audit_verdict(audit_result)
        note(f"[auditor] verdict={verdict} gaps={gaps[:200]!r}")

        backstops_ok = tests_ok and artifacts_ok
        passed = backstops_ok and verdict != "FAIL"
        _checkpoint(checkpoint_dir, f"iter{i}", {
            "tests_ok": tests_ok, "artifacts_ok": artifacts_ok,
            "verdict": verdict, "gaps": gaps, "passed": passed,
        })
        if passed:
            note(f"[done] job passed on iteration {i}")
            return JobOutcome(True, i, verdict, tests_ok, artifacts_ok, gaps, plan_text, log)

        # Feed gaps back to the executor.
        reasons = []
        if not tests_ok and plan.success_criteria.tests:
            reasons.append(f"tests failing: `{plan.success_criteria.tests}`")
        if not artifacts_ok:
            reasons.append(f"missing artifacts: {plan.success_criteria.artifacts}")
        if gaps:
            reasons.append(f"auditor gaps: {gaps}")
        executor_goal = (
            f"{plan.goal}\n\nThe previous attempt did not pass. Fix these issues, then re-run tests:\n- "
            + "\n- ".join(reasons or ["address remaining gaps"])
        )

    note("[done] job did NOT pass within max_iterations")
    return JobOutcome(False, plan.max_iterations, verdict, tests_ok, artifacts_ok, gaps, plan_text, log)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _extract_audit_verdict(result) -> tuple[str, str]:
    """Pull the structured submit_for_audit verdict from the run's steps (only accepted channel)."""
    for step in reversed(result.state.steps):
        if step.tool_name == "submit_for_audit" and isinstance(step.observation, dict):
            status = str(step.observation.get("status", "")).upper()
            if status in {"PASS", "FAIL"}:
                return status, str(step.observation.get("gaps", ""))
    return "NONE", ""


def _snapshot_protected(workspace: Path, protect: list[str], _print) -> dict[str, bytes]:
    """Capture pristine bytes of seeded grader/fixture files before the agent runs."""
    snap: dict[str, bytes] = {}
    for rel in protect or []:
        p = workspace / rel
        if p.is_file():
            snap[rel] = p.read_bytes()
        else:
            _print(f"[backstop] WARNING: protected file not found at job start: {rel}")
    if snap:
        _print(f"[backstop] tamper-guarding {len(snap)} seeded file(s): {sorted(snap)}")
    return snap


def _restore_protected(workspace: Path, snapshot: dict[str, bytes], note) -> None:
    """Restore pristine copies before scoring; flag any file the agent altered/removed."""
    for rel, original in (snapshot or {}).items():
        p = workspace / rel
        try:
            current = p.read_bytes() if p.is_file() else None
        except OSError:
            current = None
        if current != original:
            note(f"[backstop] TAMPER: protected file {rel!r} was modified/removed -> restoring pristine copy")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(original)


def _run_tests_command(command: str | None, cwd: Path, note, *, success_token: str | None = None) -> bool:
    if not command:
        return True  # no tests configured -> not a gate
    try:
        proc = subprocess.run(["bash", "-lc", command], cwd=str(cwd),
                              capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        note("[backstop] tests timed out")
        return False
    if proc.returncode != 0:
        note(f"[backstop] tests exit={proc.returncode}: {proc.stdout[-300:]}{proc.stderr[-300:]}")
        return False
    # Exit-0 is necessary but not sufficient: an optional success token in stdout closes
    # the "reduce the test to print('OK')/sys.exit(0)" gaming gap.
    if success_token and success_token not in proc.stdout:
        note(f"[backstop] tests exited 0 but required token {success_token!r} not in stdout -> FAIL")
        return False
    return True


def _artifacts_ok(artifacts: list[str], cwd: Path, *, symbols: dict | None = None) -> bool:
    for rel in artifacts or []:
        if not (cwd / rel).exists():
            return False
    # Content check: a required artifact must DEFINE the named symbols (not just exist),
    # so an empty/stub file no longer satisfies the gate.
    for rel, required in (symbols or {}).items():
        if not _defines_symbols(cwd / rel, required):
            return False
    return True


def _defines_symbols(path: Path, required: list[str]) -> bool:
    """True iff the python file defines every required top-level name (def/class/assignment)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError, UnicodeDecodeError):
        return False
    defined: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            defined.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return all(name in defined for name in (required or []))


def _checkpoint(checkpoint_dir: str | None, name: str, data: dict) -> None:
    if not checkpoint_dir:
        return
    d = Path(checkpoint_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"checkpoint-{name}.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI entrypoint (invoked by the sbatch runner on the node)
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="slurm-agentic-on-node")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--db", default=".agentic/job.db")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default=None)
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model-timeout", type=float, default=120.0)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--result", default=None, help="Write the JobOutcome JSON here.")
    args = parser.parse_args(argv)

    plan = PlanConfig.load(args.plan)
    outcome = run_job(
        plan, workspace=args.workspace, db_path=args.db,
        provider=args.provider, model_name=args.model, ollama_host=args.ollama_host,
        api_base=args.api_base, api_key=args.api_key,
        model_timeout=args.model_timeout, checkpoint_dir=args.checkpoint_dir,
    )
    if args.result:
        Path(args.result).write_text(json.dumps(outcome.to_dict(), indent=2, default=str), encoding="utf-8")
    print(json.dumps({"passed": outcome.passed, "iterations": outcome.iterations,
                      "verdict": outcome.verdict}, indent=2))
    return 0 if outcome.passed else 1


if __name__ == "__main__":
    sys.exit(main())
