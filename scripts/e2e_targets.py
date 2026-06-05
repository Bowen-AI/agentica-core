"""Cross-target e2e: exercise ALL functionality on local / ssh / slurm.

For each target it runs the agentic JOB (Planner->Executor->Auditor + deterministic
backstop) end-to-end and asserts it passes. Locally it also exercises chat
(agentic + plain) and the plan draft/refine flow via the API server.

    # local only (needs a local ollama + model)
    python scripts/e2e_targets.py --local-model gemma4:e4b

    # add real remote targets (ssh box + slurm cluster)
    python scripts/e2e_targets.py --local-model gemma4:e4b \
        --ssh examples/pinotage.yaml --slurm examples/discovery_debug.yaml

Verified runs (2026-06): local PASS, ssh pinotage PASS (job-6ab1555c),
slurm discovery a40 PASS (9210485).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agentica_core  # noqa: F401
from agentica_core import apiserver, job
from agentica_core.config import PlanConfig, SuccessCriteria
from agentica_core.on_node_runner import run_job

HELLO_PLAN = {
    "title": "hello", "goal": "Use the write_file tool to create hello.txt with content exactly: hello",
    "lines": [{"id": "L1", "text": "write hello.txt"}],
    "tests": "grep -qx hello hello.txt", "artifacts": ["hello.txt"],
}

results: list[tuple[str, str, bool, str]] = []


def record(target: str, name: str, ok: bool, detail: str = ""):
    results.append((target, name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {target:14s} {name:18s} {detail}")


def e2e_local(model: str, host: str):
    # chat (plain + agentic) + plan draft/refine via the API state
    st = apiserver.State(ollama_host=host, model=model, workspace="sample_workspace",
                         db_path=tempfile.mktemp(suffix=".db"))
    try:
        ans = st.complete([{"role": "user", "content": "What is 2+2? one number"}])
        record("local", "chat-plain", "4" in ans, f"ans={ans[:24]!r}")
    except Exception as exc:  # noqa: BLE001
        record("local", "chat-plain", False, str(exc))

    plan = apiserver.draft_plan(st, "create hello.txt containing hello", None)
    record("local", "plan-draft", bool(plan["lines"]), f"{len(plan['lines'])} lines")
    refined = apiserver.refine_plan(st, plan, [{"line": "L1", "text": "be explicit"}])
    record("local", "plan-refine", bool(refined["lines"]), f"{len(refined['lines'])} lines")

    # job (local thread)
    pc = PlanConfig(title="hello", goal=HELLO_PLAN["goal"], workspace=tempfile.mkdtemp(),
                    checklist=["hello.txt == hello"],
                    success_criteria=SuccessCriteria(tests="grep -qx hello hello.txt", artifacts=["hello.txt"]),
                    max_iterations=3, max_steps_per_iteration=6)
    out = run_job(pc, workspace=pc.workspace, db_path=str(Path(pc.workspace) / "j.db"),
                  provider="ollama", model_name=model, ollama_host=host, model_timeout=300,
                  _print=lambda *a: None)
    record("local", "job", out.passed, f"verdict={out.verdict} tests_ok={out.tests_ok}")


def e2e_remote(label: str, cluster_path: str, timeout_s: float = 900):
    """Submit the hello job to a real ssh/slurm target and poll to completion."""
    tmp = Path(tempfile.mkdtemp())
    plan_yaml = tmp / "plan.yaml"
    plan_yaml.write_text(apiserver.plan_to_yaml(HELLO_PLAN, "./ws"), encoding="utf-8")
    out: list[str] = []
    rc = job.submit(cluster_path, str(plan_yaml), sync_code=True, _print=out.append)
    if rc != 0:
        return record(label, "job-submit", False, "submit rc=%d" % rc)
    jid = jobdir = None
    for line in out:
        for tok in line.split():
            if tok.startswith("job_id="):
                jid = tok.split("=", 1)[1]
            if tok.startswith("jobdir="):
                jobdir = tok.split("=", 1)[1]
    record(label, "job-submit", bool(jid), f"job_id={jid}")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        s: list[str] = []
        job.status(cluster_path, jid, jobdir=jobdir, _print=s.append)
        joined = " ".join(s)
        if "passed=True" in joined:
            return record(label, "job-result", True, "passed")
        if "passed=False" in joined:
            return record(label, "job-result", False, "failed")
        time.sleep(15)
    record(label, "job-result", False, "timeout")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-model", default="gemma4:e4b")
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    ap.add_argument("--ssh", default=None, help="cluster.yaml or alias for a plain ssh GPU box")
    ap.add_argument("--slurm", default=None, help="cluster.yaml or alias for a SLURM cluster")
    ap.add_argument("--no-local", action="store_true")
    args = ap.parse_args()

    if not args.no_local:
        print("== LOCAL ==")
        e2e_local(args.local_model, args.ollama_host)
    if args.ssh:
        print("== SSH ==")
        e2e_remote("ssh", args.ssh)
    if args.slurm:
        print("== SLURM ==")
        e2e_remote("slurm", args.slurm)

    npass = sum(1 for r in results if r[2])
    print(f"\n=== {npass}/{len(results)} checks passed ===")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
