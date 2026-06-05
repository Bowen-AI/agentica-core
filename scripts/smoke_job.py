"""Mode-2 smoke: run the Planner->Executor->Auditor loop with a REAL local model.

Trivial task so a small model can complete it; the deterministic backstops
(test command + artifact) are authoritative regardless of model quality.

    python scripts/smoke_job.py [--model gemma4:e4b]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agentica_core  # noqa: F401
from agentica_core.config import PlanConfig, SuccessCriteria
from agentica_core.on_node_runner import run_job


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4:e4b")
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    args = ap.parse_args()

    ws = Path(tempfile.mkdtemp(prefix="smoke-job-"))
    plan = PlanConfig(
        title="hello file",
        goal=("Create a file named hello.txt in the workspace whose contents are exactly the "
              "word: hello . Use the write_file tool."),
        workspace=str(ws),
        checklist=["hello.txt exists and contains 'hello'"],
        success_criteria=SuccessCriteria(
            tests="grep -qx hello hello.txt", artifacts=["hello.txt"]),
        max_iterations=2, max_steps_per_iteration=6,
    )
    outcome = run_job(plan, workspace=str(ws), db_path=str(ws / "job.db"),
                      provider="ollama", model_name=args.model, ollama_host=args.ollama_host)
    print("\n=== OUTCOME ===")
    print(f"passed={outcome.passed} iterations={outcome.iterations} verdict={outcome.verdict} "
          f"tests_ok={outcome.tests_ok} artifacts_ok={outcome.artifacts_ok}")
    hello = ws / "hello.txt"
    if hello.exists():
        print(f"hello.txt -> {hello.read_text()!r}")
    return 0 if outcome.passed else 1


if __name__ == "__main__":
    sys.exit(main())
