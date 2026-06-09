"""agentica-core: open-weight agentic AI on a SLURM cluster.

Two modes, both built on AgenticLocal's ``agentic_loop`` engine:

1. Interactive gateway -- a localhost web chat + OpenAI-compatible ``/v1`` API
   backed by a model served on a remote GPU node (reached over an SSH tunnel).
2. Agentic job -- submit a plan file to SLURM as a batch job that drives a
   Planner -> Executor -> Auditor loop to completion.

This package depends on ``agentic_loop``. To make the sibling dev checkout work
without an explicit install, we add ../AgenticLocal to sys.path if needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

__version__ = "0.2.3"


def _ensure_agentic_loop_importable() -> None:
    try:
        import agentic_loop  # noqa: F401

        return
    except ImportError:
        pass
    # Fall back to the sibling checkout layout: <parent>/AgenticLocal.
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent.parent / "AgenticLocal",  # ~/Github/AgenticLocal
        here.parent.parent / "AgenticLocal",
    ]
    for candidate in candidates:
        if (candidate / "agentic_loop" / "__init__.py").exists():
            sys.path.insert(0, str(candidate))
            return


_ensure_agentic_loop_importable()
