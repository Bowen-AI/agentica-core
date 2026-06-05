import sys
from pathlib import Path

# Make the repo root (and thus agentica_core) importable without an install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agentica_core  # noqa: E402,F401  (also wires up agentic_loop on sys.path)
