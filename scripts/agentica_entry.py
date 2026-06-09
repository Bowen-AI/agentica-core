"""PyInstaller entry point: a single-file `agentica-core` binary.

Bundles agentica_core + agentic_loop + PyYAML so the Agentica AppImage can run the
backend (`agentica-core serve-api ...`) with no Python install.
"""

import sys

# Debug affordance for the frozen binary: `kill -USR1 <pid>` dumps every thread's
# stack to stderr (py-spy needs root on macOS, so this is the only way to see
# where a frozen process is stuck in the field).
try:
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)
except Exception:  # noqa: BLE001 - diagnostics must never block startup
    pass

from agentica_core.cli import main

if __name__ == "__main__":
    sys.exit(main())
