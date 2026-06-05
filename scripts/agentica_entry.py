"""PyInstaller entry point: a single-file `agentica-core` binary.

Bundles agentica_core + agentic_loop + PyYAML so the Agentica AppImage can run the
backend (`agentica-core serve-api ...`) with no Python install.
"""

import sys

from agentica_core.cli import main

if __name__ == "__main__":
    sys.exit(main())
