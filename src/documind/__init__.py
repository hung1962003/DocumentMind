"""DocuMind package entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """CLI entry for `uv run documind` / `documind`."""
    root = Path(__file__).resolve().parents[2]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from main import main as run_main

    run_main()
