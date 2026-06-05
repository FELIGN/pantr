"""Pytest configuration: make `src` and the repository root importable.

`src` is added so the package resolves without installation; the repository root is
added so test modules can import shared, non-collected helpers via the ``tests``
namespace package (e.g. ``from tests._thb_assembly import ...``).
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_on_sys_path() -> None:
    """Prepend the repository `src` directory and the repository root to `sys.path`."""
    repo_root: Path = Path(__file__).resolve().parents[1]
    for path in (str(repo_root / "src"), str(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


_ensure_on_sys_path()
