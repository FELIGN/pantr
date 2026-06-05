"""Pytest configuration: make `src` and the repository root importable.

`src` is added so the package resolves without installation.  The repository root is
added so test modules can import shared, non-collected helpers from the regular
``tests`` package (e.g. ``from tests._thb_assembly import ...``); under pytest this
already resolves via ``rootdir``, so the addition is a belt-and-suspenders guard for
direct (non-pytest) invocation.
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
