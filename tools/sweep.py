#!/usr/bin/env python
"""Launcher for the adversarial parameter sweep.

Puts this checkout's ``src`` and ``tools`` on ``sys.path`` -- so the sweep always
exercises the worktree's own source, never whatever an editable install points at --
and hands over to :mod:`adversarial_sweep.__main__`, which configures Numba's bounds
check before importing anything compiled.

Usage::

    conda run -n pantr python tools/sweep.py --profile smoke
    conda run -n pantr python tools/sweep.py --profile full --journal sweep.jsonl
    conda run -n pantr python tools/sweep.py --list-groups
"""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools"))
sys.path.insert(0, str(_ROOT / "src"))

from adversarial_sweep.__main__ import main  # noqa: E402  -- needs the path set up first

if __name__ == "__main__":
    raise SystemExit(main())
