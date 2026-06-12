"""Pytest configuration: make `src` and the repository root importable.

`src` is added so the package resolves without installation.  The repository root is
added so test modules can import shared, non-collected helpers from the regular
``tests`` package (e.g. ``from tests._thb_assembly import ...``); under pytest this
already resolves via ``rootdir``, so the addition is a belt-and-suspenders guard for
direct (non-pytest) invocation.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest


def _ensure_on_sys_path() -> None:
    """Prepend the repository `src` directory and the repository root to `sys.path`."""
    repo_root: Path = Path(__file__).resolve().parents[1]
    for path in (str(repo_root / "src"), str(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


_ensure_on_sys_path()


@pytest.fixture(autouse=True)
def _isolate_thread_state() -> Iterator[None]:
    """Keep the Numba thread count and pantr's thread-policy state test-local.

    MPI-engaging entry points (:mod:`pantr.mpi`) apply a sticky process-level default
    of one Numba thread per rank; without isolation, a single such test would
    serialize every later parallel-kernel test in the session.

    Yields:
        None
    """
    import numba as nb  # noqa: PLC0415

    prev = int(nb.get_num_threads())
    yield
    from pantr.mpi import _thread_policy  # noqa: PLC0415

    _thread_policy._reset_policy_for_testing()
    nb.set_num_threads(prev)
