"""Optional MPI-parallel distribution layer for PaNTr.

This subpackage hosts the MPI-dependent and dolfinx-bridge code for distributing
B-spline and THB-spline spaces across ranks. Everything here is optional: the
serial core (:mod:`pantr.grid`, :mod:`pantr.bspline`, ...) never imports it
(enforced by an import-linter contract), and ``import pantr.mpi`` succeeds even
when ``mpi4py`` is absent -- only helpers that genuinely need MPI raise at call
time, via :func:`require_mpi`.

The default ``pip install pantr`` pulls in ``mpi4py``. Build with the
``PANTR_NO_MPI`` environment variable set to drop that dependency, yielding a
serial-only, MPI-free install.

Main exports:
- :data:`HAS_MPI`: whether ``mpi4py`` is importable in the current environment.
- :func:`mpi_available`: runtime check for ``mpi4py`` availability.
- :func:`require_mpi`: lazily import and return the ``mpi4py.MPI`` module, or raise.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Final


def mpi_available() -> bool:
    """Report whether ``mpi4py`` can be imported in this environment.

    Uses :func:`importlib.util.find_spec`, so it never triggers an actual import
    or MPI initialization.

    Returns:
        bool: ``True`` if ``mpi4py`` is installed and importable, ``False`` otherwise.
    """
    return importlib.util.find_spec("mpi4py") is not None


HAS_MPI: Final[bool] = mpi_available()
"""bool: Whether ``mpi4py`` is importable, evaluated once at import time."""


def require_mpi() -> ModuleType:
    """Lazily import and return the ``mpi4py.MPI`` module, or raise a clear error.

    Call this from code paths that genuinely need MPI. Keeping the import lazy
    lets the serial core and a bare ``import pantr.mpi`` work without ``mpi4py``.

    Returns:
        ModuleType: The imported ``mpi4py.MPI`` module.

    Raises:
        ImportError: If ``mpi4py`` is not installed, with guidance on how to obtain it.
    """
    if not mpi_available():
        raise ImportError(
            "pantr.mpi requires 'mpi4py', which is not installed. It ships by default "
            "with pantr; reinstall without the PANTR_NO_MPI build flag, or run "
            "'pip install mpi4py' in an environment with an MPI library."
        )
    return importlib.import_module("mpi4py.MPI")


__all__ = [
    "HAS_MPI",
    "mpi_available",
    "require_mpi",
]
