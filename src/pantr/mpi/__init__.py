"""Optional MPI-parallel distribution layer for PaNTr.

This subpackage will host the MPI-dependent and dolfinx-bridge code for distributing
B-spline and THB-spline spaces across ranks. Everything here is optional: the
serial core (:mod:`pantr.grid`, :mod:`pantr.bspline`, ...) never imports it
(enforced by an import-linter contract), and ``import pantr.mpi`` succeeds even
when ``mpi4py`` is absent -- only helpers that genuinely need MPI raise at call
time, via :func:`require_mpi`.

``mpi4py`` is an opt-in dependency: a plain ``pip install pantr`` is serial-only
and MPI-free, while ``pip install "pantr[mpi]"`` adds ``mpi4py`` (and requires an
MPI library).

Main exports:
- :data:`HAS_MPI`: whether ``mpi4py`` was importable at module load time.
- :func:`mpi_available`: live runtime check for ``mpi4py`` availability.
- :func:`require_mpi`: lazily import and return the ``mpi4py.MPI`` module, or raise.
- :func:`from_dolfinx`: build a :class:`pantr.grid.Partition` from a dolfinx mesh.
- :class:`DistributedSpace`: the per-rank handle to an MPI-distributed space.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Final

from ._distributed_space import DistributedSpace
from ._from_dolfinx import from_dolfinx


def mpi_available() -> bool:
    """Report whether ``mpi4py`` can be imported in this environment.

    Uses :func:`importlib.util.find_spec`, so it never triggers an actual import
    or MPI initialization.

    Returns:
        bool: ``True`` if ``mpi4py`` is installed and importable, ``False`` otherwise.
    """
    return importlib.util.find_spec("mpi4py") is not None


HAS_MPI: Final[bool] = mpi_available()
"""bool: Whether ``mpi4py`` was importable at module load time.

This value is frozen at import and will not reflect environment changes made
afterwards. Use :func:`mpi_available` for a live probe. In code paths that
actually need MPI, always call :func:`require_mpi` — it checks freshly and
raises a clear error if MPI is unavailable or broken.
"""


def require_mpi() -> ModuleType:
    """Lazily import and return the ``mpi4py.MPI`` module, or raise a clear error.

    Call this from code paths that genuinely need MPI. Keeping the import lazy
    lets the serial core and a bare ``import pantr.mpi`` work without ``mpi4py``.

    Returns:
        ModuleType: The imported ``mpi4py.MPI`` module.

    Raises:
        ImportError: If ``mpi4py`` is not installed or fails to load (e.g. the
            underlying MPI runtime library is missing or ABI-incompatible).
    """
    if not mpi_available():
        raise ImportError(
            "pantr.mpi requires 'mpi4py', which is not installed. Install the optional "
            "MPI extra with 'pip install \"pantr[mpi]\"', or 'pip install mpi4py', in an "
            "environment with an MPI library."
        )
    try:
        return importlib.import_module("mpi4py.MPI")
    except Exception as exc:
        raise ImportError(
            "pantr.mpi found 'mpi4py' but failed to import 'mpi4py.MPI'. "
            "This usually means the MPI runtime library (e.g. libmpi.so) is missing "
            "or the mpi4py build is incompatible with the current environment. "
            "Install an MPI library (e.g. 'conda install openmpi'), or reinstall "
            "mpi4py against it (e.g. 'pip install --no-binary mpi4py mpi4py')."
        ) from exc


__all__ = [
    "HAS_MPI",
    "DistributedSpace",
    "from_dolfinx",
    "mpi_available",
    "require_mpi",
]
