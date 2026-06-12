"""Optional MPI-parallel distribution layer for PaNTr.

This subpackage will host the MPI-dependent and dolfinx-bridge code for distributing
B-spline and THB-spline spaces across ranks. Everything here is optional: the
serial core (:mod:`pantr.grid`, :mod:`pantr.bspline`, ...) never imports it
(enforced by an import-linter contract), and ``import pantr.mpi`` succeeds even
when ``mpi4py`` is absent -- only helpers that genuinely need MPI raise at call
time, via :func:`require_mpi`.

The default ``pip install pantr`` pulls in ``mpi4py``. Build with the
``PANTR_NO_MPI`` environment variable set to drop that dependency, yielding a
serial-only, MPI-free install.

Main exports:
- :data:`HAS_MPI`: whether ``mpi4py`` was importable at module load time.
- :func:`mpi_available`: live runtime check for ``mpi4py`` availability.
- :func:`require_mpi`: lazily import and return the ``mpi4py.MPI`` module, or raise.
- :func:`from_dolfinx`: build a :class:`pantr.grid.Partition` from a dolfinx mesh.
- :class:`DistributedSpace`: the per-rank handle to an MPI-distributed space.
- :func:`configure_threads`: explicitly set this rank's Numba thread count.

A process-level thread policy coordinates MPI with PaNTr's Numba parallelism: the
first use of any MPI-engaging entry point limits this process to **one Numba thread
per rank**, unless threads were explicitly configured (``NUMBA_NUM_THREADS``,
``pantr.set_num_threads``, ``pantr.num_threads``, or :func:`configure_threads`).
Every new MPI entry point added to this package must call
``_ensure_default_thread_policy()`` from :mod:`pantr.mpi._thread_policy` before any
other work -- there is no structural chokepoint, since communicators are duck-typed.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Final

from ._distributed_space import DistributedSpace
from ._from_dolfinx import from_dolfinx
from ._thread_policy import configure_threads


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
            "pantr.mpi requires 'mpi4py', which is not installed. It ships by default "
            "with pantr; reinstall without the PANTR_NO_MPI build flag, or run "
            "'pip install mpi4py' in an environment with an MPI library."
        )
    try:
        return importlib.import_module("mpi4py.MPI")
    except Exception as exc:
        raise ImportError(
            "pantr.mpi found 'mpi4py' but failed to import 'mpi4py.MPI'. "
            "This usually means the MPI runtime library (e.g. libmpi.so) is missing "
            "or the mpi4py build is incompatible with the current environment. "
            "Install an MPI library (e.g. 'conda install openmpi') or reinstall "
            "pantr without MPI via 'PANTR_NO_MPI=1 pip install pantr'."
        ) from exc


__all__ = [
    "HAS_MPI",
    "DistributedSpace",
    "configure_threads",
    "from_dolfinx",
    "mpi_available",
    "require_mpi",
]
