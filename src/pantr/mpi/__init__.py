"""Optional MPI-parallel distribution layer for PaNTr.

This subpackage hosts the MPI-dependent and dolfinx-bridge code for distributing
B-spline and THB-spline spaces across ranks. It ships with every install of PaNTr but
is **backend-gated**: the serial core (:mod:`pantr.grid`, :mod:`pantr.bspline`, ...)
never imports it (enforced by an import-linter contract), and ``import pantr.mpi``
succeeds even when ``mpi4py`` is absent -- only helpers that genuinely need MPI raise at
call time, via :func:`require_mpi`.

``mpi4py`` is the optional *backend*, not extra PaNTr code: a plain ``pip install pantr``
is serial-only and MPI-free, while ``pip install "pantr[mpi]"`` simply installs ``mpi4py``
for you (equivalently ``pip install mpi4py``; an MPI library is also required).

Main exports:
- :data:`HAS_MPI`: whether ``mpi4py`` was importable at module load time.
- :func:`mpi_available`: live runtime check for ``mpi4py`` availability.
- :func:`require_mpi`: lazily import and return the ``mpi4py.MPI`` module, or raise.
- :func:`from_dolfinx`: build a :class:`pantr.grid.Partition` from a dolfinx mesh.
- :class:`DistributedSpace`: the per-rank handle to an MPI-distributed space.
- :func:`create_distributed_space`: build a :class:`DistributedSpace` from a space.
- :class:`DistributedFunction`: the per-rank handle to an MPI-distributed function.
- :func:`create_distributed_function`: build a :class:`DistributedFunction` from a function.
- :func:`configure_threads`: explicitly set this rank's Numba thread count.
- :func:`quasi_interpolate_bspline_distributed`: distributed B-spline quasi-interpolation.
- :func:`quasi_interpolate_thb_spline_distributed`: distributed THB-spline quasi-interpolation.

A process-level thread policy coordinates MPI with PaNTr's Numba parallelism: the
first use of any MPI-engaging entry point limits this process to **one Numba thread
per rank**, unless threads were explicitly configured (``NUMBA_NUM_THREADS``,
``pantr.set_num_threads``, ``pantr.num_threads``, or :func:`configure_threads`).
Every new MPI entry point added to this package must call
``_ensure_default_thread_policy()`` from ``pantr.mpi._thread_policy`` before any
other work -- there is no structural chokepoint, since communicators are duck-typed.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Final

from ._create import create_distributed_space
from ._distributed_function import DistributedFunction, create_distributed_function
from ._distributed_space import DistributedSpace
from ._from_dolfinx import from_dolfinx
from ._qi import quasi_interpolate_bspline_distributed
from ._thb_qi import quasi_interpolate_thb_spline_distributed
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
            "pantr.mpi requires 'mpi4py', which is not installed. Install the optional "
            "MPI extra with 'pip install \"pantr[mpi]\"', or 'pip install mpi4py', in an "
            "environment with an MPI library."
        )
    try:
        return importlib.import_module("mpi4py.MPI")
    except Exception as exc:
        err = ImportError("pantr.mpi found 'mpi4py' but failed to import 'mpi4py.MPI'.")
        err.add_note("The MPI runtime library (e.g. libmpi.so) may be missing or ABI-incompatible.")
        err.add_note("Fix: install an MPI library (e.g. 'conda install openmpi').")
        err.add_note(
            "Fix: reinstall mpi4py against it (e.g. 'pip install --no-binary mpi4py mpi4py')."
        )
        raise err from exc


__all__ = [
    "HAS_MPI",
    "DistributedFunction",
    "DistributedSpace",
    "configure_threads",
    "create_distributed_function",
    "create_distributed_space",
    "from_dolfinx",
    "mpi_available",
    "quasi_interpolate_bspline_distributed",
    "quasi_interpolate_thb_spline_distributed",
    "require_mpi",
]
