"""Process-level Numba thread policy for MPI runs.

Without coordination, every MPI rank evaluating PaNTr's parallel kernels spawns one
Numba thread per logical core, so a node running ``R`` ranks executes ``R x n_cores``
compute threads -- heavy oversubscription. To prevent this, every MPI-engaging entry
point of :mod:`pantr.mpi` (:class:`~pantr.mpi.DistributedSpace`,
:func:`~pantr.mpi.from_dolfinx`, and any future ones) applies a process-level default
on first use: the Numba thread pool of this process is limited to **one thread per
rank** (flat MPI).

The default is sticky (applied at most once per process) and never overrides explicit
user configuration; see :func:`_ensure_default_thread_policy`. The ``NUMBA_NUM_THREADS``
cap is never modified, so the count can always be raised back at runtime (e.g. for
rank-0 postprocessing) via ``pantr.set_num_threads`` or :func:`configure_threads`.

This module never imports ``mpi4py`` -- it only touches Numba (and, on request,
``threadpoolctl``) state -- so it works with duck-typed communicators (serial tests)
and in MPI-free installs alike.

Note:
    Numba's thread mask is **thread-local**: the policy and
    :func:`configure_threads` only affect kernels launched from the calling thread --
    in SPMD practice, the rank's main thread. Kernels launched from other
    user-created threads keep the full default width; call
    ``pantr.set_num_threads`` inside each such thread if needed.
"""

from __future__ import annotations

import threading
import warnings
from typing import Any, Final

from .. import _parallel

_DEFAULT_THREADS_PER_RANK: Final[int] = 1
"""Numba threads granted to each MPI rank by the default policy (flat MPI)."""

_policy_applied: Final[threading.Event] = threading.Event()
"""Set once the default policy has run in this process (sticky guard)."""

_blas_state: Final[dict[str, Any]] = {"limiter": None}
"""Holds the live ``threadpoolctl.threadpool_limits`` object (persistent BLAS limit)."""


def configure_threads(threads_per_rank: int = 1, *, limit_blas: bool = False) -> None:
    """Explicitly set the number of Numba threads this process (MPI rank) uses.

    The explicit alternative to the default policy: call it on every rank -- at any
    point, including before any :mod:`pantr.mpi` object is built -- to choose how many
    threads each rank may use in a hybrid MPI + threads run. Marks the thread count as
    explicitly configured, so the default one-thread policy never overrides it. Also
    the right call when running serial PaNTr under your own ``mpiexec`` without any
    :mod:`pantr.mpi` machinery.

    Args:
        threads_per_rank (int): Numba threads this process may use. Must be >= 1 and
            at most ``numba.config.NUMBA_NUM_THREADS``. Defaults to 1.
        limit_blas (bool): If ``True``, also persistently limit this process's
            BLAS/LAPACK thread pools (OpenBLAS, MKL, ...) to *threads_per_rank* via
            the optional ``threadpoolctl`` package; emits a ``UserWarning`` when it is
            not installed. A successive call replaces the previous limit. Defaults to
            ``False``.

    Raises:
        ValueError: If *threads_per_rank* is less than 1 or exceeds
            ``numba.config.NUMBA_NUM_THREADS``.

    Note:
        Numba's thread mask is thread-local; see the module docstring.
    """
    _parallel.set_num_threads(threads_per_rank)
    _policy_applied.set()
    if limit_blas:
        try:
            from threadpoolctl import threadpool_limits  # noqa: PLC0415
        except ImportError:
            warnings.warn(
                "limit_blas=True requires the 'threadpoolctl' package. "
                "Install it with: pip install threadpoolctl",
                stacklevel=2,
            )
        else:
            if _blas_state["limiter"] is not None:
                _blas_state["limiter"].restore_original_limits()
            _blas_state["limiter"] = threadpool_limits(limits=threads_per_rank)


def _ensure_default_thread_policy() -> None:
    """Apply the default MPI thread policy (1 Numba thread per rank), at most once.

    Called by every MPI-engaging entry point of :mod:`pantr.mpi` before any other
    work. No-op when the policy already ran in this process or when the thread count
    was explicitly configured (``NUMBA_NUM_THREADS`` environment variable,
    ``pantr.set_num_threads`` / ``pantr.num_threads``, or :func:`configure_threads`).
    Sticky: once applied, later user changes are never overridden by subsequent
    entry-point calls.

    Note:
        Every new MPI entry point added to :mod:`pantr.mpi` must call this hook
        first; there is no structural chokepoint (communicators are duck-typed and
        ``mpi4py`` is never imported here).
    """
    if _policy_applied.is_set() or _parallel._threads_explicitly_configured():
        return
    _parallel._set_num_threads_raw(_DEFAULT_THREADS_PER_RANK)
    _policy_applied.set()


def _reset_policy_for_testing() -> None:
    """Reset all sticky thread-policy state (testing only).

    Clears the policy-applied flag, the explicit-configuration flag in
    :mod:`pantr._parallel`, and releases any persistent BLAS limit installed by
    :func:`configure_threads`. The Numba thread count itself is not restored;
    callers must snapshot and restore it around tests.
    """
    _policy_applied.clear()
    _parallel._user_configured.clear()
    if _blas_state["limiter"] is not None:
        _blas_state["limiter"].restore_original_limits()
        _blas_state["limiter"] = None


__all__ = ["configure_threads"]
