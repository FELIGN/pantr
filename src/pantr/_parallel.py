"""Thread-count control for PaNTr's parallel kernels.

Provides a thin wrapper around Numba's thread-pool configuration so that
users can limit (or disable) the parallelism used by PaNTr without
touching environment variables or Numba internals directly.

Typical usage::

    import pantr

    # Query / set globally
    pantr.set_num_threads(4)
    print(pantr.get_num_threads())

    # Scoped override (restores previous value on exit)
    with pantr.num_threads(1):
        result = space.tabulate_basis(pts)  # runs serially

    # Also throttle BLAS threads
    with pantr.num_threads(4, limit_blas=True):
        result = space.tabulate_basis(pts)
"""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Generator
from typing import Any, Final

import numba as nb
from threadpoolctl import threadpool_limits

_user_configured: Final[threading.Event] = threading.Event()
"""Set by the public setters; queried by default policies (e.g. the per-rank MPI
default in :mod:`pantr.mpi`) so they never override an explicit user decision."""


def get_num_threads() -> int:
    """Return the current number of threads used by parallel kernels.

    Returns:
        int: Active Numba thread-pool size.
    """
    return int(nb.get_num_threads())


def _set_num_threads_raw(n: int) -> None:
    """Set the Numba thread count without marking it as explicit configuration.

    Internal hook for default policies (e.g. the per-rank default applied by
    :mod:`pantr.mpi`): performs the same validation as :func:`set_num_threads`
    but leaves the explicit-configuration flag untouched, so a policy-applied
    value remains distinguishable from a user decision.

    Args:
        n (int): Desired thread count.  Must be >= 1 and at most
            ``numba.config.NUMBA_NUM_THREADS``.

    Raises:
        ValueError: If *n* is less than 1 or exceeds the maximum.
    """
    max_threads: int = nb.config.NUMBA_NUM_THREADS
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n > max_threads:
        raise ValueError(f"n must be <= NUMBA_NUM_THREADS ({max_threads}), got {n}")
    nb.set_num_threads(n)


def set_num_threads(n: int) -> None:
    """Set the number of threads used by parallel kernels.

    Calling this marks the thread count as explicitly configured: default
    policies (e.g. the per-rank MPI default in :mod:`pantr.mpi`) will never
    override it afterwards.

    Args:
        n (int): Desired thread count.  Must be >= 1 and at most
            ``numba.config.NUMBA_NUM_THREADS``.

    Raises:
        ValueError: If *n* is less than 1 or exceeds the maximum.
    """
    _set_num_threads_raw(n)
    _user_configured.set()


def _threads_explicitly_configured() -> bool:
    """Report whether the user explicitly configured the thread count.

    ``True`` when :func:`set_num_threads` was called directly, when the
    ``NUMBA_NUM_THREADS`` environment variable is set, or when the
    :func:`num_threads` context manager was used.  The flag is set on both
    entry *and* exit of :func:`num_threads` (exit restores the previous count
    via :func:`set_num_threads`), so any use of that context manager permanently
    marks the process as explicitly configured for the rest of its lifetime.

    Returns:
        bool: Whether explicit thread configuration is in effect.
    """
    return _user_configured.is_set() or "NUMBA_NUM_THREADS" in os.environ


@contextlib.contextmanager
def num_threads(n: int, *, limit_blas: bool = False) -> Generator[None, None, None]:
    """Context manager that temporarily sets the parallel thread count.

    On entry the Numba thread-pool size is changed to *n*; on exit the
    previous value is restored.  When *limit_blas* is ``True``, BLAS/LAPACK
    thread pools (OpenBLAS, MKL, ...) are also limited to *n* threads for the
    duration of the block, via ``threadpoolctl``.

    Args:
        n (int): Desired thread count for the block.
        limit_blas (bool): If ``True``, also limit BLAS/LAPACK threads
            via ``threadpoolctl``.  Defaults to ``False``.

    Yields:
        None

    Note:
        Using this context manager permanently marks the thread count as explicitly
        configured (via :func:`set_num_threads` on both entry and exit).  Default
        policies -- e.g. the per-rank MPI default in :mod:`pantr.mpi` -- will not
        override the thread count for the rest of the process lifetime, even after
        the ``with`` block exits.

    Example:
        >>> with pantr.num_threads(1):
        ...     # all pantr operations run serially here
        ...     pass
    """
    prev = get_num_threads()
    set_num_threads(n)

    blas_ctx: Any = None
    if limit_blas:
        blas_ctx = threadpool_limits(limits=n)
        blas_ctx.__enter__()

    try:
        yield
    finally:
        if blas_ctx is not None:
            blas_ctx.__exit__(None, None, None)
        set_num_threads(prev)


__all__ = [
    "get_num_threads",
    "num_threads",
    "set_num_threads",
]
