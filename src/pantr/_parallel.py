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

    # Also throttle BLAS threads (requires threadpoolctl)
    with pantr.num_threads(4, limit_blas=True):
        result = space.tabulate_basis(pts)
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Generator
from typing import Any

import numba as nb


def get_num_threads() -> int:
    """Return the current number of threads used by parallel kernels.

    Returns:
        int: Active Numba thread-pool size.
    """
    return int(nb.get_num_threads())  # type: ignore[attr-defined]


def set_num_threads(n: int) -> None:
    """Set the number of threads used by parallel kernels.

    Args:
        n (int): Desired thread count.  Must be >= 1 and at most
            ``numba.config.NUMBA_NUM_THREADS``.

    Raises:
        ValueError: If *n* is less than 1 or exceeds the maximum.
    """
    max_threads: int = nb.config.NUMBA_NUM_THREADS  # type: ignore[attr-defined]
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n > max_threads:
        raise ValueError(f"n must be <= NUMBA_NUM_THREADS ({max_threads}), got {n}")
    nb.set_num_threads(n)  # type: ignore[attr-defined]


@contextlib.contextmanager
def num_threads(n: int, *, limit_blas: bool = False) -> Generator[None, None, None]:
    """Context manager that temporarily sets the parallel thread count.

    On entry the Numba thread-pool size is changed to *n*; on exit the
    previous value is restored.  When *limit_blas* is ``True`` and the
    optional ``threadpoolctl`` package is installed, BLAS/LAPACK thread
    pools (OpenBLAS, MKL, ...) are also limited to *n* threads for the
    duration of the block.

    Args:
        n (int): Desired thread count for the block.
        limit_blas (bool): If ``True``, also limit BLAS/LAPACK threads
            via ``threadpoolctl``.  Emits a warning when
            ``threadpoolctl`` is not installed.  Defaults to ``False``.

    Yields:
        None

    Example:
        >>> with pantr.num_threads(1):
        ...     # all pantr operations run serially here
        ...     pass
    """
    prev = get_num_threads()
    set_num_threads(n)

    blas_ctx: Any = None
    if limit_blas:
        try:
            from threadpoolctl import threadpool_limits  # noqa: PLC0415

            blas_ctx = threadpool_limits(limits=n)
            blas_ctx.__enter__()
        except ImportError:
            warnings.warn(
                "limit_blas=True requires the 'threadpoolctl' package. "
                "Install it with: pip install threadpoolctl",
                stacklevel=2,
            )

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
