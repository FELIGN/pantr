"""Batch-parallel root finding kernels for Bernstein polynomials.

Provides ``nb_prange``-parallelized wrappers that solve multiple independent
polynomials concurrently. These are the **only** kernels in the root-finding
module that use ``parallel=True``.

Main exports:

- :func:`_find_roots_batch_core` -- find roots of many same-degree polynomials.
- :func:`_solve_monotone_root_batch_core` -- batch monotone root solver.
"""

from __future__ import annotations

import numpy as np
from numpy import typing as npt

from pantr._numba_compat import nb_jit, nb_prange
from pantr.bezier._clipping_core import _clip_roots_core, _dedup_roots_core
from pantr.bezier._yuksel_core import (
    _solve_monotone_root_kernel,
    _yuksel_roots,
)

_CLIP_MIN_DEGREE: int = 6
"""Minimum polynomial degree for which Bezier clipping is considered.

Duplicated from ``_find_roots.py`` (Numba kernels cannot import Python-level
module constants at call time) -- keep in sync.
"""

_CLIP_COEFF_RANGE_LIMIT: float = 1e8
"""Maximum coefficient dynamic range for which Bezier clipping is used.

Duplicated from ``_find_roots.py`` -- keep in sync.
"""


@nb_jit(nopython=True, cache=True)
def _dispatch_and_find(
    coeff: npt.NDArray[np.float32 | np.float64],
    param_tol: float,
    geom_tol: float,
) -> tuple[npt.NDArray[np.float64], int]:
    """Auto-dispatch root finding for a single polynomial.

    Selects between Yuksel and Bezier clipping based on degree and coefficient
    dynamic range. Returns the found roots as a sorted, deduplicated array
    together with the count of valid entries.

    Args:
        coeff (npt.NDArray[np.float32 | np.float64]): 1-D Bernstein coefficients.
        param_tol (float): Parametric tolerance.
        geom_tol (float): Geometric tolerance.

    Returns:
        tuple[npt.NDArray[np.float64], int]: ``(roots_array, count)`` where
            only the first ``count`` entries are valid roots, sorted in
            ascending order.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`pantr.bezier.find_roots_batch`
        instead.
    """
    n = len(coeff) - 1
    if n < 1:
        return np.empty(0, dtype=np.float64), 0

    # All-zero check.
    all_zero = True
    for i in range(n + 1):
        if abs(coeff[i]) > geom_tol:
            all_zero = False
            break
    if all_zero:
        return np.empty(0, dtype=np.float64), 0

    use_clipping = False
    if n >= _CLIP_MIN_DEGREE:
        c_max = 0.0
        c_min_nonzero = float("inf")
        for i in range(n + 1):
            av = abs(coeff[i])
            c_max = max(c_max, av)
            if av > 0.0 and av < c_min_nonzero:
                c_min_nonzero = av
        coeff_range = c_max / c_min_nonzero if c_min_nonzero < float("inf") else float("inf")
        if coeff_range <= _CLIP_COEFF_RANGE_LIMIT:
            use_clipping = True

    if use_clipping:
        raw_roots, n_roots = _clip_roots_core(coeff, param_tol, geom_tol)
    else:
        raw_roots, n_roots = _yuksel_roots(coeff, param_tol)

    if n_roots == 0:
        return raw_roots, 0

    # Sort and deduplicate with the shared derivative-aware merge (capped for
    # multiple roots), matching the single-polynomial path.
    return _dedup_roots_core(raw_roots, n_roots, coeff, param_tol, geom_tol)


@nb_jit(nopython=True, cache=True, parallel=True)
def _find_roots_batch_core(
    coeffs: npt.NDArray[np.float32 | np.float64],
    param_tol: float,
    geom_tol: float,
    out_roots: npt.NDArray[np.float32 | np.float64],
    out_counts: npt.NDArray[np.intp],
) -> None:
    """Find roots of multiple same-degree Bernstein polynomials in parallel.

    Each polynomial is solved independently using auto-dispatch between Yuksel
    and Bezier clipping. Parallelized over polynomials with ``nb_prange``.

    Args:
        coeffs (npt.NDArray[np.float32 | np.float64]): Batch of coefficients with shape
            ``(n_polys, degree + 1)``.
        param_tol (float): Parametric tolerance.
        geom_tol (float): Geometric tolerance.
        out_roots (npt.NDArray[np.float32 | np.float64]): Pre-allocated output array of
            shape ``(n_polys, degree)`` to receive roots. Unused entries are
            not zeroed.
        out_counts (npt.NDArray[np.intp]): Pre-allocated output array of
            shape ``(n_polys,)`` to receive root counts.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`pantr.bezier.find_roots_batch`
        instead.
    """
    n_polys = coeffs.shape[0]
    for i in nb_prange(n_polys):
        coeff_i = coeffs[i].copy()
        roots, count = _dispatch_and_find(coeff_i, param_tol, geom_tol)
        # A degree-n polynomial has at most n roots; clamp as a memory-safety
        # backstop so a dedup artifact can never overflow the output row.
        count = min(count, out_roots.shape[1])
        out_counts[i] = count
        for j in range(count):
            out_roots[i, j] = roots[j]


@nb_jit(nopython=True, cache=True, parallel=True)
def _solve_monotone_root_batch_core(
    coeffs: npt.NDArray[np.float32 | np.float64],
    param_tol: float,
    out_roots: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Solve for monotone roots on multiple polynomials in parallel.

    Each polynomial is solved independently using the Newton/bisection hybrid.
    Parallelized over polynomials with ``nb_prange``. Entries where no root
    exists are left as ``NaN`` (the caller pre-fills the array).

    Args:
        coeffs (npt.NDArray[np.float32 | np.float64]): Batch of coefficients
            with shape ``(n_polys, degree + 1)``.
        param_tol (float): Parameter-space termination tolerance.
        out_roots (npt.NDArray[np.float32 | np.float64]): Pre-allocated output
            array of shape ``(n_polys,)`` for root values. Must be
            pre-filled with ``NaN``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call
        :func:`pantr.bezier.solve_monotone_root_batch` instead.
    """
    n_polys = coeffs.shape[0]
    for i in nb_prange(n_polys):
        coeff_i = coeffs[i].copy()
        root = _solve_monotone_root_kernel(coeff_i, param_tol)
        if not np.isnan(root):
            out_roots[i] = root


def _warmup_numba_functions() -> None:
    """Trigger Numba compilation of the batch kernels and their direct dependencies.

    Called from the background warmup thread in ``pantr.__init__``.
    """
    coeffs = np.array([[1.0, -1.0, 0.5]], dtype=np.float64)
    out_roots = np.empty((1, 2), dtype=np.float64)
    out_counts = np.zeros(1, dtype=np.intp)
    _find_roots_batch_core(coeffs, 1e-12, 1e-12, out_roots, out_counts)

    out_mono = np.full(1, np.nan, dtype=np.float64)
    _solve_monotone_root_batch_core(coeffs, 1e-12, out_mono)
