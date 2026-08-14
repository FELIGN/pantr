"""Numba kernels for B-spline knot insertion via the Oslo algorithm.

Implements the discrete B-spline recurrence from Cohen, Lyche & Riesenfeld
(1980), which maps old control points to new ones in a single pass.

The recurrence is run **row by row over a band of ``degree + 1`` entries**
(Lyche & Mørken, *Spline Methods*, section 4.2.3), not over a dense matrix: row
``i`` of the refinement matrix is a discrete B-spline supported on the columns
``[mu(i) - degree, mu(i)]``, so everything a dense sweep computes outside that
window is exactly zero.  One row costs ``O(degree^2)`` instead of
``O(n * degree)``.

Both forms are available.  :func:`_insert_knots_1d_core` never materialises the
matrix, while :func:`_compute_oslo_matrix_1d_core` scatters the bands into the
dense array its callers expect.

Note:
    Inputs are assumed to be correct (no validation performed).
    For general use, call the Layer 2 helpers in ``_bspline_knot_insertion`` instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .._numba_compat import nb_jit


@nb_jit(nopython=True, cache=True)
def _compute_oslo_rows_1d_core(
    degree: int,
    old_knots: npt.NDArray[Any],
    new_knots: npt.NDArray[Any],
) -> tuple[npt.NDArray[Any], npt.NDArray[np.int64]]:
    r"""Compute the Oslo refinement matrix one banded row at a time.

    Row ``i`` of the refinement matrix holds the discrete B-splines
    :math:`\alpha_{j,p+1}(i)`, which vanish outside ``j`` in
    ``[mu(i) - degree, mu(i)]`` where ``mu(i)`` is the old-knot span containing
    ``new_knots[i]``.  Only that window is computed, by the recurrence

    .. math::

        \alpha_j^{(k+1)}
        = \frac{x_k - t_j}{t_{j+k} - t_j}\,\alpha_j^{(k)}
        + \frac{t_{j+k+1} - x_k}{t_{j+k+1} - t_{j+1}}\,\alpha_{j+1}^{(k)},
        \qquad x_k = \tau_{i+k},

    run for ``k = 1 .. degree`` from the seed ``alpha_mu = 1``.  A term whose
    denominator vanishes is dropped, and since the two terms that consume
    :math:`\alpha_j^{(k)}` share the denominator :math:`t_{j+k} - t_j`, one test
    per entry settles both — which is what lets the level be run in place with a
    single carried value.

    Unlike the Cox-de Boor triangle for basis functions, the evaluation point
    changes with the level (:math:`x_k = \tau_{i+k}`), so no knot differences can
    be reused across levels.

    Args:
        degree (int): Polynomial degree of the B-spline.
        old_knots (npt.NDArray[Any]): Original knot vector of shape
            ``(n + degree + 2,)``.
        new_knots (npt.NDArray[Any]): Refined (merged) knot vector of shape
            ``(m + degree + 2,)``.  Must be a superset of ``old_knots``.

    Returns:
        tuple[npt.NDArray[Any], npt.NDArray[np.int64]]: ``(alphas, first_col)``
        where ``alphas`` has shape ``(m + 1, degree + 1)`` and ``first_col`` has
        shape ``(m + 1,)``.  Entry ``alphas[i, l]`` is the coefficient of old
        control point ``first_col[i] + l``.  ``first_col[i]`` is negative when
        the span sits within the first ``degree`` knots, which happens on
        non-clamped (periodic) knot vectors; the corresponding leading entries
        are meaningless and callers must skip columns outside ``[0, n]``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call the Layer 2 helper ``_insert_knots_bspline_1d_impl``
        instead.
    """
    p = degree
    n = old_knots.shape[0] - p - 2  # num old control points - 1
    m = new_knots.shape[0] - p - 2  # num new control points - 1

    alphas = np.zeros((m + 1, p + 1), dtype=old_knots.dtype)
    first_col = np.empty(m + 1, dtype=np.int64)

    for i in range(m + 1):
        # Span of the old knot vector containing new_knots[i].  The right
        # endpoint lands past the last span and is clamped back onto it.
        mu = np.searchsorted(old_knots, new_knots[i], side="right") - 1
        mu = min(mu, n)
        mu = max(mu, 0)
        first_col[i] = mu - p

        band = alphas[i]
        band[p] = 1.0

        for k in range(1, p + 1):
            x = new_knots[i + k]
            saved = 0.0
            # The band grows one entry to the left per level; entries left of
            # column 0 only ever feed columns further left, so they are skipped.
            l_start = max(p - k + 1, p - mu)
            for level_index in range(l_start, p + 1):
                j = mu - p + level_index
                denom = old_knots[j + k] - old_knots[j]
                if denom > 0.0:
                    to_left = (old_knots[j + k] - x) / denom * band[level_index]
                    to_right = (x - old_knots[j]) / denom * band[level_index]
                else:
                    to_left = 0.0
                    to_right = 0.0
                band[level_index - 1] = saved + to_left
                saved = to_right
            band[p] = saved

    return alphas, first_col


@nb_jit(nopython=True, cache=True)
def _compute_oslo_matrix_1d_core(
    degree: int,
    old_knots: npt.NDArray[Any],
    new_knots: npt.NDArray[Any],
) -> npt.NDArray[Any]:
    """Compute the Oslo refinement matrix via the discrete B-spline recurrence.

    Returns the matrix ``alpha`` of shape ``(m+1, n+1)`` such that the new
    control points ``Q = alpha @ P`` reproduce the original geometry exactly.
    Assembled by scattering the banded rows of
    :func:`_compute_oslo_rows_1d_core`, which reproduces the dense recurrence
    entry for entry: outside the band the dense sweep computes exact zeros.

    Args:
        degree (int): Polynomial degree of the B-spline.
        old_knots (npt.NDArray[Any]): Original knot vector of shape
            ``(n + degree + 2,)``.
        new_knots (npt.NDArray[Any]): Refined (merged) knot vector of shape
            ``(m + degree + 2,)``.  Must be a superset of ``old_knots``.

    Returns:
        npt.NDArray[Any]: Refinement matrix of shape ``(m+1, n+1)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call the Layer 2 helper ``_insert_knots_bspline_1d_impl``
        instead.
    """
    n = old_knots.shape[0] - degree - 2  # num old control points - 1
    m = new_knots.shape[0] - degree - 2  # num new control points - 1

    alphas, first_col = _compute_oslo_rows_1d_core(degree, old_knots, new_knots)

    result = np.zeros((m + 1, n + 1), dtype=old_knots.dtype)
    for i in range(m + 1):
        base = first_col[i]
        for level_index in range(degree + 1):
            col = base + level_index
            if 0 <= col <= n:
                result[i, col] = alphas[i, level_index]
    return result


@nb_jit(nopython=True, cache=True)
def _insert_knots_1d_core(
    degree: int,
    old_knots: npt.NDArray[Any],
    ctrl: npt.NDArray[Any],
    new_knots: npt.NDArray[Any],
) -> npt.NDArray[Any]:
    """Apply the Oslo algorithm to compute new control points after knot insertion.

    Args:
        degree (int): Polynomial degree of the B-spline.
        old_knots (npt.NDArray[Any]): Original knot vector of shape
            ``(n + degree + 2,)``.
        ctrl (npt.NDArray[Any]): Control point matrix of shape ``(n+1, rank)``.
        new_knots (npt.NDArray[Any]): Refined (merged) knot vector of shape
            ``(m + degree + 2,)``.  Must be a superset of ``old_knots``.

    Returns:
        npt.NDArray[Any]: New control point matrix of shape ``(m+1, rank)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call the Layer 2 helper ``_insert_knots_bspline_1d_impl``
        instead.

        The refinement matrix is never materialised: each new control point is
        the combination of the ``degree + 1`` old ones its band selects.  The
        summation order therefore differs from the dense matrix product this
        replaced, so results may move in the last bits.
    """
    n = old_knots.shape[0] - degree - 2
    m = new_knots.shape[0] - degree - 2
    rank = ctrl.shape[1]

    alphas, first_col = _compute_oslo_rows_1d_core(degree, old_knots, new_knots)

    result = np.zeros((m + 1, rank), dtype=ctrl.dtype)
    for i in range(m + 1):
        base = first_col[i]
        for level_index in range(degree + 1):
            col = base + level_index
            if 0 <= col <= n:
                weight = alphas[i, level_index]
                for r in range(rank):
                    result[i, r] += weight * ctrl[col, r]
    return result


def _warmup_numba_functions() -> None:
    """Precompile numba functions with float64 signatures for faster first call.

    This function triggers compilation of the numba-decorated functions
    with float64 arrays, ensuring they are cached and ready for use.
    """
    old_knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    new_knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
    ctrl = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]], dtype=np.float64)
    degree = 2

    _compute_oslo_rows_1d_core(degree, old_knots, new_knots)
    _compute_oslo_matrix_1d_core(degree, old_knots, new_knots)
    _insert_knots_1d_core(degree, old_knots, ctrl, new_knots)


__all__ = [
    "_compute_oslo_matrix_1d_core",
    "_compute_oslo_rows_1d_core",
    "_insert_knots_1d_core",
]
