"""Numba kernels for B-spline knot insertion via the Oslo algorithm.

Implements the discrete B-spline recurrence from Cohen, Lyche, Riesenfeld (1980)
to compute a refinement matrix that maps old control points to new ones in a
single pass.

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
def _compute_oslo_matrix_1d_core(
    degree: int,
    old_knots: npt.NDArray[Any],
    new_knots: npt.NDArray[Any],
) -> npt.NDArray[Any]:
    """Compute the Oslo refinement matrix via the discrete B-spline recurrence.

    Returns the matrix ``alpha`` of shape ``(m+1, n+1)`` such that the new
    control points ``Q = alpha @ P`` reproduce the original geometry exactly.

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

    # Two (m+1) x (n+2) buffers: column n+1 is a permanent ghost column of zeros,
    # used so that the recurrence can safely access alpha[i, j+1] when j == n.
    alpha_prev = np.zeros((m + 1, n + 2), dtype=old_knots.dtype)
    alpha_curr = np.zeros((m + 1, n + 2), dtype=old_knots.dtype)

    # Base case (order 1): alpha[i, j] = 1 if old_knots[j] <= new_knots[i] < old_knots[j+1].
    # For the right endpoint (new_knots[i] == old_knots[-1]) searchsorted returns an
    # index beyond n, which we clamp to n.
    for i in range(m + 1):
        si = new_knots[i]
        j = np.searchsorted(old_knots, si, side="right") - 1
        j = min(j, n)
        j = max(j, 0)
        alpha_prev[i, j] = 1.0

    # Recurrence: build up from order 1 to order degree+1.
    for k in range(2, degree + 2):
        # Clear current buffer.
        for i in range(m + 1):
            for jj in range(n + 2):
                alpha_curr[i, jj] = 0.0

        for i in range(m + 1):
            sik = new_knots[i + k - 1]  # s_{i+k-1} in the Oslo notation
            for j in range(n + 1):
                val = 0.0
                # First term: (sik - t_j) / (t_{j+k-1} - t_j)
                denom1 = old_knots[j + k - 1] - old_knots[j]
                if denom1 > 0.0:
                    val += (sik - old_knots[j]) / denom1 * alpha_prev[i, j]
                # Second term: (t_{j+k} - sik) / (t_{j+k} - t_{j+1})
                denom2 = old_knots[j + k] - old_knots[j + 1]
                if denom2 > 0.0:
                    val += (old_knots[j + k] - sik) / denom2 * alpha_prev[i, j + 1]
                alpha_curr[i, j] = val

        # Swap buffers.
        for i in range(m + 1):
            for jj in range(n + 2):
                alpha_prev[i, jj] = alpha_curr[i, jj]

    # Extract the (m+1, n+1) block (drop ghost column).
    result = np.zeros((m + 1, n + 1), dtype=old_knots.dtype)
    for i in range(m + 1):
        for j in range(n + 1):
            result[i, j] = alpha_prev[i, j]
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
    """
    oslo = _compute_oslo_matrix_1d_core(degree, old_knots, new_knots)
    result: npt.NDArray[Any] = oslo @ ctrl
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

    _compute_oslo_matrix_1d_core(degree, old_knots, new_knots)
    _insert_knots_1d_core(degree, old_knots, ctrl, new_knots)


__all__ = [
    "_compute_oslo_matrix_1d_core",
    "_insert_knots_1d_core",
]
