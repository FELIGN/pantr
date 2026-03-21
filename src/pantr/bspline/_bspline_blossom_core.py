"""Numba kernel for B-spline blossom (polar form) evaluation.

Implements the generalized de Boor algorithm to evaluate the symmetric
multilinear polar form of a B-spline at an arbitrary sequence of parameter
values.

Note:
    Inputs are assumed to be correct (no validation performed).
    For general use, call the Layer 2 helper ``_evaluate_blossom_1d`` in
    ``_bspline_blossom`` instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .._numba_compat import nb_jit


@nb_jit(nopython=True, cache=True)
def _evaluate_blossom_1d_core(
    knots: npt.NDArray[Any],
    degree: int,
    control_points: npt.NDArray[Any],
    u_values: npt.NDArray[Any],
) -> npt.NDArray[Any]:
    """Evaluate the blossom of a 1D B-spline at ``degree`` parameter values.

    Computes the symmetric multilinear polar form ``f[u_0, ..., u_{p-1}]``
    using the generalized de Boor algorithm.  The blossom satisfies
    ``f[t, t, ..., t] = f(t)`` (diagonal property) and is symmetric in its
    arguments.

    Args:
        knots (npt.NDArray[Any]): Knot vector of shape ``(n + degree + 2,)``.
        degree (int): Polynomial degree ``p``.
        control_points (npt.NDArray[Any]): Control point matrix of shape
            ``(n + 1, rank)``.
        u_values (npt.NDArray[Any]): Sorted ascending array of ``p`` parameter
            values at which to evaluate the blossom.

    Returns:
        npt.NDArray[Any]: Blossom value of shape ``(rank,)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        ``u_values`` must be sorted in ascending order and have length ``p``.
        For general use, call ``_evaluate_blossom_1d`` instead.
    """
    rank = control_points.shape[1]

    if degree == 0:
        # Degree-0 blossom: locate the single control point whose support
        # contains u (there are no u_values since p=0).
        # For a degree-0 B-spline, B_i has support [t_i, t_{i+1}).
        # By convention we pick the last active knot span.
        n = control_points.shape[0] - 1
        result = np.empty(rank, dtype=control_points.dtype)
        for r in range(rank):
            result[r] = control_points[n, r]
        return result

    # Find knot span for the largest u value (u_values[-1]).
    u_last = u_values[degree - 1]
    n = control_points.shape[0] - 1
    k = n  # default: rightmost span
    for idx in range(n + degree):
        if knots[idx] <= u_last < knots[idx + 1]:
            k = idx
            break

    # Local copy of control points d[j] = P[k-p+j] for j in 0..p.
    d = np.empty((degree + 1, rank), dtype=control_points.dtype)
    for j in range(degree + 1):
        idx = k - degree + j
        for r in range(rank):
            d[j, r] = control_points[idx, r]

    # Generalized de Boor recurrence: level l uses u_values[l-1].
    for level in range(1, degree + 1):
        u = u_values[level - 1]
        for j in range(degree, level - 1, -1):
            # d[j] corresponds to global index k - degree + j
            global_j = k - degree + j
            denom = knots[global_j + degree + 1 - level] - knots[global_j]
            alpha = (u - knots[global_j]) / denom if denom > 0.0 else 0.0
            for r in range(rank):
                d[j, r] = (1.0 - alpha) * d[j - 1, r] + alpha * d[j, r]

    result = np.empty(rank, dtype=control_points.dtype)
    for r in range(rank):
        result[r] = d[degree, r]
    return result


def _warmup_numba_functions() -> None:
    """Precompile Numba functions with float64 signatures for faster first call.

    Triggers compilation of the Numba-decorated functions with float64 arrays
    so they are cached and ready for use on first call.
    """
    knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    ctrl = np.array([[0.0], [0.5], [1.0]], dtype=np.float64)
    u_vals = np.array([0.3, 0.6], dtype=np.float64)
    _evaluate_blossom_1d_core(knots, 2, ctrl, u_vals)


__all__ = ["_evaluate_blossom_1d_core"]
