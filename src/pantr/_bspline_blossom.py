"""Layer 2 helper for B-spline blossom (polar form) evaluation.

Provides :func:`_evaluate_blossom_1d`, a validated entry point that delegates
to the Numba kernel in :mod:`pantr._bspline_blossom_core`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._bspline_blossom_core import _evaluate_blossom_1d_core


def _evaluate_blossom_1d(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    control_points: npt.NDArray[np.float32 | np.float64],
    u_values: npt.NDArray[np.float32 | np.float64],
    tol: float,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate the blossom of a 1D B-spline at ``degree`` parameter values.

    Computes the symmetric multilinear polar form ``f[u_0, ..., u_{p-1}]``
    using the generalized de Boor algorithm.  The diagonal property guarantees
    ``f[t, ..., t] == f(t)``, and the blossom is symmetric in its arguments.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): Knot vector of shape
            ``(n + degree + 2,)``.
        degree (int): Polynomial degree ``p``.
        control_points (npt.NDArray[np.float32 | np.float64]): Control point
            matrix of shape ``(n + 1, rank)``.
        u_values (npt.NDArray[np.float32 | np.float64]): Array of exactly
            ``p`` parameter values.  Need not be sorted; values may coincide.
        tol (float): Tolerance for domain-membership checks.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Blossom value of shape
        ``(rank,)``.

    Raises:
        ValueError: If ``len(u_values) != degree``.
        ValueError: If any entry of ``u_values`` lies outside the domain
            ``[knots[degree], knots[-degree-1]]`` (beyond ``tol``).
        ValueError: If ``knots`` and ``u_values`` have incompatible dtypes.
    """
    if u_values.shape[0] != degree:
        raise ValueError(f"u_values must have length degree={degree}, got {u_values.shape[0]}.")

    if degree == 0:
        # Degree-0: no u_values to check; return the single control point.
        return _evaluate_blossom_1d_core(
            knots, degree, control_points, np.empty(0, dtype=knots.dtype)
        )

    a = float(knots[degree])
    b = float(knots[knots.shape[0] - degree - 1])
    for u in u_values:
        if float(u) < a - tol or float(u) > b + tol:
            raise ValueError(
                f"u_values entry {float(u)!r} is outside domain [{a}, {b}] (tol={tol})."
            )

    # Sort ascending before passing to the kernel (symmetry — order does not
    # affect the mathematical result, but the kernel relies on ascending order
    # to find the correct knot span).
    u_sorted: npt.NDArray[np.float32 | np.float64] = np.sort(u_values)
    return _evaluate_blossom_1d_core(knots, degree, control_points, u_sorted)


__all__ = ["_evaluate_blossom_1d"]
