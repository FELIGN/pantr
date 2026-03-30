"""Bézier degree elevation, reduction, and minimization.

This module provides :func:`_degree_elevate_bezier`, which raises the polynomial
degree of a Bézier in one or more parametric directions while preserving the
same geometric mapping, :func:`_degree_reduce_bezier`, which computes a
least-squares degree-reduced approximation, and :func:`_minimize_degree_bezier`,
which automatically finds the lowest degree that preserves accuracy.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ._bezier_core import _degree_elevate_bezier_1d_core, _degree_reduce_bezier_1d_core

if TYPE_CHECKING:
    from . import Bezier

_AUTO_REDUCTION_TOL_FACTOR: float = 1.0e3
"""Factor multiplied by machine epsilon for automatic degree reduction."""


def _degree_elevate_bezier(
    bezier: Bezier,
    increments: tuple[int, ...],
) -> Bezier:
    """Degree-elevate a Bézier in one or more parametric directions.

    For each direction with a positive increment, applies the Bézier degree
    elevation kernel using the moveaxis/reshape pattern.

    Args:
        bezier (~pantr.bezier.Bezier): The Bézier to elevate.
        increments (tuple[int, ...]): Degree increment per direction. All
            values must be non-negative; at least one must be positive.

    Returns:
        ~pantr.bezier.Bezier: New Bézier with elevated degrees and updated
        control points.

    Note:
        Inputs are assumed to be validated by the caller (Layer 1).
    """
    from . import Bezier as BezierCls  # noqa: PLC0415

    ctrl: npt.NDArray[np.float32 | np.float64] = bezier.control_points
    degrees = bezier.degree

    for d in range(bezier.dim):
        inc = increments[d]
        if inc == 0:
            continue

        p = degrees[d]

        # Move target direction to axis 0, flatten the rest.
        moved = np.moveaxis(ctrl, d, 0)
        orig_shape = moved.shape
        pts_2d = moved.reshape(orig_shape[0], -1)

        if not pts_2d.flags.c_contiguous:
            pts_2d = np.ascontiguousarray(pts_2d)

        new_pts_2d = _degree_elevate_bezier_1d_core(p, pts_2d, inc)

        # Restore shape and move axis back.
        new_shape = (new_pts_2d.shape[0], *orig_shape[1:])
        new_moved = new_pts_2d.reshape(new_shape)
        ctrl = np.moveaxis(new_moved, 0, d)

        # Update degrees for subsequent iterations.
        degrees = (*degrees[:d], p + inc, *degrees[d + 1 :])

    return BezierCls(ctrl, is_rational=bezier.is_rational)


def _degree_reduce_bezier(
    bezier: Bezier,
    decrements: tuple[int, ...],
) -> Bezier:
    """Degree-reduce a Bézier in one or more parametric directions.

    For each direction with a positive decrement, applies the Bézier degree
    reduction kernel using the moveaxis/reshape pattern.  The reduction is a
    least-squares approximation (not exact in general).

    Args:
        bezier (~pantr.bezier.Bezier): The Bézier to reduce.
        decrements (tuple[int, ...]): Degree decrement per direction. All
            values must be non-negative; at least one must be positive.  No
            decrement may exceed the current degree in that direction.

    Returns:
        ~pantr.bezier.Bezier: New Bézier with reduced degrees and updated
        control points.

    Note:
        Inputs are assumed to be validated by the caller (Layer 1).
    """
    from . import Bezier as BezierCls  # noqa: PLC0415

    ctrl: npt.NDArray[np.float32 | np.float64] = bezier.control_points
    degrees = bezier.degree

    for d in range(bezier.dim):
        dec = decrements[d]
        if dec == 0:
            continue

        p = degrees[d]

        # Move target direction to axis 0, flatten the rest.
        moved = np.moveaxis(ctrl, d, 0)
        orig_shape = moved.shape
        pts_2d = moved.reshape(orig_shape[0], -1)

        if not pts_2d.flags.c_contiguous:
            pts_2d = np.ascontiguousarray(pts_2d)

        new_pts_2d = _degree_reduce_bezier_1d_core(p, pts_2d, dec)

        # Restore shape and move axis back.
        new_shape = (new_pts_2d.shape[0], *orig_shape[1:])
        new_moved = new_pts_2d.reshape(new_shape)
        ctrl = np.moveaxis(new_moved, 0, d)

        # Update degrees for subsequent iterations.
        degrees = (*degrees[:d], p - dec, *degrees[d + 1 :])

    return BezierCls(ctrl, is_rational=bezier.is_rational)


def _squared_l2_norm(
    coeffs: npt.NDArray[np.floating[Any]],
) -> float:
    r"""Compute the squared L2 norm of a Bernstein polynomial.

    Uses the Bernstein inner product formula:

    .. math::

        \int_0^1 B_{i,n}(x) B_{j,n}(x)\,dx
        = \frac{1}{2n+1} \frac{\binom{n}{i} \binom{n}{j}}{\binom{2n}{i+j}}

    extended to tensor products for multivariate polynomials.

    Args:
        coeffs (npt.NDArray[np.floating[Any]]): Bernstein coefficients
            (any shape).

    Returns:
        float: The squared L2 norm ``||p||_2^2``.

    Note:
        Implementation follows algoim (Saye, J. Comput. Phys. 448, 2022).
    """
    ndim = coeffs.ndim
    shape = coeffs.shape

    # Precompute binomial rows and Gram factors per dimension
    binom_rows = []
    binom_double = []
    for d in range(ndim):
        n = shape[d] - 1
        brow = np.array([math.comb(n, i) for i in range(n + 1)], dtype=np.float64)
        bdbl = np.array([math.comb(2 * n, k) for k in range(2 * n + 1)], dtype=np.float64)
        binom_rows.append(brow)
        binom_double.append(bdbl)

    # Flatten and compute all pairwise products with Gram weights
    flat = coeffs.ravel().astype(np.float64)
    indices = np.array(np.unravel_index(np.arange(flat.size), shape)).T  # (size, ndim)

    # For each pair (i, j): gram_weight = product over dims of binom(n_d, i_d)*binom(n_d, j_d) /
    #                                      binom(2*n_d, i_d + j_d)
    # This is O(size^2) but polynomials are small in practice.
    delta = 0.0
    for idx_i in range(flat.size):
        for idx_j in range(flat.size):
            g = 1.0
            for d in range(ndim):
                ii = indices[idx_i, d]
                jj = indices[idx_j, d]
                g *= (binom_rows[d][ii] * binom_rows[d][jj]) / binom_double[d][ii + jj]
            delta += flat[idx_i] * flat[idx_j] * g

    for d in range(ndim):
        delta /= 2.0 * shape[d] - 1.0

    return abs(delta)


def _minimize_degree_bezier(
    bezier: Bezier,
    tol: float | None = None,
) -> Bezier:
    """Automatically reduce the degree of a Bézier while maintaining accuracy.

    Iterates over each parametric direction and repeatedly tries to reduce
    the degree by 1.  A reduction is accepted when the round-trip
    (reduce then elevate) relative L2 error stays below ``tol``.  For
    vector-valued Bézier, all rank components are checked simultaneously.

    Args:
        bezier (~pantr.bezier.Bezier): The Bézier to simplify.
        tol (float | None): Relative tolerance for accepting a degree
            reduction.  If *None*, uses
            :data:`_AUTO_REDUCTION_TOL_FACTOR` ``* eps``.

    Returns:
        ~pantr.bezier.Bezier: A new Bézier with the lowest degree that
        preserves accuracy within ``tol``.  If no reduction is possible,
        returns a copy of the input.

    Note:
        Implementation follows algoim (Saye, J. Comput. Phys. 448, 2022).
    """
    from . import Bezier as BezierCls  # noqa: PLC0415

    ctrl: npt.NDArray[np.floating[Any]] = bezier.control_points  # (*orders, rank)
    n_param = bezier.dim
    eps = float(np.finfo(ctrl.dtype).eps)
    if tol is None:
        tol = _AUTO_REDUCTION_TOL_FACTOR * eps

    if tol <= 0.0:
        return BezierCls(ctrl.copy(), is_rational=bezier.is_rational)

    result = ctrl
    changed = False

    for dim in range(n_param):
        while result.shape[dim] >= 2:  # noqa: PLR2004
            degree = result.shape[dim] - 1

            # Reduce all rank components together along this dimension
            moved = np.moveaxis(result, dim, 0)
            shape_after = moved.shape
            flat = moved.reshape(shape_after[0], -1)
            if not flat.flags.c_contiguous:
                flat = np.ascontiguousarray(flat)
            reduced_flat = _degree_reduce_bezier_1d_core(degree, flat, 1)
            reduced = np.moveaxis(
                reduced_flat.reshape(reduced_flat.shape[0], *shape_after[1:]), 0, dim
            )

            # Elevate back to original shape for error check
            moved_r = np.moveaxis(reduced, dim, 0)
            shape_r = moved_r.shape
            flat_r = moved_r.reshape(shape_r[0], -1)
            if not flat_r.flags.c_contiguous:
                flat_r = np.ascontiguousarray(flat_r)
            elevated_flat = _degree_elevate_bezier_1d_core(degree - 1, flat_r, 1)
            elevated = np.moveaxis(
                elevated_flat.reshape(elevated_flat.shape[0], *shape_r[1:]), 0, dim
            )

            # Check L2 error summed across all rank components
            diff = elevated - result
            diff_norm = sum(_squared_l2_norm(diff[..., r]) for r in range(result.shape[-1]))
            orig_norm = sum(_squared_l2_norm(result[..., r]) for r in range(result.shape[-1]))

            if orig_norm > 0.0:
                rel_error = math.sqrt(abs(diff_norm)) / math.sqrt(abs(orig_norm))
            else:
                rel_error = math.sqrt(abs(diff_norm))

            if rel_error < tol:
                result = reduced
                changed = True
            else:
                break

    if not changed:
        return BezierCls(ctrl.copy(), is_rational=bezier.is_rational)
    return BezierCls(result, is_rational=bezier.is_rational)
