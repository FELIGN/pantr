"""Bézier degree elevation, reduction, and automatic degree reduction.

This module provides :func:`_degree_elevate_bezier`, which raises the polynomial
degree of a Bézier in one or more parametric directions while preserving the
same geometric mapping, :func:`_degree_reduce_bezier`, which computes a
least-squares degree-reduced approximation, and :func:`_auto_reduce_degree_bezier`,
which automatically reduces degree in each direction while the approximation
error remains below a relative tolerance.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ._bezier_core import _degree_elevate_bezier_1d_core, _degree_reduce_bezier_1d_core

if TYPE_CHECKING:
    from . import Bezier


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


@lru_cache(maxsize=64)
def _bernstein_gram_1d(n: int) -> npt.NDArray[np.float64]:
    r"""Compute the Gram matrix of the Bernstein basis of degree ``n`` on [0, 1].

    The entry ``G[i, j]`` equals the L2 inner product of Bernstein
    polynomials ``B_i^n`` and ``B_j^n``:

    .. math::

        G_{ij} = \frac{\binom{n}{i}\,\binom{n}{j}}
                       {\binom{2n}{i+j}\,(2n+1)}

    The returned array is read-only to protect the cache.

    Args:
        n (int): Polynomial degree (``n >= 0``).

    Returns:
        npt.NDArray[np.float64]: Symmetric positive-definite matrix of shape
        ``(n + 1, n + 1)``.
    """
    size = n + 1
    gram = np.empty((size, size), dtype=np.float64)
    denom_factor = 2 * n + 1
    for i in range(size):
        binom_ni = math.comb(n, i)
        for j in range(i + 1):
            val = binom_ni * math.comb(n, j) / (math.comb(2 * n, i + j) * denom_factor)
            gram[i, j] = val
            gram[j, i] = val
    gram.flags.writeable = False
    return gram


def _squared_l2_norm_bernstein(
    ctrl: npt.NDArray[np.float32 | np.float64],
    degrees: tuple[int, ...],
) -> float:
    r"""Compute the squared L2 norm of a Bernstein polynomial.

    For a polynomial with control-point tensor ``ctrl`` of shape
    ``(*degrees_plus_1, rank)``, computes

    .. math::

        \|f\|_{L^2}^2
          = \sum_r \mathbf{c}_r^{\!\top}
            (G_0 \otimes \cdots \otimes G_{N-1})\,\mathbf{c}_r

    using successive mode-*d* contractions with the 1-D Gram matrices.
    The result may be very slightly negative due to round-off; callers
    should take ``abs()`` before square-rooting.

    Args:
        ctrl (npt.NDArray[np.float32 | np.float64]): Control-point array of
            shape ``(*degrees_plus_1, rank)``.
        degrees (tuple[int, ...]): Polynomial degree per parametric direction.

    Returns:
        float: The squared L2 norm (scalar).
    """
    ndim = len(degrees)
    ctrl_f64 = np.asarray(ctrl, dtype=np.float64)
    temp = ctrl_f64
    for d in range(ndim):
        gram = _bernstein_gram_1d(degrees[d])
        temp = np.moveaxis(np.tensordot(gram, temp, axes=([1], [d])), 0, d)
    return float(np.sum(ctrl_f64 * temp))


def _reduce_ctrl_along(
    ctrl: npt.NDArray[np.float32 | np.float64],
    degree: int,
    direction: int,
) -> npt.NDArray[np.float32 | np.float64]:
    """Reduce degree by 1 along *direction*, returning the new control-point array.

    Applies the moveaxis/reshape -> kernel -> reshape/moveaxis pattern for a
    single direction with decrement 1.

    Args:
        ctrl (npt.NDArray[np.float32 | np.float64]): Control-point array.
        degree (int): Current degree in *direction*.
        direction (int): Parametric direction to reduce.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Reduced control points (one
        fewer entry along *direction*).
    """
    moved = np.moveaxis(ctrl, direction, 0)
    orig_shape = moved.shape
    pts_2d = moved.reshape(orig_shape[0], -1)
    if not pts_2d.flags.c_contiguous:
        pts_2d = np.ascontiguousarray(pts_2d)
    new_2d = _degree_reduce_bezier_1d_core(degree, pts_2d, 1)
    return np.moveaxis(new_2d.reshape(new_2d.shape[0], *orig_shape[1:]), 0, direction)


def _elevate_ctrl_along(
    ctrl: npt.NDArray[np.float32 | np.float64],
    degree: int,
    direction: int,
) -> npt.NDArray[np.float32 | np.float64]:
    """Elevate degree by 1 along *direction*, returning the new control-point array.

    Args:
        ctrl (npt.NDArray[np.float32 | np.float64]): Control-point array.
        degree (int): Current degree in *direction*.
        direction (int): Parametric direction to elevate.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Elevated control points (one
        more entry along *direction*).
    """
    moved = np.moveaxis(ctrl, direction, 0)
    orig_shape = moved.shape
    pts_2d = moved.reshape(orig_shape[0], -1)
    if not pts_2d.flags.c_contiguous:
        pts_2d = np.ascontiguousarray(pts_2d)
    new_2d = _degree_elevate_bezier_1d_core(degree, pts_2d, 1)
    return np.moveaxis(new_2d.reshape(new_2d.shape[0], *orig_shape[1:]), 0, direction)


def _auto_reduce_degree_bezier(
    bezier: Bezier,
    tol: float,
) -> Bezier:
    """Automatically reduce the degree of a non-rational Bezier.

    For each parametric direction, iteratively attempts to reduce the degree
    by one.  A reduction is accepted when the L2 norm of the round-trip error
    (reduce then re-elevate) is below ``tol`` times the L2 norm of the
    current polynomial.  This follows the *autoReduction* strategy described
    in R. I. Saye, *J. Comput. Phys.* 448, 110720 (2022).

    Args:
        bezier (~pantr.bezier.Bezier): A non-rational Bezier.
        tol (float): Relative tolerance (must be positive).

    Returns:
        ~pantr.bezier.Bezier: A new Bezier with (potentially) lower degrees,
        or the original object if no reduction was possible.

    Note:
        Inputs are assumed to be validated by the caller (Layer 1).
    """
    from . import Bezier as BezierCls  # noqa: PLC0415

    ctrl: npt.NDArray[np.float32 | np.float64] = bezier.control_points
    degrees = list(bezier.degree)
    ndim = bezier.dim
    tol_sq = tol * tol
    changed = False

    d = 0
    while d < ndim:
        if degrees[d] < 1:
            d += 1
            continue

        p = degrees[d]

        # Reduce by 1 in direction d, then elevate back.
        reduced_ctrl = _reduce_ctrl_along(ctrl, p, d)
        re_elevated = _elevate_ctrl_along(reduced_ctrl, p - 1, d)

        # Compute error in-place to avoid an extra allocation.
        error = re_elevated - ctrl

        deg_tuple = tuple(degrees)
        error_norm_sq = _squared_l2_norm_bernstein(error, deg_tuple)
        orig_norm_sq = _squared_l2_norm_bernstein(ctrl, deg_tuple)

        if abs(error_norm_sq) < tol_sq * abs(orig_norm_sq):
            ctrl = reduced_ctrl
            degrees[d] = p - 1
            changed = True
            # Stay on the same direction -- try reducing again.
        else:
            d += 1

    if changed:
        return BezierCls(ctrl, is_rational=False)
    return bezier
