"""Bézier derivative (hodograph) computation.

This module provides :func:`_derivative_bezier`, which computes the exact
first derivative of a Bézier in a given parametric direction, returning a
new :class:`~pantr.bezier.Bezier` whose value at every point equals the
derivative of the original.

Works for non-rational and rational Bézier of any parametric dimension.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .bezier import Bezier


def _derivative_ctrl_1d(
    degree: int,
    ctrl: npt.NDArray[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    """Compute derivative control points for a 1D non-rational Bézier.

    Applies the standard Bézier derivative formula:
    ``Q[i] = p * (P[i+1] - P[i])``

    Args:
        degree (int): Polynomial degree of the original Bézier (``p >= 1``).
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points of shape
            ``(p+1, ...)``.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Derivative control points of
        shape ``(p, ...)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_derivative_bezier` instead.
    """
    return np.asarray(degree * (ctrl[1:] - ctrl[:-1]), dtype=ctrl.dtype)


def _derivative_nonrational_1d(bezier: Bezier) -> Bezier:
    """Compute the derivative of a non-rational 1D Bézier.

    Args:
        bezier (~pantr.bezier.Bezier): A 1D non-rational Bézier with degree >= 1.

    Returns:
        ~pantr.bezier.Bezier: Derivative Bézier of degree ``p - 1``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_derivative_bezier` instead.
    """
    from .bezier import Bezier as BezierCls  # noqa: PLC0415

    p = bezier.degree[0]
    ctrl = bezier.control_points
    new_ctrl = _derivative_ctrl_1d(p, ctrl)
    return BezierCls(new_ctrl, is_rational=False)


def _derivative_nonrational_nd(bezier: Bezier, direction: int) -> Bezier:
    """Compute the partial derivative of a non-rational nD Bézier.

    Applies the 1D derivative formula along the given parametric direction
    using the moveaxis/reshape/restore pattern.

    Args:
        bezier (~pantr.bezier.Bezier): A non-rational nD Bézier.
        direction (int): Parametric direction for differentiation.

    Returns:
        ~pantr.bezier.Bezier: Derivative Bézier with degree ``p_d - 1`` in
        direction ``d`` and unchanged degrees in other directions.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_derivative_bezier` instead.
    """
    from .bezier import Bezier as BezierCls  # noqa: PLC0415

    p = bezier.degree[direction]
    ctrl = bezier.control_points

    # Move target direction to axis 0, flatten the rest.
    moved = np.moveaxis(ctrl, direction, 0)
    orig_shape = moved.shape
    pts_2d = moved.reshape(orig_shape[0], -1)

    if not pts_2d.flags.c_contiguous:
        pts_2d = np.ascontiguousarray(pts_2d)

    new_pts_2d = _derivative_ctrl_1d(p, pts_2d)

    # Restore shape and move axis back.
    new_shape = (new_pts_2d.shape[0], *orig_shape[1:])
    new_moved = new_pts_2d.reshape(new_shape)
    new_ctrl = np.moveaxis(new_moved, 0, direction)

    return BezierCls(new_ctrl, is_rational=False)


def _derivative_nonrational(bezier: Bezier, direction: int) -> Bezier:
    """Compute the partial derivative of a non-rational Bézier.

    Dispatches to the 1D or nD implementation.

    Args:
        bezier (~pantr.bezier.Bezier): A non-rational Bézier.
        direction (int): Parametric direction for differentiation.

    Returns:
        ~pantr.bezier.Bezier: Derivative Bézier.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_derivative_bezier` instead.
    """
    if bezier.dim == 1:
        return _derivative_nonrational_1d(bezier)
    return _derivative_nonrational_nd(bezier, direction)


def _tile_scalar_bezier(w: Bezier, target_rank: int) -> Bezier:
    """Tile a rank-1 Bézier to match a target rank.

    Repeats the single control point component along the last axis so that
    the resulting Bézier has ``target_rank`` components, each identical to
    the original scalar function.

    Args:
        w (~pantr.bezier.Bezier): A rank-1 (scalar) non-rational Bézier.
        target_rank (int): Desired number of output components.

    Returns:
        ~pantr.bezier.Bezier: Bézier with ``target_rank`` identical components.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    from .bezier import Bezier as BezierCls  # noqa: PLC0415

    ctrl = np.repeat(w.control_points, target_rank, axis=-1)
    return BezierCls(ctrl, is_rational=False)


def _derivative_rational(bezier: Bezier, direction: int) -> Bezier:
    """Compute the partial derivative of a rational Bézier.

    Applies the quotient rule: for ``f = A/w``,
    ``f' = (A'w - Aw') / w^2``.

    The numerator and denominator are computed via Bézier products and
    combined into a rational Bézier of degree ``2p`` in the given direction.

    Args:
        bezier (~pantr.bezier.Bezier): A rational Bézier.
        direction (int): Parametric direction for differentiation.

    Returns:
        ~pantr.bezier.Bezier: Rational derivative Bézier.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_derivative_bezier` instead.
    """
    from ._bezier_degree import _degree_elevate_bezier  # noqa: PLC0415
    from ._bezier_product import _multiply_bezier  # noqa: PLC0415
    from .bezier import Bezier as BezierCls  # noqa: PLC0415

    ctrl = bezier.control_points
    vec_rank = ctrl.shape[-1] - 1  # number of non-weight components

    # Decompose into numerator A (weighted coords) and denominator w (weights).
    a_bezier = BezierCls(ctrl[..., :-1], is_rational=False)
    w_bezier = BezierCls(ctrl[..., -1:], is_rational=False)

    # Compute non-rational derivatives.
    a_prime = _derivative_nonrational(a_bezier, direction)
    w_prime = _derivative_nonrational(w_bezier, direction)

    # Tile scalar Béziers to match vector rank for multiply().
    w_tiled = _tile_scalar_bezier(w_bezier, vec_rank)
    w_prime_tiled = _tile_scalar_bezier(w_prime, vec_rank)

    # Compute products: A'*w and A*w' (degree 2p-1 each).
    num1 = _multiply_bezier(a_prime, w_tiled)
    num2 = _multiply_bezier(a_bezier, w_prime_tiled)

    # Subtract CPs (same degree guaranteed).
    numerator_ctrl = num1.control_points - num2.control_points
    numerator = BezierCls(numerator_ctrl, is_rational=False)

    # Compute denominator: w^2 (degree 2p).
    denom = _multiply_bezier(w_bezier, w_bezier)

    # Elevate numerator degree by 1 in the target direction.
    increments = [0] * bezier.dim
    increments[direction] = 1
    numerator = _degree_elevate_bezier(numerator, tuple(increments))

    # Assemble rational Bézier.
    result_ctrl = np.concatenate([numerator.control_points, denom.control_points], axis=-1)
    return BezierCls(result_ctrl, is_rational=True)


def _derivative_bezier(bezier: Bezier, direction: int) -> Bezier:
    """Compute the first partial derivative of a Bézier.

    This is the main entry point for derivative computation. Dispatches
    to the appropriate implementation based on the Bézier's properties.

    Args:
        bezier (~pantr.bezier.Bezier): The Bézier to differentiate.
        direction (int): Parametric direction for differentiation. Must be
            in ``[0, dim)``.

    Returns:
        ~pantr.bezier.Bezier: A new Bézier representing the derivative.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :meth:`~pantr.bezier.Bezier.derivative` instead.
    """
    if bezier.is_rational:
        return _derivative_rational(bezier, direction)
    return _derivative_nonrational(bezier, direction)
