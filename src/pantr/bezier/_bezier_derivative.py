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

from .._array_utils import _flatten_along_axis, _unflatten_along_axis

if TYPE_CHECKING:
    from . import Bezier


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


def _derivative_keep_degree_ctrl_1d(
    degree: int,
    ctrl: npt.NDArray[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    """Compute derivative control points with degree preservation for a 1D non-rational Bézier.

    Fuses the derivative formula ``Q[i] = p * (P[i+1] - P[i])`` with a
    single degree elevation, so the result has the same degree ``p`` as the
    input. The combined formula is:

    ``R[j] = (p - j) * (P[j+1] - P[j]) + j * (P[j] - P[j-1])``

    for ``j = 0, ..., p``.

    Args:
        degree (int): Polynomial degree of the original Bézier (``p >= 1``).
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points of shape
            ``(p+1, ...)``.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Derivative control points of
        shape ``(p+1, ...)``, representing a degree-``p`` Bézier.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_derivative_bezier` instead.
    """
    p = degree
    n = p + 1  # number of output CPs = same as input
    result = np.empty_like(ctrl[:n])
    # j = 0: R[0] = p * (P[1] - P[0])
    result[0] = p * (ctrl[1] - ctrl[0])
    # j = 1, ..., p-1
    for j in range(1, p):
        result[j] = (p - j) * (ctrl[j + 1] - ctrl[j]) + j * (ctrl[j] - ctrl[j - 1])
    # j = p: R[p] = p * (P[p] - P[p-1])
    result[p] = p * (ctrl[p] - ctrl[p - 1])
    return result


def _derivative_ctrl_nd(
    coeffs: npt.NDArray[np.float32 | np.float64],
    dim: int,
) -> npt.NDArray[np.float32 | np.float64]:
    """Compute the partial derivative of an N-D coefficient array along one axis.

    Applies :func:`_derivative_ctrl_1d` along dimension ``dim`` using the
    shared flatten/unflatten helpers, reducing the size in that dimension
    by 1.

    Args:
        coeffs (npt.NDArray[np.float32 | np.float64]): Coefficient array with
            shape ``(n_0, ..., n_{N-1})``.  ``n_dim >= 2``.
        dim (int): Direction to differentiate.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Derivative coefficients with shape
        ``(n_0, ..., n_dim - 1, ..., n_{N-1})``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_derivative_bezier` instead.
    """
    degree = coeffs.shape[dim] - 1
    pts_2d, trailing_shape = _flatten_along_axis(coeffs, dim)
    result = _derivative_ctrl_1d(degree, pts_2d)
    return _unflatten_along_axis(result, trailing_shape, dim)


def _derivative_nonrational(bezier: Bezier, direction: int) -> Bezier:
    """Compute the partial derivative of a non-rational Bézier.

    Applies the 1D derivative formula along the given parametric direction
    using the shared flatten/unflatten helpers.

    Args:
        bezier (~pantr.bezier.Bezier): A non-rational Bézier.
        direction (int): Parametric direction for differentiation.

    Returns:
        ~pantr.bezier.Bezier: Derivative Bézier with degree ``p_d - 1`` in
        direction ``d`` and unchanged degrees in other directions.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_derivative_bezier` instead.
    """
    from . import Bezier as BezierCls  # noqa: PLC0415

    new_ctrl = _derivative_ctrl_nd(bezier.control_points, direction)
    return BezierCls(new_ctrl, is_rational=False)


def _derivative_keep_degree_nonrational(bezier: Bezier, direction: int) -> Bezier:
    """Compute the partial derivative of a non-rational Bézier, preserving degree.

    Fuses the derivative and degree elevation into a single operation along
    the given direction using the shared flatten/unflatten helpers.

    Args:
        bezier (~pantr.bezier.Bezier): A non-rational Bézier.
        direction (int): Parametric direction for differentiation.

    Returns:
        ~pantr.bezier.Bezier: Derivative Bézier with the same degree as the
        input in direction ``d`` and unchanged degrees in other directions.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_derivative_bezier` instead.
    """
    from . import Bezier as BezierCls  # noqa: PLC0415

    p = bezier.degree[direction]
    ctrl = bezier.control_points

    pts_2d, trailing_shape = _flatten_along_axis(ctrl, direction)
    new_pts_2d = _derivative_keep_degree_ctrl_1d(p, pts_2d)
    new_ctrl = _unflatten_along_axis(new_pts_2d, trailing_shape, direction)

    return BezierCls(new_ctrl, is_rational=False)


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
    from . import Bezier as BezierCls  # noqa: PLC0415

    ctrl = np.repeat(w.control_points, target_rank, axis=-1)
    return BezierCls(ctrl, is_rational=False)


def _derivative_rational(bezier: Bezier, direction: int) -> Bezier:
    """Compute the partial derivative of a rational Bézier.

    Applies the quotient rule: for ``f = A/w``,
    ``f' = (A'w - Aw') / w^2``.

    Uses degree-preserving derivatives for ``A'`` and ``w'`` so that the
    products ``A'w`` and ``Aw'`` are degree ``2p`` (matching ``w^2``),
    eliminating the need for a separate numerator degree elevation.

    Args:
        bezier (~pantr.bezier.Bezier): A rational Bézier.
        direction (int): Parametric direction for differentiation.

    Returns:
        ~pantr.bezier.Bezier: Rational derivative Bézier.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_derivative_bezier` instead.
    """
    from . import Bezier as BezierCls  # noqa: PLC0415
    from ._bezier_product import _multiply_bezier  # noqa: PLC0415

    ctrl = bezier.control_points
    vec_rank = ctrl.shape[-1] - 1  # number of non-weight components

    # Decompose into numerator A (weighted coords) and denominator w (weights).
    a_bezier = BezierCls(ctrl[..., :-1], is_rational=False)
    w_bezier = BezierCls(ctrl[..., -1:], is_rational=False)

    # Degree-preserving derivatives: A' and w' remain degree p.
    a_prime = _derivative_keep_degree_nonrational(a_bezier, direction)
    w_prime = _derivative_keep_degree_nonrational(w_bezier, direction)

    # Tile scalar Béziers to match vector rank for multiply().
    w_tiled = _tile_scalar_bezier(w_bezier, vec_rank)
    w_prime_tiled = _tile_scalar_bezier(w_prime, vec_rank)

    # Products are degree p + p = 2p (same as w^2).
    num1 = _multiply_bezier(a_prime, w_tiled)
    num2 = _multiply_bezier(a_bezier, w_prime_tiled)

    # Subtract CPs (same degree guaranteed).
    numerator_ctrl = num1.control_points - num2.control_points

    # Compute denominator: w^2 (degree 2p).
    denom = _multiply_bezier(w_bezier, w_bezier)

    # Assemble rational Bézier (numerator and denom are both degree 2p).
    result_ctrl = np.concatenate([numerator_ctrl, denom.control_points], axis=-1)
    return BezierCls(result_ctrl, is_rational=True)


def _derivative_bezier(bezier: Bezier, direction: int, *, keep_degree: bool = False) -> Bezier:
    """Compute the first partial derivative of a Bézier.

    This is the main entry point for derivative computation. Dispatches
    to the appropriate implementation based on the Bézier's properties.

    Args:
        bezier (~pantr.bezier.Bezier): The Bézier to differentiate.
        direction (int): Parametric direction for differentiation. Must be
            in ``[0, dim)``.
        keep_degree (bool): If ``True``, the result has the same degree as the
            input by fusing derivative and degree elevation. Defaults to
            ``False``.

    Returns:
        ~pantr.bezier.Bezier: A new Bézier representing the derivative.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :meth:`~pantr.bezier.Bezier.derivative` instead.
    """
    if bezier.is_rational:
        return _derivative_rational(bezier, direction)
    if keep_degree:
        return _derivative_keep_degree_nonrational(bezier, direction)
    return _derivative_nonrational(bezier, direction)
