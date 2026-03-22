"""B-spline derivative (hodograph) computation.

This module provides :func:`_derivative_bspline`, which computes the exact
first derivative of a B-spline in a given parametric direction, returning a
new :class:`~pantr.bspline.Bspline` whose value at every point equals the
derivative of the original.

Works for non-rational and rational (NURBS) B-splines of any parametric
dimension, and correctly preserves per-direction boundary structure
(open / periodic / non-open).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ._bspline_knots import _get_Bspline_num_basis_1D_impl
from ._bspline_space_1d import BsplineSpace1D
from ._bspline_space_nd import BsplineSpace

if TYPE_CHECKING:
    from . import Bspline


def _derivative_ctrl_1d(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    ctrl: npt.NDArray[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    """Compute derivative control points for a 1D non-rational B-spline.

    Applies the standard B-spline derivative formula:
    ``Q[i] = p * (P[i+1] - P[i]) / (T[i+p+1] - T[i+1])``

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): Knot vector of the
            original B-spline, length ``n + p + 1``.
        degree (int): Polynomial degree of the original B-spline (``p >= 1``).
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points of shape
            ``(n, ...)``, where ``n`` is the number of basis functions.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Derivative control points of
        shape ``(n - 1, ...)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_derivative_bspline` instead.
    """
    n = ctrl.shape[0]
    denom = knots[degree + 1 : n + degree] - knots[1:n]
    # Reshape denom for broadcasting: (n-1,) -> (n-1, 1, 1, ...)
    for _ in range(ctrl.ndim - 1):
        denom = denom[..., np.newaxis]
    diff = ctrl[1:] - ctrl[:-1]
    return np.where(denom == 0.0, 0.0, degree * diff / denom)


def _derivative_nonrational_1d(bspline: Bspline) -> Bspline:
    """Compute the derivative of a non-rational 1D B-spline.

    Handles both open and periodic knot vectors. For periodic B-splines,
    expands the control points to full representation before applying the
    derivative formula, then trims back to the periodic count.

    Args:
        bspline (~pantr.bspline.Bspline): A 1D non-rational B-spline with
            degree >= 1.

    Returns:
        ~pantr.bspline.Bspline: Derivative B-spline of degree ``p - 1``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_derivative_bspline` instead.
    """
    from . import Bspline as BsplineCls  # noqa: PLC0415

    space_1d = bspline.space.spaces[0]
    knots = space_1d.knots
    p = space_1d.degree
    ctrl = bspline.control_points
    is_periodic = space_1d.periodic

    # For periodic: expand CPs to full representation.
    if is_periodic:
        n_periodic = space_1d.num_basis
        n_full = len(knots) - p - 1
        indices = np.arange(n_full) % n_periodic
        ctrl = ctrl[indices]

    new_knots = knots[1:-1]
    new_degree = p - 1
    new_ctrl = _derivative_ctrl_1d(knots, p, ctrl)

    # For periodic: trim to periodic CP count.
    if is_periodic:
        tol = float(space_1d.tolerance)
        n_periodic_new = _get_Bspline_num_basis_1D_impl(new_knots, new_degree, True, tol)
        new_ctrl = new_ctrl[:n_periodic_new]

    new_space = BsplineSpace([BsplineSpace1D(new_knots, new_degree, periodic=is_periodic)])
    return BsplineCls(new_space, new_ctrl, is_rational=False)


def _derivative_nonrational_nd(bspline: Bspline, direction: int) -> Bspline:
    """Compute the partial derivative of a non-rational nD B-spline.

    Applies the 1D derivative formula along the given parametric direction
    using the moveaxis/reshape/restore pattern. Other directions are unchanged.

    Args:
        bspline (~pantr.bspline.Bspline): A non-rational nD B-spline.
        direction (int): Parametric direction for differentiation.

    Returns:
        ~pantr.bspline.Bspline: Derivative B-spline with degree ``p_d - 1`` in
        direction ``d`` and unchanged degrees in other directions.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_derivative_bspline` instead.
    """
    from . import Bspline as BsplineCls  # noqa: PLC0415

    space_d = bspline.space.spaces[direction]
    knots = space_d.knots
    p = space_d.degree
    ctrl = bspline.control_points
    is_periodic = space_d.periodic

    # For periodic: expand CPs along the direction axis.
    if is_periodic:
        n_periodic = space_d.num_basis
        n_full = len(knots) - p - 1
        indices = np.arange(n_full) % n_periodic
        ctrl = np.take(ctrl, indices, axis=direction)

    # Move target direction to axis 0, flatten the rest.
    moved = np.moveaxis(ctrl, direction, 0)
    orig_shape = moved.shape
    pts_2d = moved.reshape(orig_shape[0], -1)

    if not pts_2d.flags.c_contiguous:
        pts_2d = np.ascontiguousarray(pts_2d)

    new_pts_2d = _derivative_ctrl_1d(knots, p, pts_2d)

    # Restore shape and move axis back.
    new_shape = (new_pts_2d.shape[0], *orig_shape[1:])
    new_moved = new_pts_2d.reshape(new_shape)
    new_ctrl = np.moveaxis(new_moved, 0, direction)

    # For periodic: trim to periodic CP count.
    new_knots = knots[1:-1]
    new_degree = p - 1

    if is_periodic:
        tol = float(space_d.tolerance)
        n_periodic_new = _get_Bspline_num_basis_1D_impl(new_knots, new_degree, True, tol)
        slices: list[slice] = [slice(None)] * new_ctrl.ndim
        slices[direction] = slice(0, n_periodic_new)
        new_ctrl = new_ctrl[tuple(slices)]

    # Build new spaces list: replace direction d, keep others.
    new_spaces = list(bspline.space.spaces)
    new_spaces[direction] = BsplineSpace1D(new_knots, new_degree, periodic=is_periodic)
    new_space = BsplineSpace(new_spaces)

    return BsplineCls(new_space, new_ctrl, is_rational=False)


def _refine_to_common_space_1d(
    a: Bspline,
    b: Bspline,
    direction: int,
) -> tuple[Bspline, Bspline]:
    """Refine two B-splines to share a common knot vector in one direction.

    Computes the knots missing from each B-spline in direction ``direction``
    and inserts them so both share the same knot vector in that direction.

    Args:
        a (~pantr.bspline.Bspline): First B-spline.
        b (~pantr.bspline.Bspline): Second B-spline.
        direction (int): Parametric direction to refine.

    Returns:
        tuple[~pantr.bspline.Bspline, ~pantr.bspline.Bspline]: Both B-splines
        refined to the same knot vector in direction ``direction``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    from ._bspline_knots import _get_unique_knots_and_multiplicity_impl  # noqa: PLC0415

    space_a = a.space.spaces[direction]
    space_b = b.space.spaces[direction]
    tol = max(float(space_a.tolerance), float(space_b.tolerance))
    dtype = a.control_points.dtype

    unique_a, mults_a = _get_unique_knots_and_multiplicity_impl(
        space_a.knots, space_a.degree, tol, in_domain=True
    )
    unique_b, mults_b = _get_unique_knots_and_multiplicity_impl(
        space_b.knots, space_b.degree, tol, in_domain=True
    )

    # Find knots to insert into a (present in b but missing/lower-mult in a).
    knots_to_insert_a = _compute_missing_knots(unique_a, mults_a, unique_b, mults_b, tol, dtype)
    # Find knots to insert into b (present in a but missing/lower-mult in b).
    knots_to_insert_b = _compute_missing_knots(unique_b, mults_b, unique_a, mults_a, tol, dtype)

    if knots_to_insert_a.size > 0:
        if a.dim == 1:
            a = a.insert_knots(knots_to_insert_a)
        else:
            per_dim: list[npt.NDArray[np.float32 | np.float64] | None] = [None] * a.dim
            per_dim[direction] = knots_to_insert_a
            a = a.insert_knots(per_dim)
    if knots_to_insert_b.size > 0:
        if b.dim == 1:
            b = b.insert_knots(knots_to_insert_b)
        else:
            per_dim_b: list[npt.NDArray[np.float32 | np.float64] | None] = [None] * b.dim
            per_dim_b[direction] = knots_to_insert_b
            b = b.insert_knots(per_dim_b)

    return a, b


def _compute_missing_knots(  # noqa: PLR0913
    unique_self: npt.NDArray[np.float32 | np.float64],
    mults_self: npt.NDArray[np.int_],
    unique_other: npt.NDArray[np.float32 | np.float64],
    mults_other: npt.NDArray[np.int_],
    tol: float,
    dtype: npt.DTypeLike,
) -> npt.NDArray[np.float32 | np.float64]:
    """Compute knots present in other but missing or lower-multiplicity in self.

    For each unique knot in ``unique_other``, computes how many additional
    insertions are needed in ``unique_self`` to match the multiplicity.

    Args:
        unique_self (npt.NDArray[np.float32 | np.float64]): Unique knots of self.
        mults_self (npt.NDArray[np.int_]): Multiplicities of self.
        unique_other (npt.NDArray[np.float32 | np.float64]): Unique knots of other.
        mults_other (npt.NDArray[np.int_]): Multiplicities of other.
        tol (float): Tolerance for coincidence tests.
        dtype (npt.DTypeLike): Output dtype.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Sorted flat array of knots to insert.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    knots_list: list[float] = []
    j = 0
    for i in range(len(unique_other)):
        xi = unique_other[i]
        m_other = int(mults_other[i])
        # Find matching knot in self.
        m_self = 0
        while j < len(unique_self) and float(unique_self[j]) < float(xi) - tol:
            j += 1
        if j < len(unique_self) and abs(float(unique_self[j]) - float(xi)) <= tol:
            m_self = int(mults_self[j])
        n_insert = m_other - m_self
        if n_insert > 0:
            knots_list.extend([float(xi)] * n_insert)
    if len(knots_list) == 0:
        return np.empty(0, dtype=dtype)
    return np.array(knots_list, dtype=dtype)


def _derivative_keep_degree_nonrational(bspline: Bspline, direction: int) -> Bspline:
    """Compute derivative with degree preservation for a non-rational B-spline.

    Computes the derivative (degree ``p - 1``) and degree-elevates by 1 to
    restore the original degree ``p``.  Works directly with arrays and the
    degree-elevation kernel to avoid creating an intermediate
    :class:`~pantr.bspline.Bspline` object.

    Args:
        bspline (~pantr.bspline.Bspline): A non-rational B-spline.
        direction (int): Parametric direction for differentiation.

    Returns:
        ~pantr.bspline.Bspline: Derivative B-spline of the same degree as the
        input.  Periodic directions in the differentiated axis are converted
        to open representation (degree elevation does not support periodic).

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_derivative_bspline` instead.
    """
    from . import Bspline as BsplineCls  # noqa: PLC0415
    from ._bspline_degree_core import _degree_elevate_1d_core  # noqa: PLC0415

    space_d = bspline.space.spaces[direction]
    p = space_d.degree

    # Degree elevation requires open knot vectors. For periodic B-splines,
    # convert to open representation first, then recurse.
    if space_d.periodic:
        return _derivative_keep_degree_nonrational(bspline.to_open_bspline(), direction)

    knots = space_d.knots
    ctrl = bspline.control_points

    # Move target direction to axis 0, flatten the rest.
    moved = np.moveaxis(ctrl, direction, 0)
    orig_shape = moved.shape
    pts_2d = moved.reshape(orig_shape[0], -1)

    if not pts_2d.flags.c_contiguous:
        pts_2d = np.ascontiguousarray(pts_2d)

    # Derivative (degree p → p-1) + degree elevation (p-1 → p) on arrays.
    deriv_pts = _derivative_ctrl_1d(knots, p, pts_2d)
    deriv_knots = knots[1:-1]
    elevated_pts, elevated_knots = _degree_elevate_1d_core(p - 1, deriv_pts, deriv_knots, 1)

    # Restore shape and move axis back.
    new_shape = (elevated_pts.shape[0], *orig_shape[1:])
    new_moved = elevated_pts.reshape(new_shape)
    new_ctrl = np.moveaxis(new_moved, 0, direction)

    # Build new spaces: replace direction d, keep others unchanged.
    # Result is open in the differentiated direction.
    new_spaces = list(bspline.space.spaces)
    new_spaces[direction] = BsplineSpace1D(elevated_knots, p)
    new_space = BsplineSpace(new_spaces)

    return BsplineCls(new_space, new_ctrl, is_rational=False)


def _tile_scalar_bspline(w: Bspline, target_rank: int) -> Bspline:
    """Tile a rank-1 B-spline to match a target rank.

    Repeats the single control point component along the last axis so that
    the resulting B-spline has ``target_rank`` components, each identical to
    the original scalar function.

    Args:
        w (~pantr.bspline.Bspline): A rank-1 (scalar) non-rational B-spline.
        target_rank (int): Desired number of output components.

    Returns:
        ~pantr.bspline.Bspline: B-spline with ``target_rank`` identical
        components.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    from . import Bspline as BsplineCls  # noqa: PLC0415

    ctrl = np.repeat(w.control_points, target_rank, axis=-1)
    return BsplineCls(w.space, ctrl, is_rational=False)


def _derivative_rational(bspline: Bspline, direction: int) -> Bspline:
    """Compute the partial derivative of a rational (NURBS) B-spline.

    Applies the quotient rule: for ``f = A/w``,
    ``f' = (A'w - Aw') / w^2``.

    Uses degree-preserving derivatives for ``A'`` and ``w'`` so that the
    products ``A'w`` and ``Aw'`` are degree ``2p`` (matching ``w^2``),
    eliminating the need for a separate numerator degree elevation.

    Since ``multiply()`` requires operands to have the same rank, the scalar
    weight B-splines (rank 1) are tiled to match the vector rank before
    multiplication.

    Args:
        bspline (~pantr.bspline.Bspline): A rational B-spline.
        direction (int): Parametric direction for differentiation.

    Returns:
        ~pantr.bspline.Bspline: Rational derivative B-spline.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_derivative_bspline` instead.
    """
    from . import Bspline as BsplineCls  # noqa: PLC0415

    ctrl = bspline.control_points
    vec_rank = ctrl.shape[-1] - 1  # number of non-weight components

    # Decompose into numerator A (weighted coords) and denominator w (weights).
    a_bspline = BsplineCls(bspline.space, ctrl[..., :-1], is_rational=False)
    w_bspline = BsplineCls(bspline.space, ctrl[..., -1:], is_rational=False)

    # Degree-preserving derivatives: A' and w' remain degree p.
    a_prime = _derivative_keep_degree_nonrational(a_bspline, direction)
    w_prime = _derivative_keep_degree_nonrational(w_bspline, direction)

    # Tile scalar B-splines to match vector rank for multiply().
    w_tiled = _tile_scalar_bspline(w_bspline, vec_rank)
    w_prime_tiled = _tile_scalar_bspline(w_prime, vec_rank)

    # Products are degree p + p = 2p (same as w^2).
    num1 = a_prime.multiply(w_tiled)
    num2 = a_bspline.multiply(w_prime_tiled)

    # Subtract CPs (same space guaranteed).
    numerator_ctrl = num1.control_points - num2.control_points
    numerator = BsplineCls(num1.space, numerator_ctrl, is_rational=False)

    # Compute denominator: w^2 (degree 2p).
    denom = w_bspline.multiply(w_bspline)

    # Refine both to a common knot vector in the target direction.
    numerator, denom = _refine_to_common_space_1d(numerator, denom, direction)

    # Assemble rational B-spline.
    result_ctrl = np.concatenate([numerator.control_points, denom.control_points], axis=-1)
    return BsplineCls(numerator.space, result_ctrl, is_rational=True)


def _derivative_nonrational(bspline: Bspline, direction: int) -> Bspline:
    """Compute the partial derivative of a non-rational B-spline.

    Dispatches to the 1D or nD implementation based on the B-spline's
    dimension.

    Args:
        bspline (~pantr.bspline.Bspline): A non-rational B-spline.
        direction (int): Parametric direction for differentiation.

    Returns:
        ~pantr.bspline.Bspline: Derivative B-spline.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_derivative_bspline` instead.
    """
    if bspline.dim == 1:
        return _derivative_nonrational_1d(bspline)
    return _derivative_nonrational_nd(bspline, direction)


def _derivative_bspline(bspline: Bspline, direction: int, *, keep_degree: bool = False) -> Bspline:
    """Compute the first partial derivative of a B-spline.

    This is the main entry point for derivative computation. It dispatches
    to the appropriate implementation based on the B-spline's properties
    (rational vs non-rational, dimension, periodicity).

    Args:
        bspline (~pantr.bspline.Bspline): The B-spline to differentiate.
        direction (int): Parametric direction for differentiation. Must be
            in ``[0, dim)``.
        keep_degree (bool): If ``True``, the result has the same degree as the
            input by applying degree elevation after differentiation. Defaults
            to ``False``.

    Returns:
        ~pantr.bspline.Bspline: A new B-spline representing the derivative.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :meth:`~pantr.bspline.Bspline.derivative` instead.
    """
    if not keep_degree:
        if bspline.is_rational:
            return _derivative_rational(bspline, direction)
        return _derivative_nonrational(bspline, direction)

    # keep_degree=True: non-rational uses array-level derivative + elevation.
    if not bspline.is_rational:
        return _derivative_keep_degree_nonrational(bspline, direction)

    # Rational: derivative already uses degree-preserving A'/w', producing
    # degree 2p which exceeds the original degree p — no elevation needed.
    return _derivative_rational(bspline, direction)
