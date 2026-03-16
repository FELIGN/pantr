"""Core B-spline evaluation implementations.

This module provides low-level routines for evaluating a :class:`Bspline`
at arbitrary parametric points. The main entry point is
:func:`_evaluate_Bspline`, which dispatches to the 1D basis-combine kernel for
1D splines and to the sequential-contraction implementation for multi-dimensional
splines.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import product as iproduct
from typing import TYPE_CHECKING

import numpy as np
from numpy import typing as npt

from ._basis_utils import _validate_out_array_1D
from ._bspline_basis_core import (
    _compute_basis_deriv_nurbs_book_impl,
    _compute_basis_nurbs_book_impl,
)
from ._numba_compat import nb_jit
from .quad import PointsLattice

if TYPE_CHECKING:
    from .bspline import Bspline
    from .bspline_space_1D import BsplineSpace1D
    from .quad import PointsLattice


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _evaluate_Bspline_basis_combine_1D(  # noqa: PLR0913
    control_points: npt.NDArray[np.float32 | np.float64],
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    periodic: bool,
    tol: float,
    pts: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate a B-spline by computing basis functions then combining with control points.

    For each evaluation point, computes the (degree+1) non-zero B-spline basis
    functions via the Cox-de Boor recurrence (Algorithm A2.2 from Piegl & Tiller),
    then forms the result as a linear combination of the corresponding control points.
    This separates the O(p^2) scalar recurrence from the rank dimension, reducing
    the work per point from O(p^2 * rank) to O(p^2 + p * rank).

    Args:
        control_points (npt.NDArray[np.float32 | np.float64]): Control-point array of
            shape (n_basis, rank).
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        periodic (bool): Whether the B-spline is periodic.
        tol (float): Tolerance for numerical comparisons.
        pts (npt.NDArray[np.float32 | np.float64]): 1-D array of evaluation points,
            shape (n_pts,).
        out (npt.NDArray[np.float32 | np.float64]): Pre-allocated output array of shape
            (n_pts, rank) and matching dtype.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The `out` array filled with evaluated
        values, shape (n_pts, rank).

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_evaluate_Bspline_1D` instead.
    """
    order = degree + 1
    n_pts = pts.size
    dtype = knots.dtype
    zero = dtype.type(0.0)
    num_cp = control_points.shape[0]

    basis = np.empty((n_pts, order), dtype=dtype)
    first_basis = np.empty(n_pts, dtype=np.int64)

    _compute_basis_nurbs_book_impl(knots, degree, periodic, tol, pts, basis, first_basis)

    rank = control_points.shape[1]
    for i in range(n_pts):
        s = first_basis[i]
        for k in range(rank):
            total = zero
            for j in range(order):
                idx = s + j
                if periodic:
                    idx = idx % num_cp
                total = total + basis[i, j] * control_points[idx, k]
            out[i, k] = total

    return out


def _evaluate_Bspline_1D(
    spline: Bspline,
    pts: npt.NDArray[np.float32 | np.float64] | PointsLattice,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate the 1D B-spline at the given points.

    Dispatches to the de Boor algorithm for general B-splines and handles
    rational B-splines by dividing the numerator by the weight coordinate.

    Args:
        spline (Bspline): A 1D B-spline object containing space, control points,
            and rational flag.
        pts (npt.NDArray[np.float32 | np.float64] | PointsLattice): Evaluation
            points. If a PointsLattice, must be 1D. Otherwise must be a 1D array
            of shape (n_pts,) matching the B-spline's dtype.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have shape (n_pts, rank) and dtype matching the B-spline.
            This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: B-spline values at the given
        points. Shape is (n_pts,) for scalar fields or (n_pts, rank) for
        vector-valued B-splines. For rational B-splines the weight column is
        divided out and not included in the output.

    Raises:
        ValueError: If the B-spline is not 1D, if the points lattice is not 1D,
            or if the points dtype does not match the B-spline dtype.
    """
    if spline.dim != 1:
        raise ValueError("B-spline must be 1D")

    # Convert PointsLattice to ndarray if necessary
    pts_array: npt.NDArray[np.float32 | np.float64]
    if isinstance(pts, PointsLattice):
        if pts.dim != 1:
            raise ValueError("Points lattice must be 1D")
        pts_array = pts._pts_per_dir[0]
    else:
        pts_array = pts

    if pts_array.dtype != spline.dtype:
        raise ValueError("Points dtype must match B-spline dtype")

    n_pts = pts_array.shape[0]
    expected_shape = (n_pts, spline.control_points.shape[-1])
    expected_dtype = spline.dtype

    # Allocate output array if not provided
    out_array: npt.NDArray[np.float32 | np.float64]
    if out is None:
        out_array = np.empty(expected_shape, dtype=expected_dtype)
    else:
        _validate_out_array_1D(out, expected_shape, expected_dtype)
        out_array = out

    spline_1D = spline.space.spaces[0]

    _evaluate_Bspline_basis_combine_1D(
        spline.control_points,
        spline_1D.knots,
        spline_1D.degree,
        spline_1D.periodic,
        spline_1D.tolerance,
        pts_array,
        out_array,
    )

    if spline.is_rational:
        out_array[:, :-1] = out_array[:, :-1] / out_array[:, -1:]
        # Return only the physical coordinates, excluding weights
        return out_array[:, :-1].squeeze()

    # For scalar-valued B-splines, return 1D array
    return out_array.squeeze()


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _evaluate_Bspline_basis_combine_deriv_1D(  # noqa: PLR0913
    control_points: npt.NDArray[np.float32 | np.float64],
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    periodic: bool,
    tol: float,
    n_deriv: int,
    pts: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate B-spline derivatives by computing derivative basis functions and combining.

    For each evaluation point, computes the (degree+1) non-zero B-spline basis
    derivatives up to order ``n_deriv`` via Algorithm A2.3 from Piegl & Tiller,
    then forms each derivative as a linear combination of the corresponding
    control points.

    Args:
        control_points (npt.NDArray[np.float32 | np.float64]): Control-point array of
            shape ``(n_basis, rank)``.
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        periodic (bool): Whether the B-spline is periodic.
        tol (float): Tolerance for numerical comparisons.
        n_deriv (int): Maximum derivative order to compute (>= 0).
        pts (npt.NDArray[np.float32 | np.float64]): 1-D array of evaluation points,
            shape ``(n_pts,)``.
        out (npt.NDArray[np.float32 | np.float64]): Pre-allocated output array of shape
            ``(n_pts, n_deriv+1, rank)`` and matching dtype. Written in-place.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_evaluate_Bspline_deriv_1D` instead.
    """
    order = degree + 1
    n_pts = pts.size
    dtype = knots.dtype
    zero = dtype.type(0.0)
    num_cp = control_points.shape[0]
    rank = control_points.shape[1]

    basis_deriv = np.empty((n_pts, n_deriv + 1, order), dtype=dtype)
    first_basis = np.empty(n_pts, dtype=np.int64)

    _compute_basis_deriv_nurbs_book_impl(
        knots, degree, periodic, tol, n_deriv, pts, basis_deriv, first_basis
    )

    for i in range(n_pts):
        s = first_basis[i]
        for k in range(n_deriv + 1):
            for r in range(rank):
                total = zero
                for j in range(order):
                    idx = s + j
                    if periodic:
                        idx = idx % num_cp
                    total = total + basis_deriv[i, k, j] * control_points[idx, r]
                out[i, k, r] = total


def _evaluate_Bspline_deriv_1D(  # noqa: PLR0912
    spline: Bspline,
    pts: npt.NDArray[np.float32 | np.float64] | PointsLattice,
    orders: Sequence[int],
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate a specific derivative of a 1D B-spline at the given points.

    Computes the derivative of order ``orders[0]`` at each evaluation point.
    For rational B-splines the quotient rule (Algorithm A4.2 from Piegl & Tiller)
    is applied to return the derivative of the projected mapping.

    Args:
        spline (Bspline): A 1D B-spline object containing space, control points,
            and rational flag.
        pts (npt.NDArray[np.float32 | np.float64] | PointsLattice): Evaluation
            points. If a :class:`~pantr.quad.PointsLattice`, must be 1D. Otherwise
            must be a 1D array of shape ``(n_pts,)`` matching the B-spline's dtype.
        orders (Sequence[int]): Must have length 1. ``orders[0]`` is the derivative
            order to return. Must be >= 0.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional pre-allocated
            output array with shape ``(n_pts,)`` for scalar or ``(n_pts, rank)``
            for vector output. Filled in-place and returned. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Derivative values of shape
        ``(n_pts,)`` for scalar output or ``(n_pts, rank)`` for vector-valued
        output. For rational B-splines the weight column is divided out and not
        included in the output.

    Raises:
        ValueError: If the B-spline is not 1D, if ``orders[0] < 0``, if the
            points lattice is not 1D, or if the points dtype does not match the
            B-spline dtype.
    """
    if spline.dim != 1:
        raise ValueError("B-spline must be 1D")

    n_deriv = orders[0]
    if n_deriv < 0:
        raise ValueError(f"orders[0] must be >= 0, got {n_deriv}")

    pts_array: npt.NDArray[np.float32 | np.float64]
    if isinstance(pts, PointsLattice):
        if pts.dim != 1:
            raise ValueError("Points lattice must be 1D")
        pts_array = pts._pts_per_dir[0]
    else:
        pts_array = pts

    if pts_array.dtype != spline.dtype:
        raise ValueError("Points dtype must match B-spline dtype")

    n_pts = pts_array.shape[0]
    cp_size = spline.control_points.shape[-1]
    spline_1D = spline.space.spaces[0]

    if spline.is_rational:
        rank = cp_size - 1
        # Compute all homogeneous derivatives 0..n_deriv (needed for quotient rule).
        hom_array = np.empty((n_pts, n_deriv + 1, cp_size), dtype=spline.dtype)
        _evaluate_Bspline_basis_combine_deriv_1D(
            spline.control_points,
            spline_1D.knots,
            spline_1D.degree,
            spline_1D.periodic,
            spline_1D.tolerance,
            n_deriv,
            pts_array,
            hom_array,
        )
        # Algorithm A4.2: build result for orders 0..n_deriv, return only n_deriv.
        result_all = np.empty((n_pts, n_deriv + 1, rank), dtype=spline.dtype)
        for k in range(n_deriv + 1):
            v = hom_array[:, k, :-1].copy()
            for i in range(1, k + 1):
                v -= math.comb(k, i) * hom_array[:, i, -1:] * result_all[:, k - i, :]
            result_all[:, k, :] = v / hom_array[:, 0, -1:]
        final: npt.NDArray[np.float32 | np.float64] = result_all[:, n_deriv, :]
        if rank == 1:
            final = final[:, 0]
        if out is not None:
            _validate_out_array_1D(out, final.shape, spline.dtype)
            out[:] = final
            return out
        return final

    # Non-rational: compute all derivatives 0..n_deriv and extract row n_deriv.
    buf = np.empty((n_pts, n_deriv + 1, cp_size), dtype=spline.dtype)
    _evaluate_Bspline_basis_combine_deriv_1D(
        spline.control_points,
        spline_1D.knots,
        spline_1D.degree,
        spline_1D.periodic,
        spline_1D.tolerance,
        n_deriv,
        pts_array,
        buf,
    )
    final_nd: npt.NDArray[np.float32 | np.float64] = buf[:, n_deriv, :]
    if cp_size == 1:
        result_1d: npt.NDArray[np.float32 | np.float64] = final_nd[:, 0]
        if out is not None:
            _validate_out_array_1D(out, result_1d.shape, spline.dtype)
            out[:] = result_1d
            return out
        return result_1d
    if out is not None:
        _validate_out_array_1D(out, final_nd.shape, spline.dtype)
        out[:] = final_nd
        return out
    return final_nd


def _evaluate_Bspline_deriv_multi_dim_pts_array(
    cp: npt.NDArray[np.float32 | np.float64],
    spaces_1D: tuple[BsplineSpace1D, ...],
    pts: npt.NDArray[np.float32 | np.float64],
    orders: tuple[int, ...],
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate a partial derivative of a multi-dimensional B-spline at points.

    Computes the partial derivative of orders ``orders[d]`` in each direction
    ``d`` for all evaluation points simultaneously. Uses outer-product contraction
    with the appropriate derivative basis row for each direction.

    Args:
        cp (npt.NDArray[np.float32 | np.float64]): Control-point array of shape
            ``(*num_basis, cp_size)``.
        spaces_1D (tuple[BsplineSpace1D, ...]): 1D B-spline spaces, one per direction.
        pts (npt.NDArray[np.float32 | np.float64]): Evaluation points of shape
            ``(n_pts, dim)``.
        orders (tuple[int, ...]): Derivative order for each direction. Each entry
            must be >= 0 and ``len(orders) == len(spaces_1D)``.
        out (npt.NDArray[np.float32 | np.float64]): Pre-allocated output array of
            shape ``(n_pts, cp_size)`` and matching dtype. Filled in-place.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_evaluate_Bspline_deriv_multi_dim` instead.
    """
    dim = len(spaces_1D)
    n_pts = pts.shape[0]

    dBs: list[npt.NDArray[np.float32 | np.float64]] = []
    first_idxs: list[npt.NDArray[np.int_]] = []
    for d, space_d in enumerate(spaces_1D):
        dB_d, first_d = space_d.tabulate_basis_derivatives(
            np.ascontiguousarray(pts[:, d]), orders[d]
        )
        dBs.append(dB_d)  # (n_pts, orders[d]+1, degree_d+1)
        first_idxs.append(first_d)

    # Extract the specific derivative row for each direction.
    Bs = [dBs[d][:, orders[d], :] for d in range(dim)]  # each (n_pts, degree_d+1)
    basis_orders = tuple(int(B.shape[1]) for B in Bs)

    # Build advanced index arrays (same pattern as _evaluate_Bspline_multi_dim_pts_array).
    index_list: list[npt.NDArray[np.intp]] = []
    for d, (first_d, space_d) in enumerate(zip(first_idxs, spaces_1D, strict=True)):
        order_d = basis_orders[d]
        idx_d = (
            first_d[:, np.newaxis].astype(np.intp)
            + np.arange(order_d, dtype=np.intp)[np.newaxis, :]
        )  # (n_pts, order_d)
        if space_d.periodic:
            idx_d = idx_d % space_d.num_basis
        shape: tuple[int, ...] = (n_pts,) + (1,) * d + (order_d,) + (1,) * (dim - d - 1)
        index_list.append(idx_d.reshape(shape))

    idx_with_last: tuple[npt.NDArray[np.intp] | slice, ...] = (*tuple(index_list), slice(None))
    cp_local: npt.NDArray[np.float32 | np.float64] = cp[idx_with_last]
    # shape: (n_pts, order_0, ..., order_{D-1}, cp_size)

    weights: npt.NDArray[np.float32 | np.float64] = Bs[0].reshape(
        (n_pts, basis_orders[0]) + (1,) * (dim - 1)
    )
    for d in range(1, dim):
        w_shape: tuple[int, ...] = (n_pts,) + (1,) * d + (basis_orders[d],) + (1,) * (dim - d - 1)
        weights = weights * Bs[d].reshape(w_shape)

    total_local = int(np.prod(basis_orders))
    out[:] = (
        (cp_local * weights[..., np.newaxis]).reshape(n_pts, total_local, cp.shape[-1]).sum(axis=1)
    )


def _evaluate_Bspline_deriv_multi_dim_lattice(
    cp: npt.NDArray[np.float32 | np.float64],
    spaces_1D: tuple[BsplineSpace1D, ...],
    pts: PointsLattice,
    orders: tuple[int, ...],
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate a partial derivative of a multi-dimensional B-spline on a lattice.

    Computes the partial derivative of orders ``orders[d]`` in each direction
    ``d`` via sequential contraction, using the appropriate derivative basis row for
    each direction.

    Args:
        cp (npt.NDArray[np.float32 | np.float64]): Control-point array of shape
            ``(*num_basis, cp_size)``.
        spaces_1D (tuple[BsplineSpace1D, ...]): 1D B-spline spaces, one per direction.
        pts (PointsLattice): Evaluation lattice. ``pts.dim`` must equal
            ``len(spaces_1D)``.
        orders (tuple[int, ...]): Derivative order for each direction. Each entry
            must be >= 0 and ``len(orders) == len(spaces_1D)``.
        out (npt.NDArray[np.float32 | np.float64]): Pre-allocated output array of
            shape ``(*pts_grid_shape, cp_size)`` and matching dtype. Filled in-place.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_evaluate_Bspline_deriv_multi_dim` instead.
    """
    Bs: list[npt.NDArray[np.float32 | np.float64]] = []
    first_idxs: list[npt.NDArray[np.int_]] = []
    for d, (space_d, pts_d) in enumerate(zip(spaces_1D, pts.pts_per_dir, strict=True)):
        dB_d, first_d = space_d.tabulate_basis_derivatives(pts_d, orders[d])
        Bs.append(dB_d[:, orders[d], :])  # (m_d, degree_d+1) — specific derivative row
        first_idxs.append(first_d)

    # Sequential contraction identical to _evaluate_Bspline_multi_dim_lattice.
    current: npt.NDArray[np.float32 | np.float64] = cp
    for d, (B_d, first_d, space_d) in enumerate(zip(Bs, first_idxs, spaces_1D, strict=True)):
        m_d = int(B_d.shape[0])
        order_d = int(B_d.shape[1])

        idx_d = (
            first_d[:, np.newaxis] + np.arange(order_d, dtype=np.intp)[np.newaxis, :]
        )  # (m_d, order_d)
        if space_d.periodic:
            idx_d = idx_d % space_d.num_basis

        current_moved = np.moveaxis(current, d, 0)
        gathered_moved: npt.NDArray[np.float32 | np.float64] = current_moved[idx_d]

        n_trail = gathered_moved.ndim - 2
        B_exp = B_d.reshape((m_d, order_d) + (1,) * n_trail)
        contracted: npt.NDArray[np.float32 | np.float64] = (gathered_moved * B_exp).sum(axis=1)
        current = np.moveaxis(contracted, 0, d)

    np.copyto(out, current)


def _evaluate_Bspline_deriv_multi_dim(  # noqa: PLR0912, PLR0915
    spline: Bspline,
    pts: npt.NDArray[np.float32 | np.float64] | PointsLattice,
    orders: Sequence[int],
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate a partial derivative of a multi-dimensional B-spline.

    Computes the single partial derivative specified by ``orders``, where
    ``orders[d]`` is the derivative order in parametric direction ``d``. For
    rational B-splines the generalised quotient rule is applied so that the
    result is the derivative of the projected (inhomogeneous) mapping.

    Args:
        spline (Bspline): A multi-dimensional B-spline (``dim >= 2``).
        pts (npt.NDArray[np.float32 | np.float64] | PointsLattice): Evaluation
            points. Either a 2-D array of shape ``(n_pts, dim)`` or a
            :class:`~pantr.quad.PointsLattice`.
        orders (Sequence[int]): One non-negative integer per parametric direction.
            ``len(orders)`` must equal ``spline.dim``.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional pre-allocated
            output array whose shape matches the return value. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Mixed partial derivative values.
        Shape is ``(*pts_base_shape,)`` for scalar output or
        ``(*pts_base_shape, rank)`` for vector-valued output, where
        ``pts_base_shape`` is ``(n_pts,)`` for a points array or
        ``(*pts_grid_shape,)`` for a :class:`~pantr.quad.PointsLattice`.
        For rational B-splines the weight column is divided out.

    Raises:
        ValueError: If ``len(orders) != spline.dim``, if any order is negative, if
            the pts dimension or dtype does not match the B-spline, or if ``out``
            has an incorrect shape or dtype.
    """
    dim = spline.dim
    orders_tuple = tuple(orders)

    if len(orders_tuple) != dim:
        raise ValueError(f"len(orders) must equal spline.dim ({dim}), got {len(orders_tuple)}")
    for d, od in enumerate(orders_tuple):
        if od < 0:
            raise ValueError(f"orders[{d}] must be >= 0, got {od}")

    dtype = spline.dtype
    cp = spline.control_points
    cp_size = cp.shape[-1]

    pts_base_shape: tuple[int, ...]
    if isinstance(pts, PointsLattice):
        if pts.dim != dim:
            raise ValueError(
                f"Points lattice dimension {pts.dim} does not match B-spline dimension {dim}"
            )
        if pts.dtype != dtype:
            raise ValueError("Points dtype must match B-spline dtype")
        pts_base_shape = tuple(int(p.shape[0]) for p in pts.pts_per_dir)
    else:
        if pts.ndim != 2 or pts.shape[1] != dim:  # noqa: PLR2004
            raise ValueError(f"pts must be a 2D array with {dim} columns")
        if pts.dtype != dtype:
            raise ValueError("Points dtype must match B-spline dtype")
        pts_base_shape = (pts.shape[0],)

    out_shape = (*pts_base_shape, cp_size)

    if spline.is_rational:
        # For the generalised quotient rule we need all partial derivatives
        # hom[i] for multi-indices 0 <= i[d] <= orders[d].
        # hom_all shape: (*pts_base_shape, orders[0]+1, ..., orders[D-1]+1, cp_size)
        hom_shape = (*pts_base_shape, *(od + 1 for od in orders_tuple), cp_size)
        hom_all: npt.NDArray[np.float32 | np.float64] = np.empty(hom_shape, dtype=dtype)

        if isinstance(pts, PointsLattice):
            # Precompute per-direction basis derivatives.
            dBs_lat: list[npt.NDArray[np.float32 | np.float64]] = []
            first_idxs_lat: list[npt.NDArray[np.int_]] = []
            max_order = max(orders_tuple, default=0)
            for space_d, pts_d in zip(spline.space.spaces, pts.pts_per_dir, strict=True):
                dB_d, first_d = space_d.tabulate_basis_derivatives(pts_d, max_order)
                dBs_lat.append(dB_d)
                first_idxs_lat.append(first_d)
            for multi_idx in iproduct(*[range(od + 1) for od in orders_tuple]):
                # Sequential contraction for this multi-index.
                current: npt.NDArray[np.float32 | np.float64] = cp
                for d, (dB_d, first_d, space_d) in enumerate(
                    zip(dBs_lat, first_idxs_lat, spline.space.spaces, strict=True)
                ):
                    B_d = dB_d[:, multi_idx[d], :]  # (m_d, degree_d+1)
                    m_d = int(B_d.shape[0])
                    order_d = int(B_d.shape[1])
                    idx_d = (
                        first_d[:, np.newaxis] + np.arange(order_d, dtype=np.intp)[np.newaxis, :]
                    )
                    if space_d.periodic:
                        idx_d = idx_d % space_d.num_basis
                    current_moved = np.moveaxis(current, d, 0)
                    gathered = current_moved[idx_d]
                    n_trail = gathered.ndim - 2
                    B_exp = B_d.reshape((m_d, order_d) + (1,) * n_trail)
                    contracted = (gathered * B_exp).sum(axis=1)
                    current = np.moveaxis(contracted, 0, d)
                hom_all[(Ellipsis,) + multi_idx + (slice(None),)] = current  # noqa: RUF005
        else:
            # Precompute per-direction basis derivatives and gather cp_local once.
            dBs_pts: list[npt.NDArray[np.float32 | np.float64]] = []
            first_idxs_pts: list[npt.NDArray[np.int_]] = []
            for d, space_d in enumerate(spline.space.spaces):
                dB_d, first_d = space_d.tabulate_basis_derivatives(
                    np.ascontiguousarray(pts[:, d]), orders_tuple[d]
                )
                dBs_pts.append(dB_d)
                first_idxs_pts.append(first_d)
            n_pts = pts.shape[0]
            basis_orders_pts = tuple(int(dB.shape[2]) for dB in dBs_pts)
            index_list_pts: list[npt.NDArray[np.intp]] = []
            for d, (first_d, space_d) in enumerate(
                zip(first_idxs_pts, spline.space.spaces, strict=True)
            ):
                order_d = basis_orders_pts[d]
                idx_d = (
                    first_d[:, np.newaxis].astype(np.intp)
                    + np.arange(order_d, dtype=np.intp)[np.newaxis, :]
                )
                if space_d.periodic:
                    idx_d = idx_d % space_d.num_basis
                s: tuple[int, ...] = (n_pts,) + (1,) * d + (order_d,) + (1,) * (dim - d - 1)
                index_list_pts.append(idx_d.reshape(s))
            idx_tup: tuple[npt.NDArray[np.intp] | slice, ...] = (
                *tuple(index_list_pts),
                slice(None),
            )
            cp_local: npt.NDArray[np.float32 | np.float64] = cp[idx_tup]
            total_local = int(np.prod(basis_orders_pts))
            cp_flat = cp_local.reshape(n_pts, total_local, cp_size)
            for multi_idx in iproduct(*[range(od + 1) for od in orders_tuple]):
                w: npt.NDArray[np.float32 | np.float64] = dBs_pts[0][:, multi_idx[0], :].reshape(
                    (n_pts, basis_orders_pts[0]) + (1,) * (dim - 1)
                )
                for d in range(1, dim):
                    ws: tuple[int, ...] = (
                        (n_pts,) + (1,) * d + (basis_orders_pts[d],) + (1,) * (dim - d - 1)
                    )
                    w = w * dBs_pts[d][:, multi_idx[d], :].reshape(ws)
                w_flat = w.reshape(n_pts, total_local)
                hom_all[(Ellipsis,) + multi_idx + (slice(None),)] = (  # noqa: RUF005
                    (cp_flat * w_flat[..., np.newaxis]).sum(axis=1)
                )

        # Generalised quotient rule (ascending total-order iteration).
        rank_value = cp_size - 1
        result_shape_rat = (*pts_base_shape, *(od + 1 for od in orders_tuple), rank_value)
        result_all: npt.NDArray[np.float32 | np.float64] = np.empty(result_shape_rat, dtype=dtype)
        zero_idx = (0,) * dim
        W = hom_all[(Ellipsis,) + zero_idx + (-1,)]  # noqa: RUF005  # (*pts_base_shape,)

        all_multi_indices = sorted(iproduct(*[range(od + 1) for od in orders_tuple]), key=sum)
        for k_idx in all_multi_indices:
            hom_N_k = hom_all[(Ellipsis,) + k_idx + (slice(None),)][..., :-1]  # noqa: RUF005
            v: npt.NDArray[np.float32 | np.float64] = hom_N_k.copy()
            for i_idx in all_multi_indices:
                if i_idx == zero_idx:
                    continue
                if sum(i_idx) > sum(k_idx):
                    break
                if not all(i_idx[d] <= k_idx[d] for d in range(dim)):
                    continue
                coeff = 1
                for d in range(dim):
                    coeff *= math.comb(k_idx[d], i_idx[d])
                hom_W_i = hom_all[(Ellipsis,) + i_idx + (-1,)]  # noqa: RUF005
                k_minus_i = tuple(k_idx[d] - i_idx[d] for d in range(dim))
                res_km_i = result_all[(Ellipsis,) + k_minus_i + (slice(None),)]  # noqa: RUF005
                v = v - coeff * hom_W_i[..., np.newaxis] * res_km_i
            result_all[(Ellipsis,) + k_idx + (slice(None),)] = v / W[..., np.newaxis]  # noqa: RUF005

        final_rat: npt.NDArray[np.float32 | np.float64] = result_all[
            (Ellipsis,) + orders_tuple + (slice(None),)  # noqa: RUF005
        ]
        if rank_value == 1:
            final_rat = final_rat[..., 0]
        if out is not None:
            _validate_out_array_1D(out, final_rat.shape, dtype)
            out[:] = final_rat
            return out
        return final_rat

    # Non-rational: extract specific derivative row directly.
    # Kernel always writes to (*pts_base_shape, cp_size); out matches the final
    # squeezed shape ((*pts_base_shape,) for scalar, (*pts_base_shape, rank) for vector).
    buf: npt.NDArray[np.float32 | np.float64]
    final_nd: npt.NDArray[np.float32 | np.float64]
    if cp_size == 1:
        # Scalar: allocate internal buf; out must be (*pts_base_shape,)
        buf = np.empty(out_shape, dtype=dtype)
        if isinstance(pts, PointsLattice):
            _evaluate_Bspline_deriv_multi_dim_lattice(
                cp, spline.space.spaces, pts, orders_tuple, buf
            )
        else:
            _evaluate_Bspline_deriv_multi_dim_pts_array(
                cp, spline.space.spaces, pts, orders_tuple, buf
            )
        final_nd = buf[..., 0]
        if out is not None:
            _validate_out_array_1D(out, final_nd.shape, dtype)
            out[:] = final_nd
            return out
        return final_nd
    else:
        # Vector: out has same shape as buf; pass directly to kernel if provided.
        if out is None:
            buf = np.empty(out_shape, dtype=dtype)
        else:
            _validate_out_array_1D(out, out_shape, dtype)
            buf = out
        if isinstance(pts, PointsLattice):
            _evaluate_Bspline_deriv_multi_dim_lattice(
                cp, spline.space.spaces, pts, orders_tuple, buf
            )
        else:
            _evaluate_Bspline_deriv_multi_dim_pts_array(
                cp, spline.space.spaces, pts, orders_tuple, buf
            )
        return buf


def _evaluate_Bspline_deriv(
    spline: Bspline,
    pts: npt.NDArray[np.float32 | np.float64] | PointsLattice,
    orders: Sequence[int],
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate a B-spline derivative, dispatching on parametric dimension.

    Args:
        spline (Bspline): The B-spline object.
        pts (npt.NDArray[np.float32 | np.float64] | PointsLattice): Evaluation
            points.
        orders (Sequence[int]): One non-negative derivative order per parametric
            direction. ``len(orders)`` must equal ``spline.dim``.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional pre-allocated
            output array. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Derivative values at the given points.
    """
    if spline.dim == 1:
        return _evaluate_Bspline_deriv_1D(spline, pts, orders, out)
    else:
        return _evaluate_Bspline_deriv_multi_dim(spline, pts, orders, out)


def _evaluate_Bspline_multi_dim_lattice(
    cp: npt.NDArray[np.float32 | np.float64],
    spaces_1D: tuple[BsplineSpace1D, ...],
    pts: PointsLattice,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate a multi-dimensional B-spline on a point lattice via sequential contraction.

    For each parametric direction ``d``, the current tensor is reduced by moving
    axis ``d`` to position 0, gathering ``order_d`` support entries per evaluation
    point, contracting with the 1D basis values, and moving the result back to
    axis ``d``. After all directions are processed, ``out`` is filled with the
    result of shape ``(*pts_grid_shape, cp.shape[-1])``.

    Args:
        cp (npt.NDArray[np.float32 | np.float64]): Control-point array of shape
            ``(*num_basis, k)`` where ``k`` is the number of values per control point.
        spaces_1D (tuple[BsplineSpace1D, ...]): 1D B-spline spaces, one per direction.
        pts (PointsLattice): Evaluation lattice. ``pts.dim`` must equal ``len(spaces_1D)``.
        out (npt.NDArray[np.float32 | np.float64]): Pre-allocated output array of shape
            ``(*pts_grid_shape, k)`` and matching dtype. Filled in-place.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_evaluate_Bspline_multi_dim` instead.
    """
    Bs: list[npt.NDArray[np.float32 | np.float64]] = []
    first_idxs: list[npt.NDArray[np.int_]] = []
    for space_d, pts_d in zip(spaces_1D, pts.pts_per_dir, strict=True):
        B_d, first_d = space_d.tabulate_basis(pts_d)
        Bs.append(B_d)
        first_idxs.append(first_d)

    # Sequential contraction over parametric directions.
    # After processing direction d, 'current' has shape
    # (m_0, ..., m_d, n_{d+1}, ..., n_{D-1}, k).
    current: npt.NDArray[np.float32 | np.float64] = cp
    for d, (B_d, first_d, space_d) in enumerate(zip(Bs, first_idxs, spaces_1D, strict=True)):
        m_d = int(B_d.shape[0])
        order_d = int(B_d.shape[1])

        idx_d = (
            first_d[:, np.newaxis] + np.arange(order_d, dtype=np.intp)[np.newaxis, :]
        )  # (m_d, order_d)
        if space_d.periodic:
            idx_d = idx_d % space_d.num_basis

        # Move axis d to position 0, gather order_d entries per evaluation
        # point, contract with B_d, then move the result back to axis d.
        current_moved = np.moveaxis(current, d, 0)
        gathered_moved: npt.NDArray[np.float32 | np.float64] = current_moved[idx_d]
        # shape: (m_d, order_d, m_0, ..., m_{d-1}, n_{d+1}, ..., n_{D-1}, k)

        n_trail = gathered_moved.ndim - 2
        B_exp = B_d.reshape((m_d, order_d) + (1,) * n_trail)
        contracted: npt.NDArray[np.float32 | np.float64] = (gathered_moved * B_exp).sum(axis=1)

        current = np.moveaxis(contracted, 0, d)

    np.copyto(out, current)


def _evaluate_Bspline_multi_dim_pts_array(
    cp: npt.NDArray[np.float32 | np.float64],
    spaces_1D: tuple[BsplineSpace1D, ...],
    pts: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate a multi-dimensional B-spline at an array of points via gather-and-contract.

    For each evaluation point, the local control-point patch is gathered from
    ``cp`` using broadcasting advanced indexing, then contracted with the
    outer-product of the 1D basis values. The result is written into ``out``.

    Args:
        cp (npt.NDArray[np.float32 | np.float64]): Control-point array of shape
            ``(*num_basis, k)`` where ``k`` is the number of values per control point.
        spaces_1D (tuple[BsplineSpace1D, ...]): 1D B-spline spaces, one per direction.
        pts (npt.NDArray[np.float32 | np.float64]): Evaluation points of shape
            ``(n_pts, dim)``.
        out (npt.NDArray[np.float32 | np.float64]): Pre-allocated output array of shape
            ``(n_pts, k)`` and matching dtype. Filled in-place.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_evaluate_Bspline_multi_dim` instead.
    """
    dim = len(spaces_1D)
    n_pts = pts.shape[0]

    Bs: list[npt.NDArray[np.float32 | np.float64]] = []
    first_idxs: list[npt.NDArray[np.int_]] = []
    for d, space_d in enumerate(spaces_1D):
        B_d, first_d = space_d.tabulate_basis(np.ascontiguousarray(pts[:, d]))
        Bs.append(B_d)
        first_idxs.append(first_d)

    orders = tuple(int(B.shape[1]) for B in Bs)

    # Build advanced index arrays.
    # idx_d[i, j] = first_idxs[d][i] + j, reshaped for broadcasting to
    # (n_pts, order_0, ..., order_{D-1}).
    index_list: list[npt.NDArray[np.intp]] = []
    for d, (first_d, space_d) in enumerate(zip(first_idxs, spaces_1D, strict=True)):
        order_d = orders[d]
        idx_d = (
            first_d[:, np.newaxis].astype(np.intp)
            + np.arange(order_d, dtype=np.intp)[np.newaxis, :]
        )  # (n_pts, order_d)
        if space_d.periodic:
            idx_d = idx_d % space_d.num_basis
        shape: tuple[int, ...] = (n_pts,) + (1,) * d + (order_d,) + (1,) * (dim - d - 1)
        index_list.append(idx_d.reshape(shape))

    # Gather local control-point patch via broadcasting advanced indexing.
    # cp_local[i, j_0, ..., j_{D-1}, r] = cp[idx_0[i,j_0], ..., idx_{D-1}[i,j_{D-1}], r]
    # mypy cannot model a tuple of integer ndarrays followed by a slice;
    # the numpy operation is correct and the result dtype is preserved.
    idx_with_last: tuple[npt.NDArray[np.intp] | slice, ...] = (*tuple(index_list), slice(None))
    cp_local: npt.NDArray[np.float32 | np.float64] = cp[idx_with_last]
    # shape: (n_pts, order_0, ..., order_{D-1}, k)

    # Build outer-product weights: weights[i, j_0, ..., j_{D-1}] = prod_d B_d[i, j_d]
    weights: npt.NDArray[np.float32 | np.float64] = Bs[0].reshape(
        (n_pts, orders[0]) + (1,) * (dim - 1)
    )
    for d in range(1, dim):
        w_shape: tuple[int, ...] = (n_pts,) + (1,) * d + (orders[d],) + (1,) * (dim - d - 1)
        weights = weights * Bs[d].reshape(w_shape)
    # shape: (n_pts, order_0, ..., order_{D-1})

    total_local = int(np.prod(orders))
    out[:] = (
        (cp_local * weights[..., np.newaxis]).reshape(n_pts, total_local, cp.shape[-1]).sum(axis=1)
    )


def _evaluate_Bspline_multi_dim(
    spline: Bspline,
    pts: npt.NDArray[np.float32 | np.float64] | PointsLattice,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate a multi-dimensional B-spline at the given points.

    Evaluates 1D basis functions for each parametric direction and combines
    them with the control points through sequential contractions, avoiding the
    O(prod(order_d)) memory cost of assembling the full tensor-product basis.

    Args:
        spline (Bspline): A multi-dimensional B-spline object (``dim >= 2``).
        pts (npt.NDArray[np.float32 | np.float64] | PointsLattice): Evaluation
            points. Either a :class:`~pantr.quad.PointsLattice` (one 1D array
            per parametric direction) or a 2D array of shape
            ``(n_pts, spline.dim)`` containing row-wise parameter coordinates.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional
            pre-allocated output buffer. Must have shape ``(*pts_shape, rank)``
            and dtype matching the B-spline, where ``pts_shape`` is
            ``(m_0, ..., m_{D-1})`` for a :class:`~pantr.quad.PointsLattice`
            or ``(n_pts,)`` for a points array. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: B-spline values at the given
        points. Shape is ``(m_0, ..., m_{D-1})`` for scalar fields or
        ``(m_0, ..., m_{D-1}, rank)`` for vector-valued B-splines when
        ``pts`` is a :class:`~pantr.quad.PointsLattice`, and ``(n_pts,)``
        or ``(n_pts, rank)`` when ``pts`` is a points array. For rational
        B-splines the weight column is divided out and not included in
        the output.

    Raises:
        ValueError: If the pts dimension does not match the B-spline dimension,
            if the pts dtype does not match the B-spline dtype, or if ``out``
            has an incorrect shape or dtype.
    """
    dim = spline.dim
    dtype = spline.dtype
    cp = spline.control_points

    out_array: npt.NDArray[np.float32 | np.float64]

    if isinstance(pts, PointsLattice):
        if pts.dim != dim:
            raise ValueError(
                f"Points lattice dimension {pts.dim} does not match B-spline dimension {dim}"
            )
        if pts.dtype != dtype:
            raise ValueError("Points dtype must match B-spline dtype")

        pts_grid_shape = tuple(int(p.shape[0]) for p in pts.pts_per_dir)
        expected_shape = (*pts_grid_shape, cp.shape[-1])
        if out is None:
            out_array = np.empty(expected_shape, dtype=dtype)
        else:
            _validate_out_array_1D(out, expected_shape, dtype)
            out_array = out

        _evaluate_Bspline_multi_dim_lattice(cp, spline.space.spaces, pts, out_array)

    else:
        if pts.ndim != 2 or pts.shape[1] != dim:  # noqa: PLR2004
            raise ValueError(f"pts must be a 2D array with {dim} columns")
        if pts.dtype != dtype:
            raise ValueError("Points dtype must match B-spline dtype")

        expected_shape = (pts.shape[0], cp.shape[-1])
        if out is None:
            out_array = np.empty(expected_shape, dtype=dtype)
        else:
            _validate_out_array_1D(out, expected_shape, dtype)
            out_array = out

        _evaluate_Bspline_multi_dim_pts_array(cp, spline.space.spaces, pts, out_array)

    if spline.is_rational:
        out_array[..., :-1] = out_array[..., :-1] / out_array[..., -1:]
        return out_array[..., :-1].squeeze()

    return out_array.squeeze()


def _evaluate_Bspline(
    spline: Bspline,
    pts: npt.NDArray[np.float32 | np.float64] | PointsLattice,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate the B-spline at the given points, dispatching on parametric dimension.

    Args:
        spline (Bspline): The B-spline object.
        pts (npt.NDArray[np.float32 | np.float64] | PointsLattice): The points at which
            to evaluate the B-spline.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional pre-allocated output
            array. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The B-spline values at the given points.
    """
    if spline.dim == 1:
        return _evaluate_Bspline_1D(spline, pts, out)
    else:
        return _evaluate_Bspline_multi_dim(spline, pts, out)


def _warmup_numba_functions() -> None:
    """Precompile Numba functions with float64 signatures for faster first call.

    Triggers compilation of the basis-combine evaluation kernel with representative
    float64 arrays. The compiled code is cached by Numba (``cache=True``) so
    subsequent cold-start calls do not pay JIT overhead.
    """
    knots_dummy = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    pts_dummy = np.array([0.5], dtype=np.float64)
    cp_dummy = np.array([[0.0], [1.0], [0.0]], dtype=np.float64)
    tol_dummy = 1e-10
    degree_dummy = 2
    out_dummy = np.empty((1, 1), dtype=np.float64)

    _evaluate_Bspline_basis_combine_1D(
        cp_dummy, knots_dummy, degree_dummy, False, tol_dummy, pts_dummy, out_dummy
    )

    n_deriv_dummy = 1
    out_deriv_dummy = np.empty((1, n_deriv_dummy + 1, 1), dtype=np.float64)
    _evaluate_Bspline_basis_combine_deriv_1D(
        cp_dummy,
        knots_dummy,
        degree_dummy,
        False,
        tol_dummy,
        n_deriv_dummy,
        pts_dummy,
        out_deriv_dummy,
    )


# Precompile numba functions on module import
# (Moved to central thread in __init__.py)
