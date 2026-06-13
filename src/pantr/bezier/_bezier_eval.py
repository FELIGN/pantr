"""Bézier evaluation helpers.

This module provides Layer 2 evaluation functions for :class:`~pantr.bezier.Bezier`
objects. It validates inputs, allocates output arrays, and dispatches to Layer 3
Numba kernels in :mod:`_bezier_core` (for 1D) or performs sequential contraction
via NumPy (for nD).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..basis._basis_utils import _validate_out_array
from ._bezier_core import _evaluate_bezier_1d_core, _evaluate_bezier_deriv_1d_core
from ._bezier_utils import _tabulate_bernstein_1d_fast

if TYPE_CHECKING:
    from ..quad import PointsLattice
    from . import Bezier


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def _evaluate_bezier(
    bezier: Bezier,
    pts: npt.NDArray[np.float32 | np.float64] | PointsLattice,
    out: npt.NDArray[np.float32 | np.float64] | None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate a Bézier at the given parametric points.

    Dispatches to the 1D fused kernel or the nD contraction path depending on
    ``bezier.dim``.

    Args:
        bezier (~pantr.bezier.Bezier): The Bézier to evaluate.
        pts (npt.NDArray[np.float32 | np.float64] | PointsLattice): Parametric
            evaluation points.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional pre-allocated
            output array. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Values at the given points.

    Raises:
        ValueError: If the points dtype or shape does not match the Bézier.
    """
    if bezier.dim == 1:
        return _evaluate_bezier_1d(bezier, pts, out)
    return _evaluate_bezier_nd(bezier, pts, out)


def _evaluate_bezier_1d(
    bezier: Bezier,
    pts: npt.NDArray[np.float32 | np.float64] | PointsLattice,
    out: npt.NDArray[np.float32 | np.float64] | None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate a 1D Bézier at the given points.

    Args:
        bezier (~pantr.bezier.Bezier): A 1D Bézier.
        pts (npt.NDArray[np.float32 | np.float64] | PointsLattice): 1D
            evaluation points.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Values of shape ``(n_pts,)``
        for scalar or ``(n_pts, rank)`` for vector output.

    Raises:
        ValueError: If the points dtype does not match the Bézier dtype.
    """
    from ..quad import PointsLattice as PL  # noqa: PLC0415

    dtype = bezier.dtype
    ctrl = bezier.control_points

    if isinstance(pts, PL):
        if pts.dim != 1:
            raise ValueError("PointsLattice must be 1D for a 1D Bézier.")
        pts_array = pts.pts_per_dir[0]
    else:
        pts_array = np.asarray(pts)
    if pts_array.ndim != 1:
        raise ValueError("Points must be a 1D array for a 1D Bézier.")
    if pts_array.dtype != dtype:
        raise ValueError(f"Points dtype ({pts_array.dtype}) must match Bézier dtype ({dtype}).")

    n_pts = pts_array.shape[0]
    cp_size = ctrl.shape[-1]

    # Allocate raw output (includes weight column for rational)
    out_raw = np.empty((n_pts, cp_size), dtype=dtype)
    _evaluate_bezier_1d_core(ctrl, pts_array, out_raw)

    result = _project_rational(bezier, out_raw)

    if out is not None:
        _validate_out_array(out, result.shape, dtype)
        out[:] = result
        return out
    return result


def _evaluate_bezier_nd(
    bezier: Bezier,
    pts: npt.NDArray[np.float32 | np.float64] | PointsLattice,
    out: npt.NDArray[np.float32 | np.float64] | None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate an nD Bézier at the given points.

    Uses sequential contraction: evaluates Bernstein basis in each parametric
    direction and contracts with control points via ``np.einsum`` (arbitrary
    points) or ``np.tensordot`` (lattice).

    Args:
        bezier (~pantr.bezier.Bezier): An nD Bézier (``dim >= 2``).
        pts (npt.NDArray[np.float32 | np.float64] | PointsLattice): Evaluation
            points — either a 2D array of shape ``(n_pts, dim)`` or a
            :class:`~pantr.quad.PointsLattice`.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Values at the given points.

    Raises:
        ValueError: If the points shape or dtype does not match the Bézier.
    """
    from ..quad import PointsLattice as PL  # noqa: PLC0415

    dim = bezier.dim
    dtype = bezier.dtype
    ctrl = bezier.control_points
    degrees = bezier.degree

    if isinstance(pts, PL):
        return _evaluate_bezier_nd_lattice(bezier, ctrl, pts, degrees, dtype, dim, out)
    return _evaluate_bezier_nd_pts_array(bezier, ctrl, pts, degrees, dtype, dim, out)


def _evaluate_bezier_nd_pts_array(  # noqa: PLR0913
    bezier: Bezier,
    ctrl: npt.NDArray[np.float32 | np.float64],
    pts: npt.NDArray[np.float32 | np.float64],
    degrees: tuple[int, ...],
    dtype: npt.DTypeLike,
    dim: int,
    out: npt.NDArray[np.float32 | np.float64] | None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate nD Bézier at arbitrary points using sequential einsum contraction.

    Args:
        bezier (~pantr.bezier.Bezier): The Bézier.
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points.
        pts (npt.NDArray[np.float32 | np.float64]): Points of shape
            ``(n_pts, dim)``.
        degrees (tuple[int, ...]): Polynomial degrees per direction.
        dtype (npt.DTypeLike): Floating-point dtype.
        dim (int): Parametric dimension.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Values of shape ``(n_pts,)`` or
        ``(n_pts, rank)``.

    Note:
        Inputs are assumed to be partially validated by the caller.
    """
    pts = np.asarray(pts)
    if pts.ndim != 2 or pts.shape[1] != dim:  # noqa: PLR2004
        raise ValueError(f"pts must be a 2D array with {dim} columns.")
    if pts.dtype != dtype:
        raise ValueError(f"Points dtype ({pts.dtype}) must match Bézier dtype ({dtype}).")

    # Evaluate Bernstein basis per direction
    bases: list[npt.NDArray[np.float32 | np.float64]] = [
        _tabulate_bernstein_1d_fast(degrees[d], pts[:, d], dtype) for d in range(dim)
    ]

    # Sequential contraction
    # After contracting direction 0: result has shape (n_pts, n_1, ..., n_{d-1}, rank)
    # After contracting direction 1: result has shape (n_pts, n_2, ..., n_{d-1}, rank)
    # etc.
    result: npt.NDArray[np.float32 | np.float64] = np.einsum("pi,i...->p...", bases[0], ctrl)
    for d in range(1, dim):
        result = np.einsum("pj,pj...->p...", bases[d], result)

    result = _project_rational(bezier, result)

    if out is not None:
        _validate_out_array(out, result.shape, dtype)
        out[:] = result
        return out
    return result


def _evaluate_bezier_nd_lattice(  # noqa: PLR0913
    bezier: Bezier,
    ctrl: npt.NDArray[np.float32 | np.float64],
    pts: PointsLattice,
    degrees: tuple[int, ...],
    dtype: npt.DTypeLike,
    dim: int,
    out: npt.NDArray[np.float32 | np.float64] | None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate nD Bézier on a lattice using sequential tensordot contraction.

    Args:
        bezier (~pantr.bezier.Bezier): The Bézier.
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points.
        pts (~pantr.quad.PointsLattice): Lattice of evaluation points.
        degrees (tuple[int, ...]): Polynomial degrees per direction.
        dtype (npt.DTypeLike): Floating-point dtype.
        dim (int): Parametric dimension.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Values with shape
        ``(*grid_shape,)`` or ``(*grid_shape, rank)``.

    Note:
        Inputs are assumed to be partially validated by the caller.
    """
    if pts.dim != dim:
        raise ValueError(f"PointsLattice dim ({pts.dim}) must match Bézier dim ({dim}).")

    # Evaluate Bernstein basis per direction on lattice points
    bases: list[npt.NDArray[np.float32 | np.float64]] = []
    for d in range(dim):
        pts_d = pts.pts_per_dir[d]
        if pts_d.dtype != dtype:
            raise ValueError(
                f"PointsLattice dtype ({pts_d.dtype}) must match Bézier dtype ({dtype})."
            )
        bases.append(_tabulate_bernstein_1d_fast(degrees[d], pts_d, dtype))

    # Sequential tensordot contraction
    # ctrl has shape (n_0, n_1, ..., n_{d-1}, rank)
    # basis_d has shape (m_d, n_d)
    # Contract axis d of result with axis 1 of basis_d
    result: npt.NDArray[np.float32 | np.float64] = ctrl
    for d in range(dim):
        # Contract axis d with basis_d: (m_d, n_d) @ axis d of result
        result = np.tensordot(bases[d], result, axes=([1], [d]))
        # tensordot puts the m_d axis first; move it to position d
        result = np.moveaxis(result, 0, d)

    result = _project_rational(bezier, result)

    if out is not None:
        _validate_out_array(out, result.shape, dtype)
        out[:] = result
        return out
    return result


def _project_rational(
    bezier: Bezier,
    raw: npt.NDArray[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    """Project rational output by dividing by weights and squeezing scalars.

    Args:
        bezier (~pantr.bezier.Bezier): The Bézier (used for ``is_rational`` and
            ``rank`` checks).
        raw (npt.NDArray[np.float32 | np.float64]): Raw evaluation output with
            shape ``(..., cp_size)``.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Projected result with shape
        ``(...)`` for scalar or ``(..., rank)`` for vector output.
    """
    if bezier.is_rational:
        raw[..., :-1] = raw[..., :-1] / raw[..., -1:]
        result = raw[..., :-1]
    else:
        result = raw

    if result.shape[-1] == 1:
        return result[..., 0]
    return result


# ---------------------------------------------------------------------------
# evaluate_derivatives
# ---------------------------------------------------------------------------


def _evaluate_bezier_deriv(
    bezier: Bezier,
    pts: npt.NDArray[np.float32 | np.float64] | PointsLattice,
    orders: Sequence[int],
    out: npt.NDArray[np.float32 | np.float64] | None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate partial derivatives of a Bézier.

    Dispatches to the 1D fused kernel or the nD contraction path.

    Args:
        bezier (~pantr.bezier.Bezier): The Bézier to differentiate.
        pts (npt.NDArray[np.float32 | np.float64] | PointsLattice): Parametric
            evaluation points.
        orders (Sequence[int]): Derivative order per parametric direction.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Derivative values at the given
        points.

    Raises:
        ValueError: If ``len(orders) != bezier.dim`` or any order is negative.
    """
    orders_tuple = tuple(orders)
    if len(orders_tuple) != bezier.dim:
        raise ValueError(f"len(orders) ({len(orders_tuple)}) must match dim ({bezier.dim}).")
    if any(o < 0 for o in orders_tuple):
        raise ValueError("All derivative orders must be non-negative.")

    if bezier.dim == 1:
        return _evaluate_bezier_deriv_1d(bezier, pts, orders_tuple[0], out)
    return _evaluate_bezier_deriv_nd(bezier, pts, orders_tuple, out)


def _evaluate_bezier_deriv_1d(
    bezier: Bezier,
    pts: npt.NDArray[np.float32 | np.float64] | PointsLattice,
    n_deriv: int,
    out: npt.NDArray[np.float32 | np.float64] | None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate a specific derivative of a 1D Bézier.

    For non-rational Bézier: uses the fused derivative kernel directly.
    For rational Bézier: computes all homogeneous derivatives 0..n_deriv, then
    applies Algorithm A4.2 (quotient rule) from Piegl & Tiller.

    Args:
        bezier (~pantr.bezier.Bezier): A 1D Bézier.
        pts (npt.NDArray[np.float32 | np.float64] | PointsLattice): Evaluation
            points.
        n_deriv (int): Derivative order (>= 0).
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Derivative values of shape
        ``(n_pts,)`` or ``(n_pts, rank)``.
    """
    from ..quad import PointsLattice as PL  # noqa: PLC0415

    dtype = bezier.dtype
    ctrl = bezier.control_points

    if isinstance(pts, PL):
        if pts.dim != 1:
            raise ValueError("PointsLattice must be 1D for a 1D Bézier.")
        pts_array = pts.pts_per_dir[0]
    else:
        pts_array = np.asarray(pts)
    if pts_array.ndim != 1:
        raise ValueError("Points must be a 1D array for a 1D Bézier.")
    if pts_array.dtype != dtype:
        raise ValueError(f"Points dtype ({pts_array.dtype}) must match Bézier dtype ({dtype}).")

    n_pts = pts_array.shape[0]
    cp_size = ctrl.shape[-1]

    if bezier.is_rational:
        return _evaluate_bezier_deriv_1d_rational(
            ctrl, pts_array, n_deriv, n_pts, cp_size, dtype, out
        )
    return _evaluate_bezier_deriv_1d_non_rational(
        ctrl, pts_array, n_deriv, n_pts, cp_size, dtype, out
    )


def _evaluate_bezier_deriv_1d_non_rational(  # noqa: PLR0913
    ctrl: npt.NDArray[np.float32 | np.float64],
    pts_array: npt.NDArray[np.float32 | np.float64],
    n_deriv: int,
    n_pts: int,
    cp_size: int,
    dtype: npt.DTypeLike,
    out: npt.NDArray[np.float32 | np.float64] | None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate the n_deriv-th derivative of a non-rational 1D Bézier.

    Args:
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points ``(p+1, rank)``.
        pts_array (npt.NDArray[np.float32 | np.float64]): Evaluation points.
        n_deriv (int): Derivative order.
        n_pts (int): Number of points.
        cp_size (int): Number of control point components.
        dtype (npt.DTypeLike): Floating-point dtype.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Derivative values.
    """
    deriv_all = np.empty((n_pts, n_deriv + 1, cp_size), dtype=dtype)
    _evaluate_bezier_deriv_1d_core(ctrl, pts_array, n_deriv, deriv_all)
    result = deriv_all[:, n_deriv, :]

    if cp_size == 1:
        result = result[:, 0]

    if out is not None:
        _validate_out_array(out, result.shape, dtype)
        out[:] = result
        return out
    return result


def _evaluate_bezier_deriv_1d_rational(  # noqa: PLR0913
    ctrl: npt.NDArray[np.float32 | np.float64],
    pts_array: npt.NDArray[np.float32 | np.float64],
    n_deriv: int,
    n_pts: int,
    cp_size: int,
    dtype: npt.DTypeLike,
    out: npt.NDArray[np.float32 | np.float64] | None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Apply the quotient rule for the n_deriv-th derivative of a rational 1D Bézier.

    Uses Algorithm A4.2 from Piegl & Tiller.

    Args:
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points ``(p+1, rank+1)``.
        pts_array (npt.NDArray[np.float32 | np.float64]): Evaluation points.
        n_deriv (int): Derivative order.
        n_pts (int): Number of points.
        cp_size (int): Number of control point components (rank + 1).
        dtype (npt.DTypeLike): Floating-point dtype.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Derivative values of the
        projected (physical) mapping.
    """
    rank = cp_size - 1

    # Compute all homogeneous derivatives 0..n_deriv
    hom_all = np.empty((n_pts, n_deriv + 1, cp_size), dtype=dtype)
    _evaluate_bezier_deriv_1d_core(ctrl, pts_array, n_deriv, hom_all)

    # Algorithm A4.2 quotient rule
    W = hom_all[:, 0, -1:]  # (n_pts, 1)
    results: list[npt.NDArray[np.float32 | np.float64]] = []
    for k in range(n_deriv + 1):
        v: npt.NDArray[np.float32 | np.float64] = hom_all[:, k, :-1].copy()
        for i in range(1, k + 1):
            v = v - math.comb(k, i) * hom_all[:, i, -1:] * results[k - i]
        results.append(v / W)

    final = results[n_deriv]
    if rank == 1:
        final = final[:, 0]

    if out is not None:
        _validate_out_array(out, final.shape, dtype)
        out[:] = final
        return out
    return final


# ---------------------------------------------------------------------------
# nD derivative evaluation
# ---------------------------------------------------------------------------


def _evaluate_bezier_deriv_nd(
    bezier: Bezier,
    pts: npt.NDArray[np.float32 | np.float64] | PointsLattice,
    orders: tuple[int, ...],
    out: npt.NDArray[np.float32 | np.float64] | None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate partial derivatives of an nD Bézier.

    For non-rational: evaluates the derivative basis in each direction and
    contracts with control points.
    For rational: computes all homogeneous partial derivatives up to the
    requested order, then applies the generalised quotient rule.

    Args:
        bezier (~pantr.bezier.Bezier): An nD Bézier.
        pts (npt.NDArray[np.float32 | np.float64] | PointsLattice): Evaluation
            points.
        orders (tuple[int, ...]): Derivative order per direction.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Partial derivative values.
    """
    from ..quad import PointsLattice as PL  # noqa: PLC0415

    dim = bezier.dim
    dtype = bezier.dtype
    ctrl = bezier.control_points
    degrees = bezier.degree

    # --- Resolve points ---
    is_lattice = isinstance(pts, PL)
    if is_lattice:
        assert isinstance(pts, PL)
        if pts.dim != dim:
            raise ValueError(f"PointsLattice dim ({pts.dim}) must match Bézier dim ({dim}).")
        pts_per_dir: list[npt.NDArray[np.float32 | np.float64]] = list(pts.pts_per_dir)
        for d in range(dim):
            if pts_per_dir[d].dtype != dtype:
                raise ValueError(
                    f"PointsLattice dtype ({pts_per_dir[d].dtype}) must match "
                    f"Bézier dtype ({dtype})."
                )
    else:
        pts_arr = np.asarray(pts)
        if pts_arr.ndim != 2 or pts_arr.shape[1] != dim:  # noqa: PLR2004
            raise ValueError(f"pts must be a 2D array with {dim} columns.")
        if pts_arr.dtype != dtype:
            raise ValueError(f"Points dtype ({pts_arr.dtype}) must match Bézier dtype ({dtype}).")
        pts_per_dir = [pts_arr[:, d] for d in range(dim)]

    if bezier.is_rational:
        return _evaluate_bezier_deriv_nd_rational(
            bezier, ctrl, pts_per_dir, orders, degrees, dtype, dim, is_lattice, out
        )
    return _evaluate_bezier_deriv_nd_non_rational(
        bezier, ctrl, pts_per_dir, orders, degrees, dtype, dim, is_lattice, out
    )


def _evaluate_bezier_deriv_nd_non_rational(  # noqa: PLR0913
    bezier: Bezier,
    ctrl: npt.NDArray[np.float32 | np.float64],
    pts_per_dir: list[npt.NDArray[np.float32 | np.float64]],
    orders: tuple[int, ...],
    degrees: tuple[int, ...],
    dtype: npt.DTypeLike,
    dim: int,
    is_lattice: bool,
    out: npt.NDArray[np.float32 | np.float64] | None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate partial derivatives of a non-rational nD Bézier.

    For each parametric direction ``d``, evaluates the ``orders[d]``-th
    derivative of the Bernstein basis and contracts with the control points
    via sequential contraction.

    Args:
        bezier (~pantr.bezier.Bezier): The Bézier.
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points.
        pts_per_dir (list[npt.NDArray[np.float32 | np.float64]]): Points arrays
            per direction.
        orders (tuple[int, ...]): Derivative order per direction.
        degrees (tuple[int, ...]): Polynomial degrees per direction.
        dtype (npt.DTypeLike): Floating-point dtype.
        dim (int): Parametric dimension.
        is_lattice (bool): Whether the points come from a lattice.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Partial derivative values.
    """
    from ..basis._basis_core import (  # noqa: PLC0415
        _tabulate_Bernstein_basis_deriv_1D_core,
    )

    # Evaluate derivative bases per direction
    bases: list[npt.NDArray[np.float32 | np.float64]] = []
    for d in range(dim):
        pts_d = pts_per_dir[d]
        n_pts_d = pts_d.shape[0]
        order_d = orders[d]
        p_d = degrees[d]

        if order_d == 0:
            basis_d = _tabulate_bernstein_1d_fast(p_d, pts_d, dtype)
        else:
            # Derivative basis
            deriv_all_d = np.empty((n_pts_d, order_d + 1, p_d + 1), dtype=dtype)
            _tabulate_Bernstein_basis_deriv_1D_core(np.int32(p_d), pts_d, order_d, deriv_all_d)
            basis_d = deriv_all_d[:, order_d, :]
        bases.append(basis_d)

    # Sequential contraction
    if is_lattice:
        result: npt.NDArray[np.float32 | np.float64] = ctrl
        for d in range(dim):
            result = np.tensordot(bases[d], result, axes=([1], [d]))
            result = np.moveaxis(result, 0, d)
    else:
        result = np.einsum("pi,i...->p...", bases[0], ctrl)
        for d in range(1, dim):
            result = np.einsum("pj,pj...->p...", bases[d], result)

    # Squeeze scalar
    if result.shape[-1] == 1:
        result = result[..., 0]

    if out is not None:
        _validate_out_array(out, result.shape, dtype)
        out[:] = result
        return out
    return result


def _evaluate_bezier_deriv_nd_rational(  # noqa: PLR0913, PLR0912
    bezier: Bezier,
    ctrl: npt.NDArray[np.float32 | np.float64],
    pts_per_dir: list[npt.NDArray[np.float32 | np.float64]],
    orders: tuple[int, ...],
    degrees: tuple[int, ...],
    dtype: npt.DTypeLike,
    dim: int,
    is_lattice: bool,
    out: npt.NDArray[np.float32 | np.float64] | None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate partial derivatives of a rational nD Bézier.

    Computes all homogeneous partial derivatives up to the requested order in
    each direction, then applies the generalised quotient rule (Algorithm A4.2
    extended to nD).

    Args:
        bezier (~pantr.bezier.Bezier): The Bézier (rational).
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points.
        pts_per_dir (list[npt.NDArray[np.float32 | np.float64]]): Points per dir.
        orders (tuple[int, ...]): Derivative order per direction.
        degrees (tuple[int, ...]): Polynomial degrees per direction.
        dtype (npt.DTypeLike): Floating-point dtype.
        dim (int): Parametric dimension.
        is_lattice (bool): Whether the points come from a lattice.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Partial derivative values of the
        projected mapping.
    """
    import itertools  # noqa: PLC0415

    from ..basis._basis_core import (  # noqa: PLC0415
        _tabulate_Bernstein_basis_deriv_1D_core,
    )

    cp_size = ctrl.shape[-1]
    rank = cp_size - 1

    # Evaluate all derivative bases per direction up to orders[d]
    all_bases: list[list[npt.NDArray[np.float32 | np.float64]]] = []
    for d in range(dim):
        pts_d = pts_per_dir[d]
        n_pts_d = pts_d.shape[0]
        p_d = degrees[d]
        max_order_d = orders[d]

        if max_order_d == 0:
            all_bases.append([_tabulate_bernstein_1d_fast(p_d, pts_d, dtype)])
        else:
            deriv_all_d = np.empty((n_pts_d, max_order_d + 1, p_d + 1), dtype=dtype)
            _tabulate_Bernstein_basis_deriv_1D_core(np.int32(p_d), pts_d, max_order_d, deriv_all_d)
            all_bases.append([deriv_all_d[:, k, :] for k in range(max_order_d + 1)])

    # Compute all homogeneous partial derivatives
    hom_all: dict[tuple[int, ...], npt.NDArray[np.float32 | np.float64]] = {}
    for k_idx in itertools.product(*(range(o + 1) for o in orders)):
        # Select bases for this multi-index
        bases_k = [all_bases[d][k_idx[d]] for d in range(dim)]

        # Contract with ctrl
        if is_lattice:
            result_k: npt.NDArray[np.float32 | np.float64] = ctrl
            for d in range(dim):
                result_k = np.tensordot(bases_k[d], result_k, axes=([1], [d]))
                result_k = np.moveaxis(result_k, 0, d)
        else:
            result_k = np.einsum("pi,i...->p...", bases_k[0], ctrl)
            for d in range(1, dim):
                result_k = np.einsum("pj,pj...->p...", bases_k[d], result_k)

        hom_all[k_idx] = result_k

    # Apply generalised quotient rule (A4.2 extended to nD)
    W = hom_all[tuple(0 for _ in range(dim))][..., -1:]
    results: dict[tuple[int, ...], npt.NDArray[np.float32 | np.float64]] = {}

    # Iterate all multi-indices in ascending total order
    for k_idx in itertools.product(*(range(o + 1) for o in orders)):
        v: npt.NDArray[np.float32 | np.float64] = hom_all[k_idx][..., :-1].copy()
        for j_idx in itertools.product(*(range(k + 1) for k in k_idx)):
            if j_idx == tuple(0 for _ in range(dim)):
                continue
            # Check j_idx <= k_idx component-wise
            if any(j > k for j, k in zip(j_idx, k_idx, strict=True)):
                continue
            binom_prod = 1
            for d in range(dim):
                binom_prod *= math.comb(k_idx[d], j_idx[d])
            diff_idx = tuple(k - j for k, j in zip(k_idx, j_idx, strict=True))
            v = v - binom_prod * hom_all[j_idx][..., -1:] * results[diff_idx]
        results[k_idx] = v / W

    final = results[orders]
    if rank == 1:
        final = final[..., 0]

    if out is not None:
        _validate_out_array(out, final.shape, dtype)
        out[:] = final
        return out
    return final
