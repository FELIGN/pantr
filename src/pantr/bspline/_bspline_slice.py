"""B-spline slicing (dimension reduction by fixing one parametric direction).

This module provides :func:`_slice_bspline`, which fixes one parametric
direction of a B-spline at a given value and returns a B-spline with one
fewer dimension.  For a 1D B-spline the result is a plain NumPy array
(the evaluated point).

The core algorithm is de Boor corner cutting on the control points along
the sliced axis.  When the parameter value coincides with a knot of high
multiplicity, fewer de Boor iterations are needed; at a C0 knot
(multiplicity ``>= p``) the result is obtained in O(1) by direct control
point lookup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from .._array_utils import _flatten_along_axis
from ._bspline_knots import _count_multiplicity, _find_span
from ._bspline_space_nd import BsplineSpace

if TYPE_CHECKING:
    from . import Bspline


def _slice_bspline_1d(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    ctrl: npt.NDArray[np.float32 | np.float64],
    value: float,
) -> npt.NDArray[np.float32 | np.float64]:
    """Slice a 1D B-spline's control points at a parameter value via corner cutting.

    Uses the CurvePntByCornerCut algorithm (Piegl-Tiller A5.1) to evaluate
    the B-spline at a single parameter value.  The result is computed from
    ``p - s + 1`` local control points in ``p - s`` triangular iterations,
    where ``s`` is the knot multiplicity at ``u``.

    When the parameter value coincides with a knot of multiplicity ``s``,
    only ``p - s`` iterations are needed.  At a C0 knot (``s >= p``)
    the control point is returned directly with no iteration.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): Knot vector of length
            ``n_basis + degree + 1``.
        degree (int): Polynomial degree.
        tol (float): Tolerance for knot coincidence tests.
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points of shape
            ``(n_basis, n_cols)``.
        value (float): Parameter value at which to slice.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Result array of shape ``(n_cols,)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_slice_bspline` instead.
    """
    n_basis = ctrl.shape[0]
    k, u = _find_span(knots, degree, n_basis, value, tol)
    s = _count_multiplicity(knots, k, degree, u, tol)

    # C0 optimization: when multiplicity >= degree, only one basis function
    # is non-zero at u (namely B_{k-p}), so the result is that control point.
    if s >= degree:
        return cast(npt.NDArray[np.float32 | np.float64], ctrl[k - degree].copy())

    # CurvePntByCornerCut (Piegl-Tiller A5.1).
    r = degree - s
    R = ctrl[k - degree : k - degree + r + 1].copy()  # shape (r+1, n_cols)

    _zero_denom_tol = 1e-300
    for j in range(1, r + 1):
        for i in range(r - j + 1):
            left_knot = float(knots[k - degree + j + i])
            right_knot = float(knots[i + k + 1])
            denom = right_knot - left_knot
            alpha = 0.0 if abs(denom) < _zero_denom_tol else (u - left_knot) / denom
            R[i] = alpha * R[i + 1] + (1.0 - alpha) * R[i]

    return cast(npt.NDArray[np.float32 | np.float64], R[0])


def _slice_bspline(
    bspline: Bspline,
    axis: int,
    value: float,
) -> Bspline | npt.NDArray[np.float32 | np.float64]:
    """Slice a B-spline by fixing one parametric direction at a given value.

    Reduces the parametric dimension by one.  For a 1D B-spline, returns
    the evaluated point as a NumPy array.  For higher dimensions, returns
    a new :class:`~pantr.bspline.Bspline`.

    Args:
        bspline (~pantr.bspline.Bspline): The B-spline to slice.
        axis (int): Parametric direction to fix (0-indexed, must be in
            ``[0, dim)``).
        value (float): Parameter value at which to slice (must be in the
            domain of the specified direction).

    Returns:
        ~pantr.bspline.Bspline | npt.NDArray[np.float32 | np.float64]:
        A B-spline with ``dim - 1`` dimensions (when ``dim >= 2``),
        or a NumPy array of shape ``(rank,)`` (when ``dim == 1``).
        Rational B-splines preserve the NURBS structure when ``dim >= 2``;
        for ``dim == 1`` the result is projected to physical coordinates.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :meth:`~pantr.bspline.Bspline.slice` instead.
    """
    from . import Bspline as BsplineCls  # noqa: PLC0415

    space_d = bspline.space.spaces[axis]
    knots = space_d.knots
    p = space_d.degree
    tol = float(space_d.tolerance)
    ctrl = bspline.control_points
    is_periodic = space_d.periodic

    # For periodic: expand CPs to full representation along the target axis.
    if is_periodic:
        n_periodic = space_d.num_basis
        n_full = len(knots) - p - 1
        indices = np.arange(n_full) % n_periodic
        ctrl = np.take(ctrl, indices, axis=axis)

    pts_2d, trailing_shape = _flatten_along_axis(ctrl, axis)

    # Apply 1D de Boor corner cutting.
    result_1d = _slice_bspline_1d(knots, p, tol, pts_2d, value)

    if bspline.dim == 1:
        # Result is a point. For rational B-splines, project to physical coords.
        # Control points shape is (n_basis, rank + 1) for rational.
        if bspline.is_rational:
            weight = result_1d[-1]
            return cast(npt.NDArray[np.float32 | np.float64], result_1d[:-1] / weight)
        return result_1d

    # Restore shape: the sliced axis is removed.
    new_ctrl = result_1d.reshape(trailing_shape)

    # Build new spaces list: remove the sliced direction.
    new_spaces = [s for i, s in enumerate(bspline.space.spaces) if i != axis]
    new_space = BsplineSpace(new_spaces)

    return BsplineCls(new_space, new_ctrl, is_rational=bspline.is_rational)
