"""Core B-spline evaluation implementations using the de Boor algorithm.

This module provides low-level routines for evaluating a :class:`Bspline`
at arbitrary parametric points. The main entry point is
:func:`_evaluate_Bspline`, which dispatches to the 1D de Boor kernel for
1D splines. Multi-dimensional evaluation is currently not implemented.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy import typing as npt

from ._basis_utils import _validate_out_array_1D
from ._bspline_knots import (
    _get_Bspline_num_basis_1D_impl,
    _get_last_knot_smaller_equal_impl,
)
from ._numba_compat import nb_jit
from .quad import PointsLattice

if TYPE_CHECKING:
    from .bspline import Bspline
    from .quad import PointsLattice


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _evaluate_Bspline_de_Boor_1D(  # noqa: PLR0913
    control_points: npt.NDArray[np.float32 | np.float64],
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    periodic: bool,
    tol: float,
    pts: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate the B-spline at the given points using de Boor's algorithm.

    Args:
        control_points (npt.NDArray[np.float32 | np.float64]): Control points.
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        periodic (bool): Whether the B-spline is periodic.
        tol (float): Tolerance for numerical comparisons.
        pts (npt.NDArray[np.float32 | np.float64]): Points to evaluate at.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Evaluated values at points.
    """
    n_pts = pts.size
    dtype = knots.dtype
    zero = dtype.type(0.0)
    one = dtype.type(1.0)

    knot_ids = _get_last_knot_smaller_equal_impl(knots, pts)
    num_basis = _get_Bspline_num_basis_1D_impl(knots, degree, periodic, tol)
    order = degree + 1

    d = np.empty((order, control_points.shape[1]), dtype=dtype)

    for i in range(n_pts):
        pt = pts[i]
        k = knot_ids[i]

        # Determine the index of the first control point
        # We clamp to ensure we don't go out of bounds (e.g. at the end of the domain)
        s = np.minimum(k - degree, num_basis - order)

        # Initialize working array d with control points
        for j in range(order):
            idx = s + j
            if periodic:
                idx = idx % num_basis
            d[j, :] = control_points[idx, :]

        # De Boor recursion
        for r in range(1, order):
            for j in range(degree, r - 1, -1):
                # Calculate alpha
                denom_idx1 = s + j + order - r
                denom_idx0 = s + j

                denom = knots[denom_idx1] - knots[denom_idx0]
                numer = pt - knots[denom_idx0]

                alpha = zero if denom < tol else numer / denom

                d[j, :] = (one - alpha) * d[j - 1, :] + alpha * d[j, :]

        out[i, :] = d[degree, :]

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
    # For rational splines, we need an extra column for the weight
    n_cols = spline.control_points.shape[-1]
    expected_shape = (n_pts, n_cols)
    expected_dtype = spline.dtype

    # Allocate output array if not provided
    out_array: npt.NDArray[np.float32 | np.float64]
    if out is None:
        out_array = np.empty(expected_shape, dtype=expected_dtype)
    else:
        _validate_out_array_1D(out, expected_shape, expected_dtype)
        out_array = out

    spline_1D = spline.space.spaces[0]

    _evaluate_Bspline_de_Boor_1D(
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


def _evaluate_Bspline_multi_dim(
    spline: Bspline,
    pts: npt.NDArray[np.float32 | np.float64] | PointsLattice,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate the multi-dimensional B-spline at the given points.

    Args:
        spline (Bspline): A multi-dimensional B-spline object.
        pts (npt.NDArray[np.float32 | np.float64] | PointsLattice): Evaluation
            points.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output
            array. Defaults to None.

    Raises:
        NotImplementedError: Always raised; multi-dimensional evaluation is not
            yet implemented.
    """
    raise NotImplementedError("Not implemented")


def _evaluate_Bspline(
    spline: Bspline,
    pts: npt.NDArray[np.float32 | np.float64] | PointsLattice,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate the B-spline basis functions at the given points.

    Args:
        spline (BsplineSpace1D): The B-spline space.
        pts (npt.NDArray[np.float32 | np.float64]): The points at which to evaluate the
            B-spline basis functions.
        out (npt.NDArray[np.float32 | np.float64] | None): The output array.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The B-spline values at the given points.
    """
    if spline.dim == 1:
        return _evaluate_Bspline_1D(spline, pts, out)
    else:
        return _evaluate_Bspline_multi_dim(spline, pts, out)


def _warmup_numba_functions() -> None:
    """Precompile Numba functions with float64 signatures for faster first call.

    Triggers compilation of the de Boor evaluation kernel with representative
    float64 arrays. The compiled code is cached by Numba (``cache=True``) so
    subsequent cold-start calls do not pay JIT overhead.
    """
    knots_dummy = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    pts_dummy = np.array([0.5], dtype=np.float64)
    cp_dummy = np.array([[0.0], [1.0], [0.0]], dtype=np.float64)
    tol_dummy = 1e-10
    degree_dummy = 2
    out_dummy = np.empty((1, 1), dtype=np.float64)

    _evaluate_Bspline_de_Boor_1D(
        cp_dummy, knots_dummy, degree_dummy, False, tol_dummy, pts_dummy, out_dummy
    )


# Precompile numba functions on module import
# (Moved to central thread in __init__.py)
