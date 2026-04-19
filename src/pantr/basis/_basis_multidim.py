"""Multidimensional basis function evaluation.

This module provides functions for combining 1D basis function evaluations
into multidimensional tensor-product bases.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import numpy.typing as npt

from ..quad import PointsLattice
from ._basis_utils import (
    _compute_output_shape_multidimensional,
    _validate_out_array,
)


def _compute_basis_combinator_matrix_for_points_lattice(
    evaluators_1D: tuple[Callable[[npt.ArrayLike], npt.NDArray[np.float32 | np.float64]]],
    pts: PointsLattice,
    order: Literal["C", "F"] = "C",
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Combine 1D basis functions evaluated at a points lattice.

    This function efficiently evaluates a set of 1D basis functions at a tensor-product grid
    of points, exploiting the structured nature of the grid to minimize redundant computation.
    It handles both C-order (last index varies fastest) and F-order (first index varies fastest)
    meshgrid layouts, and correctly combines the results according to the specified function
    and point orderings.

    Args:
        evaluators_1D (tuple[Callable[[npt.ArrayLike], npt.NDArray[np.float32 | np.float64]]]):
            Tuple of 1D basis function evaluators.
            Each evaluator takes a 1D array of points and returns an array of basis function values.
        pts (PointsLattice): Points lattice.
        order (Literal["C", "F"]): Ordering of the basis functions and points:
            'F' for Fortran-order (first index varies fastest),
            'C' for C-order (last index varies fastest). Default is 'C'.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype if provided. This follows NumPy's
            style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Array of shape (n_points, n_basis_functions)
        containing the combined basis function values. Both dimensions (points and basis functions)
        are ordered according to the specified order. If `out` was provided, returns the same array.

    Raises:
        ValueError: If the dimension of the points is less than 1,
        or the number of evaluators is not equal to the dimension of the points,
        or if `out` is provided and has incorrect shape or dtype.
    """
    dim = pts.dim
    if dim != len(evaluators_1D):
        raise ValueError("The number of evaluators must be equal to the dimension of the points.")

    pts_per_dir = pts.pts_per_dir

    # Compute the expected output shape and evaluate all dimensions
    # We'll reuse these evaluations in the computation below
    vals_per_dim: list[npt.NDArray[np.float32 | np.float64]] = []
    n_basis_per_dim: list[int] = []
    n_pts_per_dim: list[int] = []

    for dir in range(dim):
        vals_1D = evaluators_1D[dir](pts_per_dir[dir])
        vals_per_dim.append(vals_1D)
        n_basis_per_dim.append(vals_1D.shape[1])
        n_pts_per_dim.append(vals_1D.shape[0])

    n_points = int(np.prod(n_pts_per_dim))
    n_basis_functions = int(np.prod(n_basis_per_dim))
    expected_shape = _compute_output_shape_multidimensional(n_points, n_basis_functions)

    # Determine dtype from the points lattice
    expected_dtype = pts.dtype

    # Validate or allocate output array
    if out is None:
        out = np.empty(expected_shape, dtype=expected_dtype)
    else:
        _validate_out_array(out, expected_shape, expected_dtype)

    # Same ordering (C or F) is used for both points and functions.
    op_str = "pi,qj->qpji" if order == "F" else "pi,qj->pqij"

    # Handle the 1D case separately
    if dim == 1:
        out[:] = vals_per_dim[0]
    else:
        # Use the pre-computed evaluations
        current = vals_per_dim[0].copy()  # We need to modify this, so make a copy

        for dir in range(1, dim - 1):
            vals_1D = vals_per_dim[dir]
            n_rows = current.shape[0] * vals_1D.shape[0]
            n_cols = current.shape[1] * vals_1D.shape[1]
            # einsum produces (n_pts_0, n_pts_1, n_basis_0, n_basis_1)
            # or (n_pts_1, n_pts_0, n_basis_1, n_basis_0).
            # This is then reshaped to (n_pts_total, n_basis_total) according to the order.
            current = np.einsum(op_str, current, vals_1D).reshape(n_rows, n_cols)

        # Last iteration: write directly to out to avoid copy
        # Reshape out to the 4D shape that einsum expects
        vals_1D = vals_per_dim[dim - 1]
        final_4d_shape = (
            (current.shape[0], vals_1D.shape[0], current.shape[1], vals_1D.shape[1])
            if order == "C"
            else (vals_1D.shape[0], current.shape[0], vals_1D.shape[1], current.shape[1])
        )
        np.einsum(op_str, current, vals_1D, out=out.reshape(final_4d_shape))

    return out


def _compute_basis_combinator_matrix_for_points_array(
    evaluators_1D: tuple[Callable[[npt.ArrayLike], npt.NDArray[np.float32 | np.float64]]],
    pts: npt.ArrayLike,
    funcs_order: Literal["C", "F"] = "C",
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate and combine 1D basis functions at a collection of points.

    Args:
        evaluators_1D (tuple[Callable[[npt.ArrayLike], npt.NDArray[np.float32 | np.float64]]]):
            Tuple of 1D basis function evaluators.
            Each evaluator takes a 1D array of points and returns an array of basis function values.
        pts (npt.ArrayLike): Points to evaluate the basis functions at.
        funcs_order (Literal["C", "F"]): Ordering of the basis functions: 'F' for Fortran-order
            (first index varies fastest) or 'C' for C-order (last index varies fastest).
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype if provided. This follows NumPy's
            style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Array of shape (n_points, n_basis_functions)
        containing the combined basis function values. If `out` was provided,
        returns the same array.

    Raises:
        ValueError: If the dimension of the points is not 2,
        the dtype of the points is not float32 or float64,
        or the number of evaluators is less than 1,
        or is not equal to the dimension of the points,
        or if `out` is provided and has incorrect shape or dtype.
    """
    if not isinstance(pts, np.ndarray):
        pts = np.array(pts)

    if pts.ndim != 2:  # noqa: PLR2004
        raise ValueError("Input points must be a 2D array.")

    if pts.dtype not in (np.float32, np.float64):
        pts = pts.astype(np.float64)

    dim = pts.shape[1]
    if dim < 1:
        raise ValueError("The dimension of the points must be at least 1.")
    if len(evaluators_1D) != dim:
        raise ValueError("The number of evaluators must be equal to the dimension of the points.")

    # Compute the expected output shape and evaluate all dimensions
    # We'll reuse these evaluations in the computation below
    n_pts = pts.shape[0]
    vals_per_dim: list[npt.NDArray[np.float32 | np.float64]] = []
    n_basis_per_dim: list[int] = []

    for dir in range(dim):
        vals_1D = evaluators_1D[dir](pts[:, dir])
        vals_per_dim.append(vals_1D)
        n_basis_per_dim.append(vals_1D.shape[1])

    n_basis_functions = int(np.prod(n_basis_per_dim))
    expected_shape = _compute_output_shape_multidimensional(n_pts, n_basis_functions)
    expected_dtype = pts.dtype

    # Validate or allocate output array
    if out is None:
        out = np.empty(expected_shape, dtype=expected_dtype)
    else:
        _validate_out_array(out, expected_shape, expected_dtype)

    # Handle the 1D case separately
    if dim == 1:
        out[:] = vals_per_dim[0]
    else:
        funcs_order_str = "ji" if funcs_order == "F" else "ij"
        op_str = f"pi,pj->p{funcs_order_str}"

        # Use the pre-computed evaluations
        current = vals_per_dim[0].copy()  # We need to modify this, so make a copy
        for dir in range(1, dim - 1):
            vals_1D = vals_per_dim[dir]
            current = np.einsum(op_str, current, vals_1D).reshape(n_pts, -1)

        # Last iteration: write directly to out to avoid copy
        # Reshape out to the 3D shape that einsum expects
        current_n_basis = current.shape[1]
        n_basis_1D = vals_per_dim[dim - 1].shape[1]
        final_shape = (
            (n_pts, current_n_basis, n_basis_1D)
            if funcs_order_str == "ij"
            else (n_pts, n_basis_1D, current_n_basis)
        )
        np.einsum(op_str, current, vals_per_dim[dim - 1], out=out.reshape(final_shape))

    return out


def _compute_basis_1D_combinator_matrix(
    evaluators_1D: tuple[Callable[[npt.ArrayLike], npt.NDArray[np.float32 | np.float64]]],
    pts: npt.ArrayLike | PointsLattice,
    order: Literal["C", "F"] = "C",
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Combine 1D basis function evaluations into a multidimensional tensor-product basis.

    Evaluates and combines a tuple of 1D basis function evaluators at the given points,
    supporting both general arrays
    of points and tensor-product grids (points lattices).
    Enforces the specified ordering (C or F) for the basis functions
    (and points in the case of a points lattice).

    Args:
        evaluators_1D (tuple[Callable[[npt.ArrayLike], npt.NDArray[np.float32 | np.float64]]]):
            Tuple of 1D basis function evaluators.
            Each evaluator takes a 1D array of points and returns an array of basis function values.
        pts (npt.ArrayLike | PointsLattice): Points to evaluate the basis functions at.
            It may be a general 2D numpy array of points (where each row is a point)
            or a points lattice.
        order (str): Ordering of the basis functions (and points in the case of a points lattice):
            'F' for Fortran-order (first index varies fastest),
            'C' for C-order (last index varies fastest). Defaults to 'C'.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype if provided. This follows NumPy's
            style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Array of shape (n_points, n_basis_functions)
        containing the combined basis function values.
        Functions are ordered according to the specified order.
        In the case of a points lattice, points are also ordered according to the specified order.
        If `out` was provided, returns the same array.

    Raises:
        ValueError: If the dimension of the points is not 2,
        the dtype of the points is not float32 or float64,
        or the number of evaluators is less than 1,
        or is not equal to the dimension of the points,
        or if `out` is provided and has incorrect shape or dtype.
    """
    if isinstance(pts, PointsLattice):
        return _compute_basis_combinator_matrix_for_points_lattice(evaluators_1D, pts, order, out)
    else:
        return _compute_basis_combinator_matrix_for_points_array(evaluators_1D, pts, order, out)


__all__ = ["_compute_basis_1D_combinator_matrix"]
