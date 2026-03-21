"""Multidimensional B-spline basis function evaluation.

This module provides functions for evaluating B-spline basis functions
in multiple dimensions using tensor products of 1D bases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from .._basis_utils import (
    _validate_out_array_1D,
    _validate_out_array_first_basis,
)
from ..quad import PointsLattice
from ._bspline_knots import _is_in_domain_impl

if TYPE_CHECKING:
    from ._bspline_space_nd import BsplineSpace


def _validate_and_check_points_in_domain(
    spline: BsplineSpace,
    pts: npt.NDArray[np.float32 | np.float64] | PointsLattice,
) -> None:
    """Validate points array/lattice structure and check all points are in domain.

    This function validates that the points structure matches the spline dimensions
    and checks that all points are within the B-spline domain for each direction.

    Args:
        spline (BsplineSpace): B-spline object defining the basis.
        pts (npt.NDArray[np.float32 | np.float64] | PointsLattice): Evaluation points.
            Must be a 2D array with shape (num_pts, dim) or a PointsLattice object.

    Raises:
        ValueError: If pts is not a 2D array (for arrays), if pts dimension does not
            match spline dimension, or if any point is outside the knot vector domain.
    """
    if isinstance(pts, PointsLattice):
        if pts.dim != spline.dim:
            raise ValueError(f"pts must have {spline.dim} columns")
        for dir in range(spline.dim):
            if not np.all(
                _is_in_domain_impl(
                    spline.spaces[dir].knots,
                    spline.spaces[dir].degree,
                    pts.pts_per_dir[dir],
                    spline.spaces[dir].tolerance,
                )
            ):
                raise ValueError(
                    f"One or more values in pts.pts_per_dir[{dir}] are outside the knot vector"
                    f" domain {spline.spaces[dir].domain}"
                )
    else:
        if pts.ndim != 2:  # noqa: PLR2004
            raise ValueError("pts must be a 2D array")
        if pts.shape[1] != spline.dim:
            raise ValueError(f"pts must have {spline.dim} columns")

        splines_1D = spline.spaces
        for dir in range(spline.dim):
            spline_1D = splines_1D[dir]
            pts_dir = np.ascontiguousarray(pts[:, dir])
            if not np.all(
                _is_in_domain_impl(spline_1D.knots, spline_1D.degree, pts_dir, spline_1D.tolerance)
            ):
                raise ValueError(
                    f"One or more values in pts[:, {dir}] are outside the knot vector"
                    f" domain {spline.domain}"
                )


def _multiply_with_conditional_output(
    B_left: npt.NDArray[np.float32 | np.float64],
    B_right: npt.NDArray[np.float32 | np.float64],
    is_last: bool,
    out_basis: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Multiply two arrays with conditional use of output array.

    This helper function multiplies two arrays, using the provided output array
    only if it's the last iteration and the output array is provided.

    Args:
        B_left (npt.NDArray[np.float32 | np.float64]): Left operand for multiplication.
        B_right (npt.NDArray[np.float32 | np.float64]): Right operand for multiplication.
        is_last (bool): Whether this is the last multiplication operation.
        out_basis (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. Used only if `is_last` is True.
            Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The result of multiplication. If `out_basis`
            was provided and `is_last` is True, returns the same array.
    """
    if is_last and out_basis is not None:
        np.multiply(B_left, B_right, out=out_basis)
        return out_basis
    return np.multiply(B_left, B_right)


def _tabulate_Bspline_basis_for_points_array_impl(
    spline: BsplineSpace,
    pts: npt.NDArray[np.float32 | np.float64],
    out_basis: npt.NDArray[np.float32 | np.float64] | None = None,
    out_first_basis: npt.NDArray[np.int_] | None = None,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    """Evaluate multi-dimensional B-spline basis functions at given points.

    Args:
        spline (BsplineSpace): B-spline object defining the basis.
        pts (npt.NDArray[np.float32 | np.float64]): Evaluation points.
            Must be a 2D array with shape (num_pts, dim).
        out_basis (npt.NDArray[np.float32 | np.float64] | None): Optional output array where the
            basis values will be stored. If None, a new array is allocated. Must have the
            correct shape and dtype if provided. This follows NumPy's style for output arrays.
            Defaults to None.
        out_first_basis (npt.NDArray[np.int_] | None): Optional output array where the
            first basis indices will be stored. If None, a new array is allocated. Must have
            the correct shape (num_pts, dim) and dtype np.int_ if provided. This follows NumPy's
            style for output arrays. Defaults to None.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]: Tuple containing:
            - basis_values: (npt.NDArray[np.float32 | np.float64])
              Array of shape (num_pts, order[0], order[1], ..., order[d-1])
              containing the basis function values evaluated at each point.
              If `out_basis` was provided, returns the same array.
            - first_basis_indices: (npt.NDArray[np.int_])
              2D integer array indicating the index of the first nonzero basis function
              for each evaluation point in each direction. The shape is (num_pts, dim).
              If `out_first_basis` was provided, returns the same array.

    Raises:
        ValueError: If pts is not a 2D array or does not have the correct number of columns.
        ValueError: If one or more values in pts are outside the knot vector domain, or
            if `out_basis` or `out_first_basis` is provided and has incorrect shape or dtype.
    """
    _validate_and_check_points_in_domain(spline, pts)

    splines_1D = spline.spaces
    order = tuple(int(degree + 1) for degree in spline.degrees)
    num_pts = pts.shape[0]
    expected_basis_shape = (num_pts, *order)
    expected_dtype = np.dtype(spline.dtype)
    expected_first_basis_shape = (num_pts, spline.dim)

    if out_basis is None:
        out_basis = cast(
            npt.NDArray[np.float32 | np.float64],
            np.empty(expected_basis_shape, dtype=expected_dtype),
        )
    else:
        _validate_out_array_1D(out_basis, expected_basis_shape, expected_dtype)

    if out_first_basis is None:
        out_first_basis = np.empty(expected_first_basis_shape, dtype=np.int_)
    else:
        _validate_out_array_first_basis(out_first_basis, expected_first_basis_shape)

    # Combine 1D basis along each direction using outer product to form the
    # tensor-product multidimensional basis.
    # The multidimensional basis Bs will have shape (num_pts, order[0], order[1], ..., order[d-1])

    # Start with the basis functions of the first direction
    pts_0 = np.ascontiguousarray(pts[:, 0])
    first_out = out_basis if (spline.dim == 1 and out_basis is not None) else None
    B_multi, first_idx_0 = splines_1D[0].tabulate_basis(pts_0, out_basis=first_out)

    out_first_basis[:, 0] = first_idx_0

    for dir in range(1, spline.dim):
        pts_dir = np.ascontiguousarray(pts[:, dir])
        Bdir, first_idx_dir = splines_1D[dir].tabulate_basis(pts_dir)
        out_first_basis[:, dir] = first_idx_dir
        # At each step, expand B_multi to add a new axis at the end,
        # and outer product with the next B_1D
        expanded_dir_shape = (num_pts,) + ((1,) * dir) + (order[dir],)
        Bdir_view = Bdir.reshape(expanded_dir_shape)

        is_last_dir = dir == (spline.dim - 1)
        B_multi = _multiply_with_conditional_output(
            B_multi[..., np.newaxis], Bdir_view, is_last_dir, out_basis
        )

    return out_basis, out_first_basis


def _tabulate_Bspline_basis_for_points_lattice_impl(
    spline: BsplineSpace,
    pts: PointsLattice,
    out_basis: npt.NDArray[np.float32 | np.float64] | None = None,
    out_first_basis: npt.NDArray[np.int_] | None = None,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    """Evaluate B-spline basis functions at points in a lattice structure.

    This function computes the tensor product of the 1D B-spline basis
    functions evaluated over the lattice points defined by 'pts'. The output
    arrays capture the non-zero local basis values and their corresponding
    global starting indices efficiently for the full grid.

    Args:
        spline (BsplineSpace): B-spline object defining the basis (knots, degree, etc.).
        pts (PointsLattice): Evaluation points defined as a tensor product lattice.
        out_basis (npt.NDArray[np.float32 | np.float64] | None): Optional output array where the
            basis values will be stored. If None, a new array is allocated. Must have the
            correct shape and dtype if provided. This follows NumPy's style for output arrays.
            Defaults to None.
        out_first_basis (npt.NDArray[np.int_] | None): Optional output array where the
            first basis indices will be stored. If None, a new array is allocated. Must have
            the correct shape (n_pts_0, n_pts_1, ..., n_pts_d, dim) and dtype np.int_ if
            provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]: Tuple containing:
            - basis_values: (npt.NDArray[np.float32 | np.float64])
              Array of shape (n_pts_0, n_pts_1, ..., n_pts_d, k_0, k_1, ..., k_d)
              where 'n_pts_i' is the number of points in dimension $i$, and $k_i$
              is the number of local non-zero basis functions (typically degree + 1).
              This array contains the tensor product of basis function values evaluated
              at each grid point. If `out_basis` was provided, returns the same array.
            - first_basis_indices: (npt.NDArray[np.int_])
              Array of shape (n_pts_0, n_pts_1, ..., n_pts_d, dim) indicating the global
              index of the first nonzero basis function for each evaluation point in each direction.
              If `out_first_basis` was provided, returns the same array.

    Raises:
        ValueError: If pts dimension does not match spline dimension.
        ValueError: If one or more values in pts are outside the knot vector domain.
        ValueError: If one or more values in pts are outside the corresponding knot vector domain,
            or if `out_basis` or `out_first_basis` is provided and has incorrect shape or dtype.
    """
    _validate_and_check_points_in_domain(spline, pts)

    # 1. Compute 1D components
    results_1d = [s.tabulate_basis(p) for s, p in zip(spline.spaces, pts.pts_per_dir, strict=True)]
    Bs_tuple, first_idxs = zip(*results_1d, strict=True)
    Bs: list[npt.NDArray[np.float32 | np.float64]] = list(Bs_tuple)

    # 2. Combine basis functions using tensor product (Broadcasting approach)
    ndim = spline.dim

    pts_shape = tuple(B.shape[0] for B in Bs)
    order_shape = tuple(B.shape[1] for B in Bs)
    expected_basis_shape = pts_shape + order_shape
    expected_dtype = Bs[0].dtype
    expected_first_basis_shape = (*pts_shape, ndim)

    if out_basis is None:
        out_basis = cast(
            npt.NDArray[np.float32 | np.float64],
            np.empty(expected_basis_shape, dtype=expected_dtype),
        )
    else:
        _validate_out_array_1D(out_basis, expected_basis_shape, expected_dtype)

    if out_first_basis is None:
        out_first_basis = np.empty(expected_first_basis_shape, dtype=np.int_)
    else:
        _validate_out_array_first_basis(out_first_basis, expected_first_basis_shape)

    # 3. Combine first_indices into a multi-dimensional array (Meshgrid approach)
    # Result shape: (n_pts_0, n_pts_1, ..., n_pts_d, dim)
    first_mesh = np.meshgrid(*first_idxs, indexing="ij")
    for axis, grid in enumerate(first_mesh):
        out_first_basis[..., axis] = grid

    # Handle 1D case separately
    first_B = Bs[0]
    if ndim == 1:
        np.copyto(out_basis, first_B.reshape(expected_basis_shape))
        return out_basis, out_first_basis

    # Initialize with first array for multi-dimensional case
    new_shape = [1] * (2 * ndim)
    new_shape[0] = first_B.shape[0]
    new_shape[ndim] = first_B.shape[1]
    B_multi: npt.NDArray[np.float32 | np.float64] = first_B.reshape(new_shape)

    # Multi-dimensional case: iterate through remaining dimensions
    remaining_Bs = list(Bs[1:])
    for i, B in enumerate(remaining_Bs, start=1):
        new_shape = [1] * (2 * ndim)
        new_shape[i] = B.shape[0]
        new_shape[ndim + i] = B.shape[1]
        B_view = B.reshape(new_shape)

        is_last_iteration = i == len(remaining_Bs)
        B_multi = _multiply_with_conditional_output(B_multi, B_view, is_last_iteration, out_basis)

    return out_basis, out_first_basis


def _tabulate_Bspline_basis_impl(
    spline: BsplineSpace,
    pts: npt.NDArray[np.float32 | np.float64] | PointsLattice,
    out_basis: npt.NDArray[np.float32 | np.float64] | None = None,
    out_first_basis: npt.NDArray[np.int_] | None = None,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    """Evaluate multi-dimensional B-spline basis functions at given points.

    Args:
        spline (BsplineSpace): B-spline object defining the basis.
        pts (npt.NDArray[np.float32 | np.float64] | PointsLattice): Evaluation points.
            Must be a 2D array with shape (num_pts, dim) or a PointsLattice object.
        out_basis (npt.NDArray[np.float32 | np.float64] | None): Optional output array where the
            basis values will be stored. If None, a new array is allocated. Must have the
            correct shape and dtype if provided. This follows NumPy's style for output arrays.
            Defaults to None.
        out_first_basis (npt.NDArray[np.int_] | None): Optional output array where the
            first basis indices will be stored. If None, a new array is allocated. Must have
            the correct shape and dtype np.int_ if provided. This follows NumPy's style for
            output arrays. Defaults to None.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]: Tuple containing:
            - basis_values: (npt.NDArray[np.float32 | np.float64])
              Array of shape (num_pts, order[0], order[1], ..., order[d-1])
              containing the basis function values evaluated at each point.
              If `out_basis` was provided, returns the same array.
            - first_basis_indices: (npt.NDArray[np.int_])
              Array of shape (num_pts, dim) or (n_pts_0, n_pts_1, ..., n_pts_d, dim) indicating
              the global index of the first nonzero basis function for each evaluation point.
              If `out_first_basis` was provided, returns the same array.

    Raises:
        ValueError: If pts dimension does not match spline dimension.
        ValueError: If one or more values in pts are outside the knot vector domain.
        ValueError: If one or more values in pts are outside the corresponding knot vector domain,
            or if `out_basis` or `out_first_basis` is provided and has incorrect shape or dtype.
    """
    if isinstance(pts, PointsLattice):
        return _tabulate_Bspline_basis_for_points_lattice_impl(
            spline, pts, out_basis=out_basis, out_first_basis=out_first_basis
        )
    else:
        return _tabulate_Bspline_basis_for_points_array_impl(
            spline, pts, out_basis=out_basis, out_first_basis=out_first_basis
        )


__all__ = ["_tabulate_Bspline_basis_impl"]
