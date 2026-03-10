"""Core B-spline basis function evaluation implementations.

This module provides core functions for evaluating B-spline basis functions
using Cox-de Boor recursion and Bernstein-like evaluation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import numba as nb
import numpy as np
import numpy.typing as npt

from ._basis_1D import _tabulate_Bernstein_basis_1D_impl
from ._basis_utils import (
    _compute_final_output_shape_1D,
    _normalize_points_1D,
    _validate_out_array_1D,
    _validate_out_array_first_basis,
)
from ._bspline_knots import (
    _get_Bspline_num_basis_1D_impl,
    _get_last_knot_smaller_equal_impl,
    _is_in_domain_impl,
)

F = TypeVar("F", bound=Callable[..., Any])

if TYPE_CHECKING:
    from .bspline_space_1D import BsplineSpace1D

    # During type-checking, make the decorator a no-op that preserves types.
    def nb_jit(*args: object, **kwargs: object) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            return func

        return decorator
else:
    # At runtime, use the real Numba decorator.
    nb_jit = nb.jit  # type: ignore[attr-defined]


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _compute_basis_Cox_de_Boor_impl(  # noqa: PLR0913
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    periodic: bool,
    tol: float,
    pts: npt.NDArray[np.float32 | np.float64],
    out_basis: npt.NDArray[np.float32 | np.float64],
    out_first_basis: npt.NDArray[np.int_],
) -> None:
    """Evaluate B-spline basis functions using Cox-de Boor recursion.

    This function implements Algorithm 2.23 from "Spline Methods Draft" by Tom Lyche.
    Results are written directly to the output arrays (C-style).

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        periodic (bool): Whether the B-spline is periodic.
        tol (float): Tolerance for numerical comparisons.
        pts (npt.NDArray[np.float32 | np.float64]): Points (1D array) to evaluate basis
            functions at.
        out_basis (npt.NDArray[np.float32 | np.float64]): Output array for basis values.
            Must have shape (n_pts, degree+1) and dtype matching the `knots` dtype.
        out_first_basis (npt.NDArray[np.int_]): Output array for first basis indices.
            Must have shape (n_pts,) and dtype int.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    # See Spline Methods Draft, by Tom Lychee. Algorithm 2.23

    order = degree + 1
    n_pts = pts.size

    knot_ids = _get_last_knot_smaller_equal_impl(knots, pts)

    dtype = knots.dtype
    zero = dtype.type(0.0)
    one = dtype.type(1.0)

    # Initialize basis array
    out_basis.fill(zero)
    out_basis[:, -1] = one

    # Here we account for the case where the evaluation point
    # coincides with the last knot.
    num_basis = _get_Bspline_num_basis_1D_impl(knots, degree, periodic, tol)
    out_first_basis[:] = np.minimum(knot_ids - degree, num_basis - order)

    for pt_id in range(n_pts):
        knot_id = knot_ids[pt_id]

        if knot_id == (knots.size - 1):
            continue

        pt = pts[pt_id]
        basis_i = out_basis[pt_id, :]
        local_knots = knots[knot_id - degree + 1 : knot_id + order]

        for sub_degree in range(1, order):
            k0, k1 = local_knots[0], local_knots[sub_degree]
            diff = k1 - k0
            inv_diff = zero if diff < tol else one / diff

            for bs_id in range(degree - sub_degree, degree):
                basis_i[bs_id] *= (pt - k0) * inv_diff

                k0, k1 = local_knots[bs_id], local_knots[bs_id + sub_degree]
                diff = k1 - k0
                inv_diff = zero if diff < tol else one / diff

                basis_i[bs_id] += (k1 - pt) * inv_diff * basis_i[bs_id + 1]

            basis_i[-1] *= (pt - k0) * inv_diff


def _tabulate_Bspline_basis_Bernstein_like_1D(
    spline: BsplineSpace1D,
    pts: npt.NDArray[np.float32 | np.float64],
    out_basis: npt.NDArray[np.float32 | np.float64] | None = None,
    out_first_basis: npt.NDArray[np.int_] | None = None,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    """Evaluate B-spline basis functions when they reduce to Bernstein polynomials.

    This function is used when the B-spline has Bézier-like knots, allowing
    direct evaluation using Bernstein basis functions.

    Args:
        spline (BsplineSpace1D): B-spline object with Bézier-like knots.
        pts (npt.NDArray[np.float32 | np.float64]): Evaluation points (already normalized to 1D).
        out_basis (npt.NDArray[np.float32 | np.float64] | None): Optional output array where the
            basis values will be stored. If None, a new array is allocated. Must have the
            correct shape (num_pts, degree+1) and dtype if provided. This follows NumPy's
            style for output arrays. Defaults to None.
        out_first_basis (npt.NDArray[np.int_] | None): Optional output array where the
            first basis indices will be stored. If None, a new array is allocated. Must have
            the correct shape (num_pts,) and dtype np.int_ if provided. This follows NumPy's
            style for output arrays. Defaults to None.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]: Tuple of
            (basis_values, first_basis_indices) where basis_values is an array of shape
            (number pts, degree+1) that contains the Bernstein basis function values and
            first_basis_indices contains the indices of the first non-zero basis function
            for each point. If `out_basis` or `out_first_basis` was provided,
            returns the same array(s).

    Raises:
        ValueError: If the B-spline does not have Bézier-like knots.
        ValueError: If `out_basis` or `out_first_basis` is provided and has incorrect shape
            or dtype.
    """
    if not spline.has_Bezier_like_knots():
        raise ValueError("B-spline does not have Bézier-like knots.")

    # map the points to the reference interval [0, 1]
    k0, k1 = spline.domain
    pts_normalized = (pts - k0) / (k1 - k0)

    num_pts = pts.size
    expected_first_basis_shape = (num_pts,)

    if out_first_basis is None:
        out_first_basis = np.empty(expected_first_basis_shape, dtype=np.int_)
    else:
        _validate_out_array_first_basis(out_first_basis, expected_first_basis_shape)

    # the first basis function is always the 0
    out_first_basis.fill(0)

    # Compute Bernstein basis - pass out_basis directly since pts_normalized is already 1D
    # and _tabulate_Bernstein_basis_1D_impl will handle shape validation
    B = _tabulate_Bernstein_basis_1D_impl(spline.degree, pts_normalized, out=out_basis)

    return B, out_first_basis


def _tabulate_Bspline_basis_1D_impl(
    spline: BsplineSpace1D,
    pts: npt.ArrayLike,
    out_basis: npt.NDArray[np.float32 | np.float64] | None = None,
    out_first_basis: npt.NDArray[np.int_] | None = None,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    """Evaluate B-spline basis functions at given points.

    This function automatically selects the most efficient evaluation method:
    - For Bézier-like knots: direct Bernstein evaluation
    - For general knots: Cox-de Boor recursion

    In both cases it calls vectorized or numba implementations.

    Args:
        spline (BsplineSpace1D): B-spline object defining the basis.
        pts (npt.ArrayLike): Evaluation points.
        out_basis (npt.NDArray[np.float32 | np.float64] | None): Optional output array where the
            basis values will be stored. If None, a new array is allocated. Must have the
            correct shape and dtype if provided. This follows NumPy's style for output arrays.
            Defaults to None.
        out_first_basis (npt.NDArray[np.int_] | None): Optional output array where the
            first basis indices will be stored. If None, a new array is allocated. Must have
            the correct shape and dtype np.int_ if provided. This follows NumPy's style for
            output arrays. Defaults to None.

    Returns:
        tuple[
            npt.NDArray[np.float32] | npt.NDArray[np.float64],
            npt.NDArray[np.int_]
        ]: Tuple containing:
            - basis_values: (npt.NDArray[np.float32] | npt.NDArray[np.float64])
              Array of shape matching `pts` with the last dimension length (degree+1),
              containing the basis function values evaluated at each point.
              If `out_basis` was provided, returns the same array.
            - first_basis_indices: (npt.NDArray[np.int_])
              1D integer array indicating the index of the first nonzero basis function
              for each evaluation point. The length is the same as the number of evaluation points.
              If `out_first_basis` was provided, returns the same array.

    Raises:
        ValueError: If any evaluation points are outside the B-spline domain, or if `out_basis`
            or `out_first_basis` is provided and has incorrect shape or dtype.

    Example:
        >>> bspline = BsplineSpace1D([0, 0, 0, 0.25, 0.7, 0.7, 1, 1, 1], 2)
        >>> _tabulate_Bspline_basis_1D_impl(bspline, [0.0, 0.5, 0.75, 1.0])
        (array([[1.        , 0.        , 0.        ],
                [0.12698413, 0.5643739 , 0.30864198],
                [0.69444444, 0.27777778, 0.02777778],
                [0.        , 0.        , 1.        ]]),
         array([0, 1, 3, 3]))
    """
    input_shape = np.shape(pts)
    pts = _normalize_points_1D(pts)

    if not np.all(_is_in_domain_impl(spline.knots, spline.degree, pts, spline.tolerance)):
        raise ValueError(
            f"One or more values in pts are outside the knot vector domain {spline.domain}"
        )

    num_pts = pts.shape[0]
    n_basis = spline.degree + 1
    expected_final_shape = _compute_final_output_shape_1D(input_shape, n_basis)
    expected_dtype = pts.dtype
    expected_first_basis_shape = input_shape

    if out_basis is None:
        out_basis = np.empty(expected_final_shape, dtype=expected_dtype)
    _validate_out_array_1D(out_basis, expected_final_shape, expected_dtype)
    basis_normalized = out_basis.reshape(num_pts, n_basis)

    if out_first_basis is None:
        out_first_basis = np.empty(expected_first_basis_shape, dtype=np.int_)
    _validate_out_array_first_basis(out_first_basis, expected_first_basis_shape)
    first_indices_normalized = out_first_basis.reshape(num_pts)

    if spline.has_Bezier_like_knots():
        _tabulate_Bspline_basis_Bernstein_like_1D(
            spline, pts, basis_normalized, first_indices_normalized
        )
    else:
        _compute_basis_Cox_de_Boor_impl(
            spline.knots,
            spline.degree,
            spline.periodic,
            spline.tolerance,
            pts,
            basis_normalized,
            first_indices_normalized,
        )

    return out_basis, out_first_basis


def _warmup_numba_functions() -> None:
    """Precompile numba functions with float64 signatures for faster first call.

    This function triggers compilation of the numba-decorated functions
    with float64 arrays, ensuring they are cached and ready for use.
    """
    # Small dummy arrays for warmup
    knots_dummy = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    pts_dummy = np.array([0.5], dtype=np.float64)
    tol_dummy = 1e-10
    degree_dummy = 2
    n_pts_dummy = pts_dummy.size
    basis_dummy = np.empty((n_pts_dummy, degree_dummy + 1), dtype=np.float64)
    first_basis_dummy = np.empty(n_pts_dummy, dtype=np.int_)

    # Warmup Cox-de Boor implementation with float64
    _compute_basis_Cox_de_Boor_impl(
        knots_dummy, degree_dummy, False, tol_dummy, pts_dummy, basis_dummy, first_basis_dummy
    )


# Precompile numba functions on module import
# (Moved to central thread in __init__.py)


__all__ = [
    "_compute_basis_Cox_de_Boor_impl",
    "_tabulate_Bspline_basis_1D_impl",
    "_tabulate_Bspline_basis_Bernstein_like_1D",
]
