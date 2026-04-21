"""B-spline extraction operators.

This module provides functions for computing extraction operators that transform
between different basis representations (Bernstein, Lagrange, cardinal B-spline)
and B-spline basis functions.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._numba_compat import nb_jit
from ..basis import LagrangeVariant
from ..basis._basis_utils import _validate_out_array
from ..change_basis import (
    _cached_cardinal_to_bernstein_matrix,
    _cached_lagrange_to_bernstein_matrix,
)
from ._bspline_knots import (
    _check_spline_info,
    _get_Bspline_cardinal_intervals_1D_impl,
    _get_multiplicity_of_first_knot_in_domain_impl,
    _get_unique_knots_and_multiplicity_impl,
)


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _tabulate_Bspline_Bezier_1D_extraction_core(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    r"""Core implementation to compute Bézier extraction operators, writing to output array.

    This function computes the extraction operators that transform Bernstein
    into B-spline basis functions for each interval.
    For each interval \( i \), the Bézier extraction operator \( C_i \) satisfies:

        \[
        N_i(x) = C_i @ B(ξ)
        \]

    where:
      - N_i(x) is the vector of B-spline basis functions nonzero on the interval \( i \),
        evaluated at \( x \),
      - \( B(ξ) \) is the vector of Bernstein basis functions on the reference interval \([0, 1]\),
        evaluated at \( ξ \),
      - \( C_i \) is the extraction matrix for interval \( i \),
      - \( x \) is the physical coordinate, \( ξ \) is the local (reference) referred to \([0, 1]\).

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        tol (float): Tolerance for numerical comparisons.
        out (npt.NDArray[np.float32 | np.float64]): Output array where results will be written.
            Must have the correct shape (n_elems, degree+1, degree+1) and dtype matching knots
            (no validation performed inside this numba-compiled function).

    Note:
        This is a Numba-compiled function optimized for performance. It
        expects pre-validated inputs and assumes the output array has the
        correct shape and dtype. Inputs are assumed to be correct (no validation performed).
        For general use, call _tabulate_Bspline_Bezier_1D_extraction_impl instead.
    """
    unique_knots, mults = _get_unique_knots_and_multiplicity_impl(
        knots, degree, tol, in_domain=True
    )

    n_elems = len(unique_knots) - 1

    dtype = knots.dtype
    one = dtype.type(1.0)

    # Initialize identity matrix for every element.
    out.fill(0.0)
    out[:, : degree + 1, : degree + 1] = np.eye(degree + 1, dtype=dtype)

    mult = _get_multiplicity_of_first_knot_in_domain_impl(knots, degree, tol)

    # If not open first knot, additional knot insertion is needed.
    if mult < (degree + 1):
        C = out[0]
        reg = degree - mult

        t = knots[degree]
        for r in range(reg):
            lcl_knots = knots[r:]
            for k in range(1, degree - r):
                alpha = (t - lcl_knots[k]) / (lcl_knots[k + degree - r] - lcl_knots[k])
                C[:, k - 1] = alpha * C[:, k] + (one - alpha) * C[:, k - 1]

    alphas = np.zeros(degree - 1, dtype=dtype)

    knt_id = degree
    mult = 0

    for elem_id in range(n_elems):
        knt_id += mult
        mult = mults[elem_id + 1]

        if mult >= degree:
            continue

        lcl_knots = knots[knt_id : knt_id + degree + 1]
        alphas[: degree - mult] = (lcl_knots[1] - lcl_knots[0]) / (
            lcl_knots[mult + 1 :] - lcl_knots[0]
        )

        C = out[elem_id]

        reg = degree - mult
        for r in range(1, reg + 1):
            s = mult + r
            for k in range(degree, s - 1, -1):
                alpha = alphas[k - s]
                C[:, k] = alpha * C[:, k] + (one - alpha) * C[:, k - 1]

            if elem_id < (n_elems - 1):
                out[elem_id + 1, reg - r : reg + 1, reg - r] = C[degree - r : degree + 1, degree]


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _bezier_structural_identity_mask_core(
    multiplicities: npt.NDArray[np.intp],
    degree: int,
    out: npt.NDArray[np.bool_],
) -> None:
    """Compute a per-element Bézier identity mask from knot multiplicities.

    Element ``e`` spanning ``[unique_knots[e], unique_knots[e+1]]`` has an
    identity Bézier extraction operator if and only if both boundary unique
    knots have multiplicity ``>= degree + 1``, meaning the element is already
    a Bézier patch (fully isolated from its neighbours).

    Args:
        multiplicities (npt.NDArray[np.intp]): Per-unique-knot multiplicity
            array of length ``n_elements + 1`` (in-domain unique knots
            including both endpoints).
        degree (int): B-spline degree.
        out (npt.NDArray[np.bool_]): Output boolean array of length
            ``n_elements = len(multiplicities) - 1``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call
        ``spanwise_element_extraction._bezier_structural_identity_mask`` instead.
    """
    threshold = degree + 1
    n_elements = len(multiplicities) - 1
    for e in range(n_elements):
        out[e] = multiplicities[e] >= threshold and multiplicities[e + 1] >= threshold


def _tabulate_Bspline_Bezier_1D_extraction_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Create Bézier extraction operators for each interval.

    This function computes the extraction operators that transform Bernstein
    into B-spline basis functions for each interval.
    For each interval \( i \), the Bézier extraction operator \( C_i \) satisfies:

        \[
        N_i(x) = C_i @ B(ξ)
        \]

    where:
      - N_i(x) is the vector of B-spline basis functions nonzero on the interval \( i \),
        evaluated at \( x \),
      - \( B(ξ) \) is the vector of Bernstein basis functions on the reference interval \([0, 1]\),
        evaluated at \( ξ \),
      - \( C_i \) is the extraction matrix for interval \( i \),
      - \( x \) is the physical coordinate, \( ξ \) is the local (reference) referred to \([0, 1]\).

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        tol (float): Tolerance for numerical comparisons.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array where the result
            will be stored. If None, a new array is allocated. Must have the correct shape and dtype
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Array of extraction matrices with shape
            (intervals, degree+1, degree+1) where each matrix transforms
            Bernstein basis functions to B-spline basis functions for that interval.
            If `out` was provided, returns the same array.

    Raises:
        ValueError: If the knot vector or degree fails basic validation or if tol is negative.
        ValueError: If `out` is provided and has incorrect shape or dtype.
    """
    if tol < 0:
        raise ValueError("tol must be positive")

    _check_spline_info(knots, degree)

    unique_knots, _ = _get_unique_knots_and_multiplicity_impl(knots, degree, tol, in_domain=True)

    n_elems = len(unique_knots) - 1
    dtype = knots.dtype
    expected_shape = (n_elems, degree + 1, degree + 1)

    if out is None:
        out = np.empty(expected_shape, dtype=dtype)
    else:
        _validate_out_array(out, expected_shape, dtype)

    _tabulate_Bspline_Bezier_1D_extraction_core(knots, degree, tol, out)

    return out


def _tabulate_Bspline_Lagrange_1D_extraction_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    lagrange_variant: LagrangeVariant = LagrangeVariant.EQUISPACES,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Create Lagrange extraction operators for a B-spline.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        tol (float): Tolerance for numerical comparisons.
        lagrange_variant (LagrangeVariant): Lagrange point distribution
            (e.g., equispaced, gauss lobatto legendre, etc). Defaults to LagrangeVariant.EQUISPACES.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array where the result
            will be stored. If None, a new array is allocated. Must have the correct shape and dtype
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Array of extraction matrices with shape
            (n_intervals, degree+1, degree+1) where each matrix transforms
            Lagrange basis functions to B-spline basis functions for that interval.

            Each matrix C[i, :, :] transforms Bernstein basis functions
            to B-spline basis functions for the i-th interval as
                C[i, :, :] @ [Lagrange values] = [B-spline values in interval].
            If `out` was provided, returns the same array.

    Raises:
        ValueError: If the knot vector or degree fails basic validation or if tol is negative.
        ValueError: If `out` is provided and has incorrect shape or dtype.
    """
    if tol < 0:
        raise ValueError("tol must be positive")

    _check_spline_info(knots, degree)

    unique_knots, _ = _get_unique_knots_and_multiplicity_impl(knots, degree, tol, in_domain=True)
    n_elems = len(unique_knots) - 1
    dtype = knots.dtype
    expected_shape = (n_elems, degree + 1, degree + 1)

    if out is None:
        out = np.empty(expected_shape, dtype=dtype)
    else:
        _validate_out_array(out, expected_shape, dtype)

    # Compute Bezier extraction into out
    _tabulate_Bspline_Bezier_1D_extraction_impl(knots, degree, tol, out=out)

    # Transform to Lagrange extraction in-place to avoid extra copy
    # For every interval, right-multiply the Bezier-to-B-spline extraction by lagr_to_bzr in-place.
    lagr_to_bzr = _cached_lagrange_to_bernstein_matrix(degree, lagrange_variant, dtype)
    for i in range(out.shape[0]):
        # Use out[i, :, :] = out[i, :, :] @ lagr_to_bzr, but perform in-place via np.matmul
        # to avoid extra allocation.
        np.matmul(out[i, :, :], lagr_to_bzr, out=out[i, :, :])

    return out


def _tabulate_Bspline_cardinal_1D_extraction_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Create cardinal B-spline extraction operators.

    For cardinal intervals, the extraction matrix is set to the identity matrix

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        tol (float): Tolerance for numerical comparisons.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array where the result
            will be stored. If None, a new array is allocated. Must have the correct shape and dtype
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Array of extraction matrices with shape
            (n_intervals, degree+1, degree+1) where each matrix transforms
            cardinal B-spline basis functions to B-spline basis functions for that interval.

            Each matrix C[i, :, :] transforms cardinal B-spline basis functions
            to B-spline basis functions for the i-th interval as
                C[i, :, :] @ [cardinal values] = [B-spline values in interval].
            If `out` was provided, returns the same array.

    Raises:
        ValueError: If the knot vector or degree fails basic validation or if tol is negative.
        ValueError: If `out` is provided and has incorrect shape or dtype.
    """
    if tol < 0:
        raise ValueError("tol must be positive")

    _check_spline_info(knots, degree)

    unique_knots, _ = _get_unique_knots_and_multiplicity_impl(knots, degree, tol, in_domain=True)
    n_elems = len(unique_knots) - 1
    dtype = knots.dtype
    expected_shape = (n_elems, degree + 1, degree + 1)

    if out is None:
        out = np.empty(expected_shape, dtype=dtype)
    else:
        _validate_out_array(out, expected_shape, dtype)

    # Compute Bezier extraction into out
    _tabulate_Bspline_Bezier_1D_extraction_impl(knots, degree, tol, out=out)

    # Transform to cardinal extraction
    card_to_bzr = _cached_cardinal_to_bernstein_matrix(degree, dtype)
    # Transform to cardinal extraction in-place to avoid unnecessary copy
    # out[...] = out @ card_to_bzr is not strictly in-place (it creates a new array then assigns)
    # To perform an in-place transformation, use np.matmul (or @) with out as the output
    np.matmul(out, card_to_bzr, out=out)

    # Set identity for cardinal intervals
    cardinal_intervals = _get_Bspline_cardinal_intervals_1D_impl(knots, degree, tol)
    for i in np.where(cardinal_intervals)[0]:
        out[i, :, :] = np.eye(degree + 1, dtype=dtype)

    return out


def _warmup_numba_functions() -> None:
    """Precompile numba functions with float64 signatures for faster first call.

    This function triggers compilation of the numba-decorated functions
    with float64 arrays, ensuring they are cached and ready for use.
    """
    # Small dummy arrays for warmup
    knots_dummy = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    tol_dummy = 1e-10
    degree_dummy = 2
    n_elems = 1
    out_dummy = np.empty((n_elems, degree_dummy + 1, degree_dummy + 1), dtype=np.float64)

    # Warmup Bezier extraction core with float64
    _tabulate_Bspline_Bezier_1D_extraction_core(knots_dummy, degree_dummy, tol_dummy, out_dummy)

    # Warmup structural identity mask kernel
    mults_dummy = np.array([degree_dummy + 1, degree_dummy + 1], dtype=np.intp)
    mask_dummy = np.empty(1, dtype=np.bool_)
    _bezier_structural_identity_mask_core(mults_dummy, degree_dummy, mask_dummy)


# Precompile numba functions on module import
# (Moved to central thread in __init__.py)


__all__ = [
    "_tabulate_Bspline_Bezier_1D_extraction_impl",
    "_tabulate_Bspline_Lagrange_1D_extraction_impl",
    "_tabulate_Bspline_cardinal_1D_extraction_impl",
]
