"""B-spline extraction operators.

This module provides functions for computing extraction operators that transform
between different basis representations (Bernstein, Lagrange, cardinal B-spline)
and B-spline basis functions.

Layer 2: it validates, allocates and dispatches, and holds no Numba. The kernels
live in :mod:`pantr.bspline._bspline_extraction_core`, and
:mod:`pantr.bspline._extraction_backend` chooses between them and their C++ twins
in ``cpp/include/pantr/bspline/extraction.hpp``.

The **Bézier** and **Lagrange** builders are dispatched. The cardinal one is not:
it needs the cardinal-interval scan on top of a change-of-basis matrix, and that
scan is not ported, so it still runs the Bézier builder and post-multiplies here.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..basis import LagrangeVariant
from ..basis._basis_utils import _allocate_or_validate_out
from ..change_basis import (
    _cached_cardinal_to_bernstein_matrix,
    _cached_lagrange_to_bernstein_matrix,
)
from ._bspline_knots import (
    _check_spline_info,
    _get_Bspline_cardinal_intervals_1D_impl,
    _get_unique_knots_and_multiplicity_impl,
)
from ._extraction_backend import bezier_extraction_kernel, lagrange_extraction_kernel


def _prepare_extraction_out(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    out: npt.NDArray[np.float32 | np.float64] | None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Validate inputs and allocate/validate the extraction-operator output array.

    Shared prologue for the ``_tabulate_Bspline_*_1D_extraction_impl`` helpers:
    validates ``tol`` and the spline info, then allocates (or validates) the
    ``(n_intervals, degree+1, degree+1)`` output array.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        tol (float): Tolerance for numerical comparisons; must be non-negative.
        out (npt.NDArray[np.float32 | np.float64] | None): Caller-provided output
            array to validate, or ``None`` to allocate a fresh one.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The ``(n_intervals, degree+1,
        degree+1)`` output array.

    Raises:
        ValueError: If ``tol`` is negative, the knots/degree fail validation, or
            ``out`` has the wrong shape, dtype, or is not writeable.
    """
    if tol < 0:
        raise ValueError("tol must be non-negative")
    _check_spline_info(knots, degree)
    unique_knots, _ = _get_unique_knots_and_multiplicity_impl(knots, degree, tol, in_domain=True)
    n_elems = len(unique_knots) - 1
    return _allocate_or_validate_out(out, (n_elems, degree + 1, degree + 1), knots.dtype)


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
    out = _prepare_extraction_out(knots, degree, tol, out)

    bezier_extraction_kernel()(knots, degree, tol, out)

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
    out = _prepare_extraction_out(knots, degree, tol, out)

    # The matrix is resolved here rather than inside the kernel: it depends only on
    # (degree, variant, dtype), `pantr.change_basis` caches it on exactly that key
    # and dispatches its own backend, and `lagrange_variant` is a `StrEnum`, which
    # must not reach a kernel. It also refuses `degree == 0`, which is why this
    # target has no degree-0 case and its sibling mask short-circuits before asking.
    lagr_to_bzr = _cached_lagrange_to_bernstein_matrix(degree, lagrange_variant, knots.dtype)

    lagrange_extraction_kernel()(knots, degree, tol, lagr_to_bzr, out)

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
    out = _prepare_extraction_out(knots, degree, tol, out)

    # Compute Bezier extraction into out
    _tabulate_Bspline_Bezier_1D_extraction_impl(knots, degree, tol, out=out)

    # Transform to cardinal extraction
    card_to_bzr = _cached_cardinal_to_bernstein_matrix(degree, knots.dtype)
    # Transform to cardinal extraction in-place to avoid unnecessary copy
    # out[...] = out @ card_to_bzr is not strictly in-place (it creates a new array then assigns)
    # To perform an in-place transformation, use np.matmul (or @) with out as the output
    np.matmul(out, card_to_bzr, out=out)

    # Set identity for cardinal intervals
    cardinal_intervals = _get_Bspline_cardinal_intervals_1D_impl(knots, degree, tol)
    for i in np.where(cardinal_intervals)[0]:
        out[i, :, :] = np.eye(degree + 1, dtype=knots.dtype)

    return out


__all__ = [
    "_tabulate_Bspline_Bezier_1D_extraction_impl",
    "_tabulate_Bspline_Lagrange_1D_extraction_impl",
    "_tabulate_Bspline_cardinal_1D_extraction_impl",
]
