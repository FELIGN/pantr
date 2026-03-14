"""Basis function evaluation for various polynomial bases."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from enum import Enum
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import numpy.typing as npt

from ._basis_1D import (
    _tabulate_Bernstein_basis_1D_impl,
    _tabulate_cardinal_Bspline_basis_1D_impl,
    _tabulate_Lagrange_basis_1D_impl,
    _tabulate_Legendre_basis_1D_impl,
)
from ._basis_multidim import _compute_basis_1D_combinator_matrix

if TYPE_CHECKING:
    from .quad import PointsLattice


class LagrangeVariant(Enum):
    """Enumeration for Lagrange polynomial variants.

    Attributes:
        EQUISPACES (LagrangeVariant): Equispaced points.
        GAUSS_LEGENDRE (LagrangeVariant): Gauss-Legendre points (roots of Legendre polynomial).
        GAUSS_LOBATTO_LEGENDRE (LagrangeVariant): Gauss-Lobatto-Legendre points.
        CHEBYSHEV_1ST (LagrangeVariant): Chebyshev 1st kind points
            (x = [pi*(k + 0.5)/npts for k in range(npts)]).
        CHEBYSHEV_2ND (LagrangeVariant): Chebyshev 2nd kind points
            (x = [pi*k/(npts - 1) for k in range(npts)]).
    """

    EQUISPACES = "equispaces"
    GAUSS_LEGENDRE = "gauss_legendre"
    GAUSS_LOBATTO_LEGENDRE = "gauss_lobatto_legendre"
    CHEBYSHEV_1ST = "chebyshev_1st"
    CHEBYSHEV_2ND = "chebyshev_2nd"


def tabulate_Bernstein_basis_1D(
    degree: int,
    pts: npt.ArrayLike,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate the Bernstein basis polynomials of the given degree at the given points.

    Args:
        degree (int): Degree of the Bernstein polynomials. Must be non-negative.
        pts (npt.ArrayLike): Evaluation points. Can be a scalar, list, or
            numpy array. Types different from float32 or float64 are
            automatically converted to float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype if provided. This follows NumPy's
            style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Evaluated basis functions, with the same shape as
        the input points and the last dimension equal to (degree + 1).
        If `out` was provided, returns the same array.

    Raises:
        ValueError: If degree is negative, or if `out` is provided and has incorrect
            shape or dtype.

    Example:
        >>> tabulate_Bernstein_basis_1D(2, [0.0, 0.5, 0.75, 1.0])
        array([[1.    , 0.    , 0.    ],
               [0.25  , 0.5   , 0.25  ],
               [0.0625, 0.375 , 0.5625],
               [0.    , 0.    , 1.    ]])
    """
    return _tabulate_Bernstein_basis_1D_impl(degree, pts, out=out)


def tabulate_cardinal_Bspline_basis_1D(
    degree: int,
    pts: npt.ArrayLike,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Evaluate the cardinal B-spline basis polynomials of given degree at given points.

    The cardinal B-spline basis is the set of B-spline basis functions defined
    on an interval of maximum continuity that has degree-1 contiguous
    knot spans on each side with the same length as the interval itself.
    These basis functions appear in the central knot spans
    in the case of maximum regularity uniform knot vectors.

    Explicit expression:
    \[
    B_{n,i}(t) = (1/n!) * sum_{j=0}^{n-i} binom(n+1, j) * (-1)^j * (t + n - i - j)^n
    \]
    where \( B_{n,i}(t) \) is the B-spline basis function of degree \( n \) and index \( i \)
    at point \( t \), and \( binom(a, b) \) is the binomial coefficient.

    Its actual implementation is based on the Cox-de Boor recursion formula for the
    central cardinal B-spline basis.

    Args:
        degree (int): Degree of the B-spline basis. Must be non-negative.
        pts (npt.ArrayLike): Evaluation points. Can be a scalar, list, or numpy array.
            Types different from float32 or float64 are automatically converted to float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype if provided. This follows NumPy's
            style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Evaluated basis functions, with the same shape
        as the input points and the last dimension equal to (degree + 1).
        If `out` was provided, returns the same array.

    Raises:
        ValueError: If provided degree is negative, or if `out` is provided and has incorrect
            shape or dtype.

    Example:
        >>> tabulate_cardinal_Bspline_basis_1D(2, [0.0, 0.5, 1.0])
        array([[0.5    , 0.5    , 0.     ],
               [0.125  , 0.75   , 0.125  ],
               [0.03125, 0.6875 , 0.28125],
               [0.     , 0.5    , 0.5    ]])

    """
    return _tabulate_cardinal_Bspline_basis_1D_impl(degree, pts, out=out)


def tabulate_Lagrange_basis_1D(
    degree: int,
    variant: LagrangeVariant,
    pts: npt.ArrayLike,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Evaluate Lagrange basis polynomials at points using the specified variant.

    The polynomials are defined in the interval [0, 1] and are given by the formula:
    \[
    L_{n,i}(t) = \prod_{j=0}^{n} \frac{t - x_j}{x_i - x_j}
    \]
    where \( x_i \) are the points at which the basis is evaluated.

    The variant determines the points at which the basis is evaluated.
    central cardinal B-spline basis.

    Args:
        degree (int): Degree of the B-spline basis. Must be non-negative.
        variant (LagrangeVariant): Variant of the Lagrange basis.
        pts (npt.ArrayLike): Evaluation points. Can be a scalar, list, or numpy array.
            Types different from float32 or float64 are automatically converted to float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype if provided. This follows NumPy's
            style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Evaluated basis functions, with the same shape
        as the input points and the last dimension equal to (degree + 1).
        If `out` was provided, returns the same array.

    Raises:
        ValueError: If provided degree is negative, or if `out` is provided and has incorrect
            shape or dtype.
    """
    return _tabulate_Lagrange_basis_1D_impl(degree, variant, pts, out=out)


def tabulate_Legendre_basis_1D(
    degree: int,
    pts: npt.ArrayLike,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Evaluate the normalized Shifted Legendre basis polynomials at the given points.

    The polynomials are defined on the interval [0, 1] and are orthonormal:
    \[
    \int_0^1 \tilde{p}_n(x) \tilde{p}_m(x) \, dx = \delta_{nm}
    \]

    Args:
        degree (int): Degree of the Legendre basis. Must be non-negative.
        pts (npt.ArrayLike): Evaluation points. Can be a scalar, list, or numpy array.
            Types different from float32 or float64 are automatically converted to float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype if provided. This follows NumPy's
            style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Evaluated basis functions, with the same shape
        as the input points and the last dimension equal to (degree + 1).
        If `out` was provided, returns the same array.

    Raises:
        ValueError: If provided degree is negative, or if `out` is provided and has incorrect
            shape or dtype.
    """
    return _tabulate_Legendre_basis_1D_impl(degree, pts, out=out)


def tabulate_Bernstein_basis(
    degrees: Iterable[int],
    pts: npt.ArrayLike | PointsLattice,
    funcs_order: Literal["C", "F"] = "C",
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate the Bernstein basis functions at the given points.

    Evaluates Bernstein basis functions by combining 1D basis values across each dimension,
    supporting both general points (e.g., a 2D array of shape (n_pts, dim) for scattered points)
    and point lattices (points in tensor-product grids).
    Fully supports C/F-ordering for functions and points.

    Args:
        degrees (Iterable[int]): Iterable of degrees of the B-spline basis functions.
        pts (npt.ArrayLike | PointsLattice): Points at which to evaluate the basis.
            It can be a 2D array of shape (n_points, dim) for scattered points,
            or a PointsLattice object.
        funcs_order (Literal["C", "F"]): Ordering of the basis functions: 'C' for C-order
            (last index varies fastest) or 'F' for Fortran-order (first index varies fastest).
            Defaults to 'C'.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype if provided. This follows NumPy's
            style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Array of shape (n_points, n_basis_functions)
        containing the combined basis function values. If `out` was provided,
        returns the same array.

    Raises:
        ValueError: If any degree is negative, or if `out` is provided and has incorrect
            shape or dtype.
    """
    if not all(isinstance(degree, int) and degree >= 0 for degree in degrees):
        raise ValueError("All degrees must be non-negative integers")
    evaluators_1D = cast(
        tuple[Callable[[npt.ArrayLike], npt.NDArray[np.float32 | np.float64]]],
        tuple(lambda pts, d=degree: tabulate_Bernstein_basis_1D(d, pts) for degree in degrees),
    )
    return _compute_basis_1D_combinator_matrix(evaluators_1D, pts, funcs_order, out)


def tabulate_cardinal_Bspline_basis(
    degrees: Iterable[int],
    pts: npt.ArrayLike | PointsLattice,
    funcs_order: Literal["C", "F"] = "C",
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate the cardinal B-spline basis functions at the given points.

    Evaluates cardinal B-spline basis functions by combining 1D basis values across each dimension,
    supporting both general points (e.g., a 2D array of shape (n_pts, dim) for scattered points)
    and point lattices (points in tensor-product grids).
    Fully supports C/F-ordering for functions and points.

    Args:
        degrees (Iterable[int]): Iterable of degrees of the cardinal B-spline basis functions.
        pts (npt.ArrayLike | PointsLattice): Points at which to evaluate the basis.
            It can be a 2D array of shape (n_points, dim) for scattered points,
            or a PointsLattice object.
        funcs_order (Literal["C", "F"]): Ordering of the basis functions: 'C' for C-order
            (last index varies fastest) or 'F' for Fortran-order (first index varies fastest).
            Defaults to 'C'.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype if provided. This follows NumPy's
            style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Array of shape (n_points, n_basis_functions)
        containing the combined basis function values. If `out` was provided,
        returns the same array.

    Raises:
        ValueError: If any degree is negative, or if `out` is provided and has incorrect
            shape or dtype.
    """
    if not all(isinstance(degree, int) and degree >= 0 for degree in degrees):
        raise ValueError("All degrees must be non-negative integers")
    evaluators_1D = cast(
        tuple[Callable[[npt.ArrayLike], npt.NDArray[np.float32 | np.float64]]],
        tuple(
            lambda pts, d=degree: tabulate_cardinal_Bspline_basis_1D(d, pts) for degree in degrees
        ),
    )
    return _compute_basis_1D_combinator_matrix(evaluators_1D, pts, funcs_order, out)


def tabulate_Lagrange_basis(
    degrees: Iterable[int],
    variant: LagrangeVariant,
    pts: npt.ArrayLike | PointsLattice,
    funcs_order: Literal["C", "F"] = "C",
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate the Lagrange basis functions at the given points.

    Evaluates Lagrange basis functions by combining 1D basis values across each dimension,
    supporting both general points (e.g., a 2D array of shape (n_pts, dim) for scattered points)
    and point lattices (points in tensor-product grids).
    Fully supports C/F-ordering for functions and points.

    Args:
        degrees (Iterable[int]): Iterable of degrees of the Lagrange basis functions.
        variant (LagrangeVariant): Variant of the Lagrange basis.
        pts (npt.ArrayLike | PointsLattice): Points at which to evaluate the basis.
            It can be a 2D array of shape (n_points, dim) for scattered points,
            or a PointsLattice object.
        funcs_order (Literal["C", "F"]): Ordering of the basis functions: 'C' for C-order
            (last index varies fastest) or 'F' for Fortran-order (first index varies fastest).
            Defaults to 'C'.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype if provided. This follows NumPy's
            style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Array of shape (n_points, n_basis_functions)
        containing the combined basis function values. If `out` was provided,
        returns the same array.

    Raises:
        ValueError: If any degree is negative, or if `out` is provided and has incorrect
            shape or dtype.
    """
    if not all(isinstance(degree, int) and degree >= 0 for degree in degrees):
        raise ValueError("All degrees must be non-negative integers")
    evaluators_1D = cast(
        tuple[Callable[[npt.ArrayLike], npt.NDArray[np.float32 | np.float64]]],
        tuple(
            lambda pts, d=degree: tabulate_Lagrange_basis_1D(d, variant, pts) for degree in degrees
        ),
    )
    return _compute_basis_1D_combinator_matrix(evaluators_1D, pts, funcs_order, out)
