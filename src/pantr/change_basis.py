r"""Change of basis operators for various polynomial bases in 1D.

This module provides functions to create transformation matrices between different
polynomial bases including Lagrange, Bernstein, cardinal B-spline, and monomial
bases.

Architecturally, this module serves as the **bridge between different basis types**,
providing pure mathematical functions to compute the exact $(degree+1, degree+1)$
transformation matrices (e.g., $M$ such that $new\\_basis(x) = M @ old\\_basis(x)$)
without tying the dense numerical quadrature logic directly into the core Spline
space objects.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from ._basis_lagrange import _get_lagrange_points
from ._basis_utils import _validate_out_array_multidimensional
from .basis import LagrangeVariant, tabulate_Bernstein_basis_1D, tabulate_cardinal_Bspline_basis_1D
from .quad import get_gauss_legendre_quadrature_1D


def compute_Lagrange_to_Bernstein_change_basis_1D(
    degree: int,
    lagrange_variant: LagrangeVariant = LagrangeVariant.EQUISPACES,
    dtype: npt.DTypeLike = np.float64,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Construct the matrix mapping Lagrange basis evaluations to Bernstein basis evaluations.

    Note:
        Both Bernstein and Lagrange bases follow the standard ordering (see https://en.wikipedia.org/wiki/Bernstein_polynomial).

    Args:
        degree (int): Polynomial degree. Must be at least 1.
        lagrange_variant (LagrangeVariant): Lagrange point distribution
            (e.g., equispaced, gauss lobatto legendre, etc). Defaults to LagrangeVariant.EQUISPACES.
        dtype (npt.DTypeLike): Floating point type for the output matrix.
            Defaults to np.float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have shape (degree+1, degree+1) and dtype matching the `dtype` parameter
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: (degree+1, degree+1) transformation matrix C such that
            C @ [Lagrange values] = [Bernstein values]. If `out` was provided,
            returns the same array.

    Raises:
        ValueError: If degree is lower than 1, dtype is not float32 or float64, or if `out` is
            provided and has incorrect shape or dtype.
    """
    if degree < 1:
        raise ValueError("Degree must at least 1")
    if dtype not in (np.float32, np.float64):
        raise ValueError("dtype must be float32 or float64")

    expected_shape = (degree + 1, degree + 1)
    expected_dtype = dtype

    if out is None:
        out = np.empty(expected_shape, dtype=expected_dtype)
    else:
        _validate_out_array_multidimensional(out, expected_shape, expected_dtype)

    points = _get_lagrange_points(lagrange_variant, degree + 1, dtype)

    result = out.T
    tabulate_Bernstein_basis_1D(degree, points, out=result)

    return out


def compute_Bernstein_to_Lagrange_change_basis_1D(
    degree: int,
    lagrange_variant: LagrangeVariant = LagrangeVariant.EQUISPACES,
    dtype: npt.DTypeLike = np.float64,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Construct the matrix mapping Bernstein basis evaluations to Lagrange basis evaluations.

    Note:
        Both Bernstein and Lagrange bases follow the standard ordering
        (see https://en.wikipedia.org/wiki/Bernstein_polynomial).


    Args:
        degree (int): Polynomial degree. Must be at least 1.
        lagrange_variant (LagrangeVariant): Lagrange point distribution
            (e.g., equispaced, gauss lobatto legendre, etc). Defaults to LagrangeVariant.EQUISPACES.
        dtype (npt.DTypeLike): Floating point type for the output matrix.
            Defaults to np.float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have shape (degree+1, degree+1) and dtype matching the `dtype` parameter
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: (degree+1, degree+1) transformation matrix C such that
            C @ [Bernstein values] = [Lagrange values]. If `out` was provided,
            returns the same array.

    Raises:
        ValueError: If degree is lower than 1, dtype is not float32 or float64, or if `out` is
            provided and has incorrect shape or dtype.
    """
    if degree < 1:
        raise ValueError("Degree must at least 1")
    if dtype not in (np.float32, np.float64):
        raise ValueError("dtype must be float32 or float64")

    expected_shape = (degree + 1, degree + 1)
    expected_dtype = dtype

    if out is None:
        out = np.empty(expected_shape, dtype=expected_dtype)
    else:
        _validate_out_array_multidimensional(out, expected_shape, expected_dtype)

    C = compute_Lagrange_to_Bernstein_change_basis_1D(degree, lagrange_variant, dtype)
    out[:] = np.linalg.inv(C)
    return out


def _compute_change_basis_1D(
    new_basis_eval: Callable[
        [npt.NDArray[np.float32 | np.float64]], npt.NDArray[np.float32 | np.float64]
    ],
    old_basis_eval: Callable[
        [npt.NDArray[np.float32 | np.float64]], npt.NDArray[np.float32 | np.float64]
    ],
    n_quad_pts: int,
    dtype: npt.DTypeLike = np.float64,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Create a change of basis operator using numerical quadrature.

    This function computes the transformation matrix M that satisfies:
        new_basis(x) = M @ old_basis(x)

    The matrix is computed by solving the system C = G M^T where:
    - G is the Gram matrix of the new basis
    - C is the mixed inner product matrix between new and old bases

    Args:
        new_basis_eval (callable): Function that evaluates the new basis at points.
        old_basis_eval (callable): Function that evaluates the old basis at points.
        n_quad_pts (int): Number of quadrature points for numerical integration.
            Must be positive.
        dtype (npt.DTypeLike): Floating point type for the output matrix.
            Defaults to np.float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype matching the `dtype` parameter
            if provided. The shape is determined by the number of basis functions
            in the old and new bases. This follows NumPy's style for output arrays.
            Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Change of basis transformation matrix.
            If `out` was provided, returns the same array.

    Raises:
        ValueError: If number of quadrature points is not positive, dtype is not float32 or
            float64, or if `out` is provided and has incorrect shape or dtype.
    """
    if n_quad_pts < 1:
        raise ValueError("Number of quadrature points must be positive.")

    if dtype not in (np.float32, np.float64):
        raise ValueError("dtype must be float32 or float64")

    # 1. Get Gauss-Legendre quadrature points and weights for the inner product on [0, 1]
    points, weights = get_gauss_legendre_quadrature_1D(n_quad_pts, dtype)

    # 2. Pre-evaluate all basis functions at all quadrature points for efficiency
    new_basis = new_basis_eval(points)
    old_basis = old_basis_eval(points)

    # Determine output shape from basis evaluations
    n_old_basis = old_basis.shape[1]
    n_new_basis = new_basis.shape[1]
    expected_shape = (n_old_basis, n_new_basis)
    expected_dtype = dtype

    if out is None:
        out = np.empty(expected_shape, dtype=expected_dtype)
    else:
        _validate_out_array_multidimensional(out, expected_shape, expected_dtype)

    # 3. Compute the Gram matrix G for the new basis B: G_kj = <b_k, b_j>
    # The inner product <f, g> is approximated by sum(w_m * f(x_m) * g(x_m))
    weights_diag = np.diag(weights)
    G = new_basis.T @ weights_diag @ new_basis

    # 4. Compute the mixed inner product matrix C: C_ki = <b_k, a_i>
    C = new_basis.T @ weights_diag @ old_basis

    # 5. Solve the system C = G M^T for M^T, which means M = (G^-1 C)^T
    out[:] = np.linalg.solve(G, C).T
    return out


def compute_Bernstein_to_cardinal_change_basis_1D(
    degree: int,
    dtype: npt.DTypeLike = np.float64,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Create transformation matrix from Bernstein to cardinal B-spline basis.

    Args:
        degree (int): Polynomial degree. Must be non-negative.
        dtype (np.dtype): Floating point type for the output matrix.
            Defaults to np.float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have shape (degree+1, degree+1) and dtype matching the `dtype` parameter
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: (degree+1, degree+1) transformation matrix C such that
            C @ [Bernstein values] = [cardinal values]. If `out` was provided,
            returns the same array.

    Raises:
        ValueError: If degree is negative, dtype is not float32 or float64, or if `out` is
            provided and has incorrect shape or dtype.
    """
    if degree < 0:
        raise ValueError("Degree must be non-negative")

    if dtype not in (np.float32, np.float64):
        raise ValueError("dtype must be float32 or float64")

    expected_shape = (degree + 1, degree + 1)
    expected_dtype = dtype

    if out is None:
        out = np.empty(expected_shape, dtype=expected_dtype)
    else:
        _validate_out_array_multidimensional(out, expected_shape, expected_dtype)

    def bernstein(
        pts: npt.NDArray[np.float32 | np.float64],
    ) -> npt.NDArray[np.float32 | np.float64]:
        return tabulate_Bernstein_basis_1D(degree, pts)

    def cardinal(pts: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float32 | np.float64]:
        return tabulate_cardinal_Bspline_basis_1D(degree, pts)

    return _compute_change_basis_1D(
        new_basis_eval=bernstein,
        old_basis_eval=cardinal,
        n_quad_pts=degree + 1,
        dtype=dtype,
        out=out,
    )


def compute_cardinal_to_Bernstein_change_basis_1D(
    degree: int,
    dtype: npt.DTypeLike = np.float64,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Create transformation matrix from cardinal B-spline to Bernstein basis.

    Args:
        degree (int): Polynomial degree. Must be non-negative.
        dtype (np.dtype): Floating point type for the output matrix.
            Defaults to np.float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have shape (degree+1, degree+1) and dtype matching the `dtype` parameter
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: (degree+1, degree+1) transformation matrix C such that
            C @ [cardinal values] = [Bernstein values]. If `out` was provided,
            returns the same array.

    Raises:
        ValueError: If degree is negative, dtype is not float32 or float64, or if `out` is
            provided and has incorrect shape or dtype.
    """
    if degree < 0:
        raise ValueError("Degree must be non-negative")

    if dtype not in (np.float32, np.float64):
        raise ValueError("dtype must be float32 or float64")

    expected_shape = (degree + 1, degree + 1)
    expected_dtype = dtype

    if out is None:
        out = np.empty(expected_shape, dtype=expected_dtype)
    else:
        _validate_out_array_multidimensional(out, expected_shape, expected_dtype)

    def bernstein(
        pts: npt.NDArray[np.float32 | np.float64],
    ) -> npt.NDArray[np.float32 | np.float64]:
        return tabulate_Bernstein_basis_1D(degree, pts)

    def cardinal(pts: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float32 | np.float64]:
        return tabulate_cardinal_Bspline_basis_1D(degree, pts)

    return _compute_change_basis_1D(
        new_basis_eval=cardinal,
        old_basis_eval=bernstein,
        n_quad_pts=degree + 1,
        dtype=dtype,
        out=out,
    )
