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

import functools
from collections.abc import Callable
from math import comb

import numpy as np
import numpy.typing as npt

from .basis import LagrangeVariant, tabulate_bernstein_1d, tabulate_cardinal_bspline_1d
from .basis._basis_lagrange import _get_lagrange_points
from .basis._basis_utils import (
    _allocate_or_validate_out,
    _validate_float_dtype,
)
from .quad import get_gauss_legendre_1d


def compute_lagrange_to_bernstein_1d(
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
    _validate_float_dtype(dtype)

    out = _allocate_or_validate_out(out, (degree + 1, degree + 1), dtype)

    points = _get_lagrange_points(lagrange_variant, degree + 1, dtype)
    tabulate_bernstein_1d(degree, points, out=out.T)
    return out


def compute_bernstein_to_lagrange_1d(
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
    _validate_float_dtype(dtype)

    out = _allocate_or_validate_out(out, (degree + 1, degree + 1), dtype)

    C = compute_lagrange_to_bernstein_1d(degree, lagrange_variant, dtype)
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
    _validate_float_dtype(dtype)

    # 1. Get Gauss-Legendre quadrature points and weights for the inner product on [0, 1]
    points, weights = get_gauss_legendre_1d(n_quad_pts, dtype)

    # 2. Pre-evaluate all basis functions at all quadrature points for efficiency
    new_basis = new_basis_eval(points)
    old_basis = old_basis_eval(points)

    out = _allocate_or_validate_out(out, (old_basis.shape[1], new_basis.shape[1]), dtype)

    # 3. Compute the Gram matrix G for the new basis B: G_kj = <b_k, b_j>
    # The inner product <f, g> is approximated by sum(w_m * f(x_m) * g(x_m))
    weights_diag = np.diag(weights)
    G = new_basis.T @ weights_diag @ new_basis

    # 4. Compute the mixed inner product matrix C: C_ki = <b_k, a_i>
    C = new_basis.T @ weights_diag @ old_basis

    # 5. Solve the system C = G M^T for M^T, which means M = (G^-1 C)^T
    out[:] = np.linalg.solve(G, C).T
    return out


def compute_bernstein_to_cardinal_1d(
    degree: int,
    dtype: npt.DTypeLike = np.float64,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Create transformation matrix from Bernstein to cardinal B-spline basis.

    Args:
        degree (int): Polynomial degree. Must be non-negative.
        dtype (npt.DTypeLike): Floating point type for the output matrix.
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
    _validate_float_dtype(dtype)

    out = _allocate_or_validate_out(out, (degree + 1, degree + 1), dtype)

    return _compute_change_basis_1D(
        new_basis_eval=functools.partial(tabulate_bernstein_1d, degree),
        old_basis_eval=functools.partial(tabulate_cardinal_bspline_1d, degree),
        n_quad_pts=degree + 1,
        dtype=dtype,
        out=out,
    )


def compute_cardinal_to_bernstein_1d(
    degree: int,
    dtype: npt.DTypeLike = np.float64,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Create transformation matrix from cardinal B-spline to Bernstein basis.

    Args:
        degree (int): Polynomial degree. Must be non-negative.
        dtype (npt.DTypeLike): Floating point type for the output matrix.
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
    _validate_float_dtype(dtype)

    out = _allocate_or_validate_out(out, (degree + 1, degree + 1), dtype)

    return _compute_change_basis_1D(
        new_basis_eval=functools.partial(tabulate_cardinal_bspline_1d, degree),
        old_basis_eval=functools.partial(tabulate_bernstein_1d, degree),
        n_quad_pts=degree + 1,
        dtype=dtype,
        out=out,
    )


def compute_monomial_to_bernstein_1d(
    degree: int,
    dtype: npt.DTypeLike = np.float64,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Create transformation matrix from monomial to Bernstein basis on [0, 1].

    Given a polynomial of degree ``degree`` written in the monomial basis on
    ``[0, 1]``, the returned matrix ``M`` converts its coefficient vector to
    the Bernstein basis: ``bern_coeffs = M @ mono_coeffs``. Equivalently, on
    basis evaluations, ``monomial(x) = M.T @ bernstein(x)``.

    The entries are ``M[i, j] = C(i, j) / C(degree, j)`` for ``j <= i``, else
    ``0``, where ``C(n, k)`` is the binomial coefficient.

    Args:
        degree (int): Polynomial degree. Must be non-negative.
        dtype (npt.DTypeLike): Floating point type for the output matrix.
            Defaults to np.float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have shape (degree+1, degree+1) and dtype matching the `dtype` parameter
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: (degree+1, degree+1) lower-triangular
            transformation matrix ``M`` such that ``M @ [monomial coefficients] =
            [Bernstein coefficients]``. If `out` was provided, returns the same array.

    Raises:
        ValueError: If degree is negative, dtype is not float32 or float64, or if `out`
            is provided and has incorrect shape or dtype.
    """
    if degree < 0:
        raise ValueError("Degree must be non-negative")
    _validate_float_dtype(dtype)

    out = _allocate_or_validate_out(out, (degree + 1, degree + 1), dtype)
    out[:] = 0

    for i in range(degree + 1):
        for j in range(i + 1):
            out[i, j] = comb(i, j) / comb(degree, j)

    return out


@functools.lru_cache(maxsize=64)
def _cached_lagrange_to_bernstein_matrix(
    degree: int,
    lagrange_variant: LagrangeVariant,
    dtype: np.dtype[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    """Return a cached, read-only Lagrange-to-Bernstein change-of-basis matrix.

    This is the hot-path counterpart of
    :func:`compute_lagrange_to_bernstein_1d` used internally by
    the extraction-operator routines.  Because the matrix depends only on
    ``(degree, lagrange_variant, dtype)`` — never on the knot vector — it is
    safe to share a single immutable copy across all calls with matching
    arguments.

    The returned array has ``writeable=False`` to guard the cached copy against
    accidental in-place mutation.  NumPy's ``matmul`` and ``@`` operator accept
    read-only arrays as non-output arguments, so callers can use the matrix
    directly with :func:`numpy.matmul`.

    Args:
        degree (int): Polynomial degree.
        lagrange_variant (LagrangeVariant): Lagrange node distribution.
        dtype (np.dtype): Floating-point dtype (``float32`` or ``float64``).

    Returns:
        npt.NDArray[np.float32 | np.float64]: Read-only ``(degree+1, degree+1)``
            transformation matrix such that ``C @ lagrange_values = bernstein_values``.
    """
    mat = compute_lagrange_to_bernstein_1d(degree, lagrange_variant, dtype)
    mat.flags.writeable = False
    return mat


@functools.lru_cache(maxsize=64)
def _cached_cardinal_to_bernstein_matrix(
    degree: int,
    dtype: np.dtype[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    """Return a cached, read-only cardinal-B-spline-to-Bernstein change-of-basis matrix.

    This is the hot-path counterpart of
    :func:`compute_cardinal_to_bernstein_1d` used internally by
    the extraction-operator routines.  Because the matrix depends only on
    ``(degree, dtype)`` — never on the knot vector — it is safe to share a
    single immutable copy across all calls with matching arguments.

    The returned array has ``writeable=False`` to guard the cached copy against
    accidental in-place mutation.  NumPy's ``matmul`` and ``@`` operator accept
    read-only arrays as non-output arguments, so callers can use the matrix
    directly with :func:`numpy.matmul`.

    Args:
        degree (int): Polynomial degree.
        dtype (np.dtype): Floating-point dtype (``float32`` or ``float64``).

    Returns:
        npt.NDArray[np.float32 | np.float64]: Read-only ``(degree+1, degree+1)``
            transformation matrix such that ``C @ cardinal_values = bernstein_values``.
    """
    mat = compute_cardinal_to_bernstein_1d(degree, dtype)
    mat.flags.writeable = False
    return mat


__all__ = [
    "compute_bernstein_to_cardinal_1d",
    "compute_bernstein_to_lagrange_1d",
    "compute_cardinal_to_bernstein_1d",
    "compute_lagrange_to_bernstein_1d",
    "compute_monomial_to_bernstein_1d",
]
