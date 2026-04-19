"""1D basis function evaluation implementations.

This module provides wrapper functions for evaluating various 1D basis functions
(Bernstein, cardinal B-spline, Lagrange, Legendre) with input validation and
output array management.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from .._numba_compat import wait_for_jit_warmup
from ._basis_core import (
    _tabulate_Bernstein_basis_1D_core,
    _tabulate_cardinal_Bspline_basis_1D_core,
    _tabulate_Legendre_basis_1D_core,
)
from ._basis_lagrange import _tabulate_lagrange_basis_1D_core
from ._basis_utils import (
    _compute_final_output_shape_1D,
    _normalize_points_1D,
    _validate_out_array,
)

if TYPE_CHECKING:
    from . import LagrangeVariant


def _tabulate_basis_1D_impl_helper(
    n: int,
    t: npt.ArrayLike,
    core_func: Callable[
        [np.int32, npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]],
        None,
    ],
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Common implementation for tabulating 1D basis functions.

    Handles input normalization, output allocation/validation, and dtype dispatching
    for calling the appropriate core function.

    Args:
        n (int): Degree of the basis polynomials. Must be non-negative.
        t (npt.ArrayLike): Evaluation points. Can be a scalar, list, or numpy array.
            Types different from float32 or float64 are automatically converted to float64.
        core_func: Core function to call for computation. Must accept
            (np.int32, npt.NDArray[float32/float64], npt.NDArray[float32/float64]) -> None.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype if provided. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Evaluated basis functions, with the same shape
        as the input points and the last dimension equal to (degree + 1).
        If `out` was provided, returns the same array.

    Raises:
        ValueError: If degree is negative, or if `out` is provided and has incorrect
            shape or dtype.
    """
    # Ensure background JIT compilation is complete before calling Numba
    # kernels that use parallel=True (avoids concurrent-compilation crash).
    wait_for_jit_warmup()

    if n < 0:
        raise ValueError("degree must be non-negative")

    # Get input shape before normalization (handle scalars and lists)
    if isinstance(t, np.ndarray):
        input_shape = t.shape
    elif isinstance(t, list | tuple):
        input_shape = np.array(t).shape
    else:  # scalar
        input_shape = ()

    t = _normalize_points_1D(t)
    num_pts = t.shape[0]
    n_basis = n + 1

    # Determine expected shapes
    expected_normalized_shape = (num_pts, n_basis)
    expected_final_shape = _compute_final_output_shape_1D(input_shape, n_basis)

    if out is None:
        out = np.empty(expected_final_shape, dtype=t.dtype)
    else:
        _validate_out_array(out, expected_final_shape, cast(npt.DTypeLike, t.dtype))

    B_normalized = out.reshape(expected_normalized_shape)

    core_func(np.int32(n), t, B_normalized)

    return out


def _tabulate_Bernstein_basis_1D_impl(
    n: int,
    t: npt.ArrayLike,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate the Bernstein basis polynomials of the given degree at the given points.

    Args:
        n (int): Degree of the Bernstein polynomials. Must be non-negative.
        t (npt.ArrayLike): Evaluation points. Can be a scalar, list, or numpy array.
            Types different from float32 or float64 are automatically converted to float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype if provided. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Evaluated basis functions, with the same shape
        as the input points and the last dimension equal to (degree + 1).
        If `out` was provided, returns the same array.

    Raises:
        ValueError: If degree is negative, or if `out` is provided and has incorrect
            shape or dtype.

    Example:
        >>> _tabulate_Bernstein_basis_1D_impl(2, [0.0, 0.5, 0.75, 1.0])
        array([[1.    , 0.    , 0.    ],
               [0.25  , 0.5   , 0.25  ],
               [0.0625, 0.375 , 0.5625],
               [0.    , 0.    , 1.    ]])
    """
    return _tabulate_basis_1D_impl_helper(n, t, _tabulate_Bernstein_basis_1D_core, out)


def _tabulate_cardinal_Bspline_basis_1D_impl(
    n: int,
    t: npt.ArrayLike,
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
        n (int): Degree of the B-spline basis. Must be non-negative.
        t (npt.ArrayLike): Evaluation points. Can be a scalar, list, or numpy array.
            Types different from float32 or float64 are automatically converted to float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype if provided. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Evaluated basis functions, with the same shape
        as the input points and the last dimension equal to (degree + 1).
        If `out` was provided, returns the same array.

    Raises:
        ValueError: If provided degree is negative, or if `out` is provided and has incorrect
            shape or dtype.

    Example:
        >>> tabulate_cardinal_bspline_1d(2, [0.0, 0.5, 1.0])
        array([[0.5    , 0.5    , 0.     ],
               [0.125  , 0.75   , 0.125  ],
               [0.03125, 0.6875 , 0.28125],
               [0.     , 0.5    , 0.5    ]])

    """
    return _tabulate_basis_1D_impl_helper(n, t, _tabulate_cardinal_Bspline_basis_1D_core, out)


def _tabulate_Lagrange_basis_1D_impl(
    n: int,
    variant: LagrangeVariant,
    t: npt.ArrayLike,
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
        n (int): Degree of the B-spline basis. Must be non-negative.
        variant (LagrangeVariant): Variant of the Lagrange basis.
        t (npt.ArrayLike): Evaluation points. Can be a scalar, list, or numpy array.
            Types different from float32 or float64 are automatically converted to float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype if provided. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Evaluated basis functions, with the same shape
        as the input points and the last dimension equal to (degree + 1).
        If `out` was provided, returns the same array.

    Raises:
        ValueError: If provided degree is negative, or if `out` is provided and has incorrect
            shape or dtype.
    """
    # Lazy import to avoid circular dependency

    def core_func(
        n_int32: np.int32,
        t_array: npt.NDArray[np.float32 | np.float64],
        out_array: npt.NDArray[np.float32 | np.float64],
    ) -> None:
        return _tabulate_lagrange_basis_1D_core(int(n_int32), variant, t_array, out_array)

    return _tabulate_basis_1D_impl_helper(n, t, core_func, out)


def _tabulate_Legendre_basis_1D_impl(
    n: int,
    t: npt.ArrayLike,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate the normalized Shifted Legendre basis polynomials of the given degree.

    Args:
        n (int): Degree of the Legendre polynomials. Must be non-negative.
        t (npt.ArrayLike): Evaluation points. Can be a scalar, list, or numpy array.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype if provided. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Evaluated basis functions.
        If `out` was provided, returns the same array.

    Raises:
        ValueError: If degree is negative, or if `out` is provided and has incorrect
            shape or dtype.
    """
    return _tabulate_basis_1D_impl_helper(n, t, _tabulate_Legendre_basis_1D_core, out)


__all__ = [
    "_tabulate_Bernstein_basis_1D_impl",
    "_tabulate_Lagrange_basis_1D_impl",
    "_tabulate_Legendre_basis_1D_impl",
    "_tabulate_cardinal_Bspline_basis_1D_impl",
]
