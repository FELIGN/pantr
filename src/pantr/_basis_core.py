"""Core Numba-compiled implementations for 1D basis functions.

This module provides low-level, Numba-accelerated core functions for evaluating
Bernstein, cardinal B-spline, and Legendre basis polynomials.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._numba_compat import nb_jit, nb_prange


@nb_jit(
    nopython=True,
    cache=True,
    parallel=True,
)
def _tabulate_Bernstein_basis_1D_core(
    n: np.int32,
    t: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate Bernstein basis polynomials of degree n at points t.

    Computes all (n+1) Bernstein basis polynomials B_i,n(t) for i=0,...,n
    at each evaluation point in t. Writes the result to the provided output array,
    where out[j, i] contains the value of the i-th basis polynomial evaluated at point t_j.

    The implementation uses a recurrence relation for efficient computation:
    - Base case: B_0,n(t) = (1-t)^n
    - Recurrence: B_i,n(t) = B_{i-1},n(t) * ((n-i+1)/i) * (t/(1-t))

    Special handling is applied for points where t=1.0, where the recurrence
    relation would produce NaN values. At t=1.0, only the last basis function
    B_n,n(1) = 1, while all others are 0.

    Each evaluation point is independent, so the outer loop over points is
    parallelised with ``numba.prange`` for better throughput on large arrays.

    Args:
        n (np.int32): Degree of the Bernstein polynomials. Must be non-negative.
        t (npt.NDArray[np.float32 | np.float64]): 1D array of
            evaluation points. Must be a contiguous array of float32 or float64
            values. Points should typically be in [0, 1], though the function
            will compute values for any real t.
        out (npt.NDArray[np.float32 | np.float64]): Output array
            of shape (len(t), n+1) and dtype matching t. The function writes
            the evaluated basis functions to this array. Must have the correct
            shape and dtype (no validation performed inside this numba-compiled function).

    Note:
        This is a Numba-compiled function optimized for performance. It
        expects pre-normalized inputs (1D contiguous arrays) and assumes the
        output array has the correct shape and dtype. For general use,
        call _tabulate_Bernstein_basis_1D_impl instead.
    """
    if n == 0:
        # The basis is just B_0,0(pts) = 1
        for j in nb_prange(out.shape[0]):
            out[j, 0] = 1.0
        return

    # Process each point — rows are independent, so use prange.
    for j in nb_prange(t.shape[0]):
        u = t[j]
        if u == 1.0:
            # At t=1.0: only B_n,n(1) = 1, all others are 0
            for i in range(out.shape[1]):
                out[j, i] = 0.0
            out[j, n] = 1.0
        else:
            # For t != 1.0, use recurrence relation
            one_minus_u = 1.0 - u
            out[j, 0] = np.power(one_minus_u, n)
            if n > 0:
                t_over_1mt = u / one_minus_u
                for i in range(1, n + 1):
                    const_factor = (n - i + 1.0) / i
                    out[j, i] = out[j, i - 1] * const_factor * t_over_1mt


@nb_jit(
    nopython=True,
    cache=True,
    parallel=True,
)
def _tabulate_cardinal_Bspline_basis_1D_core(
    n: np.int32,
    t: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate the central cardinal B-spline basis of degree n on [0, 1].

    Computes the (n+1) nonzero B-spline basis functions active over the central
    unit span [0, 1] of a uniform knot vector with unit spacing, at each
    evaluation point in t. Values are zero outside [0, 1]. Uses the stable
    Cox-de Boor (BasisFuns) recursion specialized to span index i=0.

    Each evaluation point is independent, so the outer loop over points is
    parallelised with ``numba.prange`` for better throughput on large arrays.

    Args:
        n (np.int32): Degree of the B-spline basis (>= 0).
        t (npt.NDArray[np.float32 | np.float64]): 1D array of
            evaluation points (float32 or float64).
        out (npt.NDArray[np.float32 | np.float64]): Output array
            of shape (len(t), n+1) and dtype matching t. The function writes
            the evaluated basis functions to this array. Must have the correct
            shape and dtype (no validation performed inside this numba-compiled function).
    """
    num_pts = t.shape[0]

    if n == 0:
        # Degree-0: basis function is constant.
        for j in nb_prange(num_pts):
            out[j, 0] = 1.0
        return

    # Each row is independent — parallelise over points.
    for j in nb_prange(num_pts):
        out[j, 0] = 1.0
        u = t[j]
        one_minus_u = 1.0 - u

        for k in range(1, n + 1):
            inv_k = 1.0 / k
            saved = 0.0
            for r in range(k):
                Nr_old = out[j, r]
                term = (r + one_minus_u) * inv_k
                out[j, r] = saved + Nr_old * term
                saved = Nr_old * (1.0 - term)
            out[j, k] = saved


@nb_jit(
    nopython=True,
    cache=True,
    parallel=True,
)
def _tabulate_Legendre_basis_1D_core(
    n: np.int32,
    t: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate normalized Shifted Legendre basis polynomials of degree n at points t.

    Computes all (n+1) basis polynomials p_i(t) for i=0,...,n
    at each evaluation point in t. Writes the result to the provided output array,
    where out[j, i] contains the value of the i-th basis polynomial evaluated at point t_j.

    The implementation uses the recurrence relation for normalized shifted Legendre polynomials:
    p_0(x) = 1
    p_1(x) = sqrt(3)(2x-1)
    p_i(x) = (sqrt(2i-1)sqrt(2i+1)/i) * (2x-1) * p_{i-1}(x)
             - ((i-1)/i) * sqrt((2i+1)/(2i-3)) * p_{i-2}(x)

    The recurrence coefficients are precomputed once, and the outer loop over
    evaluation points is parallelised with ``numba.prange``.

    Args:
        n (np.int32): Degree of the Legendre polynomials. Must be non-negative.
        t (npt.NDArray[np.float32 | np.float64]): 1D array of
            evaluation points.
        out (npt.NDArray[np.float32 | np.float64]): Output array
            of shape (len(t), n+1) and dtype matching t. The function writes
            the evaluated basis functions to this array. Must have the correct
            shape and dtype (no validation performed inside this numba-compiled function).
    """
    num_pts = t.shape[0]

    if n == 0:
        for j in nb_prange(num_pts):
            out[j, 0] = 1.0
        return

    # Precompute recurrence coefficients a_i, b_i for i = 2..n.
    # This is O(n) work shared across all points.
    a_coeffs = np.empty(n + 1)
    b_coeffs = np.empty(n + 1)
    for i in range(2, n + 1):
        i_float = float(i)
        sqrt_2i_minus_1 = np.sqrt(2.0 * i_float - 1.0)
        sqrt_2i_plus_1 = np.sqrt(2.0 * i_float + 1.0)
        sqrt_2i_minus_3 = np.sqrt(2.0 * i_float - 3.0)
        a_coeffs[i] = (sqrt_2i_minus_1 * sqrt_2i_plus_1) / i_float
        b_coeffs[i] = ((i_float - 1.0) / i_float) * (sqrt_2i_plus_1 / sqrt_2i_minus_3)

    sqrt3 = np.sqrt(3.0)

    # Each row is independent — parallelise over points.
    for j in nb_prange(num_pts):
        two_x_minus_1 = 2.0 * t[j] - 1.0
        out[j, 0] = 1.0
        out[j, 1] = sqrt3 * two_x_minus_1
        for i in range(2, n + 1):
            out[j, i] = a_coeffs[i] * two_x_minus_1 * out[j, i - 1] - b_coeffs[i] * out[j, i - 2]


def _warmup_numba_functions() -> None:
    """Precompile numba functions with float64 signatures for faster first call.

    This function triggers compilation of the numba-decorated core functions
    with float64 arrays, ensuring they are cached and ready for use.
    """
    # Small dummy arrays for warmup
    t_dummy = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    out_dummy = np.empty((3, 2), dtype=np.float64)

    # Warmup each core function with float64
    _tabulate_Bernstein_basis_1D_core(np.int32(1), t_dummy, out_dummy)
    _tabulate_cardinal_Bspline_basis_1D_core(np.int32(1), t_dummy, out_dummy)
    _tabulate_Legendre_basis_1D_core(np.int32(1), t_dummy, out_dummy)


# Precompile numba functions on module import
# (Moved to central thread in __init__.py)
