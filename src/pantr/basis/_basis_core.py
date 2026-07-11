"""Core Numba-compiled implementations for 1D basis functions.

This module provides low-level, Numba-accelerated core functions for evaluating
Bernstein, cardinal B-spline, and Legendre basis polynomials.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._numba_compat import nb_jit, nb_prange

_PARALLEL_MIN_NUM_PTS = 4096
"""Minimum number of evaluation points for which the ``parallel=True`` kernels pay off.

Below this threshold the fork/join overhead of a parallel Numba kernel launch
(hundreds of microseconds) dwarfs the per-point work (tens of nanoseconds), so
the Layer-2 tabulation helpers dispatch to the serial twin kernels instead.
"""


@nb_jit(
    nopython=True,
    cache=True,
    inline="always",
)
def _bernstein_point(
    n: np.int32,
    u: np.float32 | np.float64,
    out_row: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate all Bernstein basis polynomials of degree ``n`` at one point.

    Uses the O(n) ratio recurrence, run from whichever endpoint keeps the seed
    term bounded away from underflow, exploiting the symmetry
    ``B_i,n(u) = B_{n-i},n(1-u)``:

    - ``u <= 0.5``: forward recurrence from ``u = 0``:
      ``B_0,n(u) = (1-u)^n``; ``B_i,n = B_{i-1},n * ((n-i+1)/i) * (u/(1-u))``.
    - ``u > 0.5``: mirrored recurrence from ``u = 1``:
      ``B_n,n(u) = u^n``; ``B_{i-1},n = B_i,n * (i/(n-i+1)) * ((1-u)/u)``.

    Branching on the midpoint bounds the seed (``(1-u)^n`` or ``u^n``) below
    by ``0.5^n``, so it never underflows for any representable degree ``n``.
    Without the branch, the forward seed ``(1-u)^n`` underflows to exact
    ``0.0`` once ``u`` is close enough to 1 at high degree, and every
    subsequent term (a positive multiple of the previous one) stays zero —
    including ``B_n,n``, whose true value is near 1 — silently breaking
    partition of unity. ``u == 1.0`` needs no special-casing: the mirrored
    branch handles it exactly (``u^n == 1.0``, then every step multiplies by
    ``(1-u)/u == 0.0``).

    Args:
        n (np.int32): Degree of the Bernstein polynomials (>= 0).
        u (np.float32 | np.float64): Evaluation point.
        out_row (npt.NDArray[np.float32 | np.float64]): Output row of length ``n + 1``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_tabulate_Bernstein_basis_1D_impl` instead.
    """
    if u > 0.5:
        # Mirrored recurrence from u=1: seed B_n,n(u) = u^n is >= 0.5^n, so it
        # never underflows. Low-index entries may legitimately underflow to
        # exact 0.0 (their true value is negligible near u=1).
        one_minus_u = 1.0 - u
        out_row[n] = np.power(u, n)
        one_minus_u_over_u = one_minus_u / u
        for i in range(n, 0, -1):
            const_factor = i / (n - i + 1.0)
            out_row[i - 1] = out_row[i] * const_factor * one_minus_u_over_u
    else:
        # Forward recurrence from u=0: seed B_0,n(u) = (1-u)^n is >= 0.5^n, so
        # it never underflows.
        one_minus_u = 1.0 - u
        out_row[0] = np.power(one_minus_u, n)
        t_over_1mt = u / one_minus_u
        for i in range(1, n + 1):
            const_factor = (n - i + 1.0) / i
            out_row[i] = out_row[i - 1] * const_factor * t_over_1mt


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

    Each evaluation point is independent, so the outer loop over points is
    parallelised with ``numba.prange`` for better throughput on large arrays.
    For small batches (fewer than ``_PARALLEL_MIN_NUM_PTS`` points) prefer the
    serial twin :func:`_tabulate_Bernstein_basis_1D_serial_core`, which avoids
    the parallel launch overhead.

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
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_tabulate_Bernstein_basis_1D_impl` instead.
    """
    if n == 0:
        # The basis is just B_0,0(pts) = 1
        for j in nb_prange(out.shape[0]):
            out[j, 0] = 1.0
        return

    # Process each point — rows are independent, so use prange.
    for j in nb_prange(t.shape[0]):
        _bernstein_point(n, t[j], out[j, :])


@nb_jit(
    nopython=True,
    cache=True,
)
def _tabulate_Bernstein_basis_1D_serial_core(
    n: np.int32,
    t: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate Bernstein basis polynomials of degree n at points t (serial twin).

    Identical to :func:`_tabulate_Bernstein_basis_1D_core` but compiled without
    ``parallel=True``: no fork/join overhead, which makes it the faster choice
    for small point batches.

    Args:
        n (np.int32): Degree of the Bernstein polynomials. Must be non-negative.
        t (npt.NDArray[np.float32 | np.float64]): 1D array of
            evaluation points. Must be a contiguous array of float32 or float64
            values.
        out (npt.NDArray[np.float32 | np.float64]): Output array
            of shape (len(t), n+1) and dtype matching t. Must have the correct
            shape and dtype (no validation performed).

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call _tabulate_Bernstein_basis_1D_impl instead.
    """
    if n == 0:
        for j in range(out.shape[0]):
            out[j, 0] = 1.0
        return

    for j in range(t.shape[0]):
        _bernstein_point(n, t[j], out[j, :])


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


@nb_jit(
    nopython=True,
    cache=True,
    inline="always",
)
def _bernstein_derivs_point(
    n: np.int32,
    s: np.float32 | np.float64,
    n_deriv: int,
    out_pt: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate all Bernstein basis derivatives at one point (A2.3 body, unit spans).

    Inlined (``inline="always"``) into the batch kernels so the per-point
    scratch allocations stay visible to Numba's parallel allocation hoisting.

    Args:
        n (np.int32): Degree of the Bernstein polynomials (>= 0).
        s (np.float32 | np.float64): Evaluation point in [0, 1].
        n_deriv (int): Maximum derivative order to compute (>= 0).
        out_pt (npt.NDArray[np.float32 | np.float64]): Output block of shape
            ``(n_deriv + 1, n + 1)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_tabulate_Bspline_basis_deriv_1D_impl` instead.
    """
    order = int(n) + 1
    dtype = out_pt.dtype
    zero = dtype.type(0.0)
    one = dtype.type(1.0)

    # Zero out this point's output slice (thread-local write)
    for k in range(n_deriv + 1):
        for i in range(order):
            out_pt[k, i] = zero

    # --- Build ndu table for uniform knots on [0, 1] ---
    # Upper triangle / diagonal: ndu[r, j] = B_{r, j}(s)  (r <= j)
    # Lower triangle:            ndu[j, r] = 1             (r <  j)  ← knot differences
    ndu = np.zeros((order, order), dtype=dtype)
    a_arr = np.zeros((2, n_deriv + 1), dtype=dtype)

    ndu[0, 0] = one
    for j in range(1, order):
        saved = zero
        for r in range(j):
            ndu[j, r] = one  # knot difference = 1 for uniform Bernstein knots
            temp = ndu[r, j - 1]  # divide by ndu[j, r] = 1
            ndu[r, j] = saved + (one - s) * temp
            saved = s * temp
        ndu[j, j] = saved

    # 0th derivatives = Bernstein basis values
    for j in range(order):
        out_pt[0, j] = ndu[j, int(n)]

    # --- k-th derivatives via triangular recursion (A2.3, unit knot spans) ---
    for r in range(order):
        s1 = 0
        s2 = 1
        a_arr[0, 0] = one

        for k in range(1, n_deriv + 1):
            d = zero
            rk = r - k
            pk = int(n) - k

            if r >= k:
                a_arr[s2, 0] = a_arr[s1, 0]  # divide by ndu[pk+1, rk] = 1
                d = a_arr[s2, 0] * ndu[rk, pk]

            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if (r - 1) <= pk else int(n) - r

            for jj in range(j1, j2 + 1):
                a_arr[s2, jj] = a_arr[s1, jj] - a_arr[s1, jj - 1]  # divide by 1
                d += a_arr[s2, jj] * ndu[rk + jj, pk]

            if r <= pk:
                a_arr[s2, k] = -a_arr[s1, k - 1]  # divide by ndu[pk+1, r] = 1
                d += a_arr[s2, k] * ndu[r, pk]

            out_pt[k, r] = d

            # Swap rows
            tmp_s = s1
            s1 = s2
            s2 = tmp_s

    # --- Factorial scaling: multiply k-th row by n! / (n-k)! ---
    # When k > n, fac becomes 0, zeroing rows beyond the degree (the k-th
    # derivative of a degree-n polynomial is identically zero for k > n).
    fac = int(n)
    for k in range(1, n_deriv + 1):
        for j in range(order):
            out_pt[k, j] *= fac
        fac *= int(n) - k


@nb_jit(
    nopython=True,
    cache=True,
    parallel=True,
)
def _tabulate_Bernstein_basis_deriv_1D_core(
    n: np.int32,
    t: npt.NDArray[np.float32 | np.float64],
    n_deriv: int,
    out_deriv: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate Bernstein basis function derivatives on [0, 1].

    Implements Algorithm A2.3 (DerBasisFuncs) from Piegl & Tiller specialised
    for Bernstein polynomials, where all knot differences equal one.  The outer
    loop over points is parallelised with ``numba.prange``; for small batches
    (fewer than ``_PARALLEL_MIN_NUM_PTS`` points) prefer the serial twin
    :func:`_tabulate_Bernstein_basis_deriv_1D_serial_core`, which avoids the
    parallel launch overhead.

    Args:
        n (np.int32): Degree of the Bernstein polynomials (>= 0).
        t (npt.NDArray[np.float32 | np.float64]): 1D array of evaluation points
            in [0, 1]. Points equal to 1.0 are handled as a boundary special case.
        n_deriv (int): Maximum derivative order to compute (>= 0).
        out_deriv (npt.NDArray[np.float32 | np.float64]): Output array of shape
            ``(len(t), n_deriv+1, n+1)`` and dtype matching ``t``. The function
            writes all results in-place. Must have the correct shape and dtype
            (no validation performed).

    Note:
        ``out_deriv[pt, k, i]`` is the k-th derivative of the i-th Bernstein
        basis polynomial of degree n evaluated at ``t[pt]``.  For k > n all
        entries are identically zero.  The ``ndu``-table recurrence is
        well-defined at ``t=1.0``; no special-case handling is required.

        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_tabulate_Bspline_basis_deriv_1D_impl` instead.
    """
    n_pts = t.shape[0]
    for pt_id in nb_prange(n_pts):
        _bernstein_derivs_point(n, t[pt_id], n_deriv, out_deriv[pt_id, :, :])


@nb_jit(
    nopython=True,
    cache=True,
)
def _tabulate_Bernstein_basis_deriv_1D_serial_core(
    n: np.int32,
    t: npt.NDArray[np.float32 | np.float64],
    n_deriv: int,
    out_deriv: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate Bernstein basis function derivatives on [0, 1] (serial twin).

    Identical to :func:`_tabulate_Bernstein_basis_deriv_1D_core` but compiled
    without ``parallel=True``: no fork/join overhead, which makes it the faster
    choice for small point batches.

    Args:
        n (np.int32): Degree of the Bernstein polynomials (>= 0).
        t (npt.NDArray[np.float32 | np.float64]): 1D array of evaluation points
            in [0, 1].
        n_deriv (int): Maximum derivative order to compute (>= 0).
        out_deriv (npt.NDArray[np.float32 | np.float64]): Output array of shape
            ``(len(t), n_deriv+1, n+1)`` and dtype matching ``t``. Must have the
            correct shape and dtype (no validation performed).

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_tabulate_Bspline_basis_deriv_1D_impl` instead.
    """
    n_pts = t.shape[0]
    for pt_id in range(n_pts):
        _bernstein_derivs_point(n, t[pt_id], n_deriv, out_deriv[pt_id, :, :])


def _warmup_numba_functions() -> None:
    """Precompile numba functions with float64 signatures for faster first call.

    This function triggers compilation of the numba-decorated core functions
    with float64 arrays, ensuring they are cached and ready for use.
    """
    # Small dummy arrays for warmup
    t_dummy = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    out_dummy = np.empty((3, 2), dtype=np.float64)

    # Warmup each core function with float64 (parallel and serial twins)
    _tabulate_Bernstein_basis_1D_core(np.int32(1), t_dummy, out_dummy)
    _tabulate_Bernstein_basis_1D_serial_core(np.int32(1), t_dummy, out_dummy)
    _tabulate_cardinal_Bspline_basis_1D_core(np.int32(1), t_dummy, out_dummy)
    _tabulate_Legendre_basis_1D_core(np.int32(1), t_dummy, out_dummy)

    # Warmup Bernstein derivative cores with float64 (parallel and serial twins)
    n_deriv_dummy = 2
    out_deriv_dummy = np.empty((3, n_deriv_dummy + 1, 2), dtype=np.float64)
    _tabulate_Bernstein_basis_deriv_1D_core(np.int32(1), t_dummy, n_deriv_dummy, out_deriv_dummy)
    _tabulate_Bernstein_basis_deriv_1D_serial_core(
        np.int32(1), t_dummy, n_deriv_dummy, out_deriv_dummy
    )


# Precompile numba functions on module import
# (Moved to central thread in __init__.py)
