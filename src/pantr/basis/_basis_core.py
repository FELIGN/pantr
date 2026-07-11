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

_MIRROR_THRESHOLD = 0.5
"""Midpoint used to pick the Bernstein ratio-recurrence branch in `_bernstein_point`.

Bounds the recurrence seed (``(1-u)^n`` or ``u^n``) below by ``0.5^n``, so it
never underflows regardless of degree.
"""

_MAX_SAFE_DEGREE_NO_MIRROR_FLOAT64 = 20
"""Largest float64 degree for which the plain (unmirrored) forward recurrence
cannot underflow for *any* representable ``u``.

The smallest positive ``1 - u`` for a representable float64 ``u < 1`` is half
a ULP below 1, i.e. ``2**-53`` (attained at ``u = np.nextafter(1.0, 0.0)``).
``(2**-53)**n`` first underflows to exactly ``0.0`` (below the smallest
float64 subnormal, ``2**-1074``) once ``53 * n > 1074``, i.e. at ``n = 21``.
Verified empirically: partition of unity holds within ``8*n*eps`` at that
worst-case ``u`` for every ``n`` up to 20, and fails at ``n = 21``. Used by
the tabulation kernels to dispatch to :func:`_bernstein_point_no_mirror`
(bit-identical to the pre-fix recurrence) instead of the mirrored
:func:`_bernstein_point`, avoiding the mirror branch's overhead entirely for
degrees where it can provably never be needed.
"""

_MAX_SAFE_DEGREE_NO_MIRROR_FLOAT32 = 6
"""Float32 analogue of `_MAX_SAFE_DEGREE_NO_MIRROR_FLOAT64`.

The smallest positive ``1 - u`` for representable float32 ``u < 1`` is
``2**-24``; the smallest float32 subnormal is ``2**-149``. Underflow to exact
``0.0`` first occurs once ``24 * n > 149``, i.e. at ``n = 7``. Verified
empirically (see `_MAX_SAFE_DEGREE_NO_MIRROR_FLOAT64`): safe up to ``n = 6``.
"""

_FLOAT64_ITEMSIZE = 8
"""Byte width of a float64 element, used to pick the safe-degree threshold by dtype."""


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

    - ``u <= 0.5``: forward recurrence from ``u = 0``, written directly into
      ``out_row`` in ascending order:
      ``B_0,n(u) = (1-u)^n``; ``B_i,n = B_{i-1},n * ((n-i+1)/i) * (u/(1-u))``.
    - ``u > 0.5``: the same forward recurrence run on ``1-u`` in place of
      ``u`` (seed ``u^n``, ratio ``(1-u)/u``) — which by the symmetry above
      computes ``B_i,n(1-u) = B_{n-i},n(u)`` into ``out_row[i]`` — followed by
      an in-place reversal of ``out_row`` to restore ``B_i,n(u)`` at index
      ``i``. Reusing the *ascending* loop (rather than accumulating directly
      into descending indices) avoids a measured slowdown from the
      decreasing-index store pattern a naive mirrored loop would produce.

    Branching on the midpoint bounds the seed (``(1-u)^n`` or ``u^n``) below
    by ``0.5^n``, so it never underflows for any representable degree ``n``.
    Without the branch, the forward seed ``(1-u)^n`` underflows to exact
    ``0.0`` once ``u`` is close enough to 1 at high degree, and every
    subsequent term (a positive multiple of the previous one) stays zero —
    including ``B_n,n``, whose true value is near 1 — silently breaking
    partition of unity. ``u == 1.0`` needs no special-casing: the mirrored
    branch handles it exactly (``u^n == 1.0``, then every step multiplies by
    ``(1-u)/u == 0.0``, and the reversal is a no-op on the resulting
    ``[1.0, 0.0, ..., 0.0]`` row read backwards).

    Args:
        n (np.int32): Degree of the Bernstein polynomials (>= 0).
        u (np.float32 | np.float64): Evaluation point.
        out_row (npt.NDArray[np.float32 | np.float64]): Output row of length ``n + 1``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_tabulate_Bernstein_basis_1D_impl` instead.
    """
    if u > _MIRROR_THRESHOLD:
        # Mirrored recurrence from u=1, computed as the forward recurrence on
        # 1-u (seed u^n >= 0.5^n never underflows) written ascending into
        # out_row, then reversed in place to land B_i,n(u) at index i.
        one_minus_u = 1.0 - u
        out_row[0] = np.power(u, n)
        one_minus_u_over_u = one_minus_u / u
        for i in range(1, n + 1):
            const_factor = (n - i + 1.0) / i
            out_row[i] = out_row[i - 1] * const_factor * one_minus_u_over_u
        lo = 0
        hi = n
        while lo < hi:
            tmp = out_row[lo]
            out_row[lo] = out_row[hi]
            out_row[hi] = tmp
            lo += 1
            hi -= 1
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
    inline="always",
)
def _bernstein_point_no_mirror(
    n: np.int32,
    u: np.float32 | np.float64,
    out_row: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate Bernstein basis polynomials at one point via the plain forward recurrence.

    Bit-identical to the pre-issue-#258 recurrence: ``B_0,n(u) = (1-u)^n``,
    ``B_i,n = B_{i-1},n * ((n-i+1)/i) * (u/(1-u))``, with ``u == 1.0``
    special-cased directly. Callers must only use this for degrees at or
    below `_MAX_SAFE_DEGREE_NO_MIRROR_FLOAT64` (float64) or
    `_MAX_SAFE_DEGREE_NO_MIRROR_FLOAT32` (float32), where the seed
    ``(1-u)^n`` is proven to never underflow — this function has none of
    :func:`_bernstein_point`'s mirroring, so it is unsafe for higher degrees.
    Kept as a byte-for-byte-unbranched fast path purely for performance: the
    tabulation kernels dispatch to it once per batch (not per point) so that
    low-degree callers, which can never hit the underflow this issue fixes,
    pay zero overhead for the mirror-branch machinery.

    Args:
        n (np.int32): Degree of the Bernstein polynomials (>= 0), at or below
            the relevant `_MAX_SAFE_DEGREE_NO_MIRROR_*` bound for ``u``'s dtype.
        u (np.float32 | np.float64): Evaluation point.
        out_row (npt.NDArray[np.float32 | np.float64]): Output row of length ``n + 1``.

    Note:
        Inputs are assumed to be correct (no validation performed), including
        the degree bound above: this is a private performance fast path, not
        a general-purpose entry point.
        For general use, call :func:`_tabulate_Bernstein_basis_1D_impl` instead.
    """
    if u == 1.0:
        for i in range(out_row.shape[0]):
            out_row[i] = 0.0
        out_row[n] = 1.0
    else:
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

    The degree/dtype check against `_MAX_SAFE_DEGREE_NO_MIRROR_FLOAT64` /
    `_MAX_SAFE_DEGREE_NO_MIRROR_FLOAT32` happens once per call (not once per
    point), then dispatches every point to the branch-free
    :func:`_bernstein_point_no_mirror` when safe, or the mirrored
    :func:`_bernstein_point` otherwise — this keeps the (measured) mirror
    branch's overhead confined to the rare degrees that actually need it.

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

    max_safe_degree = (
        _MAX_SAFE_DEGREE_NO_MIRROR_FLOAT64
        if t.itemsize == _FLOAT64_ITEMSIZE
        else _MAX_SAFE_DEGREE_NO_MIRROR_FLOAT32
    )
    # Process each point — rows are independent, so use prange.
    if n <= max_safe_degree:
        for j in nb_prange(t.shape[0]):
            _bernstein_point_no_mirror(n, t[j], out[j, :])
    else:
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
    for small point batches. Dispatches to the branch-free
    :func:`_bernstein_point_no_mirror` or the mirrored :func:`_bernstein_point`
    exactly as described there.

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

    max_safe_degree = (
        _MAX_SAFE_DEGREE_NO_MIRROR_FLOAT64
        if t.itemsize == _FLOAT64_ITEMSIZE
        else _MAX_SAFE_DEGREE_NO_MIRROR_FLOAT32
    )
    if n <= max_safe_degree:
        for j in range(t.shape[0]):
            _bernstein_point_no_mirror(n, t[j], out[j, :])
    else:
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

    # The Bernstein tabulation kernels dispatch on degree (see
    # `_MAX_SAFE_DEGREE_NO_MIRROR_FLOAT64`): the n=1 warmup above only compiles
    # the branch-free `_bernstein_point_no_mirror` fast path, so warm up the
    # mirrored `_bernstein_point` path too with a degree above the threshold.
    n_mirrored_dummy = _MAX_SAFE_DEGREE_NO_MIRROR_FLOAT64 + 1
    out_mirrored_dummy = np.empty((3, n_mirrored_dummy + 1), dtype=np.float64)
    _tabulate_Bernstein_basis_1D_core(np.int32(n_mirrored_dummy), t_dummy, out_mirrored_dummy)
    _tabulate_Bernstein_basis_1D_serial_core(
        np.int32(n_mirrored_dummy), t_dummy, out_mirrored_dummy
    )

    # Warmup Bernstein derivative cores with float64 (parallel and serial twins)
    n_deriv_dummy = 2
    out_deriv_dummy = np.empty((3, n_deriv_dummy + 1, 2), dtype=np.float64)
    _tabulate_Bernstein_basis_deriv_1D_core(np.int32(1), t_dummy, n_deriv_dummy, out_deriv_dummy)
    _tabulate_Bernstein_basis_deriv_1D_serial_core(
        np.int32(1), t_dummy, n_deriv_dummy, out_deriv_dummy
    )


# Precompile numba functions on module import
# (Moved to central thread in __init__.py)
