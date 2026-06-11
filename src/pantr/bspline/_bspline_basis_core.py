"""Core B-spline basis function evaluation implementations.

This module provides core functions for evaluating B-spline basis functions
using the BasisFuncs algorithm (Piegl & Tiller) and Bernstein-like evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .._numba_compat import nb_jit, nb_prange
from ..basis._basis_1D import _tabulate_Bernstein_basis_1D_impl
from ..basis._basis_core import (
    _PARALLEL_MIN_NUM_PTS,
    _tabulate_Bernstein_basis_deriv_1D_core,
    _tabulate_Bernstein_basis_deriv_1D_serial_core,
)
from ..basis._basis_utils import (
    _compute_final_output_shape_1D,
    _compute_final_output_shape_1D_deriv,
    _normalize_points_1D,
    _validate_out_array,
)
from ._bspline_knots import (
    _get_Bspline_num_basis_1D_impl,
    _get_last_knot_smaller_equal_impl,
    _is_in_domain_impl,
)

if TYPE_CHECKING:
    from ._bspline_space_1d import BsplineSpace1D


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _find_spans_and_first_basis(  # noqa: PLR0913
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    periodic: bool,
    tol: float,
    pts: npt.NDArray[np.float32 | np.float64],
    out_first_basis: npt.NDArray[np.int_],
) -> npt.NDArray[np.int_]:
    """Compute clamped knot-span indices and fill the first-basis indices.

    Shared preamble of the BasisFuncs / DerBasisFuncs kernels.  Three steps:

    1. Locate each point's raw knot span via binary search.
    2. Clamp each span index to ``knots.size - degree - 2`` (the last in-domain
       span).  Without clamping, a point at the right domain endpoint can be
       placed in an out-of-domain span, causing Cox-de Boor to read knot values
       beyond the last valid span.
    3. Compute ``out_first_basis``: for non-periodic splines the index is
       additionally clamped so the final evaluation point always addresses the
       last ``degree + 1`` active basis functions; for periodic splines the raw
       (unclamped) index is used so the evaluation loop can wrap it modulo the
       number of control points.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        periodic (bool): Whether the B-spline is periodic.
        tol (float): Tolerance for numerical comparisons.
        pts (npt.NDArray[np.float32 | np.float64]): Points (1D array) to evaluate at.
        out_first_basis (npt.NDArray[np.int_]): Output array for first basis indices.
            Must have shape (n_pts,) and dtype int.

    Returns:
        npt.NDArray[np.int_]: Clamped knot-span index per point, shape (n_pts,).

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_tabulate_Bspline_basis_1D_impl` instead.
    """
    knot_ids = _get_last_knot_smaller_equal_impl(knots, pts)

    # Clamp knot span indices to the last in-domain span.  For non-open knot
    # vectors the right domain endpoint knots[-degree-1] is not the last knot
    # in the vector, so searchsorted can place a point in an out-of-domain
    # span.  Clamping to knots.size - degree - 2 ensures the Cox-de Boor
    # recurrence always uses an in-domain span.  For open knot vectors this
    # subsumes the former ``knot_id == knots.size - 1`` special case.
    max_knot_id = knots.size - degree - 2
    knot_ids = np.minimum(knot_ids, max_knot_id)

    # For non-periodic splines, clamp first_basis so the last (degree+1) basis
    # functions are addressed by the final evaluation point.  For periodic
    # splines, the unclamped index is needed: the evaluation loop wraps it via
    # modulo to cycle through the periodic control points.
    if periodic:
        out_first_basis[:] = knot_ids - degree
    else:
        order = degree + 1
        num_basis = _get_Bspline_num_basis_1D_impl(knots, degree, periodic, tol)
        out_first_basis[:] = np.minimum(knot_ids - degree, num_basis - order)

    return knot_ids


@nb_jit(
    nopython=True,
    cache=True,
    inline="always",
)
def _basis_funcs_point(  # noqa: PLR0913
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    knot_id: int,
    pt: np.float32 | np.float64,
    N: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate the nonzero B-spline basis functions at one point (A2.2 body).

    Inlined (``inline="always"``) into the batch kernels so the per-point
    scratch allocations stay visible to Numba's parallel allocation hoisting.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        tol (float): Tolerance for numerical comparisons.
        knot_id (int): Clamped knot-span index of ``pt``.
        pt (np.float32 | np.float64): Evaluation point.
        N (npt.NDArray[np.float32 | np.float64]): Output row of length ``degree + 1``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    order = degree + 1
    dtype = knots.dtype
    zero = dtype.type(0.0)
    one = dtype.type(1.0)

    left = np.zeros(order, dtype=dtype)
    right = np.zeros(order, dtype=dtype)
    N[0] = one

    for j in range(1, order):
        left[j] = pt - knots[knot_id + 1 - j]
        right[j] = knots[knot_id + j] - pt
        saved = zero

        for r in range(j):
            denom = right[r + 1] + left[j - r]  # always >= 0 (non-decreasing knots)
            temp = zero if denom < tol else N[r] / denom
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp

        N[j] = saved


@nb_jit(
    nopython=True,
    cache=True,
    parallel=True,
)
def _compute_basis_nurbs_book_impl(  # noqa: PLR0913
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    periodic: bool,
    tol: float,
    pts: npt.NDArray[np.float32 | np.float64],
    out_basis: npt.NDArray[np.float32 | np.float64],
    out_first_basis: npt.NDArray[np.int_],
) -> None:
    """Evaluate B-spline basis functions using BasisFuncs (Piegl & Tiller A2.2).

    This function implements Algorithm A2.2 from "The NURBS Book" by Piegl & Tiller.
    Results are written directly to the output arrays (C-style).  The outer loop
    over evaluation points is parallelized via ``prange``; for small batches
    (fewer than ``_PARALLEL_MIN_NUM_PTS`` points) prefer the serial twin
    :func:`_compute_basis_nurbs_book_serial_impl`, which avoids the parallel
    launch overhead.

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
    # See The NURBS Book, by Piegl & Tiller. Algorithm A2.2 (BasisFuncs)
    n_pts = pts.size
    knot_ids = _find_spans_and_first_basis(knots, degree, periodic, tol, pts, out_first_basis)

    for pt_id in nb_prange(n_pts):
        _basis_funcs_point(knots, degree, tol, knot_ids[pt_id], pts[pt_id], out_basis[pt_id, :])


@nb_jit(
    nopython=True,
    cache=True,
)
def _compute_basis_nurbs_book_serial_impl(  # noqa: PLR0913
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    periodic: bool,
    tol: float,
    pts: npt.NDArray[np.float32 | np.float64],
    out_basis: npt.NDArray[np.float32 | np.float64],
    out_first_basis: npt.NDArray[np.int_],
) -> None:
    """Evaluate B-spline basis functions using BasisFuncs (serial twin).

    Identical to :func:`_compute_basis_nurbs_book_impl` but compiled without
    ``parallel=True``: no fork/join overhead, which makes it the faster choice
    for small point batches (FEM/IGA per-cell assembly).

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
    n_pts = pts.size
    knot_ids = _find_spans_and_first_basis(knots, degree, periodic, tol, pts, out_first_basis)

    for pt_id in range(n_pts):
        _basis_funcs_point(knots, degree, tol, knot_ids[pt_id], pts[pt_id], out_basis[pt_id, :])


@nb_jit(
    nopython=True,
    cache=True,
    inline="always",
)
def _basis_derivs_point(  # noqa: PLR0913
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    n_deriv: int,
    knot_id: int,
    pt: np.float32 | np.float64,
    out_pt: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate the nonzero B-spline basis derivatives at one point (A2.3 body).

    Inlined (``inline="always"``) into the batch kernels so the per-point
    scratch allocations stay visible to Numba's parallel allocation hoisting.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        tol (float): Tolerance for numerical comparisons.
        n_deriv (int): Maximum derivative order to compute (>= 0).
        knot_id (int): Clamped knot-span index of ``pt``.
        pt (np.float32 | np.float64): Evaluation point.
        out_pt (npt.NDArray[np.float32 | np.float64]): Output block of shape
            ``(n_deriv + 1, degree + 1)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    order = degree + 1
    dtype = knots.dtype
    zero = dtype.type(0.0)
    one = dtype.type(1.0)

    ndu = np.zeros((order, order), dtype=dtype)
    left = np.zeros(order, dtype=dtype)
    right = np.zeros(order, dtype=dtype)
    a = np.zeros((2, n_deriv + 1), dtype=dtype)

    # --- Step 1: build ndu table (A2.2 extended to retain intermediate values) ---
    ndu[0, 0] = one
    for j in range(1, order):
        left[j] = pt - knots[knot_id + 1 - j]
        right[j] = knots[knot_id + j] - pt
        saved = zero
        for r in range(j):
            ndu[j, r] = right[r + 1] + left[j - r]  # knot differences (lower triangle)
            denom = ndu[j, r]
            temp = zero if denom < tol else ndu[r, j - 1] / denom
            ndu[r, j] = saved + right[r + 1] * temp  # basis values (upper triangle)
            saved = left[j - r] * temp
        ndu[j, j] = saved

    # Store 0th-order derivatives (basis values)
    for j in range(order):
        out_pt[0, j] = ndu[j, degree]

    # --- Step 2: compute kth derivatives via triangular recursion ---
    for r in range(order):
        s1 = 0
        s2 = 1
        a[0, 0] = one

        for k in range(1, n_deriv + 1):
            d = zero
            rk = r - k
            pk = degree - k

            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]

            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if (r - 1) <= pk else degree - r

            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]

            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]

            out_pt[k, r] = d

            # swap rows
            j = s1
            s1 = s2
            s2 = j

    # --- Step 3: apply degree factorial scaling factors ---
    # When k > degree, fac becomes 0 (degree - k ≤ 0), so rows beyond degree
    # are zeroed out.  This is intentional: the k-th derivative of a degree-p
    # polynomial is identically zero for k > p.
    fac = degree
    for k in range(1, n_deriv + 1):
        for j in range(order):
            out_pt[k, j] *= fac
        fac *= degree - k


@nb_jit(
    nopython=True,
    cache=True,
    parallel=True,
)
def _compute_basis_deriv_nurbs_book_impl(  # noqa: PLR0913
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    periodic: bool,
    tol: float,
    n_deriv: int,
    pts: npt.NDArray[np.float32 | np.float64],
    out_deriv: npt.NDArray[np.float32 | np.float64],
    out_first_basis: npt.NDArray[np.int_],
) -> None:
    """Evaluate B-spline basis function derivatives using DerBasisFuncs (Piegl & Tiller A2.3).

    This function implements Algorithm A2.3 from "The NURBS Book" by Piegl & Tiller.
    Results are written directly to the output arrays (C-style).  The outer loop
    over evaluation points is parallelized via ``prange``; for small batches
    (fewer than ``_PARALLEL_MIN_NUM_PTS`` points) prefer the serial twin
    :func:`_compute_basis_deriv_nurbs_book_serial_impl`, which avoids the
    parallel launch overhead.

    The 0th-order slice ``out_deriv[pt, 0, :]`` contains the plain basis values,
    identical to the output of ``_compute_basis_nurbs_book_impl``.  For
    ``n_deriv > degree`` all rows beyond ``degree`` are identically zero.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        periodic (bool): Whether the B-spline is periodic.
        tol (float): Tolerance for numerical comparisons.
        n_deriv (int): Maximum derivative order to compute (>= 0).
        pts (npt.NDArray[np.float32 | np.float64]): Points (1D array) to evaluate.
        out_deriv (npt.NDArray[np.float32 | np.float64]): Output array for derivative values.
            Must have shape (n_pts, n_deriv+1, degree+1) and dtype matching ``knots``.
        out_first_basis (npt.NDArray[np.int_]): Output array for first basis indices.
            Must have shape (n_pts,) and dtype int.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    # See The NURBS Book, by Piegl & Tiller. Algorithm A2.3 (DerBasisFuncs)
    n_pts = pts.size
    knot_ids = _find_spans_and_first_basis(knots, degree, periodic, tol, pts, out_first_basis)

    for pt_id in nb_prange(n_pts):
        _basis_derivs_point(
            knots, degree, tol, n_deriv, knot_ids[pt_id], pts[pt_id], out_deriv[pt_id, :, :]
        )


@nb_jit(
    nopython=True,
    cache=True,
)
def _compute_basis_deriv_nurbs_book_serial_impl(  # noqa: PLR0913
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    periodic: bool,
    tol: float,
    n_deriv: int,
    pts: npt.NDArray[np.float32 | np.float64],
    out_deriv: npt.NDArray[np.float32 | np.float64],
    out_first_basis: npt.NDArray[np.int_],
) -> None:
    """Evaluate B-spline basis function derivatives using DerBasisFuncs (serial twin).

    Identical to :func:`_compute_basis_deriv_nurbs_book_impl` but compiled
    without ``parallel=True``: no fork/join overhead, which makes it the faster
    choice for small point batches (FEM/IGA per-cell assembly).

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        periodic (bool): Whether the B-spline is periodic.
        tol (float): Tolerance for numerical comparisons.
        n_deriv (int): Maximum derivative order to compute (>= 0).
        pts (npt.NDArray[np.float32 | np.float64]): Points (1D array) to evaluate.
        out_deriv (npt.NDArray[np.float32 | np.float64]): Output array for derivative values.
            Must have shape (n_pts, n_deriv+1, degree+1) and dtype matching ``knots``.
        out_first_basis (npt.NDArray[np.int_]): Output array for first basis indices.
            Must have shape (n_pts,) and dtype int.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n_pts = pts.size
    knot_ids = _find_spans_and_first_basis(knots, degree, periodic, tol, pts, out_first_basis)

    for pt_id in range(n_pts):
        _basis_derivs_point(
            knots, degree, tol, n_deriv, knot_ids[pt_id], pts[pt_id], out_deriv[pt_id, :, :]
        )


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
        _validate_out_array(out_first_basis, expected_first_basis_shape, np.int_)

    # the first basis function is always the 0
    out_first_basis.fill(0)

    # Compute Bernstein basis - pass out_basis directly since pts_normalized is already 1D
    # and _tabulate_Bernstein_basis_1D_impl will handle shape validation
    B = _tabulate_Bernstein_basis_1D_impl(spline.degree, pts_normalized, out=out_basis)

    return B, out_first_basis


def _tabulate_Bspline_basis_Bernstein_like_deriv_1D(
    spline: BsplineSpace1D,
    pts: npt.NDArray[np.float32 | np.float64],
    n_deriv: int,
    out_deriv: npt.NDArray[np.float32 | np.float64],
    out_first_basis: npt.NDArray[np.int_],
) -> None:
    """Evaluate B-spline basis derivatives for Bézier-like knots via Bernstein polynomials.

    Maps the evaluation points to the reference interval [0, 1], delegates to the
    parallel Bernstein derivative kernel, then applies the chain-rule correction
    ``(1/(b-a))^k`` to each k-th derivative slice.

    Args:
        spline (BsplineSpace1D): B-spline with Bézier-like knots.
        pts (npt.NDArray[np.float32 | np.float64]): Evaluation points (1D, already
            normalized by :func:`_normalize_points_1D`).
        n_deriv (int): Maximum derivative order to compute (>= 0).
        out_deriv (npt.NDArray[np.float32 | np.float64]): Pre-allocated output array
            of shape ``(n_pts, n_deriv+1, degree+1)`` and dtype matching ``pts``.
        out_first_basis (npt.NDArray[np.int_]): Pre-allocated output array of shape
            ``(n_pts,)`` and dtype int.

    Raises:
        ValueError: If the B-spline does not have Bézier-like knots.
    """
    if not spline.has_Bezier_like_knots():
        raise ValueError("B-spline does not have Bézier-like knots.")

    k0, k1 = spline.domain
    pts_normalized = (pts - k0) / (k1 - k0)  # map to [0, 1]

    deriv_core = (
        _tabulate_Bernstein_basis_deriv_1D_core
        if pts_normalized.shape[0] >= _PARALLEL_MIN_NUM_PTS
        else _tabulate_Bernstein_basis_deriv_1D_serial_core
    )
    deriv_core(np.int32(spline.degree), pts_normalized, n_deriv, out_deriv)

    # Chain-rule: d^k/dx^k f(x) = d^k/ds^k f(s) * (ds/dx)^k = d^k/ds^k f(s) * (1/(k1-k0))^k
    inv_span: float = 1.0 / float(k1 - k0)
    scale: float = inv_span
    for k in range(1, n_deriv + 1):
        out_deriv[:, k, :] = out_deriv[:, k, :] * scale
        scale *= inv_span

    out_first_basis.fill(0)


def _tabulate_Bspline_basis_1D_impl(
    spline: BsplineSpace1D,
    pts: npt.ArrayLike,
    out_basis: npt.NDArray[np.float32 | np.float64] | None = None,
    out_first_basis: npt.NDArray[np.int_] | None = None,
    *,
    validate: bool = True,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    """Evaluate B-spline basis functions at given points.

    This function automatically selects the most efficient evaluation method:
    - For Bézier-like knots: direct Bernstein evaluation
    - For general knots: BasisFuncs (Piegl & Tiller A2.2).  Batches smaller
      than ``_PARALLEL_MIN_NUM_PTS`` use the serial (non-parallel) kernel to
      avoid fork/join overhead; larger batches use the ``parallel=True`` kernel.

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
        validate (bool): If True (default), check that every point lies inside the
            spline domain. Pass False only when the caller guarantees in-domain
            points (e.g. points generated inside a knot span); out-of-domain
            points are then undefined behavior. Defaults to True.

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
        ValueError: If ``validate`` is True and any evaluation point is outside the
            B-spline domain, or if `out_basis` or `out_first_basis` is provided and
            has incorrect shape or dtype.

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

    if validate and not np.all(
        _is_in_domain_impl(spline.knots, spline.degree, pts, spline.tolerance)
    ):
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
    _validate_out_array(out_basis, expected_final_shape, expected_dtype)
    basis_normalized = out_basis.reshape(num_pts, n_basis)

    if out_first_basis is None:
        out_first_basis = np.empty(expected_first_basis_shape, dtype=np.int_)
    _validate_out_array(out_first_basis, expected_first_basis_shape, np.int_)
    first_indices_normalized = out_first_basis.reshape(num_pts)

    if spline.has_Bezier_like_knots():
        _tabulate_Bspline_basis_Bernstein_like_1D(
            spline, pts, basis_normalized, first_indices_normalized
        )
    else:
        kernel = (
            _compute_basis_nurbs_book_impl
            if num_pts >= _PARALLEL_MIN_NUM_PTS
            else _compute_basis_nurbs_book_serial_impl
        )
        kernel(
            spline.knots,
            spline.degree,
            spline.periodic,
            spline.tolerance,
            pts,
            basis_normalized,
            first_indices_normalized,
        )

    return out_basis, out_first_basis


def _tabulate_Bspline_basis_deriv_1D_impl(  # noqa: PLR0913
    spline: BsplineSpace1D,
    pts: npt.ArrayLike,
    n_deriv: int,
    out_deriv: npt.NDArray[np.float32 | np.float64] | None = None,
    out_first_basis: npt.NDArray[np.int_] | None = None,
    *,
    validate: bool = True,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    """Evaluate B-spline basis function derivatives at given points.

    Implements Algorithm A2.3 (DerBasisFuncs) from Piegl & Tiller.  Uses a
    fast Bernstein path for Bézier-like knots (parallel kernel + chain-rule
    scaling) and falls back to the general DerBasisFuncs kernel otherwise.
    For general knots, batches smaller than ``_PARALLEL_MIN_NUM_PTS`` use the
    serial twin to avoid fork/join overhead.
    The 0th slice of the result is identical to the output of
    :func:`_tabulate_Bspline_basis_1D_impl`.  For ``n_deriv > degree`` all
    rows beyond ``degree`` are identically zero.

    Args:
        spline (BsplineSpace1D): B-spline object defining the basis.
        pts (npt.ArrayLike): Evaluation points.
        n_deriv (int): Maximum derivative order to compute (>= 0).
        out_deriv (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            for derivative values. If None, a new array is allocated. Must have shape
            ``(*pts_shape, n_deriv+1, degree+1)`` and dtype matching ``pts`` if provided.
            Defaults to None.
        out_first_basis (npt.NDArray[np.int_] | None): Optional output array for first
            basis indices. If None, a new array is allocated. Must have shape ``pts_shape``
            and dtype ``np.int_`` if provided. Defaults to None.
        validate (bool): If True (default), check that every point lies inside the
            spline domain. Pass False only when the caller guarantees in-domain
            points (e.g. points generated inside a knot span); out-of-domain
            points are then undefined behavior. Defaults to True.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]: Tuple of
            ``(deriv_values, first_basis_indices)``.
            ``deriv_values[..., k, i]`` is the k-th derivative of the i-th local basis
            function at each point.

    Raises:
        ValueError: If ``n_deriv < 0``, if ``validate`` is True and any evaluation
            point is outside the domain, or ``out_deriv`` / ``out_first_basis`` has
            incorrect shape or dtype.

    Example:
        >>> bspline = BsplineSpace1D([0, 0, 0, 1, 1, 1], 2)
        >>> d, first = _tabulate_Bspline_basis_deriv_1D_impl(bspline, [0.5], n_deriv=1)
        >>> d.shape
        (1, 2, 3)
    """
    if n_deriv < 0:
        raise ValueError(f"n_deriv must be non-negative, got {n_deriv}")

    input_shape = np.shape(pts)
    pts = _normalize_points_1D(pts)

    if validate and not np.all(
        _is_in_domain_impl(spline.knots, spline.degree, pts, spline.tolerance)
    ):
        raise ValueError(
            f"One or more values in pts are outside the knot vector domain {spline.domain}"
        )

    num_pts = pts.shape[0]
    order = spline.degree + 1
    expected_dtype = pts.dtype
    expected_deriv_shape = _compute_final_output_shape_1D_deriv(input_shape, n_deriv, order)
    expected_first_basis_shape = input_shape

    if out_deriv is None:
        out_deriv = np.empty(expected_deriv_shape, dtype=expected_dtype)
    _validate_out_array(out_deriv, expected_deriv_shape, expected_dtype)
    deriv_normalized = out_deriv.reshape(num_pts, n_deriv + 1, order)

    if out_first_basis is None:
        out_first_basis = np.empty(expected_first_basis_shape, dtype=np.int_)
    _validate_out_array(out_first_basis, expected_first_basis_shape, np.int_)
    first_indices_normalized = out_first_basis.reshape(num_pts)

    if spline.has_Bezier_like_knots():
        _tabulate_Bspline_basis_Bernstein_like_deriv_1D(
            spline, pts, n_deriv, deriv_normalized, first_indices_normalized
        )
    else:
        kernel = (
            _compute_basis_deriv_nurbs_book_impl
            if num_pts >= _PARALLEL_MIN_NUM_PTS
            else _compute_basis_deriv_nurbs_book_serial_impl
        )
        kernel(
            spline.knots,
            spline.degree,
            spline.periodic,
            spline.tolerance,
            n_deriv,
            pts,
            deriv_normalized,
            first_indices_normalized,
        )

    return out_deriv, out_first_basis


def _warmup_numba_functions() -> None:
    """Precompile numba functions with float64 signatures for faster first call.

    Triggers compilation of the parallel and serial twin kernels (BasisFuncs,
    DerBasisFuncs) and the Bernstein derivative core with float64 arrays,
    ensuring they are cached and ready for use.
    """
    # Small dummy arrays for warmup
    knots_dummy = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    pts_dummy = np.array([0.5], dtype=np.float64)
    tol_dummy = 1e-10
    degree_dummy = 2
    n_pts_dummy = pts_dummy.size
    basis_dummy = np.empty((n_pts_dummy, degree_dummy + 1), dtype=np.float64)
    first_basis_dummy = np.empty(n_pts_dummy, dtype=np.int_)

    # Warmup BasisFuncs implementation with float64 (parallel and serial twins)
    _compute_basis_nurbs_book_impl(
        knots_dummy, degree_dummy, False, tol_dummy, pts_dummy, basis_dummy, first_basis_dummy
    )
    _compute_basis_nurbs_book_serial_impl(
        knots_dummy, degree_dummy, False, tol_dummy, pts_dummy, basis_dummy, first_basis_dummy
    )

    # Warmup DerBasisFuncs implementation with float64 (parallel and serial twins)
    n_deriv_dummy = 2
    deriv_dummy = np.empty((n_pts_dummy, n_deriv_dummy + 1, degree_dummy + 1), dtype=np.float64)
    _compute_basis_deriv_nurbs_book_impl(
        knots_dummy,
        degree_dummy,
        False,
        tol_dummy,
        n_deriv_dummy,
        pts_dummy,
        deriv_dummy,
        first_basis_dummy,
    )
    _compute_basis_deriv_nurbs_book_serial_impl(
        knots_dummy,
        degree_dummy,
        False,
        tol_dummy,
        n_deriv_dummy,
        pts_dummy,
        deriv_dummy,
        first_basis_dummy,
    )

    # Warmup Bernstein derivative core (Bézier fast path) with float64
    pts_norm_dummy = pts_dummy  # knots_dummy already has [0,1] domain
    _tabulate_Bernstein_basis_deriv_1D_core(
        np.int32(degree_dummy), pts_norm_dummy, n_deriv_dummy, deriv_dummy
    )


# Precompile numba functions on module import
# (Moved to central thread in __init__.py)


__all__ = [
    "_compute_basis_deriv_nurbs_book_impl",
    "_compute_basis_deriv_nurbs_book_serial_impl",
    "_compute_basis_nurbs_book_impl",
    "_compute_basis_nurbs_book_serial_impl",
    "_tabulate_Bspline_basis_1D_impl",
    "_tabulate_Bspline_basis_Bernstein_like_1D",
    "_tabulate_Bspline_basis_Bernstein_like_deriv_1D",
    "_tabulate_Bspline_basis_deriv_1D_impl",
]
