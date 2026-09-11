"""The Numba kernels of the general-knot B-spline basis tabulation, and nothing else.

Layer 3 in the layering of ``CLAUDE.md``: pure computation, no input validation. Every
correctness guarantee comes from the Layer 2 entry points in
:mod:`pantr.bspline._bspline_basis_core`, which validate shapes, dtypes and domain
membership before any of these runs.

**This module exists so that a catalogue can import it at module scope.**
:mod:`pantr.bspline._basis_backend` has to name these kernels, and it used to defer the
import into its two catalogue functions because the module that held them --
``_bspline_basis_core`` -- also holds the Layer 2 entry points, which ask the catalogue
which kernel to call. Holding both halves in one module closed a cycle that a
module-scope import would have tripped over. Splitting them the way
:mod:`pantr.bspline._bspline_extraction_core` is split opens it: nothing here imports
the catalogue, so the catalogue can import this directly.

The contents are unchanged from where they lived before, line for line. What moved is
which file they are in.

- :func:`_find_spans_and_first_basis`, :func:`_find_span_and_first_basis_point`: the
  span search and the first non-zero basis index.
- :func:`_basis_funcs_point`, :func:`_basis_derivs_point`: Piegl & Tiller A2.2 and A2.3
  at one point.
- The four ``_compute_basis_*_impl`` kernels: the parallel entry points and their serial
  twins, which :mod:`pantr.bspline._basis_backend` catalogues.
- :func:`_warmup_numba_functions`: compiles them at import time, as every other ``core``
  module in this package does for its own.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._numba_compat import nb_jit, nb_prange
from ._bspline_knots import (
    _get_Bspline_num_basis_1D_impl,
    _get_last_knot_smaller_equal_impl,
)


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
    # Clamp upper end (right-of-domain points → last valid span).
    knot_ids = np.minimum(knot_ids, max_knot_id)
    # Clamp lower end: ensures knot_id + 1 - j >= 0 for all j in range(1, degree+1),
    # so _basis_funcs_point never wraps into negative knot indices for points left of
    # the domain.  In-place boolean assignment avoids calling np.maximum (whose return
    # type is Any on numpy stubs).
    knot_ids[knot_ids < degree] = degree

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
def _find_span_and_first_basis_point(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    periodic: bool,
    num_basis: int,
    pt: np.float32 | np.float64,
) -> tuple[int, int]:
    """Compute the clamped knot-span and first-basis index for one point.

    Per-point body of :func:`_find_spans_and_first_basis`.  Inlined
    (``inline="always"``) into the parallel ``BasisFuncs``/``DerBasisFuncs``
    kernels so span search and Cox-de Boor evaluation both run inside the same
    ``prange`` iteration, instead of a first serial pass over all points
    followed by a second parallel pass. See :func:`_find_spans_and_first_basis`
    for the two-step clamping rationale (out-of-domain spans, negative knot
    indices); the logic here is the scalar equivalent.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        periodic (bool): Whether the B-spline is periodic.
        num_basis (int): Total number of basis functions, precomputed once by
            the caller (e.g. via :func:`_get_Bspline_num_basis_1D_impl`) since
            it does not depend on the point. Ignored when ``periodic`` is True
            (callers may pass any placeholder value in that case).
        pt (np.float32 | np.float64): Evaluation point.

    Returns:
        tuple[int, int]: ``(knot_id, first_basis)`` — the clamped knot-span
        index and the index of the first active basis function at ``pt``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_tabulate_Bspline_basis_1D_impl` or
        :func:`_tabulate_Bspline_basis_deriv_1D_impl` instead.
    """
    knot_id = int(np.searchsorted(knots, pt, side="right")) - 1

    # Clamp to the last in-domain span (see _find_spans_and_first_basis).
    max_knot_id = knots.size - degree - 2
    knot_id = min(knot_id, max_knot_id)
    knot_id = max(knot_id, degree)

    if periodic:
        first_basis = knot_id - degree
    else:
        order = degree + 1
        first_basis = knot_id - degree
        max_first_basis = num_basis - order
        first_basis = min(first_basis, max_first_basis)

    return knot_id, first_basis


@nb_jit(
    nopython=True,
    cache=True,
    inline="always",
)
def _basis_funcs_point(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
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
        knot_id (int): Clamped knot-span index of ``pt``.
        pt (np.float32 | np.float64): Evaluation point.
        N (npt.NDArray[np.float32 | np.float64]): Output row of length ``degree + 1``.

    Note:
        Inputs are assumed to be correct (no validation performed). The recurrence
        denominator is a sum of two knot differences and is always ``>= 0`` for a
        non-decreasing knot vector; it is treated as zero (and the corresponding
        term dropped) exactly when ``denom == 0`` (Piegl & Tiller A2.2's textbook
        guard). No tolerance is needed, and -- contrary to what this note used to
        claim -- the guard does **not** rest on knots meant to be equal being
        stored bitwise identical.

        What makes it sound is the shape of the step. ``denom`` is
        ``right[r + 1] + left[j - r]``, the sum of the two non-negative terms the
        step then multiplies ``temp = N[r] / denom`` by, so the two contributions
        carry weights ``right[r + 1] / denom`` and ``left[j - r] / denom``, which
        are non-negative and add to one up to a single rounding. A denominator
        small enough to make ``temp`` large is multiplied straight back by factors
        no larger than itself, so each contribution stays bounded by ``N[r]``.
        Near-duplicate knots cost accuracy in that ratio, not boundedness.
        Measured with ``snap_knots=False`` and two interior knots separated by 0 to
        64 ulp, at degrees 1 to 5 and knot bases 1.0 and 1e6: ``max|N|`` is 1.0
        exactly and the partition of unity holds to 6.7e-16.

        ``denom == 0`` is reached only when both terms are exactly zero, which is
        what an empty knot span is, and the test is scale-invariant by
        construction, so the guard is unaffected by the knot vector's parametric
        span (shift or scale).
        For general use, call :func:`_tabulate_Bspline_basis_1D_impl` instead.
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
            temp = zero if denom == zero else N[r] / denom
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
    Results are written directly to the output arrays (C-style).  Each point's span
    search and Cox-de Boor evaluation are independent, so both are fused into a
    single ``prange`` loop over evaluation points (span search alone does not
    parallelize well enough on its own to be worth a separate pass — see
    :func:`_find_spans_and_first_basis`, which remains serial for the small-batch
    twin below).  For small batches (fewer than ``_PARALLEL_MIN_NUM_PTS`` points)
    prefer the serial twin :func:`_compute_basis_nurbs_book_serial_impl`, which
    avoids the parallel launch overhead.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        periodic (bool): Whether the B-spline is periodic.
        tol (float): Absolute per-dtype tolerance, passed to
            :func:`_get_Bspline_num_basis_1D_impl` for interface consistency with
            the space's other tolerance-based conventions; unused in practice here
            since that call always passes ``periodic=False`` (the periodic case
            short-circuits ``num_basis`` to 0 above, skipping the call entirely).
            The Cox-de Boor denominator guard itself needs no tolerance; see
            :func:`_basis_funcs_point`.
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
    # num_basis is only used by the non-periodic clamp in _find_span_and_first_basis_point;
    # skip computing it for periodic splines, matching _find_spans_and_first_basis's cost.
    num_basis = 0 if periodic else _get_Bspline_num_basis_1D_impl(knots, degree, periodic, tol)

    for pt_id in nb_prange(n_pts):
        pt = pts[pt_id]
        knot_id, first_basis = _find_span_and_first_basis_point(
            knots, degree, periodic, num_basis, pt
        )
        out_first_basis[pt_id] = first_basis
        _basis_funcs_point(knots, degree, knot_id, pt, out_basis[pt_id, :])


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
        tol (float): Absolute per-dtype tolerance, threaded through to
            :func:`_find_spans_and_first_basis` for interface consistency with the
            space's other tolerance-based conventions (e.g.
            :func:`_get_Bspline_num_basis_1D_impl`'s periodic-regularity path); it
            does not affect this function's own behavior. The Cox-de Boor
            denominator guard itself needs no tolerance; see
            :func:`_basis_funcs_point`.
        pts (npt.NDArray[np.float32 | np.float64]): Points (1D array) to evaluate basis
            functions at.
        out_basis (npt.NDArray[np.float32 | np.float64]): Output array for basis values.
            Must have shape (n_pts, degree+1) and dtype matching the `knots` dtype.
        out_first_basis (npt.NDArray[np.int_]): Output array for first basis indices.
            Must have shape (n_pts,) and dtype int.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_tabulate_Bspline_basis_1D_impl` instead.
    """
    n_pts = pts.size
    knot_ids = _find_spans_and_first_basis(knots, degree, periodic, tol, pts, out_first_basis)

    for pt_id in range(n_pts):
        _basis_funcs_point(knots, degree, knot_ids[pt_id], pts[pt_id], out_basis[pt_id, :])


@nb_jit(
    nopython=True,
    cache=True,
    inline="always",
)
def _basis_derivs_point(  # noqa: PLR0913
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
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
        n_deriv (int): Maximum derivative order to compute (>= 0).
        knot_id (int): Clamped knot-span index of ``pt``.
        pt (np.float32 | np.float64): Evaluation point.
        out_pt (npt.NDArray[np.float32 | np.float64]): Output block of shape
            ``(n_deriv + 1, degree + 1)``.

    Note:
        Inputs are assumed to be correct (no validation performed). The recurrence
        denominator ``ndu[j, r]`` is a sum of two knot differences and is always
        ``>= 0`` for a non-decreasing knot vector; it is treated as zero (and the
        corresponding term dropped) exactly when ``denom == 0`` (Piegl & Tiller
        A2.3's textbook guard). No tolerance is needed, and the guard does **not**
        rest on knots meant to be equal being stored bitwise identical: as in
        :func:`_basis_funcs_point`, each ``ndu[j, r]`` divides a quantity that is
        then multiplied by two non-negative terms summing to that same denominator,
        so a small denominator cancels against an equally small numerator and the
        term stays bounded. ``denom == 0`` is reached only when both terms are
        exactly zero, which is what an empty knot span is, and the exact-zero test
        is scale-invariant by construction, so this guard is unaffected by the knot
        vector's parametric span (shift or scale).
        For general use, call :func:`_tabulate_Bspline_basis_deriv_1D_impl` instead.
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
            temp = zero if denom == zero else ndu[r, j - 1] / denom
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
    Results are written directly to the output arrays (C-style).  Each point's span
    search and derivative evaluation are independent, so both are fused into a
    single ``prange`` loop over evaluation points (mirrors
    :func:`_compute_basis_nurbs_book_impl`; see :func:`_find_spans_and_first_basis`,
    which remains serial for the small-batch twin below).  For small batches
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
        tol (float): Absolute per-dtype tolerance, passed to
            :func:`_get_Bspline_num_basis_1D_impl` for interface consistency with
            the space's other tolerance-based conventions; unused in practice here
            since that call always passes ``periodic=False`` (the periodic case
            short-circuits ``num_basis`` to 0 above, skipping the call entirely).
            The Cox-de Boor denominator guard itself needs no tolerance; see
            :func:`_basis_derivs_point`.
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
    # num_basis is only used by the non-periodic clamp in _find_span_and_first_basis_point;
    # skip computing it for periodic splines, matching _find_spans_and_first_basis's cost.
    num_basis = 0 if periodic else _get_Bspline_num_basis_1D_impl(knots, degree, periodic, tol)

    for pt_id in nb_prange(n_pts):
        pt = pts[pt_id]
        knot_id, first_basis = _find_span_and_first_basis_point(
            knots, degree, periodic, num_basis, pt
        )
        out_first_basis[pt_id] = first_basis
        _basis_derivs_point(knots, degree, n_deriv, knot_id, pt, out_deriv[pt_id, :, :])


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
        tol (float): Absolute per-dtype tolerance, threaded through to
            :func:`_find_spans_and_first_basis` for interface consistency with the
            space's other tolerance-based conventions (e.g.
            :func:`_get_Bspline_num_basis_1D_impl`'s periodic-regularity path); it
            does not affect this function's own behavior. The Cox-de Boor
            denominator guard itself needs no tolerance; see
            :func:`_basis_derivs_point`.
        n_deriv (int): Maximum derivative order to compute (>= 0).
        pts (npt.NDArray[np.float32 | np.float64]): Points (1D array) to evaluate.
        out_deriv (npt.NDArray[np.float32 | np.float64]): Output array for derivative values.
            Must have shape (n_pts, n_deriv+1, degree+1) and dtype matching ``knots``.
        out_first_basis (npt.NDArray[np.int_]): Output array for first basis indices.
            Must have shape (n_pts,) and dtype int.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_tabulate_Bspline_basis_deriv_1D_impl` instead.
    """
    n_pts = pts.size
    knot_ids = _find_spans_and_first_basis(knots, degree, periodic, tol, pts, out_first_basis)

    for pt_id in range(n_pts):
        _basis_derivs_point(
            knots, degree, n_deriv, knot_ids[pt_id], pts[pt_id], out_deriv[pt_id, :, :]
        )


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

    # Warmup Bernstein derivative core (Bézier fast path) with float64.
    #
    # Named directly rather than fetched from `bernstein_deriv_core()`: this function
    # exists to trigger Numba compilation, so it must reach the Numba kernel whatever
    # backend happens to be selected.
    from ..basis._basis_core import (  # noqa: PLC0415
        _tabulate_Bernstein_basis_deriv_1D_core,
    )

    pts_norm_dummy = pts_dummy  # knots_dummy already has [0,1] domain
    _tabulate_Bernstein_basis_deriv_1D_core(
        np.int32(degree_dummy), pts_norm_dummy, n_deriv_dummy, deriv_dummy
    )


__all__ = [
    "_compute_basis_deriv_nurbs_book_impl",
    "_compute_basis_deriv_nurbs_book_serial_impl",
    "_compute_basis_nurbs_book_impl",
    "_compute_basis_nurbs_book_serial_impl",
    "_warmup_numba_functions",
]
