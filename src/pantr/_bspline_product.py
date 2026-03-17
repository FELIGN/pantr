"""B-spline pointwise product for 1D splines.

This module provides :func:`_multiply_bspline_1d`, which computes the exact
pointwise product of two 1D B-splines via full-Bézier extraction and the
Bernstein product formula. Works for both non-rational and rational (NURBS) splines.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ._bspline_knots import _get_unique_knots_and_multiplicity_impl
from .bspline_space_1D import BsplineSpace1D
from .bspline_space_nd import BsplineSpace

if TYPE_CHECKING:
    from .bspline import Bspline


def _get_interior_breakpoints_and_mults(
    space_1d: BsplineSpace1D,
    tol: float,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    """Return interior breakpoints and their multiplicities for a 1D B-spline space.

    Calls ``_get_unique_knots_and_multiplicity_impl`` with ``in_domain=True`` and strips
    the two boundary entries, leaving only interior breakpoints.

    Args:
        space_1d (BsplineSpace1D): The 1D B-spline space to query.
        tol (float): Tolerance for grouping nearby knots.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]: Arrays
        ``(breakpoints, multiplicities)`` of interior knots only.  Both arrays are
        empty when the space has a single Bézier element.
    """
    dtype = space_1d.knots.dtype
    tol_typed = float(dtype.type(tol))
    unique, mults = _get_unique_knots_and_multiplicity_impl(
        space_1d.knots, space_1d.degree, tol_typed, in_domain=True
    )
    return unique[1:-1], mults[1:-1]


def _merge_interior_breakpoints(
    bp1: npt.NDArray[np.float32 | np.float64],
    mult1: npt.NDArray[np.int_],
    bp2: npt.NDArray[np.float32 | np.float64],
    mult2: npt.NDArray[np.int_],
    tol: float,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    """Merge two sorted interior breakpoint arrays with additive multiplicities.

    Uses a two-pointer scan.  When two breakpoints are within ``tol`` of each
    other they are considered the same knot; their position is averaged and
    their multiplicities are summed.  A breakpoint absent from one side
    contributes 0 to the sum (i.e. the other side's multiplicity is taken
    as-is).

    Args:
        bp1 (npt.NDArray[np.float32 | np.float64]): Sorted interior breakpoints
            of the first space.
        mult1 (npt.NDArray[np.int_]): Multiplicities corresponding to ``bp1``.
        bp2 (npt.NDArray[np.float32 | np.float64]): Sorted interior breakpoints
            of the second space.
        mult2 (npt.NDArray[np.int_]): Multiplicities corresponding to ``bp2``.
        tol (float): Tolerance for coincidence tests.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]: Merged
        ``(all_bp, sum_mults)`` arrays in ascending order.
    """
    n1, n2 = len(bp1), len(bp2)
    i, j = 0, 0
    all_bp_list: list[float] = []
    sum_mults_list: list[int] = []

    while i < n1 and j < n2:
        if abs(float(bp1[i]) - float(bp2[j])) <= tol:
            all_bp_list.append(float(bp1[i] + bp2[j]) / 2.0)
            sum_mults_list.append(int(mult1[i]) + int(mult2[j]))
            i += 1
            j += 1
        elif float(bp1[i]) < float(bp2[j]):
            all_bp_list.append(float(bp1[i]))
            sum_mults_list.append(int(mult1[i]))
            i += 1
        else:
            all_bp_list.append(float(bp2[j]))
            sum_mults_list.append(int(mult2[j]))
            j += 1

    while i < n1:
        all_bp_list.append(float(bp1[i]))
        sum_mults_list.append(int(mult1[i]))
        i += 1

    while j < n2:
        all_bp_list.append(float(bp2[j]))
        sum_mults_list.append(int(mult2[j]))
        j += 1

    dtype = bp1.dtype
    if len(all_bp_list) == 0:
        return np.empty(0, dtype=dtype), np.empty(0, dtype=np.int_)
    return (
        np.array(all_bp_list, dtype=dtype),
        np.array(sum_mults_list, dtype=np.int_),
    )


def _lookup_mults_in_space(
    all_bp: npt.NDArray[np.float32 | np.float64],
    bp_space: npt.NDArray[np.float32 | np.float64],
    mult_space: npt.NDArray[np.int_],
    tol: float,
) -> npt.NDArray[np.int_]:
    """Look up the multiplicity of each merged breakpoint within a single space.

    For each entry in ``all_bp``, finds the corresponding entry in ``bp_space``
    (within ``tol``) and returns its multiplicity; returns 0 for breakpoints
    absent from that space.

    Both ``all_bp`` and ``bp_space`` must be sorted in ascending order.
    Uses ``np.searchsorted`` for efficient binary search (same pattern as
    :func:`~pantr._bspline_knots._get_last_knot_smaller_equal_impl`).

    Args:
        all_bp (npt.NDArray[np.float32 | np.float64]): Merged (union) interior
            breakpoints, sorted ascending.
        bp_space (npt.NDArray[np.float32 | np.float64]): Interior breakpoints of
            one space, sorted ascending.
        mult_space (npt.NDArray[np.int_]): Multiplicities for ``bp_space``.
        tol (float): Tolerance for coincidence tests.

    Returns:
        npt.NDArray[np.int_]: Array of shape ``(len(all_bp),)`` containing the
        multiplicity in the given space for each entry of ``all_bp`` (0 if absent).
    """
    result = np.zeros(len(all_bp), dtype=np.int_)
    if bp_space.size == 0 or all_bp.size == 0:
        return result
    indices = np.searchsorted(bp_space, all_bp)
    safe_idx = np.minimum(indices, bp_space.size - 1)
    in_range = indices < bp_space.size
    matched = in_range & (np.abs(bp_space[safe_idx] - all_bp) <= tol)
    result[matched] = mult_space[safe_idx[matched]]
    return result


def _knots_for_full_bezier(
    space_1d: BsplineSpace1D,
    all_bp: npt.NDArray[np.float32 | np.float64],
    mults_in_space: npt.NDArray[np.int_],
    tol: float,
) -> npt.NDArray[np.float32 | np.float64]:
    """Compute the additional knots needed to bring a space to full-Bézier form.

    For each breakpoint ``ξ`` with existing multiplicity ``m`` in ``space_1d``,
    inserts ``degree - m`` copies of ``ξ`` so that every interior breakpoint
    reaches multiplicity ``degree`` (full-Bézier / C^0).

    Args:
        space_1d (BsplineSpace1D): The 1D B-spline space to refine.
        all_bp (npt.NDArray[np.float32 | np.float64]): Union of interior
            breakpoints (sorted ascending).
        mults_in_space (npt.NDArray[np.int_]): Multiplicity of each entry of
            ``all_bp`` in ``space_1d`` (0 if absent).
        tol (float): Tolerance (unused here, kept for API symmetry).

    Returns:
        npt.NDArray[np.float32 | np.float64]: Flat sorted array of knot values
        to insert, possibly empty.
    """
    degree = space_1d.degree
    dtype = space_1d.knots.dtype
    knots_to_insert: list[float] = []
    for xi, m in zip(all_bp, mults_in_space, strict=True):
        n_to_insert = degree - int(m)
        if n_to_insert > 0:
            knots_to_insert.extend([float(xi)] * n_to_insert)
    if len(knots_to_insert) == 0:
        return np.empty(0, dtype=dtype)
    return np.array(knots_to_insert, dtype=dtype)


def _bernstein_product_coefficients(
    b_f: npt.NDArray[np.float32 | np.float64],
    b_g: npt.NDArray[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Compute Bézier control points of the product of two Bézier segments.

    Applies the Bernstein product formula element-wise over the rank axis:

    .. math::

        d_k = \frac{1}{\binom{p+q}{k}} \sum_{i=\max(0,k-q)}^{\min(p,k)}
              \binom{p}{i} \binom{q}{k-i}\, b_f[i] \cdot b_g[k-i]

    Args:
        b_f (npt.NDArray[np.float32 | np.float64]): Control points of the first
            Bézier segment, shape ``(p+1, rank)``.
        b_g (npt.NDArray[np.float32 | np.float64]): Control points of the second
            Bézier segment, shape ``(q+1, rank)``.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Product Bézier control points of
        shape ``(p+q+1, rank)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    p = b_f.shape[0] - 1
    q = b_g.shape[0] - 1
    rank = b_f.shape[1]
    r = p + q
    dtype = b_f.dtype
    d = np.zeros((r + 1, rank), dtype=dtype)

    for k in range(r + 1):
        i_min = max(0, k - q)
        i_max = min(p, k)
        cpq_k = math.comb(r, k)
        for i in range(i_min, i_max + 1):
            j = k - i
            coeff = dtype.type(math.comb(p, i) * math.comb(q, j)) / dtype.type(cpq_k)
            d[k] += coeff * b_f[i] * b_g[j]

    return d


def _build_product_knot_vector(
    domain: tuple[np.float32 | np.float64, np.float32 | np.float64],
    all_bp: npt.NDArray[np.float32 | np.float64],
    degree_sum: int,
    dtype: npt.DTypeLike,
) -> npt.NDArray[np.float32 | np.float64]:
    """Build the full-Bézier knot vector for the product B-spline space.

    Assembles a clamped full-Bézier knot vector with:
    - ``degree_sum + 1`` copies of the left endpoint ``a``
    - For each interior breakpoint ``ξ``: ``degree_sum`` copies (C^0)
    - ``degree_sum + 1`` copies of the right endpoint ``b``

    The resulting space has exactly ``n_elements * degree_sum + 1`` basis
    functions, matching the number of control points assembled by the
    Bernstein product formula.

    Args:
        domain (tuple[np.float32 | np.float64, np.float32 | np.float64]): Domain
            endpoints ``(a, b)`` of the parametric interval.
        all_bp (npt.NDArray[np.float32 | np.float64]): Interior breakpoints of the
            union mesh, sorted ascending.
        degree_sum (int): Total polynomial degree ``p + q`` of the product space;
            also the interior knot multiplicity (full-Bézier).
        dtype (npt.DTypeLike): Floating-point dtype for the output knot vector.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Full-Bézier clamped knot vector for
        the product space.
    """
    a, b = domain
    parts: list[npt.NDArray[np.float32 | np.float64]] = [np.full(degree_sum + 1, a, dtype=dtype)]
    for xi in all_bp:
        parts.append(np.full(degree_sum, xi, dtype=dtype))
    parts.append(np.full(degree_sum + 1, b, dtype=dtype))
    result: npt.NDArray[np.float32 | np.float64] = np.concatenate(parts)
    return result


def _to_rational(f: Bspline) -> Bspline:
    """Convert a B-spline to rational form by appending a column of unit weights.

    If ``f`` is already rational, returns it unchanged.  Otherwise, creates a
    new :class:`~pantr.bspline.Bspline` with the same space and control points
    augmented by a column of ones (homogeneous weights = 1).

    Args:
        f (~pantr.bspline.Bspline): The B-spline to convert.

    Returns:
        ~pantr.bspline.Bspline: Rational B-spline equivalent to ``f``.
    """
    if f.is_rational:
        return f
    from .bspline import Bspline  # noqa: PLC0415

    n = f.control_points.shape[0]
    weights = np.ones((n, 1), dtype=f.control_points.dtype)
    new_ctrl = np.concatenate([f.control_points, weights], axis=-1)
    return Bspline(f.space, new_ctrl, is_rational=True)


def _multiply_nonrational_1d(f: Bspline, g: Bspline) -> Bspline:
    """Multiply two non-rational 1D B-splines using the full-Bézier approach.

    Refines both operands to full-Bézier form (C^0 at every interior breakpoint),
    then applies the Bernstein product formula element by element, and assembles
    the result in the product space.

    This function does not perform input validation; use :func:`_multiply_bspline_1d`
    for the validated public entry point.

    Args:
        f (~pantr.bspline.Bspline): First non-rational 1D B-spline operand.
        g (~pantr.bspline.Bspline): Second non-rational 1D B-spline operand.

    Returns:
        ~pantr.bspline.Bspline: Non-rational B-spline ``h`` such that
        ``h(t) ≈ f(t) * g(t)`` for all ``t`` in the shared domain.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_multiply_bspline_1d` instead.
    """
    from .bspline import Bspline  # noqa: PLC0415

    space_f = f.space.spaces[0]
    space_g = g.space.spaces[0]
    p = space_f.degree
    q = space_g.degree
    tol = max(float(space_f.tolerance), float(space_g.tolerance))

    # --- Step 1: gather interior breakpoints ---
    bp_f, mf = _get_interior_breakpoints_and_mults(space_f, tol)
    bp_g, mg = _get_interior_breakpoints_and_mults(space_g, tol)

    # --- Step 2: merge union of interior breakpoints ---
    all_bp, _ = _merge_interior_breakpoints(bp_f, mf, bp_g, mg, tol)
    n_elements = int(all_bp.size) + 1

    # --- Step 3: compute per-space multiplicities at union breakpoints ---
    mults_in_f = _lookup_mults_in_space(all_bp, bp_f, mf, tol)
    mults_in_g = _lookup_mults_in_space(all_bp, bp_g, mg, tol)

    # --- Step 4: refine to full-Bézier ---
    knots_f_ins = _knots_for_full_bezier(space_f, all_bp, mults_in_f, tol)
    knots_g_ins = _knots_for_full_bezier(space_g, all_bp, mults_in_g, tol)

    f_bezier: Bspline = f.insert_knots(knots_f_ins) if knots_f_ins.size > 0 else f
    g_bezier: Bspline = g.insert_knots(knots_g_ins) if knots_g_ins.size > 0 else g

    # --- Step 5: assemble product control points ---
    rank = int(f.control_points.shape[-1])
    ctrl_h = np.empty((n_elements * (p + q) + 1, rank), dtype=f.control_points.dtype)

    for e in range(n_elements):
        b_f_e = f_bezier.control_points[e * p : e * p + p + 1]
        b_g_e = g_bezier.control_points[e * q : e * q + q + 1]
        ctrl_h[e * (p + q) : e * (p + q) + p + q + 1] = _bernstein_product_coefficients(
            b_f_e, b_g_e
        )

    # --- Step 6: build product knot vector (full-Bézier, interior mult = p+q) ---
    domain = space_f.domain
    T_h = _build_product_knot_vector(domain, all_bp, p + q, f.control_points.dtype)
    space_h = BsplineSpace([BsplineSpace1D(T_h, p + q)])
    return Bspline(space_h, ctrl_h, is_rational=False)


def _multiply_rational_1d(f: Bspline, g: Bspline) -> Bspline:
    """Multiply two rational 1D B-splines (NURBS) using homogeneous coordinates.

    Decomposes each rational operand into its numerator (weighted coordinates)
    and denominator (weights) B-splines, multiplies each pair independently
    using :func:`_multiply_nonrational_1d`, then reassembles the result.

    Both ``f`` and ``g`` must already be rational (``is_rational=True``).

    Args:
        f (~pantr.bspline.Bspline): First rational 1D B-spline operand.
        g (~pantr.bspline.Bspline): Second rational 1D B-spline operand.

    Returns:
        ~pantr.bspline.Bspline: Rational B-spline ``h`` such that
        ``h(t) = f(t) * g(t)`` for all ``t`` in the shared domain.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_multiply_bspline_1d` instead.
    """
    from .bspline import Bspline  # noqa: PLC0415

    # Split into numerator and denominator non-rational B-splines.
    N_f = Bspline(f.space, f.control_points[:, :-1])
    D_f = Bspline(f.space, f.control_points[:, -1:])
    N_g = Bspline(g.space, g.control_points[:, :-1])
    D_g = Bspline(g.space, g.control_points[:, -1:])

    H_N = _multiply_nonrational_1d(N_f, N_g)
    H_D = _multiply_nonrational_1d(D_f, D_g)

    ctrl_h = np.concatenate([H_N.control_points, H_D.control_points], axis=-1)
    return Bspline(H_N.space, ctrl_h, is_rational=True)


def _multiply_bspline_1d(f: Bspline, g: Bspline) -> Bspline:
    """Compute the exact pointwise product of two 1D B-splines.

    Given B-splines ``f`` and ``g`` over the same 1D parametric domain, returns
    a new B-spline ``h`` such that ``h(t) = f(t) * g(t)`` for all ``t`` in the
    domain.  The result lives in the product space of degree ``p + q``.

    Rational operands are handled via homogeneous-coordinate decomposition.  A
    non-rational operand is silently promoted to rational (unit weights) when the
    other is rational.

    Args:
        f (~pantr.bspline.Bspline): First 1D B-spline operand.
        g (~pantr.bspline.Bspline): Second 1D B-spline operand.

    Returns:
        ~pantr.bspline.Bspline: Product B-spline ``h = f * g``.

    Raises:
        ValueError: If either ``f`` or ``g`` has ``dim != 1``.
        ValueError: If ``f`` and ``g`` have different dtypes.
        ValueError: If ``f`` and ``g`` have different ranks.
        ValueError: If ``f`` and ``g`` have different parametric domains (beyond
            the shared tolerance).
        NotImplementedError: If either ``f`` or ``g`` has ``periodic=True``.
    """
    if f.dim != 1:
        raise ValueError(f"f must be a 1D B-spline, got dim={f.dim}")
    if g.dim != 1:
        raise ValueError(f"g must be a 1D B-spline, got dim={g.dim}")

    if np.dtype(f.dtype) != np.dtype(g.dtype):
        raise ValueError(f"f and g must have the same dtype, got {f.dtype} and {g.dtype}")

    if f.rank != g.rank:
        raise ValueError(f"f and g must have the same rank, got {f.rank} and {g.rank}")

    space_f = f.space.spaces[0]
    space_g = g.space.spaces[0]

    if space_f.periodic or space_g.periodic:
        raise NotImplementedError("Multiplication of periodic B-splines is not supported.")

    tol = max(float(space_f.tolerance), float(space_g.tolerance))
    domain_f = space_f.domain
    domain_g = space_g.domain
    if (
        abs(float(domain_f[0]) - float(domain_g[0])) > tol
        or abs(float(domain_f[1]) - float(domain_g[1])) > tol
    ):
        raise ValueError(
            f"f and g must share the same parametric domain. Got {domain_f} and {domain_g}."
        )

    if not f.is_rational and not g.is_rational:
        return _multiply_nonrational_1d(f, g)
    else:
        return _multiply_rational_1d(_to_rational(f), _to_rational(g))
