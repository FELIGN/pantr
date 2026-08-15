r"""B-spline pointwise product for 1D splines.

This module provides :func:`_multiply_bspline_1d`, which computes the exact
pointwise product of two 1D B-splines via Bézier extraction and the Bernstein
product formula.  The result lives in the product space of degree ``r = p + q``
with the **minimal** multiplicity at every breakpoint, computed once by
:func:`_product_multiplicities`.

**Smoothness of a product.**  Write ``s_f(ξ) = p - m_f(ξ)`` for the smoothness of
``f`` at a breakpoint ``ξ`` (so ``f`` is :math:`C^{s_f}` there), with
``s_f(ξ) = +∞`` when ``ξ`` is not a knot of ``f`` at all, and likewise for ``g``.
Then ``h = f g`` is :math:`C^{\min(s_f, s_g)}` at ``ξ``, so

.. math::

    m_h(\xi) = r - \min(s_f(\xi), s_g(\xi))
             = \max \{\, m_f(\xi) + q \;:\; \xi \in T_f \,\}
                \cup \{\, m_g(\xi) + p \;:\; \xi \in T_g \,\},

the maximum being taken over the operands that actually have a knot at ``ξ``.  A
breakpoint carried by one operand alone therefore gets ``m_f(ξ) + q``, *not*
``p + q``: the two agree only when ``m_f(ξ) = p``, and the product space is
strictly larger than it needs to be whenever ``m_f(ξ) < p``.

The identity follows from the Curry-Schoenberg theorem (Curry & Schoenberg,
*J. Analyse Math.* **17**, 1966, 71-107), which identifies the spline space on a
knot vector with the space of piecewise polynomials of matching smoothness; see
Lyche & Mørken, *Spline Methods*, Thm 3.25, or Lyche, Manni & Speleers, CIME
2017, Thm 13.  Because that identification is an equality of spaces rather than
an inclusion, the multiplicity above is not merely sufficient but minimal: for
generic coefficients ``h`` is not smoother than :math:`C^{\min(s_f, s_g)}`.  The
hypotheses are mild — the end knots need not be clamped, and ``m = p + 1``
(a :math:`C^{-1}`, discontinuous, spline, which
:meth:`~pantr.bspline.BsplineSpace1D.subdivide` admits through
``regularity=-1``) is inside the range, not at its edge.  ``m_h`` is capped at
``r + 1`` because ``m_f ≤ p + 1`` and ``m_g ≤ q + 1``.

Works for both non-rational and rational (NURBS) splines.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import numpy.typing as npt

from .._control_points_utils import _append_unit_weight_column
from ..bezier._bezier_product import _bernstein_product_coefficients
from ._bspline_knots import (
    _get_Bspline_num_basis_1D_impl,
    _get_unique_knots_and_multiplicity_impl,
)
from ._bspline_space_1d import BsplineSpace1D
from ._bspline_space_nd import BsplineSpace

if TYPE_CHECKING:
    from . import Bspline


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
    Uses ``np.searchsorted`` for efficient binary search.

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


def _merge_interior_breakpoints(
    bp_f: npt.NDArray[np.float32 | np.float64],
    bp_g: npt.NDArray[np.float32 | np.float64],
    tol: float,
) -> npt.NDArray[np.float32 | np.float64]:
    """Merge two sorted interior breakpoint arrays into their union.

    Uses a two-pointer scan.  Two breakpoints closer than ``tol`` are treated as
    one and replaced by their midpoint.

    Args:
        bp_f (npt.NDArray[np.float32 | np.float64]): Sorted interior breakpoints
            of the first space.
        bp_g (npt.NDArray[np.float32 | np.float64]): Sorted interior breakpoints
            of the second space.
        tol (float): Tolerance for coincidence tests.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The merged breakpoints in ascending
        order.
    """
    n1, n2 = len(bp_f), len(bp_g)
    i, j = 0, 0
    all_bp_list: list[float] = []

    while i < n1 and j < n2:
        if abs(float(bp_f[i]) - float(bp_g[j])) <= tol:
            all_bp_list.append(float(bp_f[i] + bp_g[j]) / 2.0)
            i += 1
            j += 1
        elif float(bp_f[i]) < float(bp_g[j]):
            all_bp_list.append(float(bp_f[i]))
            i += 1
        else:
            all_bp_list.append(float(bp_g[j]))
            j += 1

    all_bp_list.extend(float(x) for x in bp_f[i:])
    all_bp_list.extend(float(x) for x in bp_g[j:])

    dtype = bp_f.dtype if bp_f.size > 0 else bp_g.dtype
    return np.array(all_bp_list, dtype=dtype)


def _product_multiplicities(
    mults_f: npt.NDArray[np.int_],
    mults_g: npt.NDArray[np.int_],
    p: int,
    q: int,
) -> npt.NDArray[np.int_]:
    r"""Compute the minimal product-space multiplicity at each merged breakpoint.

    Takes the largest of ``m_f(ξ) + q`` and ``m_g(ξ) + p`` over the operands that
    carry a knot at ``ξ``; see the module docstring for the derivation and its
    hypotheses.  An operand whose multiplicity is 0 has no knot there and
    contributes nothing; at least one operand must contribute, which holds for
    every breakpoint of the union.

    Writing ``max(m_f + q, p + q)`` for a breakpoint carried by ``f`` alone is
    *not* the same thing: that form silently assumes ``m_f ≤ p``, so it is one
    knot short of what a :math:`C^{-1}` operand (``m_f = p + 1``) needs, and up
    to ``p - m_f`` knots more than the product actually requires otherwise.

    Args:
        mults_f (npt.NDArray[np.int_]): Multiplicity in the first space at each
            merged breakpoint (``0`` where the breakpoint is absent).
        mults_g (npt.NDArray[np.int_]): Multiplicity in the second space, likewise.
        p (int): Degree of the first operand.
        q (int): Degree of the second operand.

    Returns:
        npt.NDArray[np.int_]: Product multiplicity per breakpoint, in ``[1, p+q+1]``.
    """
    from_f = np.where(mults_f > 0, mults_f + q, 0)
    from_g = np.where(mults_g > 0, mults_g + p, 0)
    return np.maximum(from_f, from_g).astype(np.int_)


class _ProductMesh(NamedTuple):
    """The breakpoint structure shared by two operands and by their product.

    Attributes:
        breakpoints (npt.NDArray[np.float32 | np.float64]): Union of the interior
            breakpoints of the two operands, ascending.
        mults_f (npt.NDArray[np.int_]): Multiplicity of each breakpoint in the
            first operand's knot vector (``0`` when absent).
        mults_g (npt.NDArray[np.int_]): Same for the second operand.
        product_mults (npt.NDArray[np.int_]): Minimal multiplicity of each
            breakpoint in the product space, from :func:`_product_multiplicities`.
    """

    breakpoints: npt.NDArray[np.float32 | np.float64]
    mults_f: npt.NDArray[np.int_]
    mults_g: npt.NDArray[np.int_]
    product_mults: npt.NDArray[np.int_]


def _build_product_mesh(
    space_f: BsplineSpace1D,
    space_g: BsplineSpace1D,
    tol: float,
) -> _ProductMesh:
    """Collect the interior breakpoints of two 1D spaces and their product multiplicities.

    Args:
        space_f (BsplineSpace1D): First operand's 1D space, of degree ``p``.
        space_g (BsplineSpace1D): Second operand's 1D space, of degree ``q``.
        tol (float): Tolerance for grouping nearby knots.

    Returns:
        _ProductMesh: The merged breakpoints with per-operand and product
        multiplicities.
    """
    bp_f, mf = _get_interior_breakpoints_and_mults(space_f, tol)
    bp_g, mg = _get_interior_breakpoints_and_mults(space_g, tol)
    all_bp = _merge_interior_breakpoints(bp_f, bp_g, tol)
    mults_f = _lookup_mults_in_space(all_bp, bp_f, mf, tol)
    mults_g = _lookup_mults_in_space(all_bp, bp_g, mg, tol)
    product_mults = _product_multiplicities(mults_f, mults_g, space_f.degree, space_g.degree)
    return _ProductMesh(all_bp, mults_f, mults_g, product_mults)


def _bezier_mults(degree: int, mults: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    """Raise interior multiplicities to Bézier form without ever lowering one.

    A Bézier (element-by-element) representation needs multiplicity ``degree`` at
    every interior breakpoint, except where the spline is already discontinuous:
    a knot of multiplicity ``degree + 1`` stays where it is, since removing it
    would change the function.

    Args:
        degree (int): Degree of the space being brought to Bézier form.
        mults (npt.NDArray[np.int_]): Current multiplicity per breakpoint
            (``0`` where the breakpoint is absent).

    Returns:
        npt.NDArray[np.int_]: Multiplicity per breakpoint in Bézier form.
    """
    return np.maximum(degree, mults).astype(np.int_)


def _bezier_element_offsets(bezier_mults: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    """Index of each element's first control point in a Bézier-form spline.

    Element ``e`` owns the ``degree + 1`` control points starting at entry ``e``
    of the returned array.  Consecutive elements overlap in their shared
    endpoint where the breakpoint between them has multiplicity ``degree``, and
    are disjoint where it has multiplicity ``degree + 1``.

    Args:
        bezier_mults (npt.NDArray[np.int_]): Interior multiplicities of the
            Bézier-form knot vector, from :func:`_bezier_mults`.

    Returns:
        npt.NDArray[np.int_]: Offsets of shape ``(len(bezier_mults) + 1,)``,
        starting at 0.
    """
    return np.concatenate([np.zeros(1, dtype=np.int_), np.cumsum(bezier_mults, dtype=np.int_)])


def _knots_for_full_bezier(
    space_1d: BsplineSpace1D,
    all_bp: npt.NDArray[np.float32 | np.float64],
    mults_in_space: npt.NDArray[np.int_],
) -> npt.NDArray[np.float32 | np.float64]:
    """Compute the additional knots needed to bring a space to full-Bézier form.

    For each breakpoint ``ξ`` with existing multiplicity ``m`` in ``space_1d``,
    inserts enough copies of ``ξ`` to reach :func:`_bezier_mults`, i.e.
    ``degree - m`` copies at an ordinary breakpoint and none at a
    :math:`C^{-1}` one, which is already isolated.

    Args:
        space_1d (BsplineSpace1D): The 1D B-spline space to refine.
        all_bp (npt.NDArray[np.float32 | np.float64]): Union of interior
            breakpoints (sorted ascending).
        mults_in_space (npt.NDArray[np.int_]): Multiplicity of each entry of
            ``all_bp`` in ``space_1d`` (0 if absent).

    Returns:
        npt.NDArray[np.float32 | np.float64]: Flat sorted array of knot values
        to insert, possibly empty.
    """
    deficits = _bezier_mults(space_1d.degree, mults_in_space) - mults_in_space
    return np.repeat(all_bp, deficits).astype(space_1d.knots.dtype)


def _build_product_knot_vector(
    domain: tuple[np.float32 | np.float64, np.float32 | np.float64],
    all_bp: npt.NDArray[np.float32 | np.float64],
    product_mults: npt.NDArray[np.int_],
    degree_sum: int,
    dtype: npt.DTypeLike,
) -> npt.NDArray[np.float32 | np.float64]:
    """Build a clamped product knot vector from prescribed interior multiplicities.

    Assembles a clamped knot vector with:

    - ``degree_sum + 1`` copies of the left endpoint ``a``
    - For each interior breakpoint ``ξ``: ``product_mults[k]`` copies
    - ``degree_sum + 1`` copies of the right endpoint ``b``

    The interior multiplicities encode the continuity
    ``C^{degree_sum - product_mults[k]}`` at breakpoint ``ξ_k``.  Callers pass
    either the minimal multiplicities of :func:`_product_multiplicities` or their
    Bézier-form counterparts from :func:`_bezier_mults`.

    Args:
        domain (tuple[np.float32 | np.float64, np.float32 | np.float64]): Domain
            endpoints ``(a, b)`` of the parametric interval.
        all_bp (npt.NDArray[np.float32 | np.float64]): Interior breakpoints of
            the union mesh, sorted ascending.
        product_mults (npt.NDArray[np.int_]): Product-space multiplicity for
            each entry of ``all_bp``.
        degree_sum (int): Total polynomial degree ``p + q`` of the product space.
        dtype (npt.DTypeLike): Floating-point dtype for the output knot vector.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Clamped knot vector for the
        product space.
    """
    a, b = domain
    parts: list[npt.NDArray[np.float32 | np.float64]] = [np.full(degree_sum + 1, a, dtype=dtype)]
    for xi, m in zip(all_bp, product_mults, strict=True):
        parts.append(np.full(int(m), xi, dtype=dtype))
    parts.append(np.full(degree_sum + 1, b, dtype=dtype))
    result: npt.NDArray[np.float32 | np.float64] = np.concatenate(parts)
    return result


def _get_boundary_mults(
    space_1d: BsplineSpace1D,
    tol: float,
) -> tuple[int, int]:
    """Return left and right boundary multiplicities of a 1D B-spline space.

    Args:
        space_1d (BsplineSpace1D): The 1D B-spline space to query.
        tol (float): Tolerance for grouping nearby knots.

    Returns:
        tuple[int, int]: ``(m_left, m_right)`` boundary multiplicities.
    """
    knots = space_1d.knots
    p = space_1d.degree
    a = float(knots[p])
    b = float(knots[-p - 1])
    m_left = int(np.sum(np.abs(knots[: p + 1] - a) <= tol))
    m_right = int(np.sum(np.abs(knots[-p - 1 :] - b) <= tol))
    return m_left, m_right


def _build_periodic_product_knot_vector(  # noqa: PLR0913
    domain: tuple[np.float32 | np.float64, np.float32 | np.float64],
    all_bp: npt.NDArray[np.float32 | np.float64],
    product_mults: npt.NDArray[np.int_],
    m_bdy: int,
    degree_sum: int,
    dtype: npt.DTypeLike,
) -> npt.NDArray[np.float32 | np.float64]:
    """Build a periodic product knot vector with ghost knots.

    Assembles a periodic knot vector with in-domain breakpoints and ghost knots
    extending the breakpoint pattern periodically beyond the domain boundaries.

    Args:
        domain (tuple[np.float32 | np.float64, np.float32 | np.float64]): Domain
            endpoints ``(a, b)``.
        all_bp (npt.NDArray[np.float32 | np.float64]): Interior breakpoints of
            the union mesh, sorted ascending.
        product_mults (npt.NDArray[np.int_]): Product-space multiplicity for
            each entry of ``all_bp``.
        m_bdy (int): Boundary multiplicity at both endpoints (same by periodicity).
        degree_sum (int): Total polynomial degree ``p + q`` of the product space.
        dtype (npt.DTypeLike): Floating-point dtype for the output knot vector.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Periodic knot vector with ghost knots.
    """
    a, b = domain
    period = float(b) - float(a)

    n_ghost = degree_sum + 1 - m_bdy  # ghost knot entries needed on each side

    # Tile = [a^{m_bdy}, xi_1^{m_1}, ..., xi_{N-1}^{m_{N-1}}] — one periodic unit.
    full_tile = np.repeat(
        np.concatenate([[float(a)], all_bp]),
        np.concatenate([[m_bdy], product_mults]),
    ).astype(dtype)
    # Interior part of tile (everything after the leading a^{m_bdy}).
    interior_tile = full_tile[m_bdy:]

    if n_ghost > 0:
        n_tiles = math.ceil(n_ghost / max(len(full_tile), 1)) + 2
        # Right ghosts: xi_1+L, ..., xi_{N-1}+L, a+2L, xi_1+2L, ...
        # (skips b = a+L; starts from interior at shift 1, then full tile at shift 2+)
        right_seq = np.concatenate(
            [interior_tile + period] + [full_tile + s * period for s in range(2, n_tiles + 2)]
        )[:n_ghost]
        # Left ghosts: ..., a-L, xi_1-L, ..., xi_{N-1}-L
        left_seq = np.concatenate([full_tile + s * period for s in range(-n_tiles, 0)])[-n_ghost:]
        right_parts: list[npt.NDArray[np.float32 | np.float64]] = [right_seq]
        left_parts: list[npt.NDArray[np.float32 | np.float64]] = [left_seq]
    else:
        right_parts = []
        left_parts = []

    # In-domain: [a^{m_bdy}, xi_1^{m_1}, ..., xi_{N-1}^{m_{N-1}}, b^{m_bdy}]
    in_domain_parts: list[npt.NDArray[np.float32 | np.float64]] = [
        np.full(m_bdy, a, dtype=dtype),
    ]
    for xi, m in zip(all_bp, product_mults, strict=True):
        in_domain_parts.append(np.full(int(m), xi, dtype=dtype))
    in_domain_parts.append(np.full(m_bdy, b, dtype=dtype))

    all_parts = left_parts + in_domain_parts + right_parts
    result: npt.NDArray[np.float32 | np.float64] = np.concatenate(all_parts)
    return result


def _ghost_open_oslo(
    h_open: Bspline,
    m_bdy: int,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
    """Build the ghost knot vector and compose the Oslo chain (ghost CPs → open CPs).

    Args:
        h_open (~pantr.bspline.Bspline): The product B-spline in open form.
        m_bdy (int): Boundary multiplicity for the ghost knot vector.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
        ``(T_ghost, oslo_chain)`` where ``T_ghost`` is the periodic/non-open knot vector
        with ghost extension and ``oslo_chain`` maps ghost-representation CPs to open CPs,
        shape ``(n_open, n_ghost_full)``.
    """
    from ._bspline_knot_insertion_core import _compute_oslo_matrix_1d_core  # noqa: PLC0415

    h_space = h_open.space.spaces[0]
    r = h_space.degree
    dtype = h_open.control_points.dtype

    unique, mults = _get_unique_knots_and_multiplicity_impl(
        h_space.knots, r, float(h_space.tolerance), in_domain=True
    )
    interior_bp, interior_mults = unique[1:-1], mults[1:-1]

    T_ghost = _build_periodic_product_knot_vector(
        h_space.domain, interior_bp, interior_mults, m_bdy, r, dtype
    )

    n_ghost = r + 1 - m_bdy
    a, b = float(T_ghost[r]), float(T_ghost[-r - 1])
    knots_ins = np.array([a] * n_ghost + [b] * n_ghost, dtype=dtype)
    T_inter = np.sort(np.concatenate([T_ghost, knots_ins]))
    oslo1 = _compute_oslo_matrix_1d_core(r, T_ghost, T_inter)

    T_per_open = T_inter[n_ghost : len(T_inter) - n_ghost]
    n_per_open = len(T_per_open) - r - 1
    oslo1_trimmed = oslo1[n_ghost : n_ghost + n_per_open, :]

    oslo2 = _compute_oslo_matrix_1d_core(r, T_per_open, h_space.knots)
    return T_ghost, oslo2 @ oslo1_trimmed


def _open_to_nonopen_product(
    h_open: Bspline,
    space_f_orig: BsplineSpace1D,
    space_g_orig: BsplineSpace1D,
) -> Bspline:
    """Convert an open product B-spline to non-open (unclamped) form.

    Builds a non-open knot vector with ghost extension (same pattern as periodic
    but without modulo wrapping), then uses the Oslo chain to convert from open CPs
    to non-open CPs via least-squares.

    Args:
        h_open (~pantr.bspline.Bspline): The product B-spline in open form.
        space_f_orig (BsplineSpace1D): Original 1D space of the first operand.
        space_g_orig (BsplineSpace1D): Original 1D space of the second operand.

    Returns:
        ~pantr.bspline.Bspline: Product B-spline in non-open form.
    """
    from . import Bspline  # noqa: PLC0415

    p, q = space_f_orig.degree, space_g_orig.degree
    tol = max(float(space_f_orig.tolerance), float(space_g_orig.tolerance))
    dtype = h_open.control_points.dtype
    r = h_open.space.spaces[0].degree

    mf_l, mf_r = _get_boundary_mults(space_f_orig, tol)
    mg_l, mg_r = _get_boundary_mults(space_g_orig, tol)
    m_bdy = min(max(mf_l + q, mg_l + p), max(mf_r + q, mg_r + p))

    T_nonopen, oslo = _ghost_open_oslo(h_open, m_bdy)
    ctrl, *_ = np.linalg.lstsq(oslo, h_open.control_points, rcond=None)
    space_no = BsplineSpace([BsplineSpace1D(T_nonopen, r)])
    return Bspline(space_no, ctrl.astype(dtype), is_rational=h_open.is_rational)


def _open_to_periodic_product(
    h_open: Bspline,
    space_f_orig: BsplineSpace1D,
    space_g_orig: BsplineSpace1D,
) -> Bspline:
    """Convert an open product B-spline to periodic form.

    Builds the periodic product knot vector with ghost knots, then computes
    the transformation matrix from periodic CPs (with modulo wrapping) to
    open CPs, and solves for the periodic CPs via least-squares.

    Args:
        h_open (~pantr.bspline.Bspline): The product B-spline in open form.
        space_f_orig (BsplineSpace1D): Original periodic 1D space of the first operand.
        space_g_orig (BsplineSpace1D): Original periodic 1D space of the second operand.

    Returns:
        ~pantr.bspline.Bspline: Product B-spline in periodic form.
    """
    from . import Bspline  # noqa: PLC0415

    p, q = space_f_orig.degree, space_g_orig.degree
    tol = max(float(space_f_orig.tolerance), float(space_g_orig.tolerance))
    dtype = h_open.control_points.dtype
    r = h_open.space.spaces[0].degree

    mf_bdy = _get_boundary_mults(space_f_orig, tol)[0]
    mg_bdy = _get_boundary_mults(space_g_orig, tol)[0]
    m_bdy = max(mf_bdy + q, mg_bdy + p)

    T_per, oslo = _ghost_open_oslo(h_open, m_bdy)

    n_full = len(T_per) - r - 1
    n_per = int(_get_Bspline_num_basis_1D_impl(T_per, r, True, float(np.dtype(dtype).type(tol))))
    W = np.zeros((n_full, n_per), dtype=dtype)
    for i in range(n_full):
        W[i, i % n_per] = np.dtype(dtype).type(1.0)

    ctrl, *_ = np.linalg.lstsq(oslo @ W, h_open.control_points, rcond=None)
    space_per = BsplineSpace([BsplineSpace1D(T_per, r, periodic=True)])
    return Bspline(space_per, ctrl.astype(dtype), is_rational=h_open.is_rational)


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
    from . import Bspline  # noqa: PLC0415

    new_ctrl = _append_unit_weight_column(f.control_points)
    return Bspline(f.space, new_ctrl, is_rational=True)


def _multiply_nonrational_1d(f: Bspline, g: Bspline) -> Bspline:
    r"""Multiply two non-rational 1D B-splines with minimal-multiplicity output.

    Refines both operands to Bézier form, applies the Bernstein product formula
    element by element to obtain control points in a Bézier representation of the
    product, then re-expresses those in the minimal product space.

    Bézier form means multiplicity ``degree`` at every interior breakpoint, so
    that each element carries its own ``degree + 1`` control points and adjacent
    elements share exactly their junction — except at a :math:`C^{-1}` knot,
    where the multiplicity is already ``degree + 1``, the elements share nothing,
    and :func:`_bezier_element_offsets` is what keeps the read and write windows
    aligned.  The intermediate product representation follows the same rule, so
    it contains the product even when the product is discontinuous.

    The final knot vector uses the minimal multiplicities of
    :func:`_product_multiplicities`; see the module docstring for the formula.

    This function does not perform input validation; use :func:`_multiply_bspline_1d`
    for the validated public entry point.

    Args:
        f (~pantr.bspline.Bspline): First non-rational 1D B-spline operand.
        g (~pantr.bspline.Bspline): Second non-rational 1D B-spline operand.

    Returns:
        ~pantr.bspline.Bspline: Non-rational B-spline ``h`` such that
        ``h(t) = f(t) * g(t)`` for all ``t`` in the shared domain.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_multiply_bspline_1d` instead.
    """
    from . import Bspline  # noqa: PLC0415

    space_f = f.space.spaces[0]
    space_g = g.space.spaces[0]
    p = space_f.degree
    q = space_g.degree
    r = p + q
    tol = max(float(space_f.tolerance), float(space_g.tolerance))

    # --- Step 1: union of the interior breakpoints, with the product multiplicities ---
    mesh = _build_product_mesh(space_f, space_g, tol)
    all_bp = mesh.breakpoints
    product_mults = mesh.product_mults
    n_elements = int(all_bp.size) + 1

    # --- Step 2: refine both operands to Bézier form ---
    knots_f_ins = _knots_for_full_bezier(space_f, all_bp, mesh.mults_f)
    knots_g_ins = _knots_for_full_bezier(space_g, all_bp, mesh.mults_g)

    f_bezier: Bspline = f.insert_knots(knots_f_ins) if knots_f_ins.size > 0 else f
    g_bezier: Bspline = g.insert_knots(knots_g_ins) if knots_g_ins.size > 0 else g

    # --- Step 3: assemble the product control points, element by element ---
    # A C^-1 breakpoint keeps its multiplicity degree+1 in Bézier form, so the
    # element windows are cumulative rather than a fixed stride.
    offsets_f = _bezier_element_offsets(_bezier_mults(p, mesh.mults_f))
    offsets_g = _bezier_element_offsets(_bezier_mults(q, mesh.mults_g))
    bezier_mults_h = _bezier_mults(r, product_mults)
    offsets_h = _bezier_element_offsets(bezier_mults_h)

    rank = int(f.control_points.shape[-1])
    ctrl_h_bezier = np.empty((int(offsets_h[-1]) + r + 1, rank), dtype=f.control_points.dtype)

    for e in range(n_elements):
        i_f, i_g, i_h = int(offsets_f[e]), int(offsets_g[e]), int(offsets_h[e])
        b_f_e = f_bezier.control_points[i_f : i_f + p + 1]
        b_g_e = g_bezier.control_points[i_g : i_g + q + 1]
        ctrl_h_bezier[i_h : i_h + r + 1] = _bernstein_product_coefficients(b_f_e, b_g_e)

    # --- Step 4: re-express the Bézier control points in the minimal product space ---
    domain = space_f.domain
    dtype = f.control_points.dtype

    T_full = _build_product_knot_vector(domain, all_bp, bezier_mults_h, r, dtype)
    space_full = BsplineSpace([BsplineSpace1D(T_full, r)])
    h_full = Bspline(space_full, ctrl_h_bezier, is_rational=False)

    # Nothing to remove when the product is already no smoother than C^0 everywhere.
    if all_bp.size == 0 or np.array_equal(product_mults, bezier_mults_h):
        return h_full

    T_opt = _build_product_knot_vector(domain, all_bp, product_mults, r, dtype)

    # Re-express the Bézier control points on the minimal knot vector.  T_opt is a
    # subsequence of T_full, so the Oslo matrix has full column rank and the
    # least-squares solve is an exact change of representation, not a fit.
    from ._bspline_knot_insertion_core import _compute_oslo_matrix_1d_core  # noqa: PLC0415

    # oslo[i,j]: coefficient of old CP j in new CP i (old=minimal, new=Bézier)
    oslo = _compute_oslo_matrix_1d_core(r, T_opt, T_full)

    ctrl_opt, _res, _rank_ls, _sv = np.linalg.lstsq(oslo, ctrl_h_bezier, rcond=None)
    ctrl_opt = ctrl_opt.astype(dtype)

    space_opt = BsplineSpace([BsplineSpace1D(T_opt, r)])
    return Bspline(space_opt, ctrl_opt, is_rational=False)


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
    from . import Bspline  # noqa: PLC0415

    # Split into numerator and denominator non-rational B-splines.
    N_f = Bspline(f.space, f.control_points[:, :-1])
    D_f = Bspline(f.space, f.control_points[:, -1:])
    N_g = Bspline(g.space, g.control_points[:, :-1])
    D_g = Bspline(g.space, g.control_points[:, -1:])

    H_N = _multiply_nonrational_1d(N_f, N_g)
    H_D = _multiply_nonrational_1d(D_f, D_g)

    ctrl_h = np.concatenate([H_N.control_points, H_D.control_points], axis=-1)
    return Bspline(H_N.space, ctrl_h, is_rational=True)


def _multiply_bspline_1d(f: Bspline, g: Bspline) -> Bspline:  # noqa: PLR0912
    """Compute the exact pointwise product of two 1D B-splines.

    Given B-splines ``f`` and ``g`` over the same 1D parametric domain, returns
    a new B-spline ``h`` such that ``h(t) = f(t) * g(t)`` for all ``t`` in the
    domain.  The result lives in the smallest space that contains the product:
    degree ``p + q``, with the interior multiplicities of
    :func:`_product_multiplicities` — the maximum of ``m_f(ξ) + q`` and
    ``m_g(ξ) + p`` over the operands that have a knot at ``ξ``.  Discontinuous
    (:math:`C^{-1}`) operands are included: the product is then discontinuous
    too, at multiplicity ``p + q + 1``.

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

    Note:
        The boundary structure of the operands is preserved in the result:

        - If both operands are open (clamped), the result is open.
        - If both operands are periodic, the result is periodic.
        - If one operand is periodic and the other is non-open (unclamped),
          or both are non-open, the result is non-open.
        - If either operand is open, the result is open.
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

    # Determine the target boundary type.
    f_is_open = space_f.has_open_knots() and not space_f.periodic
    g_is_open = space_g.has_open_knots() and not space_g.periodic
    both_periodic = space_f.periodic and space_g.periodic

    if f_is_open or g_is_open:
        target_type = "open"
    elif both_periodic:
        target_type = "periodic"
    else:
        target_type = "nonopen"

    # Save original spaces for boundary multiplicity extraction.
    space_f_orig = space_f
    space_g_orig = space_g

    # Convert non-open operands to open form for multiplication.
    if not f_is_open:
        f = f.to_open_bspline()
        space_f = f.space.spaces[0]
    if not g_is_open:
        g = g.to_open_bspline()
        space_g = g.space.spaces[0]

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

    # Compute the product in open form.
    if not f.is_rational and not g.is_rational:
        h_open = _multiply_nonrational_1d(f, g)
    else:
        h_open = _multiply_rational_1d(_to_rational(f), _to_rational(g))

    # Convert to the target boundary type.
    if target_type == "periodic":
        return _open_to_periodic_product(h_open, space_f_orig, space_g_orig)
    elif target_type == "nonopen":
        return _open_to_nonopen_product(h_open, space_f_orig, space_g_orig)
    return h_open
