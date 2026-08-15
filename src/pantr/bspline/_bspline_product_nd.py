"""N-dimensional B-spline pointwise product.

This module provides :func:`_multiply_bspline_nd`, which computes the exact
pointwise product of two N-dimensional tensor-product B-splines via Bézier
extraction and the Bernstein product formula.  The result lives in the product
space of degree ``p_d + q_d`` per direction *d*, with the **minimal** interior
multiplicities computed direction by direction by
:func:`~pantr.bspline._bspline_product._product_multiplicities`; that module's
docstring carries the formula, its derivation and its hypotheses.

Works for non-rational and rational (NURBS) splines of any parametric
dimension, including operands that are :math:`C^{-1}` in one or more
directions, and correctly preserves per-direction boundary structure
(open / periodic / non-open).
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .._array_utils import _flatten_along_axis, _unflatten_along_axis
from .._control_points_utils import _append_unit_weight_column
from ..bezier._bezier_product import _bernstein_product_coefficients_nd
from ._bspline_knots import _get_Bspline_num_basis_1D_impl, _get_unique_knots_and_multiplicity_impl
from ._bspline_product import (
    _bezier_element_offsets,
    _bezier_mults,
    _build_product_knot_vector,
    _build_product_mesh,
    _get_boundary_mults,
    _knots_for_full_bezier,
    _ProductMesh,
)
from ._bspline_space_1d import BsplineSpace1D
from ._bspline_space_nd import BsplineSpace

if TYPE_CHECKING:
    from . import Bspline


def _bezier_patch_slices(
    element_idx: tuple[int, ...],
    degrees: tuple[int, ...],
    offsets: list[npt.NDArray[np.int_]],
) -> tuple[slice, ...]:
    """Index the control points of one element in a Bézier-form tensor-product spline.

    Element ``e_d`` of direction *d* owns ``degree_d + 1`` consecutive control
    points starting at ``offsets[d][e_d]``.  The offsets are cumulative rather
    than a fixed ``e_d * degree_d`` stride, because a :math:`C^{-1}` breakpoint
    keeps multiplicity ``degree_d + 1`` in Bézier form and its two elements then
    share no control point.

    The same slices index the operands' control points for reading and the
    product's for writing.  Where two elements do share a control point the two
    writes carry the same value, so overwriting is safe.

    Args:
        element_idx (tuple[int, ...]): Per-direction element index.
        degrees (tuple[int, ...]): Per-direction polynomial degrees.
        offsets (list[npt.NDArray[np.int_]]): Per-direction element offsets, from
            :func:`~pantr.bspline._bspline_product._bezier_element_offsets`.

    Returns:
        tuple[slice, ...]: One slice per parametric direction; the trailing rank
        axis is left untouched.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    return tuple(
        slice(int(offsets[d][e]), int(offsets[d][e]) + deg + 1)
        for d, (e, deg) in enumerate(zip(element_idx, degrees, strict=True))
    )


def _project_to_optimal_nd(
    ctrl_full: npt.NDArray[np.float32 | np.float64],
    knots_full_per_dir: list[npt.NDArray[np.float32 | np.float64]],
    knots_opt_per_dir: list[npt.NDArray[np.float32 | np.float64]],
    degrees_sum: tuple[int, ...],
) -> npt.NDArray[np.float32 | np.float64]:
    """Re-express Bézier-form control points on the minimal knot vectors, per direction.

    Applies the Oslo-based solve direction by direction.  For each direction *d*
    the Oslo matrix mapping minimal → Bézier is computed; since the minimal knot
    vector is a subsequence of the Bézier one, that matrix has full column rank
    and the least-squares solve is an exact change of representation.

    Args:
        ctrl_full (npt.NDArray[np.float32 | np.float64]): Bézier-form control
            points of shape ``(n_full_0, ..., n_full_{D-1}, rank)``.
        knots_full_per_dir (list[npt.NDArray]): Bézier-form knot vector per direction.
        knots_opt_per_dir (list[npt.NDArray]): Minimal knot vector per direction.
        degrees_sum (tuple[int, ...]): Product degree per direction.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Optimal control points.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    from ._bspline_knot_insertion_core import _compute_oslo_matrix_1d_core  # noqa: PLC0415

    ndim = len(degrees_sum)
    ctrl = ctrl_full
    dtype = ctrl_full.dtype

    for d in range(ndim):
        oslo_d = _compute_oslo_matrix_1d_core(
            degrees_sum[d], knots_opt_per_dir[d], knots_full_per_dir[d]
        )
        flat, trailing_shape = _flatten_along_axis(ctrl, d)
        projected, *_ = np.linalg.lstsq(oslo_d, flat, rcond=None)
        ctrl = _unflatten_along_axis(projected, trailing_shape, d).astype(dtype)

    return ctrl


def _multiply_nonrational_nd(f: Bspline, g: Bspline) -> Bspline:  # noqa: PLR0915
    """Multiply two non-rational nD B-splines with minimal-multiplicity output.

    Refines both operands to Bézier form in all directions, applies the nD
    Bernstein product formula element by element, then re-expresses the result on
    the minimal knot vectors via per-direction Oslo solves.

    Args:
        f (~pantr.bspline.Bspline): First non-rational nD B-spline operand.
        g (~pantr.bspline.Bspline): Second non-rational nD B-spline operand.

    Returns:
        ~pantr.bspline.Bspline: Non-rational B-spline ``h`` such that
        ``h(t) = f(t) * g(t)`` for all ``t`` in the shared domain.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_multiply_bspline_nd` instead.
    """
    from . import Bspline as BsplineCls  # noqa: PLC0415

    ndim = f.dim
    dtype = f.control_points.dtype

    spaces_f = f.space.spaces
    spaces_g = g.space.spaces
    degrees_f = tuple(s.degree for s in spaces_f)
    degrees_g = tuple(s.degree for s in spaces_g)
    degrees_sum = tuple(pf + pg for pf, pg in zip(degrees_f, degrees_g, strict=True))

    tol = max(float(s.tolerance) for s in (*spaces_f, *spaces_g))

    # --- Per-direction breakpoint analysis ---
    meshes: list[_ProductMesh] = []
    bezier_mults_h_per_dir: list[npt.NDArray[np.int_]] = []
    offsets_f_per_dir: list[npt.NDArray[np.int_]] = []
    offsets_g_per_dir: list[npt.NDArray[np.int_]] = []
    offsets_h_per_dir: list[npt.NDArray[np.int_]] = []
    knots_f_ins_per_dir: list[npt.NDArray[np.float32 | np.float64] | None] = []
    knots_g_ins_per_dir: list[npt.NDArray[np.float32 | np.float64] | None] = []

    for d in range(ndim):
        mesh = _build_product_mesh(spaces_f[d], spaces_g[d], tol)
        meshes.append(mesh)

        bezier_mults_h = _bezier_mults(degrees_sum[d], mesh.product_mults)
        bezier_mults_h_per_dir.append(bezier_mults_h)
        offsets_f_per_dir.append(_bezier_element_offsets(_bezier_mults(degrees_f[d], mesh.mults_f)))
        offsets_g_per_dir.append(_bezier_element_offsets(_bezier_mults(degrees_g[d], mesh.mults_g)))
        offsets_h_per_dir.append(_bezier_element_offsets(bezier_mults_h))

        knots_f_d = _knots_for_full_bezier(spaces_f[d], mesh.breakpoints, mesh.mults_f)
        knots_g_d = _knots_for_full_bezier(spaces_g[d], mesh.breakpoints, mesh.mults_g)
        knots_f_ins_per_dir.append(knots_f_d if knots_f_d.size > 0 else None)
        knots_g_ins_per_dir.append(knots_g_d if knots_g_d.size > 0 else None)

    # --- Refine both operands to Bézier form ---
    if any(k is not None for k in knots_f_ins_per_dir):
        f = f.insert_knots(knots_f_ins_per_dir)
    if any(k is not None for k in knots_g_ins_per_dir):
        g = g.insert_knots(knots_g_ins_per_dir)

    # --- Element-wise nD Bernstein product ---
    n_elements_per_dir = tuple(int(mesh.breakpoints.size) + 1 for mesh in meshes)
    rank = f.control_points.shape[-1]
    full_shape = tuple(
        int(off[-1]) + rd + 1 for off, rd in zip(offsets_h_per_dir, degrees_sum, strict=True)
    )
    ctrl_h_bezier = np.empty((*full_shape, rank), dtype=dtype)

    for elem_idx in itertools.product(*(range(ne) for ne in n_elements_per_dir)):
        patch_f = f.control_points[_bezier_patch_slices(elem_idx, degrees_f, offsets_f_per_dir)]
        patch_g = g.control_points[_bezier_patch_slices(elem_idx, degrees_g, offsets_g_per_dir)]
        slices_h = _bezier_patch_slices(elem_idx, degrees_sum, offsets_h_per_dir)
        ctrl_h_bezier[slices_h] = _bernstein_product_coefficients_nd(patch_f, patch_g)

    # --- Build the product knot vectors (Bézier form and minimal) ---
    knots_full_per_dir: list[npt.NDArray[np.float32 | np.float64]] = []
    knots_opt_per_dir: list[npt.NDArray[np.float32 | np.float64]] = []
    needs_projection = False

    for d in range(ndim):
        domain_d = spaces_f[d].domain
        t_full_d = _build_product_knot_vector(
            domain_d, meshes[d].breakpoints, bezier_mults_h_per_dir[d], degrees_sum[d], dtype
        )
        t_opt_d = _build_product_knot_vector(
            domain_d, meshes[d].breakpoints, meshes[d].product_mults, degrees_sum[d], dtype
        )
        knots_full_per_dir.append(t_full_d)
        knots_opt_per_dir.append(t_opt_d)

        if not np.array_equal(meshes[d].product_mults, bezier_mults_h_per_dir[d]):
            needs_projection = True

    # --- Project to optimal continuity if needed ---
    if needs_projection:
        ctrl_opt = _project_to_optimal_nd(
            ctrl_h_bezier, knots_full_per_dir, knots_opt_per_dir, degrees_sum
        )
    else:
        ctrl_opt = ctrl_h_bezier

    # Build the optimal product space.
    spaces_opt = [BsplineSpace1D(knots_opt_per_dir[d], degrees_sum[d]) for d in range(ndim)]
    space_opt = BsplineSpace(spaces_opt)
    return BsplineCls(space_opt, ctrl_opt, is_rational=False)


def _to_rational_nd(f: Bspline) -> Bspline:
    """Convert an nD B-spline to rational form by appending unit weights.

    If ``f`` is already rational, returns it unchanged.  Otherwise, creates a
    new B-spline with the same space and control points augmented by a
    trailing column of ones (homogeneous weights = 1).

    Args:
        f (~pantr.bspline.Bspline): The B-spline to convert.

    Returns:
        ~pantr.bspline.Bspline: Rational B-spline equivalent to ``f``.
    """
    if f.is_rational:
        return f
    from . import Bspline as BsplineCls  # noqa: PLC0415

    new_ctrl = _append_unit_weight_column(f.control_points)
    return BsplineCls(f.space, new_ctrl, is_rational=True)


def _multiply_rational_nd(f: Bspline, g: Bspline) -> Bspline:
    """Multiply two rational nD B-splines (NURBS) using homogeneous coordinates.

    Decomposes each rational operand into its numerator (weighted coordinates)
    and denominator (weights) B-splines, multiplies each pair independently
    using :func:`_multiply_nonrational_nd`, then reassembles the result.

    Both ``f`` and ``g`` must already be rational (``is_rational=True``).

    Args:
        f (~pantr.bspline.Bspline): First rational nD B-spline operand.
        g (~pantr.bspline.Bspline): Second rational nD B-spline operand.

    Returns:
        ~pantr.bspline.Bspline: Rational B-spline ``h`` such that
        ``h(t) = f(t) * g(t)`` for all ``t`` in the shared domain.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_multiply_bspline_nd` instead.
    """
    from . import Bspline as BsplineCls  # noqa: PLC0415

    # Split into numerator and denominator non-rational B-splines.
    n_f = BsplineCls(f.space, f.control_points[..., :-1])
    d_f = BsplineCls(f.space, f.control_points[..., -1:])
    n_g = BsplineCls(g.space, g.control_points[..., :-1])
    d_g = BsplineCls(g.space, g.control_points[..., -1:])

    h_n = _multiply_nonrational_nd(n_f, n_g)
    h_d = _multiply_nonrational_nd(d_f, d_g)

    ctrl_h = np.concatenate([h_n.control_points, h_d.control_points], axis=-1)
    return BsplineCls(h_n.space, ctrl_h, is_rational=True)


def _convert_boundary_direction(  # noqa: PLR0913
    ctrl: npt.NDArray[np.float32 | np.float64],
    axis: int,
    target_type: str,
    h_knots_1d: npt.NDArray[np.float32 | np.float64],
    degree_sum_d: int,
    space_f_1d: BsplineSpace1D,
    space_g_1d: BsplineSpace1D,
) -> tuple[npt.NDArray[np.float32 | np.float64], BsplineSpace1D]:
    """Convert one direction of the open product to periodic or non-open form.

    Computes the ghost knot vector and Oslo chain for direction *axis*, then
    solves the least-squares system along that axis to recover the boundary-
    converted control points.

    Args:
        ctrl (npt.NDArray[np.float32 | np.float64]): Current control point
            array of shape ``(n_0, ..., n_{D-1}, rank)``.
        axis (int): Parametric direction to convert.
        target_type (str): ``"periodic"`` or ``"nonopen"``.
        h_knots_1d (npt.NDArray): Open product knot vector in direction *axis*.
        degree_sum_d (int): Product degree in direction *axis*.
        space_f_1d (BsplineSpace1D): Original 1D space of the first operand in
            direction *axis*.
        space_g_1d (BsplineSpace1D): Original 1D space of the second operand in
            direction *axis*.

    Returns:
        tuple[npt.NDArray, BsplineSpace1D]: Updated control points and the new
        1D space for direction *axis*.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    from ._bspline_knot_insertion_core import _compute_oslo_matrix_1d_core  # noqa: PLC0415
    from ._bspline_product import (  # noqa: PLC0415
        _build_periodic_product_knot_vector,
    )

    dtype = ctrl.dtype
    p_d = space_f_1d.degree
    q_d = space_g_1d.degree
    tol = max(float(space_f_1d.tolerance), float(space_g_1d.tolerance))
    r = degree_sum_d

    # Get interior breakpoints and mults of the open product for this direction.
    tol_typed = float(dtype.type(tol))
    unique, mults = _get_unique_knots_and_multiplicity_impl(
        h_knots_1d, r, tol_typed, in_domain=True
    )
    interior_bp, interior_mults = unique[1:-1], mults[1:-1]

    # Compute boundary multiplicity.
    mf_l, mf_r = _get_boundary_mults(space_f_1d, tol)
    mg_l, mg_r = _get_boundary_mults(space_g_1d, tol)

    if target_type == "periodic":
        mf_bdy = mf_l
        mg_bdy = mg_l
        m_bdy = max(mf_bdy + q_d, mg_bdy + p_d)
    else:
        m_bdy = min(max(mf_l + q_d, mg_l + p_d), max(mf_r + q_d, mg_r + p_d))

    # Build ghost knot vector and Oslo chain for this direction.
    domain = (h_knots_1d[r], h_knots_1d[-r - 1])
    t_ghost = _build_periodic_product_knot_vector(
        domain, interior_bp, interior_mults, m_bdy, r, dtype
    )

    n_ghost = r + 1 - m_bdy
    a_val, b_val = float(t_ghost[r]), float(t_ghost[-r - 1])
    knots_ins = np.array([a_val] * n_ghost + [b_val] * n_ghost, dtype=dtype)
    t_inter = np.sort(np.concatenate([t_ghost, knots_ins]))
    oslo1 = _compute_oslo_matrix_1d_core(r, t_ghost, t_inter)

    t_per_open = t_inter[n_ghost : len(t_inter) - n_ghost]
    n_per_open = len(t_per_open) - r - 1
    oslo1_trimmed = oslo1[n_ghost : n_ghost + n_per_open, :]

    oslo2 = _compute_oslo_matrix_1d_core(r, t_per_open, h_knots_1d)
    oslo = oslo2 @ oslo1_trimmed

    # Apply along the specified axis via the shared flatten/unflatten helpers.
    flat, trailing_shape = _flatten_along_axis(ctrl, axis)

    if target_type == "periodic":
        n_full = len(t_ghost) - r - 1
        n_per = int(
            _get_Bspline_num_basis_1D_impl(t_ghost, r, True, float(np.dtype(dtype).type(tol)))
        )
        w_mat = np.zeros((n_full, n_per), dtype=dtype)
        for i in range(n_full):
            w_mat[i, i % n_per] = dtype.type(1.0)
        sol, *_ = np.linalg.lstsq(oslo @ w_mat, flat, rcond=None)
        new_space_1d = BsplineSpace1D(t_ghost, r, periodic=True)
    else:
        sol, *_ = np.linalg.lstsq(oslo, flat, rcond=None)
        new_space_1d = BsplineSpace1D(t_ghost, r)

    ctrl = _unflatten_along_axis(sol.astype(dtype), trailing_shape, axis)
    return ctrl, new_space_1d


def _multiply_bspline_nd(f: Bspline, g: Bspline) -> Bspline:  # noqa: PLR0912, PLR0915
    """Compute the exact pointwise product of two nD B-splines.

    Given B-splines ``f`` and ``g`` over the same nD parametric domain, returns
    a new B-spline ``h`` such that ``h(t) = f(t) * g(t)`` for all ``t`` in the
    domain.  The result lives in the product space of degree ``p_d + q_d`` per
    direction with optimal continuity.

    Rational operands are handled via homogeneous-coordinate decomposition.  A
    non-rational operand is silently promoted to rational when the other is
    rational.

    Args:
        f (~pantr.bspline.Bspline): First nD B-spline operand.
        g (~pantr.bspline.Bspline): Second nD B-spline operand.

    Returns:
        ~pantr.bspline.Bspline: Product B-spline ``h = f * g``.

    Raises:
        ValueError: If ``f`` and ``g`` have different dimensions.
        ValueError: If ``f`` and ``g`` have different dtypes.
        ValueError: If ``f`` and ``g`` have different ranks.
        ValueError: If ``f`` and ``g`` have different parametric domains in
            any direction (beyond the shared tolerance).
    """
    from . import Bspline as BsplineCls  # noqa: PLC0415

    ndim = f.dim

    if g.dim != ndim:
        raise ValueError(
            f"f and g must have the same parametric dimension. Got f.dim={ndim} and g.dim={g.dim}."
        )
    if f.dtype != g.dtype:
        raise ValueError(f"f and g must have the same dtype. Got {f.dtype} and {g.dtype}.")
    if f.rank != g.rank:
        raise ValueError(f"f and g must have the same rank. Got {f.rank} and {g.rank}.")

    spaces_f = f.space.spaces
    spaces_g = g.space.spaces

    # --- Per-direction boundary type detection ---
    target_types: list[str] = []
    f_needs_open = False
    g_needs_open = False

    for d in range(ndim):
        sf, sg = spaces_f[d], spaces_g[d]
        f_is_open = sf.has_open_knots() and not sf.periodic
        g_is_open = sg.has_open_knots() and not sg.periodic
        f_is_periodic = sf.periodic
        g_is_periodic = sg.periodic

        if f_is_open or g_is_open:
            target_types.append("open")
        elif f_is_periodic and g_is_periodic:
            target_types.append("periodic")
        else:
            target_types.append("nonopen")

        if not f_is_open:
            f_needs_open = True
        if not g_is_open:
            g_needs_open = True

    # Save original spaces for boundary conversion.
    spaces_f_orig = spaces_f
    spaces_g_orig = spaces_g

    # Convert non-open operands to open form.
    if f_needs_open:
        has_nonopen = any(
            not spaces_f[d].has_open_knots() or spaces_f[d].periodic for d in range(ndim)
        )
        if has_nonopen:
            f = f.to_open_bspline()
    if g_needs_open:
        has_nonopen = any(
            not spaces_g[d].has_open_knots() or spaces_g[d].periodic for d in range(ndim)
        )
        if has_nonopen:
            g = g.to_open_bspline()

    # Validate per-direction domains.
    tol = max(float(s.tolerance) for s in (*f.space.spaces, *g.space.spaces))
    for d in range(ndim):
        domain_f = f.space.spaces[d].domain
        domain_g = g.space.spaces[d].domain
        if (
            abs(float(domain_f[0]) - float(domain_g[0])) > tol
            or abs(float(domain_f[1]) - float(domain_g[1])) > tol
        ):
            raise ValueError(
                f"f and g must share the same parametric domain in direction {d}. "
                f"Got {domain_f} and {domain_g}."
            )

    # Compute the product in open form.
    if not f.is_rational and not g.is_rational:
        h_open = _multiply_nonrational_nd(f, g)
    else:
        h_open = _multiply_rational_nd(_to_rational_nd(f), _to_rational_nd(g))

    # Convert to target boundary types per direction.
    any_nonopen = any(t != "open" for t in target_types)
    if not any_nonopen:
        return h_open

    ctrl = h_open.control_points
    result_spaces: list[BsplineSpace1D] = list(h_open.space.spaces)

    for d in range(ndim):
        if target_types[d] == "open":
            continue
        ctrl, new_space_1d = _convert_boundary_direction(
            ctrl,
            axis=d,
            target_type=target_types[d],
            h_knots_1d=h_open.space.spaces[d].knots,
            degree_sum_d=h_open.space.spaces[d].degree,
            space_f_1d=spaces_f_orig[d],
            space_g_1d=spaces_g_orig[d],
        )
        result_spaces[d] = new_space_1d

    space_result = BsplineSpace(result_spaces)
    return BsplineCls(space_result, ctrl, is_rational=h_open.is_rational)
