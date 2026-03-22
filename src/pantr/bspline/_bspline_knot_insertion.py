"""Layer 2 implementation for B-spline knot insertion and periodic conversion.

This module provides input validation, multi-dimensional looping, and subdivision
logic that wrap the Layer 3 Oslo-algorithm kernels.  It also implements the
open-to-periodic conversion via an exact change-of-basis (Oslo + wrapping matrix).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ._bspline_knot_insertion_core import _compute_oslo_matrix_1d_core, _insert_knots_1d_core
from ._bspline_knots import (
    _get_Bspline_num_basis_1D_impl,
    _get_unique_knots_and_multiplicity_impl,
    _is_in_domain_impl,
)

if TYPE_CHECKING:
    from . import Bspline, BsplineSpace1D


def _compute_inserted_knot_vector_1d(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    new_knots_to_insert: npt.NDArray[np.float32 | np.float64],
    tol: float,
) -> npt.NDArray[np.float32 | np.float64]:
    """Validate new knots and compute the merged (refined) knot vector.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): Original B-spline knot vector.
        degree (int): Polynomial degree.
        new_knots_to_insert (npt.NDArray[np.float32 | np.float64]): 1D array of knot
            values to insert.  Must already be cast to the same dtype as ``knots``.
        tol (float): Tolerance for domain and multiplicity checks.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Merged, sorted knot vector.

    Raises:
        ValueError: If ``new_knots_to_insert`` is not 1D.
        ValueError: If ``new_knots_to_insert`` is empty.
        ValueError: If any value lies outside the B-spline domain.
        ValueError: If any knot's resulting multiplicity would exceed ``degree + 1``.
    """
    if new_knots_to_insert.ndim != 1:
        raise ValueError(
            f"new_knots must be a 1D array-like, got shape {new_knots_to_insert.shape}"
        )

    if new_knots_to_insert.size == 0:
        raise ValueError("new_knots_to_insert must not be empty.")

    # Domain check.
    in_domain = _is_in_domain_impl(knots, degree, new_knots_to_insert, tol)
    if not np.all(in_domain):
        bad = new_knots_to_insert[~in_domain]
        domain_lo = float(knots[degree])
        domain_hi = float(knots[-degree - 1])
        raise ValueError(
            f"new_knots contains values outside the domain [{domain_lo}, {domain_hi}]: {bad}"
        )

    # Merge and sort.
    merged = np.sort(np.concatenate([knots, new_knots_to_insert]).astype(knots.dtype, copy=False))

    # Multiplicity check: no value may appear more than degree+1 times.
    max_allowed = degree + 1
    _, mults = _get_unique_knots_and_multiplicity_impl(merged, degree, tol, False)
    if np.any(mults > max_allowed):
        raise ValueError(
            f"Inserting these knots would exceed the maximum multiplicity of {max_allowed}. "
            f"Maximum multiplicity found: {int(np.max(mults))}."
        )

    return merged


def _insert_knots_bspline_1d_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    ctrl: npt.NDArray[np.float32 | np.float64],
    new_knots_to_insert: npt.NDArray[np.float32 | np.float64],
    tol: float,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
    """Insert knots into a 1D B-spline and compute new control points.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): Original knot vector of shape
            ``(n + degree + 2,)``.
        degree (int): Polynomial degree.
        ctrl (npt.NDArray[np.float32 | np.float64]): Control point matrix of shape
            ``(n+1, rank)``.
        new_knots_to_insert (npt.NDArray[np.float32 | np.float64]): 1D array of knot
            values to insert.
        tol (float): Tolerance for validation.

    Returns:
        tuple[npt.NDArray, npt.NDArray]: ``(refined_knots, new_ctrl)`` where
        ``refined_knots`` is the merged knot vector and ``new_ctrl`` has shape
        ``(m+1, rank)``.

    Raises:
        ValueError: If ``new_knots_to_insert`` is invalid (see
            :func:`_compute_inserted_knot_vector_1d`).
    """
    refined_knots = _compute_inserted_knot_vector_1d(knots, degree, new_knots_to_insert, tol)

    if refined_knots.shape[0] == knots.shape[0]:
        # No knots were added (empty insertion).
        return refined_knots, ctrl.copy()

    # Ensure contiguous layout for the Numba kernel.
    ctrl_c = ctrl if ctrl.flags.c_contiguous else np.ascontiguousarray(ctrl)

    new_ctrl = _insert_knots_1d_core(degree, knots, ctrl_c, refined_knots)
    return refined_knots, new_ctrl


def _insert_knots_bspline(
    bspline: Bspline,
    new_knots_per_dim: list[npt.NDArray[np.float32 | np.float64] | None],
) -> Bspline:
    """Apply knot insertion per parametric direction and return a new B-spline.

    The control points are updated so that the new B-spline represents the same
    geometry as the original one.

    Args:
        bspline (Bspline): Original B-spline.
        new_knots_per_dim (list[npt.NDArray | None]): Per-direction arrays of knot
            values to insert.  ``None`` or an empty array skips that direction.

    Returns:
        Bspline: New B-spline with the same geometry and refined knot vectors.
    """
    dim = bspline.dim
    ctrl = bspline.control_points

    from ._bspline_space_1d import BsplineSpace1D  # noqa: PLC0415

    new_spaces_1d: list[BsplineSpace1D] = []

    for i in range(dim):
        space_1d = bspline.space.spaces[i]
        nk = new_knots_per_dim[i]

        if nk is None or nk.size == 0:
            new_spaces_1d.append(space_1d)
            continue

        # Move dimension i to the 0th axis.
        moved_ctrl = np.moveaxis(ctrl, i, 0)
        orig_shape = moved_ctrl.shape

        # Reshape remaining axes into a single column dimension.
        pts_2d = moved_ctrl.reshape(orig_shape[0], -1)
        if not pts_2d.flags.c_contiguous:
            pts_2d = np.ascontiguousarray(pts_2d)

        refined_knots, new_pts_2d = _insert_knots_bspline_1d_impl(
            space_1d.knots, space_1d.degree, pts_2d, nk, space_1d.tolerance
        )

        # Restore multi-dimensional shape.
        new_shape = (new_pts_2d.shape[0], *orig_shape[1:])
        new_moved_ctrl = new_pts_2d.reshape(new_shape)

        # Move axis back to its original position.
        ctrl = np.moveaxis(new_moved_ctrl, 0, i)

        new_spaces_1d.append(BsplineSpace1D(refined_knots, space_1d.degree))

    # Assemble the new B-spline.
    from . import (  # noqa: PLC0415
        Bspline,
        BsplineSpace,
    )

    new_space = BsplineSpace(new_spaces_1d)
    return Bspline(new_space, ctrl, is_rational=bspline.is_rational)


def _to_open_bspline_1d_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    ctrl_2d: npt.NDArray[np.float32 | np.float64],
    periodic: bool,
    tol: float,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
    """Convert a single parametric direction to an open (clamped) knot vector.

    Inserts ``degree + 1 - m_left`` copies of the left boundary knot and
    ``degree + 1 - m_right`` copies of the right boundary knot so that both
    endpoints have multiplicity ``degree + 1``.  For periodic splines the
    ``n_full = len(knots) - degree - 1`` control points are reconstructed by
    modulo-wrapping the ``n_periodic`` stored rows of ``ctrl_2d``
    (``ctrl_full[i] = ctrl_2d[i % n_periodic]``), and ghost knots are trimmed
    after insertion.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): Knot vector of shape
            ``(len(knots),)``. May include ghost knots if ``periodic=True``.
        degree (int): Polynomial degree.
        ctrl_2d (npt.NDArray[np.float32 | np.float64]): Control point matrix of
            shape ``(n_stored, rank)`` where ``n_stored = num_total_basis``.
        periodic (bool): Whether the spline is periodic (has ghost knots).
        tol (float): Knot comparison tolerance.

    Returns:
        tuple[npt.NDArray, npt.NDArray]: ``(open_knots, open_ctrl)`` — the
        clamped knot vector and the corresponding control points.

    Raises:
        ValueError: If the knot vector already has full multiplicity at both
            boundaries (i.e. the spline is already open and non-periodic).
    """
    p = degree
    a = float(knots[p])
    b = float(knots[-p - 1])

    m_left = int(np.sum(np.isclose(knots[: p + 1], a, atol=tol)))
    m_right = int(np.sum(np.isclose(knots[-p - 1 :], b, atol=tol)))

    if m_left == p + 1 and m_right == p + 1 and not periodic:
        raise ValueError(
            "B-spline is already open (both boundary knots have multiplicity degree + 1)."
        )

    # For periodic splines, build the full non-periodic ctrl by modulo-wrapping.
    ctrl_full: npt.NDArray[np.float32 | np.float64]
    if periodic:
        n_stored = ctrl_2d.shape[0]
        n_full = len(knots) - p - 1
        indices = np.arange(n_full) % n_stored
        ctrl_full = ctrl_2d[indices]
    else:
        ctrl_full = ctrl_2d

    # Insert boundary knots to reach multiplicity degree + 1.
    knots_to_insert = np.array([a] * (p + 1 - m_left) + [b] * (p + 1 - m_right), dtype=knots.dtype)

    new_knots: npt.NDArray[np.float32 | np.float64]
    new_ctrl: npt.NDArray[np.float32 | np.float64]
    if knots_to_insert.size > 0:
        new_knots, new_ctrl = _insert_knots_bspline_1d_impl(
            knots, p, ctrl_full, knots_to_insert, tol
        )
    else:
        new_knots, new_ctrl = knots, ctrl_full

    # Trim ghost knots: the first p and last p entries of new_knots are ghost knots.
    open_knots = new_knots[p : len(new_knots) - p]
    n_open = len(open_knots) - p - 1
    open_ctrl = new_ctrl[p : p + n_open]

    return open_knots, open_ctrl


def _to_open_bspline_impl(bspline: Bspline) -> Bspline:
    """Convert all parametric directions of a B-spline to open (clamped) form.

    For each direction that is periodic or has unclamped boundary knots, inserts
    knots at the domain boundaries until multiplicity ``degree + 1`` is achieved
    and strips any ghost knots.  Directions that are already open are left
    unchanged.

    Raises a ``ValueError`` if every direction is already open (i.e. the spline
    is already fully open and non-periodic).

    Args:
        bspline (Bspline): Input B-spline. May be periodic, unclamped, or
            multi-dimensional.

    Returns:
        Bspline: New B-spline with open knot vectors in all directions and
        ``periodic=False``.

    Raises:
        ValueError: If the B-spline is already open in every direction.
    """
    from . import (  # noqa: PLC0415
        Bspline,
        BsplineSpace,
        BsplineSpace1D,
    )

    dim = bspline.dim
    ctrl = bspline.control_points  # shape (*num_basis, rank)

    # Check if every direction is already open.
    if all(s.has_open_knots() and not s.periodic for s in bspline.space.spaces):
        raise ValueError("B-spline is already open in every direction.")

    new_spaces_1d: list[BsplineSpace1D] = []

    for i in range(dim):
        space_1d = bspline.space.spaces[i]

        # Skip already-open directions.
        if space_1d.has_open_knots() and not space_1d.periodic:
            new_spaces_1d.append(space_1d)
            continue

        # Move dimension i to the 0th axis and flatten remaining axes.
        moved_ctrl = np.moveaxis(ctrl, i, 0)
        orig_shape = moved_ctrl.shape
        pts_2d = moved_ctrl.reshape(orig_shape[0], -1)
        if not pts_2d.flags.c_contiguous:
            pts_2d = np.ascontiguousarray(pts_2d)

        open_knots, open_pts_2d = _to_open_bspline_1d_impl(
            space_1d.knots,
            space_1d.degree,
            pts_2d,
            space_1d.periodic,
            float(space_1d.tolerance),
        )

        # Restore multi-dimensional shape.
        new_shape = (open_pts_2d.shape[0], *orig_shape[1:])
        new_moved_ctrl = open_pts_2d.reshape(new_shape)

        # Move axis back to its original position.
        ctrl = np.moveaxis(new_moved_ctrl, 0, i)

        new_spaces_1d.append(BsplineSpace1D(open_knots, space_1d.degree, periodic=False))

    new_space = BsplineSpace(new_spaces_1d)
    return Bspline(new_space, ctrl, is_rational=bspline.is_rational)


def _build_periodic_knot_vector(
    open_knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    interior_bp: npt.NDArray[np.float32 | np.float64],
    interior_mults: npt.NDArray[np.int_],
    m_bdy: int,
) -> npt.NDArray[np.float32 | np.float64]:
    """Build a periodic knot vector from an open knot vector's breakpoints.

    Constructs the in-domain part ``[a^{m_bdy}, interior, b^{m_bdy}]`` and
    extends it periodically with ghost knots on each side.  Ghost knots are
    computed by applying integer shifts to individual breakpoint values
    (``bp + k * period``) rather than shifting entire arrays, to minimize
    floating-point drift.

    Args:
        open_knots (npt.NDArray[np.float32 | np.float64]): Open knot vector.
        degree (int): Polynomial degree.
        interior_bp (npt.NDArray[np.float32 | np.float64]): Unique interior
            breakpoints (between ``a`` and ``b``, exclusive).
        interior_mults (npt.NDArray[np.int_]): Multiplicity of each interior
            breakpoint.
        m_bdy (int): Target boundary multiplicity.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Periodic knot vector with ghost
        knots.
    """
    dtype = open_knots.dtype
    p = degree
    a = dtype.type(open_knots[p])
    b = dtype.type(open_knots[-p - 1])
    period = dtype.type(b - a)

    n_ghost = p + 1 - m_bdy  # ghost knot entries needed on each side

    # Build the unique breakpoints (with multiplicity) for one period.
    # "tile" = [a, xi_1, ..., xi_{N-1}], each repeated by its multiplicity.
    bp_vals = np.concatenate([[a], interior_bp])
    bp_mults = np.concatenate([[m_bdy], interior_mults]).astype(int)

    # In-domain part: tile + b^{m_bdy}.
    in_domain_parts: list[npt.NDArray[np.float32 | np.float64]] = []
    for val, m in zip(bp_vals, bp_mults, strict=True):
        in_domain_parts.append(np.full(m, dtype.type(val), dtype=dtype))
    in_domain_parts.append(np.full(m_bdy, b, dtype=dtype))
    in_domain = np.concatenate(in_domain_parts)

    if n_ghost <= 0:
        return in_domain

    # Build ghost knots by generating shifted copies of the tile breakpoints.
    # We create a long enough sequence and slice to n_ghost entries.
    # Right ghosts: interior breakpoints + period, then a + 2*period, etc.
    right_entries: list[np.floating[Any]] = []
    shift = 1
    while len(right_entries) < n_ghost:
        for val, m in zip(bp_vals, bp_mults, strict=True):
            if shift == 1 and val == a:
                continue  # skip the leading a at shift=1 (it equals b)
            for _ in range(m):
                right_entries.append(dtype.type(val + shift * period))
                if len(right_entries) == n_ghost:
                    break
            if len(right_entries) == n_ghost:
                break
        shift += 1

    # Left ghosts: full tile shifted by -1, -2, etc.
    left_entries: list[np.floating[Any]] = []
    shift = -1
    while len(left_entries) < n_ghost:
        for val, m in zip(bp_vals, bp_mults, strict=True):
            for _ in range(m):
                left_entries.append(dtype.type(val + shift * period))
        shift -= 1
    left_entries = left_entries[-n_ghost:]

    left_arr = np.array(left_entries, dtype=dtype)
    right_arr = np.array(right_entries, dtype=dtype)

    return np.concatenate([left_arr, in_domain, right_arr])


def _to_periodic_bspline_1d_impl(
    open_knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    ctrl_2d: npt.NDArray[np.float32 | np.float64],
    m_bdy: int,
    tol: float,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
    """Convert an open (clamped) 1D B-spline to periodic form.

    Performs an exact change of basis from the open representation to a periodic
    one with the specified boundary multiplicity ``m_bdy``.  The Oslo matrix
    defines the linear map from the periodic (ghost-extended, wrapped) basis to
    the open basis; inverting this map via QR factorization recovers the periodic
    control points.

    Args:
        open_knots (npt.NDArray[np.float32 | np.float64]): Open (clamped) knot
            vector with multiplicity ``degree + 1`` at both boundaries.
        degree (int): Polynomial degree.
        ctrl_2d (npt.NDArray[np.float32 | np.float64]): Control points of shape
            ``(n_open, rank)``.
        m_bdy (int): Target boundary multiplicity for the periodic knot vector.
            Must satisfy ``1 <= m_bdy <= degree``.
        tol (float): Knot comparison tolerance.

    Returns:
        tuple[npt.NDArray, npt.NDArray]: ``(periodic_knots, periodic_ctrl)`` where
        ``periodic_knots`` includes ghost knots and ``periodic_ctrl`` has shape
        ``(n_periodic, rank)``.

    Raises:
        ValueError: If the function is not periodic (C^0 check at seam or
            residual exceeds tolerance).

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :meth:`~pantr.bspline.Bspline.to_periodic` instead.
    """
    p = degree
    dtype = open_knots.dtype

    a = float(open_knots[p])
    b = float(open_knots[-p - 1])

    # --- Quick C^0 check when knots are clamped (P_0 = f(a), P_{n-1} = f(b)) ---
    m_left = int(np.sum(np.isclose(open_knots[: p + 1], a, atol=tol)))
    m_right = int(np.sum(np.isclose(open_knots[-p - 1 :], b, atol=tol)))
    is_clamped = m_left == p + 1 and m_right == p + 1
    if is_clamped:
        eps = float(np.finfo(np.float64).eps)
        scale = max(float(np.max(np.abs(ctrl_2d))), 1.0)
        c0_tol = max(tol, 100.0 * eps * scale)
        c0_dev = float(np.max(np.abs(ctrl_2d[0] - ctrl_2d[-1])))
        if c0_dev > c0_tol:
            raise ValueError(
                "Function values do not match at the domain endpoints "
                f"(max deviation {c0_dev:.2e}, tolerance {c0_tol:.2e}). "
                "The B-spline is not periodic."
            )

    _, mults_all = _get_unique_knots_and_multiplicity_impl(open_knots, p, tol, in_domain=True)
    # Extract exact breakpoint values directly from open_knots to avoid
    # floating-point drift from the averaging in unique-knot computation.
    # The in-domain knots of open_knots are at indices [p, ..., len-p-1].
    in_domain_flat = open_knots[p : len(open_knots) - p]
    # Deduplicate preserving exact values: take the first occurrence of each group.
    bp_exact: list[np.float32 | np.float64] = [in_domain_flat[0]]
    for v in in_domain_flat[1:]:
        if abs(float(v) - float(bp_exact[-1])) > tol:
            bp_exact.append(v)
    interior_bp = np.array(bp_exact[1:-1], dtype=dtype)
    interior_mults = mults_all[1:-1]

    T_per = _build_periodic_knot_vector(open_knots, p, interior_bp, interior_mults, m_bdy)

    # --- Build the Oslo chain: T_per → open_knots ---
    n_ghost = p + 1 - m_bdy
    n_open = ctrl_2d.shape[0]

    # Step 1: Insert clamping knots into T_per → T_inter.
    knots_ins = np.array([a] * n_ghost + [b] * n_ghost, dtype=dtype)
    T_inter = np.sort(np.concatenate([T_per, knots_ins]))
    oslo1 = _compute_oslo_matrix_1d_core(p, T_per, T_inter)

    # Step 2: Trim ghost region from T_inter to isolate the open part.
    n_inter_open = len(T_inter) - 2 * n_ghost - p - 1
    oslo1_trimmed = oslo1[n_ghost : n_ghost + n_inter_open, :]

    # Step 3: If the trimmed intermediate has more CPs than the open form
    # (because open_knots has fewer knots), map through a second Oslo step.
    # Use open_knots directly (not T_inter trimmed) to avoid float artefacts.
    if n_inter_open == n_open:
        oslo = oslo1_trimmed
    else:
        T_inter_open = T_inter[n_ghost : len(T_inter) - n_ghost]
        oslo2 = _compute_oslo_matrix_1d_core(p, T_inter_open, open_knots)
        oslo = oslo2 @ oslo1_trimmed  # shape (n_open, n_full)

    # --- Build the modulo-wrapping matrix W ---
    n_full = len(T_per) - p - 1
    n_per = int(_get_Bspline_num_basis_1D_impl(T_per, p, True, tol))
    W = np.zeros((n_full, n_per), dtype=np.float64)
    for i in range(n_full):
        W[i, i % n_per] = 1.0

    # --- Exact solve via QR: M @ C_per = C_open ---
    M = oslo @ W  # shape (n_open, n_per)
    Q, R = np.linalg.qr(M)
    ctrl_per = np.linalg.solve(R, Q.T @ ctrl_2d.astype(np.float64))

    # --- Residual check ---
    # Use a tolerance relative to the control point magnitude, but at least
    # 100 * machine epsilon for the working dtype to absorb floating-point
    # noise from the QR solve and Oslo chain.
    eps = float(np.finfo(np.float64).eps)
    scale = max(float(np.max(np.abs(ctrl_2d))), 1.0)
    residual_tol = max(tol, 100.0 * eps * scale)
    residual = float(np.max(np.abs(M @ ctrl_per - ctrl_2d.astype(np.float64))))
    if residual > residual_tol:
        raise ValueError(
            f"Residual {residual:.2e} exceeds tolerance {residual_tol:.2e}. "
            "The B-spline does not have sufficient smoothness at the seam "
            "for the requested periodic regularity."
        )

    return T_per, ctrl_per.astype(dtype)


def _to_periodic_bspline_impl(
    bspline: Bspline,
    continuity: int | tuple[int, ...] | None,
) -> Bspline:
    """Convert all parametric directions of a B-spline to periodic form.

    For each direction that is not already periodic, computes an exact
    change of basis from the open/non-open representation to a periodic one
    with the requested continuity at the seam.

    Args:
        bspline (Bspline): Input B-spline.
        continuity (int | tuple[int, ...] | None): Target continuity at the seam
            per direction.  ``None`` means maximum regularity (``degree - 1``).
            An integer applies to all directions; a tuple specifies per-direction.

    Returns:
        Bspline: New periodic B-spline.

    Raises:
        ValueError: If already periodic in every direction.
        ValueError: If the function is not periodic in some direction.
    """
    from . import Bspline as BsplineCls  # noqa: PLC0415
    from . import BsplineSpace, BsplineSpace1D  # noqa: PLC0415

    dim = bspline.dim
    ctrl = bspline.control_points

    # Check if every direction is already periodic.
    if all(s.periodic for s in bspline.space.spaces):
        raise ValueError("B-spline is already periodic in every direction.")

    # Normalize continuity to a per-direction tuple.
    if continuity is None:
        cont_per_dir: tuple[int | None, ...] = tuple(None for _ in range(dim))
    elif isinstance(continuity, int):
        cont_per_dir = tuple(continuity for _ in range(dim))
    else:
        if len(continuity) != dim:
            raise ValueError(f"continuity tuple length {len(continuity)} != dimension {dim}.")
        cont_per_dir = tuple(continuity)

    # First, ensure we have an open representation (non-periodic directions
    # that are not yet open need clamping before conversion).
    # We process each direction: if periodic, skip; otherwise convert to open
    # first (if needed), then convert to periodic.
    new_spaces_1d: list[BsplineSpace1D] = []

    for i in range(dim):
        space_1d = bspline.space.spaces[i]

        if space_1d.periodic:
            new_spaces_1d.append(space_1d)
            continue

        p = space_1d.degree

        # Resolve continuity for this direction.
        c = cont_per_dir[i]
        if c is None:
            c = p - 1
        if not (0 <= c <= p - 1):
            raise ValueError(
                f"continuity must be in [0, degree-1]=[0, {p - 1}] for direction {i}, got {c}."
            )
        m_bdy = p - c

        # Ensure direction is open (clamped) before periodic conversion.
        knots_1d = space_1d.knots
        tol_1d = float(space_1d.tolerance)
        moved_ctrl = np.moveaxis(ctrl, i, 0)
        orig_shape = moved_ctrl.shape
        pts_2d = moved_ctrl.reshape(orig_shape[0], -1)
        if not pts_2d.flags.c_contiguous:
            pts_2d = np.ascontiguousarray(pts_2d)

        per_knots, per_pts_2d = _to_periodic_bspline_1d_impl(knots_1d, p, pts_2d, m_bdy, tol_1d)

        new_shape = (per_pts_2d.shape[0], *orig_shape[1:])
        new_moved_ctrl = per_pts_2d.reshape(new_shape)
        ctrl = np.moveaxis(new_moved_ctrl, 0, i)

        new_spaces_1d.append(BsplineSpace1D(per_knots, p, periodic=True))

    new_space = BsplineSpace(new_spaces_1d)
    return BsplineCls(new_space, ctrl, is_rational=bspline.is_rational)


def _compute_uniform_subdivision_knots(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    n_subdivisions: int,
    regularity: int | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Compute the new knots required to subdivide every knot span uniformly.

    For each non-zero span ``[u_i, u_{i+1})`` of the knot vector, generates
    ``n_subdivisions - 1`` uniformly-spaced interior knot values, each repeated
    ``degree - regularity`` times to achieve the requested continuity.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): Polynomial degree.
        tol (float): Tolerance for uniqueness detection.
        n_subdivisions (int): Number of equal sub-spans per existing interval.
            Must be >= 2 (callers have already validated this).
        regularity (int | None): Desired continuity order at inserted knots.
            Must be in ``[-1, degree - 1]``.  ``None`` defaults to
            ``degree - 1`` (minimum multiplicity = 1, maximum continuity).

    Returns:
        npt.NDArray[np.float32 | np.float64]: 1D array of new knot values to insert.
            May be empty if the knot vector has only one unique span.
    """
    eff_regularity = degree - 1 if regularity is None else regularity
    repeat = degree - eff_regularity  # multiplicity of each inserted knot

    unique_knots, _ = _get_unique_knots_and_multiplicity_impl(knots, degree, tol, in_domain=True)

    dtype = knots.dtype
    parts: list[npt.NDArray[np.float32 | np.float64]] = []

    for k in range(len(unique_knots) - 1):
        lo = float(unique_knots[k])
        hi = float(unique_knots[k + 1])
        # Generate n_subdivisions-1 equally-spaced interior points.
        interior = np.linspace(lo, hi, n_subdivisions + 1, dtype=np.float64)[1:-1]
        # Repeat each value `repeat` times to achieve the requested regularity.
        repeated = np.repeat(interior, repeat)
        parts.append(repeated.astype(dtype, copy=False))

    if not parts:
        return np.empty(0, dtype=dtype)

    return np.concatenate(parts)
