"""Layer 2 implementation for B-spline knot insertion.

This module provides input validation, multi-dimensional looping, and subdivision
logic that wrap the Layer 3 Oslo-algorithm kernels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ._bspline_knot_insertion_core import _insert_knots_1d_core
from ._bspline_knots import _get_unique_knots_and_multiplicity_impl, _is_in_domain_impl

if TYPE_CHECKING:
    from .bspline import Bspline, BsplineSpace1D


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

    from .bspline._bspline_space_1d import BsplineSpace1D  # noqa: PLC0415

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
    from .bspline import (  # noqa: PLC0415
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
    from .bspline import (  # noqa: PLC0415
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
