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
    from .bspline import Bspline
    from .bspline_space_1D import BsplineSpace1D


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
        ValueError: If any value lies outside the B-spline domain.
        ValueError: If any knot's resulting multiplicity would exceed ``degree + 1``.
    """
    if new_knots_to_insert.ndim != 1:
        raise ValueError(
            f"new_knots must be a 1D array-like, got shape {new_knots_to_insert.shape}"
        )

    if new_knots_to_insert.size == 0:
        return knots.copy()

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

    from .bspline_space_1D import BsplineSpace1D  # noqa: PLC0415

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
    from .bspline import Bspline  # noqa: PLC0415
    from .bspline_space_nd import BsplineSpace  # noqa: PLC0415

    new_space = BsplineSpace(new_spaces_1d)
    return Bspline(new_space, ctrl, is_rational=bspline.is_rational)


def _compute_uniform_subdivision_knots(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    n_subdivisions: int,
) -> npt.NDArray[np.float32 | np.float64]:
    """Compute the new knots required to subdivide every knot span uniformly.

    For each non-zero span ``[u_i, u_{i+1})`` of the knot vector, generates
    ``n_subdivisions - 1`` uniformly-spaced interior knots.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): Polynomial degree.
        tol (float): Tolerance for uniqueness detection.
        n_subdivisions (int): Number of equal sub-spans per existing interval.
            Must be >= 2 (callers have already validated this).

    Returns:
        npt.NDArray[np.float32 | np.float64]: 1D array of new knot values to insert.
            May be empty if the knot vector has only one unique span.
    """
    unique_knots, _ = _get_unique_knots_and_multiplicity_impl(knots, degree, tol, in_domain=True)

    dtype = knots.dtype
    parts: list[npt.NDArray[np.float32 | np.float64]] = []

    for k in range(len(unique_knots) - 1):
        lo = float(unique_knots[k])
        hi = float(unique_knots[k + 1])
        # Generate n_subdivisions-1 equally-spaced interior points.
        interior = np.linspace(lo, hi, n_subdivisions + 1, dtype=np.float64)[1:-1]
        parts.append(interior.astype(dtype, copy=False))

    if not parts:
        return np.empty(0, dtype=dtype)

    return np.concatenate(parts)
