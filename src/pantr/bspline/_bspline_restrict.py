"""Layer 2 implementation for B-spline domain restriction.

This module provides the algorithm for extracting a sub-region of the parametric
domain of a B-spline. The core logic inserts knots at the new boundaries until they
reach multiplicity ``degree + 1``, then extracts the relevant knot sub-vector and
control points. An optimization skips insertion when a bound coincides with an
already-open domain endpoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ._bspline_knot_insertion import (
    _insert_knots_bspline_1d_impl,
    _to_open_bspline_1d_impl,
)

if TYPE_CHECKING:
    from . import Bspline


def _validate_restrict_bounds(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    a_new: float,
    b_new: float,
) -> tuple[float, float]:
    """Validate and snap restriction bounds for a 1D B-spline.

    Args:
        knots: Knot vector.
        degree: Polynomial degree.
        tol: Knot comparison tolerance.
        a_new: Requested left bound.
        b_new: Requested right bound.

    Returns:
        tuple[float, float]: Snapped ``(a_new, b_new)`` bounds.

    Raises:
        ValueError: If ``a_new >= b_new`` or bounds lie outside the domain.
    """
    a = float(knots[degree])
    b = float(knots[-degree - 1])

    if a_new >= b_new:
        raise ValueError(f"Lower bound ({a_new}) must be strictly less than upper bound ({b_new}).")

    if a_new < a and not np.isclose(a_new, a, atol=tol):
        raise ValueError(f"Lower bound ({a_new}) is below the domain start ({a}).")
    if b_new > b and not np.isclose(b_new, b, atol=tol):
        raise ValueError(f"Upper bound ({b_new}) is above the domain end ({b}).")

    # Snap bounds to domain endpoints if within tolerance.
    if np.isclose(a_new, a, atol=tol):
        a_new = a
    if np.isclose(b_new, b, atol=tol):
        b_new = b

    return a_new, b_new


def _compute_boundary_knots_to_insert(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    a_new: float,
    b_new: float,
) -> npt.NDArray[np.float32 | np.float64]:
    """Compute knots to insert at the restriction boundaries.

    For each boundary, inserts enough copies to reach multiplicity ``degree + 1``.
    Skips insertion when the boundary coincides with an already-open domain endpoint.

    Args:
        knots: Knot vector (must be non-periodic/open-compatible).
        degree: Polynomial degree.
        tol: Knot comparison tolerance.
        a_new: Left bound of the restricted domain.
        b_new: Right bound of the restricted domain.

    Returns:
        npt.NDArray: 1D array of knot values to insert (may be empty).

    Raises:
        ValueError: If the bounds match the full domain and the direction is
            already open (no-op).
    """
    p = degree
    a = float(knots[p])
    b = float(knots[-p - 1])

    left_at_domain = np.isclose(a_new, a, atol=tol)
    right_at_domain = np.isclose(b_new, b, atol=tol)
    left_open = bool(np.isclose(knots[0], knots[p], atol=tol))
    right_open = bool(np.isclose(knots[-p - 1], knots[-1], atol=tol))

    if left_at_domain and right_at_domain and left_open and right_open:
        raise ValueError("Bounds match the full domain and the direction is already open.")

    knots_list: list[float] = []

    if not (left_at_domain and left_open):
        m_left = int(np.sum(np.isclose(knots, a_new, atol=tol)))
        deficit = p + 1 - m_left
        if deficit > 0:
            knots_list.extend([a_new] * deficit)

    if not (right_at_domain and right_open):
        m_right = int(np.sum(np.isclose(knots, b_new, atol=tol)))
        deficit = p + 1 - m_right
        if deficit > 0:
            knots_list.extend([b_new] * deficit)

    return np.array(knots_list, dtype=knots.dtype)


def _restrict_bspline_1d_impl(  # noqa: PLR0913
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    ctrl_2d: npt.NDArray[np.float32 | np.float64],
    periodic: bool,
    tol: float,
    bounds: tuple[float, float],
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
    """Restrict a 1D B-spline to a sub-interval of its parametric domain.

    Inserts knots at the boundaries until each has multiplicity ``degree + 1``,
    then extracts the knot sub-vector and control points corresponding to the
    restricted interval.  Skips insertion when a bound coincides with an
    already-open domain endpoint.

    For periodic splines, the direction is first converted to open form via
    :func:`_to_open_bspline_1d_impl`.

    Args:
        knots: Knot vector of shape ``(len(knots),)``.
        degree: Polynomial degree.
        ctrl_2d: Control point matrix of shape ``(n, rank)``.
        periodic: Whether the spline is periodic.
        tol: Knot comparison tolerance.
        bounds: ``(a_new, b_new)`` — left and right bounds of the restricted domain.

    Returns:
        tuple[npt.NDArray, npt.NDArray]: ``(restricted_knots, restricted_ctrl)``
        — the clamped knot vector on ``[a_new, b_new]`` and the corresponding
        control points.

    Raises:
        ValueError: If ``a_new >= b_new``.
        ValueError: If ``a_new`` or ``b_new`` lies outside the domain.
        ValueError: If the bounds match the full domain and the direction is
            already open (no-op).
    """
    p = degree
    a_new, b_new = _validate_restrict_bounds(knots, p, tol, bounds[0], bounds[1])

    # For periodic splines, convert to open form first.
    if periodic:
        knots, ctrl_2d = _to_open_bspline_1d_impl(knots, p, ctrl_2d, periodic, tol)

    # Compute and insert boundary knots.
    knots_to_insert = _compute_boundary_knots_to_insert(knots, p, tol, a_new, b_new)

    refined_knots: npt.NDArray[np.float32 | np.float64]
    refined_ctrl: npt.NDArray[np.float32 | np.float64]
    if knots_to_insert.size > 0:
        refined_knots, refined_ctrl = _insert_knots_bspline_1d_impl(
            knots, p, ctrl_2d, knots_to_insert, tol
        )
    else:
        refined_knots, refined_ctrl = knots, ctrl_2d

    # Extract the sub-region [a_new, b_new].
    # After insertion, a_new has multiplicity p+1 starting at index i_start,
    # and b_new has multiplicity p+1 ending at index i_end.
    i_start = int(np.searchsorted(refined_knots, a_new - tol))
    i_end = int(np.searchsorted(refined_knots, b_new + tol)) - 1

    restricted_knots = refined_knots[i_start : i_end + 1].copy()
    restricted_ctrl = refined_ctrl[i_start : i_end - p].copy()

    return restricted_knots, restricted_ctrl


def _restrict_bspline_impl(
    bspline: Bspline,
    bounds_per_dim: list[tuple[float, float] | None],
) -> Bspline:
    """Restrict a B-spline to a sub-region of its parametric domain.

    Applies :func:`_restrict_bspline_1d_impl` per parametric direction using the
    standard moveaxis pattern. Directions with ``None`` bounds are left unchanged.

    Args:
        bspline: Input B-spline.
        bounds_per_dim: Per-direction bounds as ``(a_new, b_new)`` or ``None``
            to skip. Must have length ``dim``.

    Returns:
        Bspline: New B-spline restricted to the specified sub-domain.

    Raises:
        ValueError: If every direction is a no-op (bounds match full domain
            with already-open knots, or ``None``).
    """
    from . import (  # noqa: PLC0415
        Bspline,
        BsplineSpace,
        BsplineSpace1D,
    )

    dim = bspline.dim
    ctrl = bspline.control_points

    any_restricted = False
    new_spaces_1d: list[BsplineSpace1D] = []

    for i in range(dim):
        space_1d = bspline.space.spaces[i]
        bounds = bounds_per_dim[i]

        if bounds is None:
            new_spaces_1d.append(space_1d)
            continue

        # Move dimension i to the 0th axis and flatten remaining axes.
        moved_ctrl = np.moveaxis(ctrl, i, 0)
        orig_shape = moved_ctrl.shape
        pts_2d: npt.NDArray[np.floating[Any]] = moved_ctrl.reshape(orig_shape[0], -1)
        pts_2d = np.ascontiguousarray(pts_2d)

        restricted_knots, restricted_pts_2d = _restrict_bspline_1d_impl(
            space_1d.knots,
            space_1d.degree,
            pts_2d,
            space_1d.periodic,
            float(space_1d.tolerance),
            bounds,
        )

        any_restricted = True

        # Restore multi-dimensional shape.
        new_shape = (restricted_pts_2d.shape[0], *orig_shape[1:])
        new_moved_ctrl = restricted_pts_2d.reshape(new_shape)
        ctrl = np.moveaxis(new_moved_ctrl, 0, i)

        new_spaces_1d.append(
            BsplineSpace1D(restricted_knots, space_1d.degree, periodic=False, snap_knots=False)
        )

    if not any_restricted:
        raise ValueError(
            "At least one direction must have non-None bounds that restrict the domain."
        )

    new_space = BsplineSpace(new_spaces_1d)
    return Bspline(new_space, ctrl, is_rational=bspline.is_rational)
