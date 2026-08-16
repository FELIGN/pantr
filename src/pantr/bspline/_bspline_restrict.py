"""Layer 2 implementation for B-spline domain restriction.

This module provides the algorithm for extracting a sub-region of the parametric
domain of a B-spline. The core logic inserts knots at the new boundaries until they
reach multiplicity ``degree + 1``, then extracts the relevant knot sub-vector and
control points. An optimization skips insertion when a bound coincides with an
already-open domain endpoint.

Tolerance policy
----------------

Restriction introduces **no tolerance of its own**, and that is the policy rather
than an omission. Every comparison it makes is between two parametric coordinates
of one knot vector -- a requested bound against a domain endpoint, against an
existing knot, against the refined vector -- so every one of them is the question
:func:`~pantr.bspline._bspline_knots._knot_tolerance` already answers: *are these
two values the same knot?* The single ``tol`` threaded through this module is
:attr:`~pantr.bspline.BsplineSpace1D.tolerance`, the absolute parametric length
``8 * eps * max(span, |knots[0]|, |knots[-1]|)`` that the space derived once at
construction. Introducing a second number here would be inventing a second notion
of knot identity for the same vector.

Three properties of that inheritance are load-bearing and are worth stating,
because each is what makes one number enough:

**The scale does not move between the stages.** ``tol`` is derived from the input
knot vector, then applied to the *refined* one after boundary insertion. Knot
insertion never changes the first or last knot, so ``_knot_scale`` is identical for
both and the tolerance means the same thing on either side. The one stage that does
change the endpoints is the periodic-to-open conversion, which clamps the vector to
its own domain and can only *shrink* the scale (by the ratio of the padded extent to
the domain, at most a small factor). The inherited tolerance is then conservative
rather than tight, which is the safe direction for a merge test.

**The extraction offsets widen, and that direction is deliberate.** The sub-vector
is cut with ``searchsorted(refined_knots, a_new - tol)`` and
``searchsorted(refined_knots, b_new + tol) - 1``: both move the cut *away* from the
interval, so a boundary knot can never be dropped by a coordinate landing a rounding
short of its own value. This is what the previous absolute tolerance got wrong. At a
domain based at ``1e6`` one ulp of the bound is about ``1.2e-10``, so the old fixed
``1e-12`` was absorbed entirely by ``b_new + tol == b_new`` and the cut fell before
the first copy of ``b_new``, silently truncating the last ``degree + 1`` knots. A
tolerance of ``8 * eps * scale`` is at least four ulp of any coordinate in the
vector, so the offset always changes the value it is added to and the cut always
lands past the whole boundary group.

**Widening cannot over-collect.** :class:`~pantr.bspline.BsplineSpace1D` snaps its
knots by the same rule, so two *distinct* stored knots differ by more than ``tol``
and a cut widened by ``tol`` cannot reach the next one. The one configuration where
that is not enough is a bound falling between two distinct knots less than
``2 * tol`` apart, which requires a mesh already at the format's resolution floor
for its magnitude -- the regime
:func:`~pantr.bspline._bspline_knots._check_snapping_kept_an_interval` reports on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .._array_utils import _flatten_along_axis, _unflatten_along_axis
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

    A bound within ``tol`` of a domain endpoint *is* that endpoint: it is accepted
    rather than rejected as out of range, and then snapped onto the stored value so
    the rest of the algorithm compares bitwise.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): Knot vector.
        degree (int): Polynomial degree.
        tol (float): Absolute parametric tolerance, from
            :attr:`~pantr.bspline.BsplineSpace1D.tolerance`; see the module
            docstring for why restriction adds nothing to it.
        a_new (float): Requested left bound.
        b_new (float): Requested right bound.

    Returns:
        tuple[float, float]: Snapped ``(a_new, b_new)`` bounds.

    Raises:
        ValueError: If ``a_new >= b_new`` or bounds lie outside the domain.
    """
    a = float(knots[degree])
    b = float(knots[-degree - 1])

    if a_new >= b_new:
        raise ValueError(f"Lower bound ({a_new}) must be strictly less than upper bound ({b_new}).")

    if a_new < a and abs(a_new - a) > tol:
        raise ValueError(f"Lower bound ({a_new}) is below the domain start ({a}).")
    if b_new > b and abs(b_new - b) > tol:
        raise ValueError(f"Upper bound ({b_new}) is above the domain end ({b}).")

    # Snap bounds to domain endpoints if within tolerance.
    if abs(a_new - a) <= tol:
        a_new = a
    if abs(b_new - b) <= tol:
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
        knots (npt.NDArray[np.float32 | np.float64]): Knot vector (must be
            non-periodic/open-compatible).
        degree (int): Polynomial degree.
        tol (float): Absolute parametric tolerance, from
            :attr:`~pantr.bspline.BsplineSpace1D.tolerance`. It decides which
            existing knots already *are* the boundary, and so how many copies the
            boundary is short of multiplicity ``degree + 1``.
        a_new (float): Left bound of the restricted domain.
        b_new (float): Right bound of the restricted domain.

    Returns:
        npt.NDArray: 1D array of knot values to insert (may be empty).

    Raises:
        ValueError: If the bounds match the full domain and the direction is
            already open (no-op).
    """
    p = degree
    a = float(knots[p])
    b = float(knots[-p - 1])

    left_at_domain = abs(a_new - a) <= tol
    right_at_domain = abs(b_new - b) <= tol
    left_open = abs(float(knots[0]) - float(knots[p])) <= tol
    right_open = abs(float(knots[-p - 1]) - float(knots[-1])) <= tol

    if left_at_domain and right_at_domain and left_open and right_open:
        raise ValueError("Bounds match the full domain and the direction is already open.")

    knots_list: list[float] = []

    if not (left_at_domain and left_open):
        m_left = int(np.sum(np.abs(knots - a_new) <= tol))
        deficit = p + 1 - m_left
        if deficit > 0:
            knots_list.extend([a_new] * deficit)

    if not (right_at_domain and right_open):
        m_right = int(np.sum(np.abs(knots - b_new) <= tol))
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
        knots (npt.NDArray[np.float32 | np.float64]): Knot vector of shape
            ``(len(knots),)``.
        degree (int): Polynomial degree.
        ctrl_2d (npt.NDArray[np.float32 | np.float64]): Control point matrix of
            shape ``(n, rank)``.
        periodic (bool): Whether the spline is periodic.
        tol (float): Absolute parametric tolerance, from
            :attr:`~pantr.bspline.BsplineSpace1D.tolerance`, serving every
            comparison here; see the module docstring for the policy.
        bounds (tuple[float, float]): ``(a_new, b_new)``, the left and right bounds
            of the restricted domain.

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

    # Extract the sub-region [a_new, b_new]. After insertion, a_new has multiplicity
    # p+1 starting at index i_start, and b_new has multiplicity p+1 ending at i_end.
    # Both offsets widen the cut away from the interval, so a boundary knot cannot be
    # dropped by a coordinate landing a rounding short of its own value, and `tol` is
    # at least four ulp of any coordinate here so the offset is never absorbed; see
    # the module docstring.
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
    shared flatten/unflatten helpers. Directions with ``None`` bounds are left unchanged.

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

        pts_2d, trailing_shape = _flatten_along_axis(ctrl, i)

        restricted_knots, restricted_pts_2d = _restrict_bspline_1d_impl(
            space_1d.knots,
            space_1d.degree,
            pts_2d,
            space_1d.periodic,
            float(space_1d.tolerance),
            bounds,
        )

        any_restricted = True

        ctrl = _unflatten_along_axis(restricted_pts_2d, trailing_shape, i)

        new_spaces_1d.append(
            BsplineSpace1D(restricted_knots, space_1d.degree, periodic=False, snap_knots=False)
        )

    if not any_restricted:
        raise ValueError(
            "At least one direction must have non-None bounds that restrict the domain."
        )

    new_space = BsplineSpace(new_spaces_1d)
    return Bspline(new_space, ctrl, is_rational=bspline.is_rational)
