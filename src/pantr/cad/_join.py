"""Join two B-spline patches along a parametric axis.

Provides :func:`join` for C0-concatenation of B-splines.
"""

from __future__ import annotations

import numpy as np
from numpy import typing as npt

from ..bspline import Bspline, BsplineSpace, BsplineSpace1D
from ._compat import _remap_bspline, make_compat
from ._validation import _promote_to_rational


def _prepare_for_join(
    bspline1: Bspline,
    bspline2: Bspline,
    axis: int,
) -> tuple[Bspline, Bspline, bool]:
    """Prepare two B-splines for joining along an axis.

    Makes them compatible on non-join axes, promotes rational if mixed,
    clamps if needed, elevates to max degree on the join axis, and
    remaps b2 so it starts where b1 ends.

    Args:
        bspline1: First B-spline.
        bspline2: Second B-spline.
        axis: Join axis.

    Returns:
        tuple: ``(b1, b2, is_rational)`` ready for concatenation.
    """
    dim = bspline1.dim

    # Make compatible on non-join axes
    non_join = [i for i in range(dim) if i != axis]
    if non_join:
        b1, b2 = make_compat(bspline1, bspline2, axes=non_join)
    else:
        b1, b2 = bspline1, bspline2

    # Promote to rational if mixed
    is_rational = b1.is_rational or b2.is_rational
    if is_rational:
        b1 = _promote_to_rational(b1)
        b2 = _promote_to_rational(b2)

    # Clamp if needed
    if any(s.periodic or not s.has_open_knots() for s in b1.space.spaces):
        b1 = b1.to_open_bspline()
    if any(s.periodic or not s.has_open_knots() for s in b2.space.spaces):
        b2 = b2.to_open_bspline()

    # Elevate to max degree on the join axis
    p1 = b1.degree[axis]
    p2 = b2.degree[axis]
    p = max(p1, p2)
    if p > p1:
        inc = [0] * dim
        inc[axis] = p - p1
        b1 = b1.elevate_degree(inc)
    if p > p2:
        inc = [0] * dim
        inc[axis] = p - p2
        b2 = b2.elevate_degree(inc)

    # Remap b2 so it starts where b1 ends
    u1_end = float(b1.space.spaces[axis].domain[1])
    u2_start = float(b2.space.spaces[axis].domain[0])
    u2_end = float(b2.space.spaces[axis].domain[1])
    new_end = u1_end + (u2_end - u2_start)
    b2 = _remap_bspline(b2, axis, (u1_end, new_end))

    return b1, b2, is_rational


def _concatenate_along_axis(
    b1: Bspline,
    b2: Bspline,
    axis: int,
    is_rational: bool,
) -> Bspline:
    """Concatenate two prepared B-splines along an axis.

    Averages the shared boundary control points, merges knot vectors,
    and builds the result.

    Args:
        b1: First B-spline (left side).
        b2: Second B-spline (right side, already remapped).
        axis: Join axis.
        is_rational: Whether the result is rational.

    Returns:
        Bspline: The concatenated B-spline.
    """
    dim = b1.dim
    p = b1.degree[axis]
    cp1 = b1.control_points
    cp2 = b2.control_points

    # Slice indices for the join axis (+ rank dimension)
    ndim = dim + 1
    idx_left: list[int | slice | None] = [slice(None)] * ndim
    idx_right: list[int | slice | None] = [slice(None)] * ndim
    idx_b1_last: list[int | slice | None] = [slice(None)] * ndim
    idx_b2_first: list[int | slice | None] = [slice(None)] * ndim

    idx_left[axis] = slice(0, -1)
    idx_right[axis] = slice(1, None)
    idx_b1_last[axis] = -1
    idx_b2_first[axis] = 0

    cp_left = cp1[tuple(idx_left)]
    cp_right = cp2[tuple(idx_right)]
    cp_mid = (cp1[tuple(idx_b1_last)] + cp2[tuple(idx_b2_first)]) / 2.0

    idx_expand: list[int | slice | None] = [slice(None)] * ndim
    idx_expand[axis] = np.newaxis
    cp_mid_expanded = cp_mid[tuple(idx_expand)]

    new_cp = np.concatenate([cp_left, cp_mid_expanded, cp_right], axis=axis)

    # Merge knot vectors
    u_junction = float(b1.space.spaces[axis].domain[1])
    kl = b1.space.spaces[axis].knots[: -p - 1]
    kr = b2.space.spaces[axis].knots[p + 1 :]
    kc = np.full(p, u_junction)
    new_knots = np.concatenate([kl, kc, kr])

    spaces = list(b1.space.spaces)
    spaces[axis] = BsplineSpace1D(new_knots, degree=p)
    new_space = BsplineSpace(spaces)

    result = Bspline(new_space, new_cp, is_rational=is_rational)

    # Try to remove junction knots for smoother join
    if p > 1:
        result = _try_remove_junction_knots(result, axis, u_junction, p)

    return result


def join(bspline1: Bspline, bspline2: Bspline, axis: int) -> Bspline:
    """Join two B-splines along a parametric axis with C0 continuity.

    The two inputs are made compatible on all non-join axes (degree and
    knots), elevated to the same degree on the join axis, and then
    concatenated.  The shared boundary control points are averaged.

    Optionally, knots are removed at the junction to recover higher
    smoothness when the geometry permits it.

    Args:
        bspline1: First B-spline (left side of the join).
        bspline2: Second B-spline (right side of the join).
        axis: Parametric axis along which to join (0-indexed).

    Returns:
        Bspline: A single B-spline spanning both inputs.

    Raises:
        ValueError: If the inputs have different parametric dimensions.
        ValueError: If *axis* is out of range.
    """
    dim = bspline1.dim
    if dim != bspline2.dim:
        raise ValueError(
            f"Both B-splines must have the same dim, got {bspline1.dim} and {bspline2.dim}."
        )
    if axis < 0 or axis >= dim:
        raise ValueError(f"axis must be in [0, {dim}), got {axis}.")

    b1, b2, is_rational = _prepare_for_join(bspline1, bspline2, axis)
    return _concatenate_along_axis(b1, b2, axis, is_rational)


def _try_remove_junction_knots(
    bspline: Bspline,
    axis: int,
    u_junction: float,
    p: int,
) -> Bspline:
    """Try removing junction knots to increase smoothness.

    Attempts to remove up to ``p - 1`` copies of the junction knot.

    Args:
        bspline: The joined B-spline.
        axis: Join axis.
        u_junction: Knot value at the junction.
        p: Degree on the join axis.

    Returns:
        Bspline: B-spline with some junction knots removed, or the
        original if removal was not possible.
    """
    dim = bspline.dim
    if dim == 1:
        return bspline.remove_knots(u_junction, num=p - 1)

    per_dim: list[npt.ArrayLike | None] = [None] * dim
    per_dim[axis] = np.array([u_junction])
    return bspline.remove_knots(per_dim, num=p - 1)
