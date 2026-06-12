"""Layer 2 implementation for B-spline splitting.

This module provides the algorithm for splitting a B-spline into two at a
given parameter value. The core logic inserts knots at the split point until
the multiplicity reaches ``degree + 1``, then extracts the left and right
sub-splines.
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


def _split_bspline_1d_impl(  # noqa: PLR0913
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    ctrl_2d: npt.NDArray[np.float32 | np.float64],
    periodic: bool,
    tol: float,
    value: float,
) -> tuple[
    tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]],
    tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]],
]:
    """Split a 1D B-spline at a parameter value.

    Inserts knots at ``value`` until multiplicity ``degree + 1``, then
    extracts the left ``[domain_start, value]`` and right
    ``[value, domain_end]`` sub-splines.

    For periodic splines, the direction is first converted to open form via
    :func:`_to_open_bspline_1d_impl`.

    Args:
        knots: Knot vector of shape ``(len(knots),)``.
        degree: Polynomial degree.
        ctrl_2d: Control point matrix of shape ``(n, rank)``.
        periodic: Whether the spline is periodic.
        tol: Knot comparison tolerance.
        value: Parameter value at which to split (must lie strictly inside
            the domain).

    Returns:
        tuple: ``((left_knots, left_ctrl), (right_knots, right_ctrl))`` — the
        knot vectors and control points for the left and right sub-splines.
    """
    p = degree

    # For periodic splines, convert to open form first.
    if periodic:
        knots, ctrl_2d = _to_open_bspline_1d_impl(knots, p, ctrl_2d, periodic, tol)

    # Compute current multiplicity of the split value.
    m_current = int(np.sum(np.isclose(knots, value, atol=tol)))
    deficit = p + 1 - m_current
    if deficit > 0:
        knots_to_insert = np.full(deficit, value, dtype=knots.dtype)
        knots, ctrl_2d = _insert_knots_bspline_1d_impl(knots, p, ctrl_2d, knots_to_insert, tol)

    # Find the split index: value now has multiplicity p+1.
    i_split = int(np.searchsorted(knots, value - tol))

    # Left sub-spline: [domain_start, value].
    left_knots = knots[: i_split + p + 1].copy()
    left_ctrl = ctrl_2d[:i_split].copy()

    # Right sub-spline: [value, domain_end].
    right_knots = knots[i_split:].copy()
    right_ctrl = ctrl_2d[i_split:].copy()

    return (left_knots, left_ctrl), (right_knots, right_ctrl)


def _split_bspline_impl(
    bspline: Bspline,
    direction: int,
    value: float,
) -> tuple[Bspline, Bspline]:
    """Split a B-spline into two at a parameter value in one direction.

    Applies :func:`_split_bspline_1d_impl` along the specified parametric
    direction via the shared flatten/unflatten helpers.

    Args:
        bspline: Input B-spline.
        direction: Parametric direction along which to split.
        value: Parameter value at which to split.

    Returns:
        tuple[Bspline, Bspline]: ``(left, right)`` — the left and right
        sub-splines.
    """
    from . import Bspline as BsplineCls  # noqa: PLC0415
    from . import BsplineSpace, BsplineSpace1D  # noqa: PLC0415

    dim = bspline.dim
    ctrl = bspline.control_points

    pts_2d, trailing_shape = _flatten_along_axis(ctrl, direction)

    space_1d = bspline.space.spaces[direction]
    (left_knots, left_pts_2d), (right_knots, right_pts_2d) = _split_bspline_1d_impl(
        space_1d.knots,
        space_1d.degree,
        pts_2d,
        space_1d.periodic,
        float(space_1d.tolerance),
        value,
    )

    left_ctrl = _unflatten_along_axis(left_pts_2d, trailing_shape, direction)
    right_ctrl = _unflatten_along_axis(right_pts_2d, trailing_shape, direction)

    # Construct the spaces: replace the split direction, keep others.
    left_spaces: list[BsplineSpace1D] = []
    right_spaces: list[BsplineSpace1D] = []
    for i in range(dim):
        if i == direction:
            left_spaces.append(
                BsplineSpace1D(left_knots, space_1d.degree, periodic=False, snap_knots=False)
            )
            right_spaces.append(
                BsplineSpace1D(right_knots, space_1d.degree, periodic=False, snap_knots=False)
            )
        else:
            left_spaces.append(bspline.space.spaces[i])
            right_spaces.append(bspline.space.spaces[i])

    left_bspline = BsplineCls(BsplineSpace(left_spaces), left_ctrl, is_rational=bspline.is_rational)
    right_bspline = BsplineCls(
        BsplineSpace(right_spaces), right_ctrl, is_rational=bspline.is_rational
    )

    return left_bspline, right_bspline
