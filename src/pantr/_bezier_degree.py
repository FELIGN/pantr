"""Bézier degree elevation.

This module provides :func:`_degree_elevate_bezier`, which raises the polynomial
degree of a Bézier in one or more parametric directions while preserving the
same geometric mapping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ._bezier_core import _degree_elevate_bezier_1d_core

if TYPE_CHECKING:
    from .bezier import Bezier


def _degree_elevate_bezier(
    bezier: Bezier,
    increments: tuple[int, ...],
) -> Bezier:
    """Degree-elevate a Bézier in one or more parametric directions.

    For each direction with a positive increment, applies the Bézier degree
    elevation kernel using the moveaxis/reshape pattern.

    Args:
        bezier (~pantr.bezier.Bezier): The Bézier to elevate.
        increments (tuple[int, ...]): Degree increment per direction. All
            values must be non-negative; at least one must be positive.

    Returns:
        ~pantr.bezier.Bezier: New Bézier with elevated degrees and updated
        control points.

    Note:
        Inputs are assumed to be validated by the caller (Layer 1).
    """
    from .bezier import Bezier as BezierCls  # noqa: PLC0415

    ctrl: npt.NDArray[np.float32 | np.float64] = bezier.control_points
    degrees = bezier.degree

    for d in range(bezier.dim):
        inc = increments[d]
        if inc == 0:
            continue

        p = degrees[d]

        # Move target direction to axis 0, flatten the rest.
        moved = np.moveaxis(ctrl, d, 0)
        orig_shape = moved.shape
        pts_2d = moved.reshape(orig_shape[0], -1)

        if not pts_2d.flags.c_contiguous:
            pts_2d = np.ascontiguousarray(pts_2d)

        new_pts_2d = _degree_elevate_bezier_1d_core(p, pts_2d, inc)

        # Restore shape and move axis back.
        new_shape = (new_pts_2d.shape[0], *orig_shape[1:])
        new_moved = new_pts_2d.reshape(new_shape)
        ctrl = np.moveaxis(new_moved, 0, d)

        # Update degrees for subsequent iterations.
        degrees = (*degrees[:d], p + inc, *degrees[d + 1 :])

    return BezierCls(ctrl, is_rational=bezier.is_rational)
