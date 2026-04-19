"""Bézier slicing (dimension reduction by fixing one parametric direction).

This module provides :func:`_slice_bezier`, which fixes one parametric
direction of a Bézier at a given value and returns a Bézier with one
fewer dimension.  For a 1D Bézier the result is a plain NumPy array
(the evaluated point).

The core algorithm is de Casteljau applied to the control points along
the sliced axis.  At the boundary values ``0`` and ``1`` the first or
last control point is returned directly in O(1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from ._bezier_core import _slice_bezier_1d_core

if TYPE_CHECKING:
    from . import Bezier


def _slice_bezier(
    bezier: Bezier,
    axis: int,
    value: float,
) -> Bezier | npt.NDArray[np.float32 | np.float64]:
    """Slice a Bézier by fixing one parametric direction at a given value.

    Reduces the parametric dimension by one.  For a 1D Bézier, returns
    the evaluated point as a NumPy array.  For higher dimensions, returns
    a new :class:`~pantr.bezier.Bezier`.

    Args:
        bezier (~pantr.bezier.Bezier): The Bézier to slice.
        axis (int): Parametric direction to fix (0-indexed, must be in
            ``[0, dim)``).
        value (float): Parameter value at which to slice (must be in
            ``[0, 1]``).

    Returns:
        ~pantr.bezier.Bezier | npt.NDArray[np.float32 | np.float64]:
        A Bézier with ``dim - 1`` dimensions (when ``dim >= 2``),
        or a NumPy array of shape ``(rank,)`` (when ``dim == 1``).
        Rational Béziers preserve the rational structure when ``dim >= 2``;
        for ``dim == 1`` the result is projected to physical coordinates.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :meth:`~pantr.bezier.Bezier.slice` instead.
    """
    from . import Bezier as BezierCls  # noqa: PLC0415

    ctrl = bezier.control_points

    # Move target axis to position 0, flatten the rest into a single axis.
    moved = np.moveaxis(ctrl, axis, 0)
    orig_shape = moved.shape
    pts_2d = moved.reshape(orig_shape[0], -1)

    pts_2d = np.ascontiguousarray(pts_2d)

    # Apply 1D de Casteljau via Numba kernel.
    result_1d = np.empty(pts_2d.shape[1], dtype=pts_2d.dtype)
    _slice_bezier_1d_core(pts_2d, value, result_1d)

    if bezier.dim == 1:
        # Result is a point.  For rational Béziers, project to physical coords.
        if bezier.is_rational:
            weight = result_1d[-1]
            return cast(npt.NDArray[np.float32 | np.float64], result_1d[:-1] / weight)
        return result_1d

    # Restore shape: remove the sliced axis dimension.
    new_shape = orig_shape[1:]
    new_ctrl = result_1d.reshape(new_shape)

    return BezierCls(new_ctrl, is_rational=bezier.is_rational)
