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

from .._array_utils import _flatten_along_axis
from ._bezier_backend import slice_kernel, slice_nd_kernel, slice_point_kernel

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
    if bezier.dim == 1:
        # The result is a point rather than a Bézier, so this is a different accessor
        # rather than a branch inside one: C++ cannot return either type from one
        # function, and the projection stays here, above the branch, exactly where the
        # oracle put it.
        raw = slice_point_kernel()(bezier, value)
        if bezier.is_rational:
            weight = raw[-1]
            return cast(npt.NDArray[np.float32 | np.float64], raw[:-1] / weight)
        return raw

    return slice_nd_kernel()(bezier, axis, value)


def _slice_point_python(bezier: Bezier, value: float) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate a one-dimensional Bézier at one parameter: the oracle for the port.

    Args:
        bezier (~pantr.bezier.Bezier): A one-dimensional Bézier.
        value (float): Parameter to evaluate at.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The raw homogeneous components, weight
        column included.

    Note:
        No input validation is performed here; Layer 2 did it above the branch.
    """
    pts_2d, _ = _flatten_along_axis(bezier.control_points, 0)
    raw = np.empty(pts_2d.shape[1], dtype=pts_2d.dtype)
    slice_kernel()(pts_2d, value, raw)
    return raw


def _slice_nd_python(bezier: Bezier, axis: int, value: float) -> Bezier:
    """Slice a Bézier of dimension at least two: the oracle for the port.

    Args:
        bezier (~pantr.bezier.Bezier): The Bézier to slice.
        axis (int): Parametric direction to fix.
        value (float): Parameter to fix it at.

    Returns:
        ~pantr.bezier.Bezier: The sliced Bézier, of one dimension less.

    Note:
        No input validation is performed here; Layer 2 did it above the branch.
    """
    from . import Bezier as BezierCls  # noqa: PLC0415

    pts_2d, trailing_shape = _flatten_along_axis(bezier.control_points, axis)
    result_1d = np.empty(pts_2d.shape[1], dtype=pts_2d.dtype)
    slice_kernel()(pts_2d, value, result_1d)

    # Restore shape: the sliced axis is removed.
    return BezierCls(result_1d.reshape(trailing_shape), is_rational=bezier.is_rational)
