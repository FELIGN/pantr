"""Bézier splitting (subdivide along one parametric direction).

This module provides :func:`_split_bezier`, which splits a Bézier into two
new Béziers at a given parameter value in one parametric direction using
the de Casteljau algorithm.

The core algorithm performs a single forward de Casteljau pass that
simultaneously produces the control points for both the left ``[0, value]``
and right ``[value, 1]`` halves, each reparametrized to ``[0, 1]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ._bezier_core import _split_bezier_1d_core

if TYPE_CHECKING:
    from . import Bezier


def _split_bezier(
    bezier: Bezier,
    direction: int,
    value: float,
) -> tuple[Bezier, Bezier]:
    """Split a Bézier into two at a parameter value in one direction.

    Uses the de Casteljau algorithm to compute the control points for the
    left ``[0, value]`` and right ``[value, 1]`` halves simultaneously,
    each reparametrized to ``[0, 1]``.

    Args:
        bezier (~pantr.bezier.Bezier): The Bézier to split.
        direction (int): Parametric direction along which to split
            (0-indexed, must be in ``[0, dim)``).
        value (float): Parameter value at which to split (must be in
            ``(0, 1)``).

    Returns:
        tuple[~pantr.bezier.Bezier, ~pantr.bezier.Bezier]: A pair
        ``(left, right)`` of Béziers, each on ``[0, 1]^dim``, representing
        the left and right halves of the original.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :meth:`~pantr.bezier.Bezier.split` instead.
    """
    from . import Bezier as BezierCls  # noqa: PLC0415

    ctrl = bezier.control_points

    # Move target axis to position 0, flatten the rest into a single axis.
    moved = np.moveaxis(ctrl, direction, 0)
    orig_shape = moved.shape
    pts_2d: npt.NDArray[np.floating[Any]] = moved.reshape(orig_shape[0], -1)

    pts_2d = np.ascontiguousarray(pts_2d)

    # Allocate outputs.
    out_left = np.empty_like(pts_2d)
    out_right = np.empty_like(pts_2d)

    _split_bezier_1d_core(pts_2d, value, out_left, out_right)

    # Restore multi-dimensional shape and move axis back.
    left_ctrl = np.moveaxis(out_left.reshape(orig_shape), 0, direction)
    right_ctrl = np.moveaxis(out_right.reshape(orig_shape), 0, direction)

    return (
        BezierCls(left_ctrl, is_rational=bezier.is_rational),
        BezierCls(right_ctrl, is_rational=bezier.is_rational),
    )
