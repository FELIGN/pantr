"""Shared validation and conversion helpers for CAD functions.

Provides utility functions used across the ``cad`` subpackage for
normalizing point arrays and promoting B-splines between rational and
non-rational representations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._control_points_utils import _append_unit_weight_column

if TYPE_CHECKING:
    from numpy import typing as npt

    from ..bspline import Bspline

_PHYSICAL_DIM = 3
"""int: Default physical dimension for all CAD primitives."""


def _pad_to_3d(point: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert a point array to a 3D float64 vector, zero-padding if shorter.

    Args:
        point: Array-like with 1 to 3 elements.

    Returns:
        npt.NDArray[np.float64]: A 1D array of shape ``(3,)``.

    Raises:
        ValueError: If the input has more than 3 elements.
    """
    p = np.asarray(point, dtype=np.float64).ravel()
    if p.size > _PHYSICAL_DIM:
        raise ValueError(f"Point must have at most {_PHYSICAL_DIM} coordinates, got {p.size}.")
    out = np.zeros(_PHYSICAL_DIM, dtype=np.float64)
    out[: p.size] = p
    return out


def _promote_to_rational(bspline: Bspline) -> Bspline:
    """Add unit weights to a non-rational B-spline, returning a rational one.

    If the B-spline is already rational, it is returned unchanged.

    Args:
        bspline: A B-spline curve, surface, or volume.

    Returns:
        Bspline: A rational B-spline with unit weights appended.
    """
    if bspline.is_rational:
        return bspline

    from ..bspline import Bspline as BsplineCls  # noqa: PLC0415

    new_cp = _append_unit_weight_column(bspline.control_points)
    return BsplineCls(bspline.space, new_cp, is_rational=True)
