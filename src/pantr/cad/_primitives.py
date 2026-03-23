"""Constructive geometry primitives: line, bilinear, and trilinear.

Provides functions to create basic B-spline objects from geometric
descriptions (points, corners).  All primitives produce rank-3 (3D)
output so they can be freely composed with higher-level operations
such as ``extrude`` and ``revolve``.
"""

from __future__ import annotations

import numpy as np
from numpy import typing as npt

from ..bspline import Bspline, BsplineSpace, BsplineSpace1D
from ._validation import _PHYSICAL_DIM, _pad_to_3d

_BILINEAR_SHAPE = (2, 2)
_TRILINEAR_SHAPE = (2, 2, 2)


def _linear_space_1d(dtype: type[np.float64] = np.float64) -> BsplineSpace1D:
    """Create a degree-1 B-spline space on [0, 1].

    Args:
        dtype: Floating-point dtype for the knot vector.

    Returns:
        BsplineSpace1D: A clamped degree-1 space with knots ``[0, 0, 1, 1]``.
    """
    knots = np.array([0.0, 0.0, 1.0, 1.0], dtype=dtype)
    return BsplineSpace1D(knots, degree=1)


def line(
    p0: npt.ArrayLike = (0.0, 0.0, 0.0),
    p1: npt.ArrayLike = (1.0, 0.0, 0.0),
) -> Bspline:
    """Construct a straight-line B-spline curve between two points.

    Creates a degree-1 (linear), non-rational B-spline curve with two
    control points.  Input points shorter than 3 elements are zero-padded
    to 3D.

    Args:
        p0: Start point.  Defaults to the origin.
        p1: End point.  Defaults to ``(1, 0, 0)``.

    Returns:
        Bspline: A 1D, degree-1, rank-3, non-rational B-spline curve.

    Example:
        >>> crv = line([0, 0], [1, 1])
        >>> crv.degree
        (1,)
        >>> crv.rank
        3
    """
    pt0 = _pad_to_3d(p0)
    pt1 = _pad_to_3d(p1)
    control_points = np.stack([pt0, pt1])  # shape (2, 3)
    space = BsplineSpace([_linear_space_1d()])
    return Bspline(space, control_points)


def bilinear(corners: npt.ArrayLike | None = None) -> Bspline:
    """Construct a bilinear B-spline surface from four corner points.

    Creates a degree (1, 1), non-rational B-spline surface with 2 x 2
    control points.

    The corner ordering follows the tensor-product convention::

        corners[0, 1]       corners[1, 1]
        o------------------o
        |  v               |
        |  ^               |
        |  |               |
        |  +-------> u     |
        o------------------o
        corners[0, 0]       corners[1, 0]

    Args:
        corners: Array of shape ``(2, 2, rank)`` with ``rank <= 3``.
            Coordinates shorter than 3D are zero-padded.
            If ``None``, defaults to the unit square
            ``[-0.5, 0.5]^2 x {0}`` in the *xy*-plane.

    Returns:
        Bspline: A 2D, degree-(1, 1), rank-3, non-rational B-spline surface.

    Raises:
        ValueError: If *corners* does not have shape ``(2, 2, rank)``
            with ``1 <= rank <= 3``.

    Example:
        >>> srf = bilinear()
        >>> srf.degree
        (1, 1)
        >>> srf.dim
        2
    """
    ndim_expected = len(_BILINEAR_SHAPE) + 1
    if corners is None:
        cp = np.zeros((*_BILINEAR_SHAPE, _PHYSICAL_DIM), dtype=np.float64)
        cp[0, 0] = [-0.5, -0.5, 0.0]
        cp[1, 0] = [+0.5, -0.5, 0.0]
        cp[0, 1] = [-0.5, +0.5, 0.0]
        cp[1, 1] = [+0.5, +0.5, 0.0]
    else:
        arr = np.asarray(corners, dtype=np.float64)
        if arr.ndim != ndim_expected or arr.shape[:-1] != _BILINEAR_SHAPE:
            raise ValueError(f"corners must have shape (2, 2, rank), got {arr.shape}.")
        rank = arr.shape[-1]
        if rank > _PHYSICAL_DIM:
            raise ValueError(f"corners rank must be at most {_PHYSICAL_DIM}, got {rank}.")
        cp = np.zeros((*_BILINEAR_SHAPE, _PHYSICAL_DIM), dtype=np.float64)
        cp[..., :rank] = arr

    sp = _linear_space_1d()
    space = BsplineSpace([sp, BsplineSpace1D(sp.knots.copy(), degree=1)])
    return Bspline(space, cp)


def trilinear(corners: npt.ArrayLike | None = None) -> Bspline:
    """Construct a trilinear B-spline volume from eight corner points.

    Creates a degree (1, 1, 1), non-rational B-spline volume with
    2 x 2 x 2 control points.

    The corner ordering follows the tensor-product convention::

           corners[0,1,1]       corners[1,1,1]
           o--------------------o
          /|                   /|
         / |                  / |            w
        o--------------------o  |            ^  v
        |  | corners[0,0,1]  |  | c[1,0,1]  | /
        |  |                 |  |            |/
        |  o-----------------|--o            +------> u
        | / corners[0,1,0]   | / corners[1,1,0]
        |/                   |/
        o--------------------o
        corners[0,0,0]       corners[1,0,0]

    Args:
        corners: Array of shape ``(2, 2, 2, rank)`` with ``rank <= 3``.
            Coordinates shorter than 3D are zero-padded.
            If ``None``, defaults to the unit cube
            ``[-0.5, 0.5]^3`` centered at the origin.

    Returns:
        Bspline: A 3D, degree-(1, 1, 1), rank-3, non-rational B-spline volume.

    Raises:
        ValueError: If *corners* does not have shape ``(2, 2, 2, rank)``
            with ``1 <= rank <= 3``.

    Example:
        >>> vol = trilinear()
        >>> vol.degree
        (1, 1, 1)
        >>> vol.dim
        3
    """
    ndim_expected = len(_TRILINEAR_SHAPE) + 1
    if corners is None:
        cp = np.zeros((*_TRILINEAR_SHAPE, _PHYSICAL_DIM), dtype=np.float64)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    cp[i, j, k] = [i - 0.5, j - 0.5, k - 0.5]
    else:
        arr = np.asarray(corners, dtype=np.float64)
        if arr.ndim != ndim_expected or arr.shape[:-1] != _TRILINEAR_SHAPE:
            raise ValueError(f"corners must have shape (2, 2, 2, rank), got {arr.shape}.")
        rank = arr.shape[-1]
        if rank > _PHYSICAL_DIM:
            raise ValueError(f"corners rank must be at most {_PHYSICAL_DIM}, got {rank}.")
        cp = np.zeros((*_TRILINEAR_SHAPE, _PHYSICAL_DIM), dtype=np.float64)
        cp[..., :rank] = arr

    sp0 = _linear_space_1d()
    sp1 = BsplineSpace1D(sp0.knots.copy(), degree=1)
    sp2 = BsplineSpace1D(sp0.knots.copy(), degree=1)
    space = BsplineSpace([sp0, sp1, sp2])
    return Bspline(space, cp)
