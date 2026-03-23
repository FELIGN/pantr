"""Constructive geometry primitives: line, circle, bilinear, and trilinear.

Provides functions to create basic B-spline objects from geometric
descriptions (points, corners, radii, angles).  All primitives produce
rank-3 (3D) output so they can be freely composed with higher-level
operations such as ``extrude`` and ``revolve``.
"""

from __future__ import annotations

import numpy as np
from numpy import typing as npt

from ..bspline import Bspline, BsplineSpace, BsplineSpace1D
from ..transform import AffineTransform
from ._validation import _PHYSICAL_DIM, _pad_to_3d

_DEGREE_CIRCLE = 2
_QUADRANT_BOUNDS = (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)
_BILINEAR_SHAPE = (2, 2)
_TRILINEAR_SHAPE = (2, 2, 2)


def _linear_space_1d(dtype: npt.DTypeLike = np.float64) -> BsplineSpace1D:
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


def _rotate_weighted(
    cw: npt.NDArray[np.float64],
    angle: float,
    axis: int = 2,
) -> npt.NDArray[np.float64]:
    """Rotate weighted homogeneous control points around a coordinate axis.

    For a point ``(w*x, w*y, w*z, w)``, the rotation is applied to the
    spatial part ``(w*x, w*y, w*z)`` while the weight ``w`` is preserved.

    Args:
        cw: Control points of shape ``(..., 4)`` in weighted homogeneous form.
        angle: Rotation angle in radians.
        axis: Coordinate axis (0, 1, or 2).

    Returns:
        npt.NDArray[np.float64]: Rotated control points, same shape as *cw*.
    """
    R = AffineTransform.rotation_3d(angle, axis=axis)
    out = cw.copy()
    out[..., :_PHYSICAL_DIM] = cw[..., :_PHYSICAL_DIM] @ R.matrix.T
    return out


def circle(
    radius: float = 1.0,
    center: npt.ArrayLike | None = None,
    angle: float | tuple[float, float] | None = None,
) -> Bspline:
    """Construct a NURBS circular arc or full circle.

    Creates a degree-2 rational B-spline curve in the *xy*-plane.  The
    arc is split into spans of at most 90 degrees each.  Interior knots
    have multiplicity 2 (equal to the degree), giving C0 continuity at
    arc junctions.  This is the standard exact representation of conics
    using rational quadratic B-splines.

    The number of spans depends on the sweep angle:

    - ``|sweep| <= 90``: 1 span, 3 control points
    - ``90 < |sweep| <= 180``: 2 spans, 5 control points
    - ``180 < |sweep| <= 270``: 3 spans, 7 control points
    - ``270 < |sweep| <= 360``: 4 spans, 9 control points

    Args:
        radius: Circle radius.  Defaults to 1.
        center: Center point (up to 3D, zero-padded).  If ``None``,
            the circle is centered at the origin.
        angle: Sweep specification.

            - ``None`` -- full circle (360 degrees).
            - ``float`` -- arc from angle 0 to the given value (radians).
            - ``(start, end)`` -- arc from *start* to *end* (radians).

    Returns:
        Bspline: A 1D, degree-2, rank-3, rational B-spline curve.

    Example:
        >>> crv = circle()
        >>> crv.degree
        (2,)
        >>> crv.is_rational
        True
    """
    if angle is None:
        cw = _build_full_circle(radius)
        spans = 4
    else:
        if isinstance(angle, tuple | list):
            start, end = angle
        else:
            start, end = 0.0, float(angle)
        sweep = end - start
        spans = int(np.searchsorted(_QUADRANT_BOUNDS, abs(sweep)))
        spans = max(spans, 1)
        cw = _build_arc(radius, start, sweep, spans)

    # Translate to center
    if center is not None:
        c = _pad_to_3d(center)
        # For weighted homogeneous: (w*x, w*y, w*z, w) -> (w*x + w*cx, ...)
        cw[:, :_PHYSICAL_DIM] += cw[:, _PHYSICAL_DIM : _PHYSICAL_DIM + 1] * c

    # Build knot vector: [0,0,0, u1,u1, u2,u2, ..., 1,1,1]
    knots = np.empty(2 * (spans + 1) + 2, dtype=np.float64)
    knots[0] = 0.0
    knots[-1] = 1.0
    knots[1:-1] = np.linspace(0.0, 1.0, spans + 1).repeat(2)

    space = BsplineSpace([BsplineSpace1D(knots, degree=_DEGREE_CIRCLE)])
    return Bspline(space, cw, is_rational=True)


def _build_full_circle(radius: float) -> npt.NDArray[np.float64]:
    """Build weighted homogeneous control points for a full circle.

    Args:
        radius: Circle radius.

    Returns:
        npt.NDArray[np.float64]: Array of shape ``(9, 4)``.
    """
    wm = np.sqrt(2.0) / 2.0
    cw = np.zeros((9, _PHYSICAL_DIM + 1), dtype=np.float64)
    cw[:, :2] = [
        [1, 0],
        [1, 1],
        [0, 1],
        [-1, 1],
        [-1, 0],
        [-1, -1],
        [0, -1],
        [1, -1],
        [1, 0],
    ]
    cw[:, :2] *= radius
    cw[:, _PHYSICAL_DIM] = 1.0
    cw[1::2, :] *= wm
    return cw


def _build_arc(
    radius: float,
    start: float,
    sweep: float,
    spans: int,
) -> npt.NDArray[np.float64]:
    """Build weighted homogeneous control points for a circular arc.

    Constructs a template arc bisected by the +X axis, then rotates it
    to the correct starting angle.  Subsequent spans are obtained by
    successive rotation of the previous span's last two control points.

    Args:
        radius: Circle radius.
        start: Start angle in radians.
        sweep: Sweep angle in radians (may be negative).
        spans: Number of quadratic arc spans.

    Returns:
        npt.NDArray[np.float64]: Array of shape ``(2*spans+1, 4)``.
    """
    alpha = sweep / (2 * spans)
    sin_a = np.sin(alpha)
    cos_a = np.cos(alpha)
    tan_a = np.tan(alpha)
    x = radius * cos_a
    y = radius * sin_a
    wm = cos_a
    xm = x + y * tan_a

    # Template arc: 3 control points bisected by +X axis
    template = np.array(
        [
            [x, -y, 0.0, 1.0],
            [wm * xm, 0.0, 0.0, wm],
            [x, y, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # Rotate template to the correct starting position
    cw = np.empty((2 * spans + 1, _PHYSICAL_DIM + 1), dtype=np.float64)
    cw[0:3] = _rotate_weighted(template, alpha + start)

    # Each subsequent span is a rotation of the previous one
    if spans > 1:
        two_alpha = 2.0 * alpha
        for i in range(1, spans):
            n = 2 * i + 1
            cw[n : n + 2] = _rotate_weighted(cw[n - 2 : n], two_alpha)

    return cw


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
         / |                  / |                 w
        o--------------------o  |                 ^  v
        |  | corners[0,0,1]  |  | corners[1,0,1]  | /
        |  |                 |  |                 |/
        |  o-----------------|--o                 +------> u
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
