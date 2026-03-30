"""Derived primitives built from core operations.

Provides higher-level geometric primitives constructed by composing
:func:`create_circle`, :func:`extrude`, :func:`create_ruled`, and :func:`revolve`.
"""

from __future__ import annotations

import numpy as np
from numpy import typing as npt

from ..bspline import Bspline, BsplineSpace, BsplineSpace1D
from ._operations import create_ruled, extrude
from ._primitives import create_circle
from ._validation import _PHYSICAL_DIM, _pad_to_3d


def create_rectangle(
    corner: npt.ArrayLike = (0.0, 0.0, 0.0),
    width: float = 1.0,
    height: float = 1.0,
) -> Bspline:
    """Construct a closed rectangular B-spline curve in the xy-plane.

    Creates a degree-1 curve with 5 control points (the first point
    repeated to close the loop) and 4 knot spans.

    Args:
        corner: Bottom-left corner (up to 3D, zero-padded).
        width: Rectangle width along the x-axis.
        height: Rectangle height along the y-axis.

    Returns:
        Bspline: A 1D, degree-1, rank-3, non-rational closed curve.

    Example:
        >>> rect = create_rectangle([0, 0], 2, 3)
        >>> rect.dim
        1
    """
    c = _pad_to_3d(corner)
    dx = np.array([width, 0.0, 0.0])
    dy = np.array([0.0, height, 0.0])

    # 5 control points: close the rectangle
    cp = np.array([c, c + dx, c + dx + dy, c + dy, c], dtype=np.float64)

    # 4 spans, degree 1: knots [0,0, 0.25,0.25, 0.5,0.5, 0.75,0.75, 1,1]
    n_spans = 4
    knots = np.empty(2 * (n_spans + 1), dtype=np.float64)
    knots[0] = 0.0
    knots[-1] = 1.0
    knots[1:-1] = np.linspace(0.0, 1.0, n_spans + 1).repeat(2)[1:-1]

    # Actually for degree 1 with 5 CPs we need 7 knots: n + p + 1 = 5+1+1=7
    # knots = [0, 0, 0.25, 0.5, 0.75, 1, 1]
    knots = np.array([0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0], dtype=np.float64)

    space = BsplineSpace([BsplineSpace1D(knots, degree=1)])
    return Bspline(space, cp)


def create_disk(
    radius_inner: float = 0.0,
    radius_outer: float = 1.0,
    center: npt.ArrayLike | None = None,
    angle: float | tuple[float, float] | None = None,
) -> Bspline:
    """Construct a disk or annular sector as a NURBS surface.

    When ``radius_inner > 0``, produces an annular sector via
    :func:`create_ruled` between inner and outer circular arcs.  When
    ``radius_inner == 0``, the inner boundary degenerates to a point.

    Args:
        radius_inner: Inner radius.  Use 0 for a full disk.
        radius_outer: Outer radius.
        center: Center point (up to 3D, zero-padded).  If ``None``,
            centered at the origin.
        angle: Sweep specification (same as :func:`create_circle`).

    Returns:
        Bspline: A 2D rational B-spline surface.

    Example:
        >>> d = create_disk(radius_outer=2.0)
        >>> d.dim
        2
    """
    outer = create_circle(radius=radius_outer, center=center, angle=angle)

    if radius_inner > 0:
        inner = create_circle(radius=radius_inner, center=center, angle=angle)
        return create_ruled(inner, outer)

    # Degenerate inner: all control points at center
    c = _pad_to_3d(center) if center is not None else np.zeros(_PHYSICAL_DIM)
    n_cp = outer.control_points.shape[0]
    rank_full = outer.control_points.shape[-1]

    # Build inner as a rational curve with all points at center
    inner_cp = np.zeros((n_cp, rank_full), dtype=np.float64)
    # Copy weights from outer arc
    inner_cp[:, _PHYSICAL_DIM] = outer.control_points[:, _PHYSICAL_DIM]
    # Set weighted coordinates: w * center
    for i in range(_PHYSICAL_DIM):
        inner_cp[:, i] = inner_cp[:, _PHYSICAL_DIM] * c[i]

    inner = Bspline(outer.space, inner_cp, is_rational=True)
    return create_ruled(inner, outer)


def create_cylinder(
    radius: float = 1.0,
    height: float = 1.0,
    center: npt.ArrayLike | None = None,
    angle: float | tuple[float, float] | None = None,
) -> Bspline:
    """Construct a cylindrical NURBS surface.

    Built by extruding a circle along the z-axis.

    Args:
        radius: Cylinder radius.
        height: Cylinder height along the z-axis.
        center: Center of the base circle (up to 3D, zero-padded).
        angle: Sweep specification (same as :func:`create_circle`).

    Returns:
        Bspline: A 2D rational B-spline surface.

    Example:
        >>> cyl = create_cylinder(radius=2, height=5)
        >>> cyl.dim
        2
    """
    return extrude(create_circle(radius=radius, center=center, angle=angle), [0, 0, height])
