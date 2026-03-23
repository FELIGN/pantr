"""Constructive operations: extrude and ruled.

Provides functions that create higher-dimensional B-spline objects
by combining existing ones: extrusion along a vector and ruled
interpolation between two patches.
"""

from __future__ import annotations

import numpy as np
from numpy import typing as npt

from ..bspline import Bspline, BsplineSpace
from ._compat import compat
from ._primitives import _linear_space_1d
from ._validation import _PHYSICAL_DIM, _pad_to_3d, _promote_to_rational

_MAX_DIM_FOR_OPERATIONS = 2


def extrude(bspline: Bspline, displacement: npt.ArrayLike) -> Bspline:
    """Extrude a B-spline curve or surface along a displacement vector.

    Creates a new B-spline with one additional parametric dimension by
    translating the input along the given vector.  The new direction is
    appended as the last parametric axis with degree 1 and knots
    ``[0, 0, 1, 1]``.

    Args:
        bspline: Input curve (dim=1) or surface (dim=2).
        displacement: Translation vector (up to 3D, zero-padded).

    Returns:
        Bspline: A B-spline with ``dim + 1`` parametric dimensions.

    Raises:
        ValueError: If ``bspline.dim > 2``.

    Example:
        >>> from pantr.cad import circle, extrude
        >>> cyl = extrude(circle(), [0, 0, 1])
        >>> cyl.dim
        2
    """
    if bspline.dim > _MAX_DIM_FOR_OPERATIONS:
        raise ValueError(f"extrude requires dim <= {_MAX_DIM_FOR_OPERATIONS}, got {bspline.dim}.")

    disp = _pad_to_3d(displacement)
    cp = bspline.control_points
    orig_shape = cp.shape[:-1]  # (*num_basis,)
    rank_full = cp.shape[-1]

    new_cp = np.empty((*orig_shape, 2, rank_full), dtype=cp.dtype)
    new_cp[..., 0, :] = cp

    if bspline.is_rational:
        # Weighted homogeneous: (w*x, w*y, w*z, w)
        # Translated: (w*x + w*dx, w*y + w*dy, w*z + w*dz, w)
        new_cp[..., 1, :] = cp
        weights = cp[..., _PHYSICAL_DIM : _PHYSICAL_DIM + 1]
        new_cp[..., 1, :_PHYSICAL_DIM] = cp[..., :_PHYSICAL_DIM] + weights * disp
    else:
        new_cp[..., 1, :] = cp + disp[:rank_full].astype(cp.dtype)

    spaces = [*bspline.space.spaces, _linear_space_1d(dtype=bspline.dtype)]
    new_space = BsplineSpace(spaces)
    return Bspline(new_space, new_cp, is_rational=bspline.is_rational)


def ruled(bspline1: Bspline, bspline2: Bspline) -> Bspline:
    """Construct a ruled surface or volume between two B-splines.

    Creates a new B-spline by linearly interpolating control points
    between *bspline1* (at parameter 0) and *bspline2* (at parameter 1)
    along a new last parametric axis with degree 1.

    The two inputs are first made compatible via :func:`compat` so they
    share the same degree and knot vectors.  If one is rational and the
    other is not, the non-rational one is promoted.

    Args:
        bspline1: First boundary (curve or surface, dim <= 2).
        bspline2: Second boundary (same dim as *bspline1*).

    Returns:
        Bspline: A B-spline with ``dim + 1`` parametric dimensions.

    Raises:
        ValueError: If the inputs have different parametric dimensions.
        ValueError: If either input has ``dim > 2``.

    Example:
        >>> from pantr.cad import circle, ruled
        >>> annulus = ruled(circle(radius=0.5), circle(radius=1.0))
        >>> annulus.dim
        2
    """
    if bspline1.dim != bspline2.dim:
        raise ValueError(
            f"Both B-splines must have the same dim, got {bspline1.dim} and {bspline2.dim}."
        )
    if bspline1.dim > _MAX_DIM_FOR_OPERATIONS:
        raise ValueError(f"ruled requires dim <= {_MAX_DIM_FOR_OPERATIONS}, got {bspline1.dim}.")

    b1, b2 = compat(bspline1, bspline2)

    # Promote to rational if needed
    is_rational = b1.is_rational or b2.is_rational
    if is_rational:
        b1 = _promote_to_rational(b1)
        b2 = _promote_to_rational(b2)

    cp1 = b1.control_points
    cp2 = b2.control_points
    rank_full = cp1.shape[-1]

    new_cp = np.empty((*cp1.shape[:-1], 2, rank_full), dtype=cp1.dtype)
    new_cp[..., 0, :] = cp1
    new_cp[..., 1, :] = cp2

    spaces = [*b1.space.spaces, _linear_space_1d(dtype=b1.dtype)]
    new_space = BsplineSpace(spaces)
    return Bspline(new_space, new_cp, is_rational=is_rational)
