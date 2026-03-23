"""Layer 2 implementation for B-spline degree elevation.

This module provides the validation and array manipulation logic to
prepare inputs for the Layer 3 degree elevation kernels and wrap their
outputs back into a new B-spline degree elevation implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._bspline_degree_core import _degree_elevate_1d_core
from ._bspline_knot_insertion import _to_open_bspline_1d_impl, _to_periodic_bspline_1d_impl
from ._bspline_knots import _get_unique_knots_and_multiplicity_impl
from ._bspline_space_1d import BsplineSpace1D
from ._bspline_space_nd import BsplineSpace

if TYPE_CHECKING:
    from . import Bspline


def _degree_elevate_bspline(bspline: Bspline, degree_increments: tuple[int, ...]) -> Bspline:
    """Elevate the degree of a B-spline.

    Args:
        bspline (Bspline): Original B-spline.
        degree_increments (tuple[int, ...]): Increments for each dimension.

    Returns:
        Bspline: New B-spline with elevated degrees.
    """
    dim = bspline.dim
    ctrl = bspline.control_points

    # Bspline variables
    orig_is_rational = bspline.is_rational

    new_spaces_1d: list[BsplineSpace1D] = []

    for i in range(dim):
        inc = degree_increments[i]
        space_1d = bspline.space.spaces[i]

        if inc > 0:
            is_periodic = space_1d.periodic

            # Move dimension i to the 0th axis
            moved_ctrl = np.moveaxis(ctrl, i, 0)
            orig_shape = moved_ctrl.shape

            # Reshape rest into 2D points block for Numba
            pts_2d = moved_ctrl.reshape(orig_shape[0], -1)

            # Ensure proper contiguous layout for Numba
            if not pts_2d.flags.c_contiguous:
                pts_2d = np.ascontiguousarray(pts_2d)

            if is_periodic:
                # Round-trip through open form to preserve periodicity.
                tol = float(space_1d.tolerance)
                _, mults = _get_unique_knots_and_multiplicity_impl(
                    space_1d.knots, space_1d.degree, tol, in_domain=True
                )
                m_bdy = int(mults[0])

                # Convert to open form.
                open_knots, open_pts_2d = _to_open_bspline_1d_impl(
                    space_1d.knots, space_1d.degree, pts_2d, True, tol
                )

                # Degree elevate the open representation.
                new_pts_2d, new_knots = _degree_elevate_1d_core(
                    space_1d.degree, open_pts_2d, open_knots, inc
                )

                # Convert back to periodic. Degree elevation increases every
                # knot multiplicity by inc, so m_bdy_new = m_bdy + inc.
                new_degree = space_1d.degree + inc
                m_bdy_new = m_bdy + inc
                per_knots, new_pts_2d = _to_periodic_bspline_1d_impl(
                    new_knots, new_degree, new_pts_2d, m_bdy_new, tol
                )

                new_space_1d = BsplineSpace1D(per_knots, new_degree, periodic=True)
            else:
                # Numba kernel
                new_pts_2d, new_knots = _degree_elevate_1d_core(
                    space_1d.degree, pts_2d, space_1d.knots, inc
                )
                new_space_1d = BsplineSpace1D(new_knots, space_1d.degree + inc)

            # Restore shape
            new_shape = (new_pts_2d.shape[0], *orig_shape[1:])
            new_moved_ctrl = new_pts_2d.reshape(new_shape)

            # Move axis back
            ctrl = np.moveaxis(new_moved_ctrl, 0, i)

            # New BsplineSpace1D
            new_spaces_1d.append(new_space_1d)
        else:
            new_spaces_1d.append(space_1d)

    # Assemble the new B-spline
    from . import Bspline  # noqa: PLC0415

    new_space = BsplineSpace(new_spaces_1d)

    return Bspline(new_space, ctrl, is_rational=orig_is_rational)
