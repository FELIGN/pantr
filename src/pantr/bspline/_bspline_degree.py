"""Layer 2 implementation for B-spline degree elevation and reduction.

This module provides the validation and array manipulation logic to
prepare inputs for the Layer 3 degree elevation/reduction kernels and
wrap their outputs back into a new B-spline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ._bspline_degree_core import _degree_elevate_1d_core, _degree_reduce_1d_core
from ._bspline_knot_insertion import _to_open_bspline_1d_impl, _to_periodic_bspline_1d_impl
from ._bspline_knot_removal import _remove_knot_bspline_1d_impl
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


def _coarsen_knots_after_reduction(  # noqa: PLR0913
    knots: npt.NDArray[np.float32 | np.float64],
    new_degree: int,
    ctrl: npt.NDArray[np.float32 | np.float64],
    orig_unique_knots: npt.NDArray[np.float32 | np.float64],
    orig_mults: npt.NDArray[np.int_],
    degree_decrement: int,
    tol_space: float,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
    """Remove excess knots from a Bézier-form B-spline after degree reduction.

    After degree reduction the output is in Bézier form (all interior
    breakpoints have multiplicity ``new_degree``).  This function removes
    excess copies of each interior knot so that the final multiplicity is
    ``max(1, m_i - t)`` where ``m_i`` is the original multiplicity and ``t``
    is the degree decrement, restoring the original continuity structure.

    Args:
        knots: Knot vector of the Bézier-form reduced B-spline.
        new_degree: Degree after reduction.
        ctrl: Control points of shape ``(n_pts, rank)``.
        orig_unique_knots: Unique interior knot values of the *original*
            B-spline (before reduction).
        orig_mults: Corresponding multiplicities of the original interior knots.
        degree_decrement: How many degrees were reduced.
        tol_space: Tolerance for knot comparison.

    Returns:
        tuple: ``(coarsened_knots, coarsened_ctrl)``.
    """
    for idx in range(len(orig_unique_knots)):
        knot_val = float(orig_unique_knots[idx])
        m_orig = int(orig_mults[idx])
        target_mult = max(1, m_orig - degree_decrement)
        # Current multiplicity in the Bézier form is new_degree.
        num_to_remove = new_degree - target_mult
        if num_to_remove <= 0:
            continue
        # Use a large tolerance for deviation since reduction is approximate.
        tol_dev = np.inf
        knots, ctrl, _removed = _remove_knot_bspline_1d_impl(
            knots, new_degree, ctrl, knot_val, num_to_remove, tol_space, tol_dev
        )
    return knots, ctrl


def _degree_reduce_bspline(bspline: Bspline, degree_decrements: tuple[int, ...]) -> Bspline:
    """Reduce the degree of a B-spline.

    For each direction with a positive decrement:

    1. Decompose to Bézier segments and reduce each via least-squares.
    2. Coarsen knots to restore the original continuity structure.
    3. Handle periodic directions by round-tripping through open form.

    Args:
        bspline (Bspline): Original B-spline.
        degree_decrements (tuple[int, ...]): Decrements for each dimension.

    Returns:
        Bspline: New B-spline with reduced degrees.
    """
    dim = bspline.dim
    ctrl = bspline.control_points

    orig_is_rational = bspline.is_rational

    new_spaces_1d: list[BsplineSpace1D] = []

    for i in range(dim):
        dec = degree_decrements[i]
        space_1d = bspline.space.spaces[i]

        if dec > 0:
            is_periodic = space_1d.periodic
            tol = float(space_1d.tolerance)

            # Record original interior knot multiplicities before reduction.
            orig_unique, orig_mults = _get_unique_knots_and_multiplicity_impl(
                space_1d.knots, space_1d.degree, tol, in_domain=True
            )
            # Exclude boundary knots (first and last unique in-domain knots)
            # from the coarsening — they are clamped and stay at full multiplicity.
            if len(orig_unique) > 2:  # noqa: PLR2004
                interior_knots = orig_unique[1:-1]
                interior_mults = orig_mults[1:-1]
            else:
                interior_knots = np.empty(0, dtype=orig_unique.dtype)
                interior_mults = np.empty(0, dtype=orig_mults.dtype)

            moved_ctrl = np.moveaxis(ctrl, i, 0)
            orig_shape = moved_ctrl.shape

            pts_2d = moved_ctrl.reshape(orig_shape[0], -1)
            pts_2d = np.ascontiguousarray(pts_2d)

            if is_periodic:
                m_bdy = int(orig_mults[0])

                open_knots, open_pts_2d = _to_open_bspline_1d_impl(
                    space_1d.knots, space_1d.degree, pts_2d, True, tol
                )

                new_pts_2d, new_knots = _degree_reduce_1d_core(
                    space_1d.degree, open_pts_2d, open_knots, dec
                )

                new_degree = space_1d.degree - dec

                # Coarsen interior knots.
                if len(interior_knots) > 0:
                    new_knots, new_pts_2d = _coarsen_knots_after_reduction(
                        new_knots,
                        new_degree,
                        new_pts_2d,
                        interior_knots,
                        interior_mults,
                        dec,
                        tol,
                    )

                m_bdy_new = m_bdy - dec
                per_knots, new_pts_2d = _to_periodic_bspline_1d_impl(
                    new_knots, new_degree, new_pts_2d, m_bdy_new, tol
                )

                new_space_1d = BsplineSpace1D(per_knots, new_degree, periodic=True)
            else:
                new_pts_2d, new_knots = _degree_reduce_1d_core(
                    space_1d.degree, pts_2d, space_1d.knots, dec
                )
                new_degree = space_1d.degree - dec

                # Coarsen interior knots.
                if len(interior_knots) > 0:
                    new_knots, new_pts_2d = _coarsen_knots_after_reduction(
                        new_knots,
                        new_degree,
                        new_pts_2d,
                        interior_knots,
                        interior_mults,
                        dec,
                        tol,
                    )

                new_space_1d = BsplineSpace1D(new_knots, new_degree)

            # Restore shape.
            new_shape = (new_pts_2d.shape[0], *orig_shape[1:])
            new_moved_ctrl = new_pts_2d.reshape(new_shape)
            ctrl = np.moveaxis(new_moved_ctrl, 0, i)

            new_spaces_1d.append(new_space_1d)
        else:
            new_spaces_1d.append(space_1d)

    from . import Bspline  # noqa: PLC0415

    new_space = BsplineSpace(new_spaces_1d)

    return Bspline(new_space, ctrl, is_rational=orig_is_rational)
