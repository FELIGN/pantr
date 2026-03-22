"""Layer 2 implementation for B-spline knot removal.

This module provides input validation, multiplicity lookup, and
multi-dimensional orchestration that wrap the Layer 3 knot-removal kernel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ._bspline_knot_removal_core import _remove_knot_1d_core
from ._bspline_knots import _get_unique_knots_and_multiplicity_impl

if TYPE_CHECKING:
    from . import Bspline, BsplineSpace1D


def _find_knot_index_and_multiplicity(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    knot_value: float,
    tol: float,
) -> tuple[int, int]:
    """Find the last index and multiplicity of a knot value in the knot vector.

    Searches the knot vector for the last occurrence of *knot_value* (within
    tolerance) and returns its index *r* together with the multiplicity *s*.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): Polynomial degree.
        knot_value (float): The knot value to locate.
        tol (float): Tolerance for numerical comparisons.

    Returns:
        tuple[int, int]: ``(r, s)`` where *r* is the index of the last
        occurrence and *s* is the multiplicity.

    Raises:
        ValueError: If *knot_value* is not found in the knot vector.
    """
    unique_knots, mults = _get_unique_knots_and_multiplicity_impl(
        knots, degree, tol, in_domain=False
    )

    # Find the matching unique knot.
    matches = np.where(np.isclose(unique_knots, knot_value, atol=tol))[0]
    if len(matches) == 0:
        raise ValueError(f"Knot value {knot_value} not found in knot vector (tolerance={tol}).")

    idx = int(matches[0])
    s = int(mults[idx])

    # Compute the last index r of this knot value in the full knot vector.
    # Sum multiplicities up to and including this knot.
    r = int(np.sum(mults[: idx + 1])) - 1

    return r, s


def _remove_knot_bspline_1d_impl(  # noqa: PLR0913
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    ctrl: npt.NDArray[np.float32 | np.float64],
    knot_value: float,
    num: int | None,
    tol_space: float,
    tol_deviation: float,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64], int]:
    """Remove a single knot value from a 1D B-spline.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): Knot vector of shape
            ``(n + degree + 2,)``.
        degree (int): Polynomial degree.
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points of shape
            ``(n + 1, rank)``.
        knot_value (float): The knot value to remove.
        num (int | None): Maximum number of removals. ``None`` removes as many
            as possible (up to the current multiplicity, capped at ``degree``).
        tol_space (float): Tolerance for knot comparison.
        tol_deviation (float): Maximum allowed geometric deviation.

    Returns:
        tuple[npt.NDArray, npt.NDArray, int]: ``(new_knots, new_ctrl, removals)``
        where *removals* is the number of knots actually removed.

    Raises:
        ValueError: If *knot_value* is not found in the knot vector.
        ValueError: If *knot_value* is a boundary knot of an open (clamped) spline.
        ValueError: If *num* is not positive.
    """
    if num is not None and num < 1:
        raise ValueError(f"num must be a positive integer or None, got {num}.")

    r, s = _find_knot_index_and_multiplicity(knots, degree, knot_value, tol_space)

    # Boundary knots of open splines cannot be removed.
    domain_lo = float(knots[degree])
    domain_hi = float(knots[-degree - 1])
    if np.isclose(knot_value, domain_lo, atol=tol_space):
        raise ValueError(f"Cannot remove boundary knot {knot_value} (domain start).")
    if np.isclose(knot_value, domain_hi, atol=tol_space):
        raise ValueError(f"Cannot remove boundary knot {knot_value} (domain end).")

    # Cap at the actual multiplicity (and at degree per the algorithm).
    max_removals = min(s, degree)
    num = max_removals if num is None else min(num, max_removals)

    # Ensure contiguous layout for the Numba kernel.
    ctrl_c = ctrl if ctrl.flags.c_contiguous else np.ascontiguousarray(ctrl)

    new_knots, new_ctrl, removals = _remove_knot_1d_core(
        degree,
        knots,
        ctrl_c,
        float(knot_value),
        r,
        s,
        num,
        tol_deviation,
    )
    return new_knots, new_ctrl, removals


def _remove_knots_bspline(
    bspline: Bspline,
    knot_values_per_dim: list[npt.NDArray[np.float32 | np.float64] | None],
    num: int | None,
    tol: float | None,
) -> Bspline:
    """Apply knot removal per parametric direction and return a new B-spline.

    For each direction, iterates over the distinct knot values to remove,
    applying single-knot removal sequentially (each removal changes the knot
    indices for subsequent values).

    Args:
        bspline (Bspline): Original B-spline (must be non-periodic, open).
        knot_values_per_dim (list[npt.NDArray | None]): Per-direction arrays of
            distinct knot values to remove. ``None`` or an empty array skips
            that direction.
        num (int | None): Maximum removals per knot value. ``None`` removes
            as many as possible.
        tol (float | None): Geometric deviation tolerance. ``None`` uses a
            default of ``1e-10``.

    Returns:
        Bspline: New B-spline with reduced knot vectors.
    """
    dim = bspline.dim
    ctrl = bspline.control_points
    tol_deviation = 1e-10 if tol is None else tol

    from ._bspline_space_1d import BsplineSpace1D  # noqa: PLC0415

    new_spaces_1d: list[BsplineSpace1D] = []

    for i in range(dim):
        space_1d = bspline.space.spaces[i]
        kv = knot_values_per_dim[i]

        if kv is None or kv.size == 0:
            new_spaces_1d.append(space_1d)
            continue

        # Move dimension i to the 0th axis.
        moved_ctrl = np.moveaxis(ctrl, i, 0)
        orig_shape = moved_ctrl.shape

        # Flatten remaining axes into a single column dimension.
        pts_2d = moved_ctrl.reshape(orig_shape[0], -1)
        if not pts_2d.flags.c_contiguous:
            pts_2d = np.ascontiguousarray(pts_2d)

        current_knots = space_1d.knots
        current_ctrl = pts_2d

        # Remove each distinct knot value sequentially.
        for val in kv:
            current_knots, current_ctrl, _ = _remove_knot_bspline_1d_impl(
                current_knots,
                space_1d.degree,
                current_ctrl,
                float(val),
                num,
                float(space_1d.tolerance),
                tol_deviation,
            )

        # Restore multi-dimensional shape.
        new_shape = (current_ctrl.shape[0], *orig_shape[1:])
        new_moved_ctrl = current_ctrl.reshape(new_shape)

        # Move axis back to its original position.
        ctrl = np.moveaxis(new_moved_ctrl, 0, i)

        new_spaces_1d.append(BsplineSpace1D(current_knots, space_1d.degree))

    # Assemble the new B-spline.
    from . import (  # noqa: PLC0415
        Bspline,
        BsplineSpace,
    )

    new_space = BsplineSpace(new_spaces_1d)
    return Bspline(new_space, ctrl, is_rational=bspline.is_rational)
