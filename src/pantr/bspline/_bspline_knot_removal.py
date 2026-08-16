"""Layer 2 implementation for B-spline knot removal.

This module provides input validation, multiplicity lookup, and
multi-dimensional orchestration that wrap the Layer 3 knot-removal kernel.

**What the kernel's tolerance grades.** ``_remove_knot_1d_core`` accepts a removal
when the Euclidean distance between two reconstructions of the same control point
stays within ``tol``. That distance is taken over *the columns of the array it is
given*, which for a rational spline are the homogeneous coordinates
``[w * x, w * y, w * z, w]`` and not the projected point. A caller's budget, on the
other hand, is a distance in projected space. The two are different quantities, and
this module is where they are reconciled: see
:func:`_homogeneous_deviation_tolerance`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt

from .._array_utils import _flatten_along_axis, _unflatten_along_axis
from ..tolerance import get_strict
from ._bspline_knot_removal_core import _remove_knot_1d_core
from ._bspline_knots import _find_knot_index_and_multiplicity

if TYPE_CHECKING:
    from . import Bspline, BsplineSpace1D

_REMOVAL_FLOOR_SAFETY: Final[float] = 2.0
"""Extra factor on the strict tier for the round-off floor of an exact removal.

With :func:`~pantr.tolerance.get_strict` this is ``8 * eps``, and it is applied per
step of the reconstruction chain: the floor is
``8 * (degree + 1) * eps * control-point scale``.

**Shape.** Removing a knot once runs Algorithm A5.8's two one-sided reconstructions
towards each other, at most ``degree + 1`` control points from each side, and reads
off how far apart they land. Each step is one divided difference of the previous
result, so an ``eps``-sized error enters at every one of the ``degree + 1`` of them
and the largest quantity any of them handles is the control-point scale. The steps
are *not* convex combinations -- the ``1 / alpha`` in the recurrence can amplify --
so this is an operation count and not a proved bound; the eight epsilons per step are
the same doubling of the strict tier that
:data:`~pantr.bspline._bspline_knots._KNOT_MERGE_SAFETY` applies, covering an FMA
contraction and a differing summation order on top of the count.

**Measured**, over 105 cases (degrees 1 to 7, ranks 1 to 3, geometry scales ``1e-6``
to ``1e9``), taking an exactly removable knot -- one just inserted -- and bisecting
for the smallest budget that still removes it: the largest observed distance is
``0.84 * eps * scale``, and the tightest margin against ``8 * (degree + 1)`` over the
whole set is a factor of 48. There is no trend with degree (the worst case is at
degree 4), so the ``degree + 1`` is headroom the derivation asks for rather than a
term fitted to the data.
"""


def _control_point_scale(ctrl: npt.NDArray[np.float32 | np.float64]) -> float:
    """Get the magnitude a control-point distance is relative to.

    ``max(bounding-box diagonal, largest ||P_i||)`` over the rows of ``ctrl``, the same
    pairing of extent and coordinate magnitude that
    :func:`~pantr.bspline._bspline_knots._knot_scale` makes in parametric space and
    :func:`~pantr.bspline._bspline_locate._geometric_scale` in physical space. The
    extent alone is not enough: a small part sitting at ``x = 1e6`` has control points
    whose differences carry an absolute error of ``eps * 1e6`` however small the part
    is.

    Computed on the array as given, so for a rational spline it is the scale of the
    *homogeneous* control points, which is the space the kernel measures in.

    Args:
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points, of any shape whose
            last axis is the coordinate axis.

    Returns:
        float: The scale, zero only for a control net that is identically zero.
    """
    flat = np.asarray(ctrl, dtype=np.float64).reshape(-1, ctrl.shape[-1])
    diagonal = float(np.linalg.norm(flat.max(axis=0) - flat.min(axis=0)))
    magnitude = float(np.linalg.norm(flat, axis=1).max())
    return max(diagonal, magnitude)


def _roundoff_deviation_floor(ctrl: npt.NDArray[np.float32 | np.float64], degree: int) -> float:
    """Get the deviation budget that admits exactly the removals round-off allows.

    ``_REMOVAL_FLOOR_SAFETY * get_strict(dtype) * (degree + 1) * scale``, i.e.
    ``8 * (degree + 1) * eps`` times :func:`_control_point_scale`; the derivation and
    the measurement are in :data:`_REMOVAL_FLOOR_SAFETY`.

    This is the default budget, and it is deliberately *not* a geometric one. A knot
    that a spline does not actually need is removable exactly, and the only thing
    standing between it and a bit-for-bit reconstruction is the round-off of the
    reconstruction itself. A budget at that level therefore means "remove what is
    genuinely redundant and nothing else", which is the one default that needs no
    arbitrary constant. Anything looser is a statement about how much geometry the
    caller is willing to lose, which only the caller can make.

    It is expressed in the units the kernel measures in, so for a rational spline it is
    a homogeneous distance and does **not** go through
    :func:`_homogeneous_deviation_tolerance`: a floor is a statement about the
    arithmetic, and the arithmetic happens in homogeneous coordinates.

    Args:
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points as the kernel will
            see them, homogeneous for a rational spline.
        degree (int): Polynomial degree in the direction being reduced.

    Returns:
        float: Absolute deviation budget, in the units of ``ctrl``.
    """
    scale = _control_point_scale(ctrl)
    return _REMOVAL_FLOOR_SAFETY * get_strict(ctrl.dtype) * (degree + 1) * scale


def _homogeneous_deviation_tolerance(
    ctrl: npt.NDArray[np.float32 | np.float64], tol_euclidean: float
) -> float:
    """Pull a projected-space deviation budget back to homogeneous control points.

    ``TOL = d * w_min / (1 + |P|_max)``, with ``|P|_max = max_i ||P_i||_2`` in
    **Euclidean** coordinates and ``w_min = min_i w_i``. This is Eq. (5.30) of Piegl &
    Tiller, *The NURBS Book*, 2nd ed. (Springer, 1997), p. 185, given there for
    Algorithm A5.8, which :func:`~pantr.bspline._bspline_knot_removal_core.
    _remove_knot_1d_core` otherwise implements faithfully.

    Why it is needed: the kernel measures one Euclidean distance ``D`` over every
    column of the array it is handed, which for a rational spline are the homogeneous
    coordinates ``[w x, w y, w z, w]``. Moving the homogeneous point by ``dP^w`` moves
    the projected point by ``(dP^w_xyz - P dP^w_w) / w``, so

        |dP| <= (|dP^w_xyz| + |P| |dP^w_w|) / w <= D (1 + |P|) / w,

    and requiring ``D <= d w_min / (1 + |P|_max)`` gives ``|dP| <= d`` for every point,
    since ``w >= w_min`` and ``|P| <= |P|_max``. Without it, a deviation carried by the
    **weight** column is graded as though it were a coordinate: measured with the
    perturbation in the weight and the coordinates at ``1e6``, the actual projected
    deviation exceeded the requested tolerance by a factor of ``3.6e5``. A circle cannot
    show this, because its weights are structurally tied to its coordinates.

    Two honest caveats.

    The bound is **tight for a weight-carried perturbation and conservative for a purely
    geometric one**, by up to a factor ``1 + |P|_max``. Splitting ``D`` optimally between
    the two parts gives ``sqrt(1 + |P|_max^2)``, which differs from ``1 + |P|_max`` by at
    most ``sqrt(2)``, so almost none of that conservatism comes from the triangle
    inequality: it comes from the kernel compressing a length and a weight into a single
    Euclidean distance, which loses the direction. Removing it would mean the kernel
    grading the two parts separately.

    That is also why ``1 + |P|_max`` adds a pure number to a length. The
    inhomogeneity is inherited from ``D``, whose components are not all of one unit,
    and is not an error in the formula; the bound above holds at every scale.

    Args:
        ctrl (npt.NDArray[np.float32 | np.float64]): Homogeneous control points, last
            column the weights, of any shape whose last axis is the coordinate axis.
        tol_euclidean (float): The caller's deviation budget, a distance in projected
            space.

    Returns:
        float: The equivalent budget on homogeneous control points.
    """
    flat = np.asarray(ctrl, dtype=np.float64).reshape(-1, ctrl.shape[-1])
    weights = flat[:, -1]
    w_min = float(np.abs(weights).min())
    projected = flat[:, :-1] / weights[:, None]
    p_max = float(np.linalg.norm(projected, axis=1).max())
    return tol_euclidean * w_min / (1.0 + p_max)


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
        tol_deviation (float): Maximum allowed control-point deviation, already in the
            units the kernel measures in (homogeneous for a rational spline); see the
            module docstring.

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
    if abs(float(knot_value) - domain_lo) <= tol_space:
        raise ValueError(f"Cannot remove boundary knot {knot_value} (domain start).")
    if abs(float(knot_value) - domain_hi) <= tol_space:
        raise ValueError(f"Cannot remove boundary knot {knot_value} (domain end).")

    # Cap at the actual multiplicity (and at degree per the algorithm).  This clamp is
    # load-bearing for memory safety, not just a sanity bound: `_remove_knot_1d_core`
    # sizes its scratch buffer for `2 * num <= degree + s` and writes outside it past
    # that, silently.  `min(s, degree)` implies that bound, so the clamp has to stay,
    # including when the caller supplied `num` explicitly.
    max_removals = min(s, degree)
    num = max_removals if num is None else min(num, max_removals)

    ctrl_c = np.ascontiguousarray(ctrl)

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
        tol (float | None): Deviation tolerance, a distance in **projected** space.
            ``None`` uses the round-off floor of :func:`_roundoff_deviation_floor`,
            which removes what is redundant to within the arithmetic and nothing more.

    Returns:
        Bspline: New B-spline with reduced knot vectors.
    """
    dim = bspline.dim
    ctrl = bspline.control_points

    from ._bspline_space_1d import BsplineSpace1D  # noqa: PLC0415

    new_spaces_1d: list[BsplineSpace1D] = []

    for i in range(dim):
        space_1d = bspline.space.spaces[i]
        kv = knot_values_per_dim[i]

        if kv is None or kv.size == 0:
            new_spaces_1d.append(space_1d)
            continue

        # Resolved per direction, from the control points as they stand: the degree and
        # the control-point scale both differ between directions, and `ctrl` is carried
        # forward from the previous one.
        if tol is None:
            tol_deviation = _roundoff_deviation_floor(ctrl, space_1d.degree)
        elif bspline.is_rational:
            tol_deviation = _homogeneous_deviation_tolerance(ctrl, tol)
        else:
            tol_deviation = tol

        pts_2d, trailing_shape = _flatten_along_axis(ctrl, i)

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

        ctrl = _unflatten_along_axis(current_ctrl, trailing_shape, i)

        new_spaces_1d.append(BsplineSpace1D(current_knots, space_1d.degree))

    # Assemble the new B-spline.
    from . import (  # noqa: PLC0415
        Bspline,
        BsplineSpace,
    )

    new_space = BsplineSpace(new_spaces_1d)
    return Bspline(new_space, ctrl, is_rational=bspline.is_rational)
