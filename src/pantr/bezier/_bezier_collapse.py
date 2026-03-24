"""Bézier collapse along axis (reduce to univariate by partial evaluation).

This module provides :func:`_collapse_along_axis`, which fixes all parametric
directions of a Bézier except one at given parameter values, producing a
univariate Bézier along the remaining direction.

The algorithm evaluates the Bernstein basis in each collapsed direction at the
given parameter value, then contracts the control point tensor against these
basis vectors using :func:`numpy.tensordot`.  This mirrors the single-pass
tensor contraction strategy used in algoim's ``collapseAlongAxis``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..basis._basis_core import _tabulate_Bernstein_basis_1D_core

if TYPE_CHECKING:
    from . import Bezier


def _collapse_along_axis(
    bezier: Bezier,
    axis: int,
    values: npt.NDArray[np.float64],
) -> Bezier:
    """Collapse a Bézier to a univariate polynomial along one parametric direction.

    Fixes all parametric directions except ``axis`` at the parameter values
    given in ``values``, producing a 1D Bézier whose control points are the
    Bernstein coefficients along ``axis``.

    The contraction is performed by evaluating the Bernstein basis in each
    collapsed direction and contracting the control point array via
    :func:`numpy.tensordot`, processing directions from highest to lowest
    to keep axis indices stable.

    Args:
        bezier (~pantr.bezier.Bezier): The Bézier to collapse.
        axis (int): Parametric direction to keep (0-indexed, must be in
            ``[0, dim)``).
        values (npt.NDArray[np.float64]): Parameter values for all directions
            except ``axis``, of shape ``(dim - 1,)``.  ``values[i]``
            corresponds to direction ``i`` for ``i < axis``, and direction
            ``i + 1`` for ``i >= axis``.

    Returns:
        ~pantr.bezier.Bezier: A 1D Bézier with degree ``bezier.degree[axis]``
        and the same rank and rationality as the input.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :meth:`~pantr.bezier.Bezier.collapse_along_axis`
        instead.
    """
    from . import Bezier as BezierCls  # noqa: PLC0415

    dim = bezier.dim

    if dim == 1:
        return BezierCls(bezier.control_points.copy(), is_rational=bezier.is_rational)

    ctrl = bezier.control_points
    dtype = bezier.dtype
    degrees = bezier.degree

    # Contract directions from highest to lowest, skipping `axis`.
    # Processing high-to-low ensures that the current array index of each
    # direction equals its original index when we reach it.
    result: npt.NDArray[np.float32 | np.float64] = ctrl
    for d in range(dim - 1, -1, -1):
        if d == axis:
            continue

        val_idx = d if d < axis else d - 1

        # Evaluate Bernstein basis at the single parameter value.
        pts_d = np.array([values[val_idx]], dtype=dtype)
        basis_d = np.empty((1, degrees[d] + 1), dtype=dtype)
        _tabulate_Bernstein_basis_1D_core(np.int32(degrees[d]), pts_d, basis_d)
        basis_1d: npt.NDArray[np.float32 | np.float64] = basis_d[0]

        # Contract along direction d.
        result = np.tensordot(basis_1d, result, axes=([0], [d]))

    return BezierCls(result, is_rational=bezier.is_rational)
