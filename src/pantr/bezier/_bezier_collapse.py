"""Bézier collapse along axis (reduce to univariate by partial evaluation).

This module provides :func:`_collapse_along_axis`, which fixes all parametric
directions of a Bézier except one at given parameter values, producing a
univariate Bézier along the remaining direction.

The algorithm evaluates the Bernstein basis in each collapsed direction at the
given parameter value, then contracts the control point tensor against these
basis vectors using :func:`numpy.tensordot`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..basis._basis_core import _tabulate_Bernstein_basis_1D_core
from ._bezier_backend import collapse_kernel

if TYPE_CHECKING:
    from . import Bezier


def _collapse_along_axis(
    bezier: Bezier,
    axis: int,
    values: npt.ArrayLike,
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
        bezier (~pantr.bezier.Bezier): The Bézier to collapse.  ``dim >= 2``
            and ``axis`` bounds are assumed valid (checked by the public method).
        axis (int): Parametric direction to keep (0-indexed).
        values (npt.ArrayLike): Parameter values for all directions except
            ``axis``, of length ``dim - 1``.  ``values[i]`` corresponds to
            direction ``i`` for ``i < axis``, and direction ``i + 1`` for
            ``i >= axis``.  Values are cast to ``bezier.dtype``; passing
            higher-precision values to a ``float32`` Bézier will silently
            reduce precision.

    Returns:
        ~pantr.bezier.Bezier: A 1D Bézier with degree ``bezier.degree[axis]``
        and the same rank and rationality as the input.

    Raises:
        ValueError: If ``values`` does not have length ``dim - 1``.
        ValueError: If any value is outside ``[0, 1]``.
    """
    dim = bezier.dim
    values_arr = np.asarray(values, dtype=bezier.dtype)

    if values_arr.ndim != 1 or values_arr.shape[0] != dim - 1:
        raise ValueError(
            f"values must have length dim - 1 = {dim - 1}, got shape {values_arr.shape}."
        )
    if np.any(values_arr < 0.0) or np.any(values_arr > 1.0):
        raise ValueError("All values must be in [0, 1].")

    return collapse_kernel()(bezier, axis, values_arr)


def _collapse_python(
    bezier: Bezier,
    axis: int,
    values: npt.NDArray[np.float32 | np.float64],
) -> Bezier:
    """Collapse with NumPy: the oracle for the port.

    Contracts the directions from highest to lowest, skipping ``axis``, so that each
    direction's current index still equals its original one when its turn comes. The
    order matters beyond bookkeeping: the contractions are not associative in floating
    point, so a different order is a different answer.

    Args:
        bezier (~pantr.bezier.Bezier): The Bézier to collapse.
        axis (int): Parametric direction to keep.
        values (npt.NDArray[np.float32 | np.float64]): One parameter per collapsed
            direction, already cast to the Bézier's dtype.

    Returns:
        ~pantr.bezier.Bezier: A one-dimensional Bézier along ``axis``.

    Note:
        No input validation is performed here; Layer 2 did it above the branch.
    """
    from . import Bezier as BezierCls  # noqa: PLC0415

    dim = bezier.dim
    dtype = bezier.dtype
    degrees = bezier.degree

    result: npt.NDArray[np.float32 | np.float64] = bezier.control_points
    for d in range(dim - 1, -1, -1):
        if d == axis:
            continue
        val_idx = d if d < axis else d - 1
        pts_d = np.array([values[val_idx]])
        basis_d = np.empty((1, degrees[d] + 1), dtype=dtype)
        _tabulate_Bernstein_basis_1D_core(np.int32(degrees[d]), pts_d, basis_d)
        basis_1d: npt.NDArray[np.float32 | np.float64] = basis_d[0]
        result = np.tensordot(basis_1d, result, axes=([0], [d]))

    return BezierCls(result, is_rational=bezier.is_rational)
