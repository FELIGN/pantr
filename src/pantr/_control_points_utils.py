"""Shared control-point manipulation helpers (reverse, permute, homogeneous).

These functions operate purely on control-point arrays and are used across the
:mod:`~pantr.bezier`, :mod:`~pantr.bspline`, and :mod:`~pantr.cad` packages.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


def _reverse_control_points(
    ctrl: npt.NDArray[np.float32 | np.float64],
    direction: int,
    *,
    in_place: bool,
) -> npt.NDArray[np.float32 | np.float64]:
    """Reverse control points along a parametric direction.

    Args:
        ctrl (npt.NDArray[np.float32 | np.float64]): Control point array with
            shape ``(*num_basis, rank)``.
        direction (int): Parametric axis to reverse (must be valid).
        in_place (bool): If ``True``, flip the array in place and return it.
            If ``False``, return a new (possibly non-contiguous) flipped array.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The reversed control points.
    """
    if in_place:
        idx = [slice(None)] * ctrl.ndim
        idx[direction] = slice(None, None, -1)
        ctrl[:] = ctrl[tuple(idx)]
        return ctrl

    return np.flip(ctrl, axis=direction)


def _permute_control_points(
    ctrl: npt.NDArray[np.float32 | np.float64],
    permutation: Sequence[int],
    dim: int,
) -> npt.NDArray[np.float32 | np.float64]:
    """Reorder control-point axes according to a permutation.

    Transposing changes the array shape, so a new contiguous array is always
    returned; callers performing an in-place permutation assign the result
    back to their own buffer.

    Args:
        ctrl (npt.NDArray[np.float32 | np.float64]): Control point array with
            shape ``(*num_basis, rank)``.
        permutation (Sequence[int]): A permutation of ``range(dim)``.
        dim (int): Number of parametric dimensions (rank axis is ``dim``).

    Returns:
        npt.NDArray[np.float32 | np.float64]: The permuted control points
        (contiguous).
    """
    return np.ascontiguousarray(np.transpose(ctrl, [*permutation, dim]))


def _append_unit_weight_column(
    control_points: npt.NDArray[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    """Append a trailing column of unit weights (homogeneous promotion).

    Converts Euclidean control points of shape ``(*num_basis, rank)`` into
    rational (homogeneous) control points of shape ``(*num_basis, rank + 1)``
    whose weights are all ``1``.

    Args:
        control_points (npt.NDArray[np.float32 | np.float64]): Control points of
            shape ``(*num_basis, rank)``.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Control points with a trailing
        unit-weight column, shape ``(*num_basis, rank + 1)``.
    """
    weights = np.ones((*control_points.shape[:-1], 1), dtype=control_points.dtype)
    return np.concatenate([control_points, weights], axis=-1)
