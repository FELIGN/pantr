"""Shared control-point manipulation helpers for reverse and permute operations.

These functions operate purely on control-point arrays and are used by both
:class:`~pantr.bezier.Bezier` and :class:`~pantr.bspline.Bspline`.
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
