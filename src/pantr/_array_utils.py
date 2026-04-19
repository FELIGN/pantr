"""Shared array-manipulation helpers for Layer 2 modules.

This module centralizes recurring NumPy array-reshape idioms that would
otherwise be duplicated across the ``bezier`` and ``bspline`` packages.
All helpers are private (not part of the public API).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


def _flatten_along_axis(
    arr: npt.NDArray[np.float32 | np.float64],
    axis: int,
) -> tuple[npt.NDArray[np.float32 | np.float64], tuple[int, ...]]:
    """Move ``axis`` to position 0 and flatten trailing dimensions.

    Returns a C-contiguous 2-D view suitable for passing to Numba kernels
    that operate along a single axis, together with the trailing shape
    needed to invert the operation via :func:`_unflatten_along_axis`.

    Args:
        arr (npt.NDArray[np.float32 | np.float64]): Input N-D array.
        axis (int): Axis to be moved to the leading position.

    Returns:
        tuple: ``(pts_2d, trailing_shape)`` where ``pts_2d`` has shape
        ``(arr.shape[axis], prod(trailing_shape))`` and is C-contiguous,
        and ``trailing_shape`` is the tuple of dimensions that were
        flattened.
    """
    moved = np.moveaxis(arr, axis, 0)
    return np.ascontiguousarray(moved.reshape(moved.shape[0], -1)), moved.shape[1:]


def _unflatten_along_axis(
    pts_2d: npt.NDArray[np.float32 | np.float64],
    trailing_shape: tuple[int, ...],
    axis: int,
) -> npt.NDArray[np.float32 | np.float64]:
    """Invert :func:`_flatten_along_axis`.

    Reshape a 2-D kernel output to ``(pts_2d.shape[0], *trailing_shape)``
    and move the leading axis back to position ``axis``.

    Args:
        pts_2d (npt.NDArray[np.float32 | np.float64]): 2-D array produced
            by a kernel operating along the leading axis.
        trailing_shape (tuple[int, ...]): Trailing shape returned by
            :func:`_flatten_along_axis`. ``pts_2d.shape[0]`` may differ
            from the original leading extent.
        axis (int): Target axis position (same value used when calling
            :func:`_flatten_along_axis`).

    Returns:
        npt.NDArray[np.float32 | np.float64]: N-D array with axis ordering
        restored.
    """
    return np.moveaxis(pts_2d.reshape(pts_2d.shape[0], *trailing_shape), 0, axis)
