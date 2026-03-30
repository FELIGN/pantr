"""Shared utilities for interpolation and fitting modules.

Provides constants and helpers used by both
:mod:`pantr.bezier._bezier_interpolate` and
:mod:`pantr.bspline._bspline_interpolate`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

SVD_TOL_FACTOR: float = 100.0
"""Factor multiplied by machine epsilon for default SVD truncation tolerance."""


def split_components(
    values: npt.NDArray[np.floating[Any]],
    grid_shape: tuple[int, ...],
) -> list[npt.NDArray[np.floating[Any]]]:
    """Split function values into per-component arrays.

    For scalar-valued data (shape matches *grid_shape* exactly), returns a
    single-element list. For vector-valued data (shape is
    ``(*grid_shape, rank)``), returns one array per trailing component.

    Args:
        values (npt.NDArray[np.floating[Any]]): Function output array.
        grid_shape (tuple[int, ...]): Expected grid shape
            ``(n_0, n_1, ...)``.

    Returns:
        list[npt.NDArray[np.floating[Any]]]: One array per output component,
        each with shape ``grid_shape``.

    Raises:
        ValueError: If *values* shape is incompatible with *grid_shape*.
    """
    if values.shape == grid_shape:
        return [values]
    if values.shape[: len(grid_shape)] == grid_shape and values.ndim == len(grid_shape) + 1:
        return [values[..., r] for r in range(values.shape[-1])]
    raise ValueError(
        f"Values have shape {values.shape}, expected {grid_shape} "
        f"(scalar) or {(*grid_shape, 'rank')} (vector)."
    )
