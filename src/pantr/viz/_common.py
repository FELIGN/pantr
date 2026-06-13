"""Shared internal constants and helpers for the visualization package."""

from __future__ import annotations

import numpy as np
from numpy import typing as npt

_MAX_PHYSICAL_DIM = 3
"""Maximum physical dimension for VTK coordinates."""


def _pad_points_to_3d(
    coords: npt.NDArray[np.float32 | np.float64], rank: int
) -> npt.NDArray[np.float64]:
    """Embed ``coords`` into a zero-padded ``(n_pts, 3)`` float64 array.

    VTK always works in 3D, so lower-rank coordinates are placed in the leading
    columns and the remainder left zero. Inputs are upcast to ``float64``.

    Args:
        coords (npt.NDArray[np.float32 | np.float64]): Points of shape
            ``(n_pts, >= rank)``.
        rank (int): Number of leading columns to copy (the geometric rank).

    Returns:
        npt.NDArray[np.float64]: Array of shape ``(n_pts, 3)`` whose first
        ``rank`` columns come from ``coords`` and whose remaining columns are 0.
    """
    pts_3d = np.zeros((coords.shape[0], _MAX_PHYSICAL_DIM), dtype=np.float64)
    pts_3d[:, :rank] = coords[:, :rank]
    return pts_3d


def _project_homogeneous(
    flat_cp: npt.NDArray[np.float32 | np.float64], is_rational: bool
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None]:
    """Split flat control points into Euclidean coords and homogeneous weights.

    For rational geometries the leading columns are divided by the trailing
    weight column; non-rational input is returned unchanged with ``None``
    weights. Inputs are upcast to ``float64``.

    Args:
        flat_cp (npt.NDArray[np.float32 | np.float64]): Flattened control points
            of shape ``(n_pts, rank)`` (non-rational) or ``(n_pts, rank + 1)``
            (rational; last column is the homogeneous weight).
        is_rational (bool): Whether the last column is a homogeneous weight.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None]:
        ``(coords, weights)`` — ``coords`` has shape ``(n_pts, rank)``; ``weights``
        is ``(n_pts,)`` for rational input, else ``None``.
    """
    flat = flat_cp.astype(np.float64, copy=False)
    if not is_rational:
        return flat, None
    weights = flat[:, -1].copy()
    coords = flat[:, :-1] / weights[:, np.newaxis]
    return coords, weights
