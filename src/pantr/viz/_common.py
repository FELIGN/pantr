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
