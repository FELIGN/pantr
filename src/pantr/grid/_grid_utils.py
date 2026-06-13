"""Shared Layer-2 helpers for the grid package.

Re-exports the ``float64`` coercion helper from :mod:`pantr.geometry` (so the
implementation lives in a single place) and provides small grid-local helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..geometry import _as_float64

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["_as_float64", "_mask_nonfinite_locate"]


def _mask_nonfinite_locate(pts: npt.NDArray[np.floating[Any]], out: npt.NDArray[np.int64]) -> None:
    """Mark located cell ids for non-finite query points as ``-1`` (outside).

    The locate kernels' binary search has no NaN/inf handling (such comparisons
    are all ``False``, silently landing in cell 0), so non-finite rows are
    masked out here.

    Args:
        pts (npt.NDArray[np.floating[Any]]): Query points, shape ``(n, ndim)``.
        out (npt.NDArray[np.int64]): Located cell ids, shape ``(n,)``; modified
            in place.
    """
    finite = np.isfinite(pts).all(axis=1)
    if not finite.all():
        out[~finite] = -1
