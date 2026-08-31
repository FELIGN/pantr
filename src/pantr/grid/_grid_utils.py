"""Shared Layer-2 helpers for the grid package.

Re-exports the ``float64`` coercion helper from :mod:`pantr.geometry` (so the
implementation lives in a single place) and provides small grid-local helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._backend import Backend, active_backend, available_backends
from ..geometry import _as_float64

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = [
    "_as_float64",
    "_mask_nonfinite_locate",
    "_normalize_point",
    "_python_backend_selected",
]


def _python_backend_selected() -> bool:
    """Report whether a wrapped grid type should hold its Python implementation.

    The types this package owns -- :class:`~pantr.grid.CellTags`,
    :class:`~pantr.grid.FacetTags`, :class:`~pantr.grid.Partition` and
    :class:`~pantr.grid.BVH` -- all choose an implementation the same way, so the
    policy is written once here rather than once per type. Each type keeps its own
    ``_impl_class``, because only that function knows which two classes are on
    offer; what is shared is the question, not the answer.

    **The choice is per process, not per instance**, and that is load-bearing:
    :mod:`pantr.geometry` argues it at length for :class:`~pantr.geometry.AABB` and
    the argument carries over unchanged. Two objects built under different backends
    could otherwise meet in one operation, and reconciling them would mean
    converting one implementation into the other -- the shape
    ``design/cross_backend_types.md`` forbids.

    Returns:
        bool: ``True`` under the Python backend, ``False`` when the C++ one is
        selected and available.

    Raises:
        RuntimeError: If the C++ backend is selected and is not available. An
            explicit request never falls back silently.
    """
    if active_backend() is Backend.PYTHON:
        return True
    if Backend.CPP not in available_backends():
        raise RuntimeError("the CPP backend is not available in this installation")
    return False


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


def _normalize_point(pt: npt.ArrayLike, ndim: int, *, name: str = "pt") -> npt.NDArray[np.float64]:
    """Coerce a single query point to a contiguous length-``ndim`` ``float64`` array.

    The singular counterpart of :meth:`pantr.grid.Grid._normalize_points`, and it
    lives here rather than beside that one so that the wrapper and the Python grid it
    wraps share a definition: both check the shape and both report it in the same
    sentence, so the message cannot drift between the two backends.

    Args:
        pt (npt.ArrayLike): The point.
        ndim (int): The grid's spatial dimension.
        name (str): The parameter's name, for the message. Defaults to ``"pt"``.

    Returns:
        npt.NDArray[np.float64]: A C-contiguous array of shape ``(ndim,)``.

    Raises:
        ValueError: If ``pt`` does not have exactly ``ndim`` entries.
        TypeError: If ``pt`` cannot be cast to ``float64``.
    """
    arr = _as_float64(pt, name=name).ravel()
    if arr.shape != (ndim,):
        raise ValueError(f"{name} must have shape ({ndim},); got {arr.shape}.")
    return np.ascontiguousarray(arr)
