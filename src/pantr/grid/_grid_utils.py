"""Shared Layer-2 helpers for the grid package.

Small validation utilities used across :mod:`pantr.grid`. Kept package-local so
the grid layer is self-contained and does not reach into another module's
private API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


def _as_float64(arr: npt.ArrayLike, *, name: str) -> npt.NDArray[np.float64]:
    """Coerce ``arr`` to a ``float64`` array, preserving rank.

    Integer and unsigned-integer inputs are cast to ``float64``; boolean and
    non-numeric inputs are rejected. Results of rank ``>= 1`` are made
    C-contiguous.

    Args:
        arr (npt.ArrayLike): Input array or array-like.
        name (str): Argument name, used in error messages.

    Returns:
        npt.NDArray[np.float64]: A ``float64`` view or copy of ``arr``. May
        alias the input when it is already C-contiguous ``float64``; treat as
        read-only.

    Raises:
        TypeError: If ``arr`` cannot be converted to an ndarray, or its dtype is
            neither integer nor floating-point (e.g. boolean, complex, object).
    """
    try:
        a = np.asarray(arr)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} could not be converted to an ndarray: {exc}") from exc
    if a.dtype.kind not in ("f", "i", "u"):
        raise TypeError(f"{name} must have a numeric (int or float) dtype; got {a.dtype!r}.")
    a = a.astype(np.float64, copy=False)
    if a.ndim >= 1 and not a.flags.c_contiguous:
        a = np.ascontiguousarray(a)
    return a
