"""Row-major (C-order) cell-index helpers for tensor-product grids.

A tensor-product grid with per-axis cell counts ``cells_per_axis`` numbers its
cells in row-major (C) order: the *last* axis varies fastest, matching
:func:`numpy.ravel_multi_index` / :func:`numpy.unravel_index` with their default
``order="C"``. This is the same convention used by
:class:`pantr.bspline.SpanwiseElementExtraction` (whose flat element index ``f``
maps to ``numpy.unravel_index(f, num_intervals)``), so a grid and an extraction
operator built on the same :class:`pantr.bspline.BsplineSpace` agree on cell
ids without any reindexing.

The helpers here centralize that convention so the rest of the package never
hard-codes a strides formula.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt


def c_order_strides(cells_per_axis: Sequence[int]) -> npt.NDArray[np.int64]:
    """Return the row-major (C-order) flat strides for ``cells_per_axis``.

    The flat id of multi-index ``(i_0, ..., i_{d-1})`` is
    ``sum_k i_k * strides[k]`` with ``strides[d-1] == 1`` and
    ``strides[k] == prod(cells_per_axis[k+1:])``.

    Args:
        cells_per_axis (Sequence[int]): Per-axis cell counts.

    Returns:
        npt.NDArray[np.int64]: Length-``d`` array of C-order strides.
    """
    d = len(cells_per_axis)
    strides = np.empty(d, dtype=np.int64)
    stride = 1
    for k in range(d - 1, -1, -1):
        strides[k] = stride
        stride *= int(cells_per_axis[k])
    return strides


def flat_to_multi(flat: int, cells_per_axis: Sequence[int]) -> tuple[int, ...]:
    """Convert a flat cell id to its per-axis multi-index (C-order).

    Args:
        flat (int): Flat cell id in ``[0, prod(cells_per_axis))``.
        cells_per_axis (Sequence[int]): Per-axis cell counts.

    Returns:
        tuple[int, ...]: Length-``d`` per-axis index tuple.

    Raises:
        IndexError: If ``flat`` is out of range.
    """
    total = 1
    for n in cells_per_axis:
        total *= int(n)
    if not 0 <= int(flat) < total:
        raise IndexError(f"flat cell id {flat!r} is out of range [0, {total}).")
    multi = np.unravel_index(int(flat), tuple(int(n) for n in cells_per_axis))
    return tuple(int(i) for i in multi)


def multi_to_flat(multi: Sequence[int], cells_per_axis: Sequence[int]) -> int:
    """Convert a per-axis multi-index to its flat cell id (C-order).

    Args:
        multi (Sequence[int]): Length-``d`` per-axis index sequence; each entry
            must satisfy ``0 <= i_k < cells_per_axis[k]``.
        cells_per_axis (Sequence[int]): Per-axis cell counts.

    Returns:
        int: Flat cell id.

    Raises:
        ValueError: If ``len(multi) != len(cells_per_axis)``.
        IndexError: If any per-axis index is out of range.
    """
    if len(multi) != len(cells_per_axis):
        raise ValueError(f"multi-index has length {len(multi)}; expected {len(cells_per_axis)}.")
    for k, (i, n) in enumerate(zip(multi, cells_per_axis, strict=True)):
        if not 0 <= int(i) < int(n):
            raise IndexError(f"axis {k} index {int(i)} out of range [0, {int(n)}).")
    flat = np.ravel_multi_index(tuple(int(i) for i in multi), tuple(int(n) for n in cells_per_axis))
    return int(flat)


__all__ = ["c_order_strides", "flat_to_multi", "multi_to_flat"]
