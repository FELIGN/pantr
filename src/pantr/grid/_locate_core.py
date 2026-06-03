"""Layer-3 Numba kernel for batch point location on a tensor-product grid.

Locating a point on an axis-aligned tensor-product grid reduces to one
independent binary search per axis: the per-axis cell index is the interval of
the axis knot vector that contains the coordinate, and the flat cell id is the
row-major (C-order) combination of the per-axis indices.

The kernel processes a batch of points in parallel. Per-axis knot vectors of
differing lengths are passed as a single concatenated ``float64`` array plus
per-axis start offsets and cell counts, which keeps the signature Numba-friendly
(no ragged list of arrays).

A point exactly on an interior breakpoint is assigned to the lower-indexed cell
sharing that face; a point on the outer boundary is assigned to the adjacent
boundary cell. Points outside the grid domain map to ``-1``.

Note:
    Inputs are assumed to be correct (no validation performed).
    For general use, call :meth:`pantr.grid.TensorProductGrid.locate_many`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._numba_compat import nb_jit, nb_prange

if TYPE_CHECKING:
    import numpy.typing as npt


@nb_jit(nopython=True, cache=True, parallel=True)
def _locate_points_core(  # noqa: PLR0913 -- flat tensor-product grid descriptor
    points: npt.NDArray[np.float64],
    knots_flat: npt.NDArray[np.float64],
    knot_starts: npt.NDArray[np.int64],
    cells_per_axis: npt.NDArray[np.int64],
    strides: npt.NDArray[np.int64],
    out: npt.NDArray[np.int64],
) -> None:
    """Locate a batch of points on an axis-aligned tensor-product grid.

    Args:
        points (npt.NDArray[np.float64]): Query points, shape ``(npts, ndim)``.
        knots_flat (npt.NDArray[np.float64]): All per-axis knot vectors
            concatenated end to end; axis ``d`` occupies
            ``knots_flat[knot_starts[d] : knot_starts[d] + cells_per_axis[d] + 1]``
            and must be strictly increasing.
        knot_starts (npt.NDArray[np.int64]): Per-axis start offset into
            ``knots_flat``. Shape ``(ndim,)``.
        cells_per_axis (npt.NDArray[np.int64]): Per-axis cell counts. Shape
            ``(ndim,)``.
        strides (npt.NDArray[np.int64]): Per-axis C-order flat strides. Shape
            ``(ndim,)``.
        out (npt.NDArray[np.int64]): Output flat cell ids, shape ``(npts,)``;
            ``-1`` for points outside the grid domain.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :meth:`pantr.grid.TensorProductGrid.locate_many`.
    """
    npts = points.shape[0]
    ndim = points.shape[1]
    for p in nb_prange(npts):
        cid = 0
        inside = True
        for d in range(ndim):
            start = knot_starts[d]
            ncells = cells_per_axis[d]
            x = points[p, d]
            if x < knots_flat[start] or x > knots_flat[start + ncells]:
                inside = False
                break
            # lower_bound: first index idx with knots[idx] >= x.
            lo_i = 0
            hi_i = ncells + 1
            while lo_i < hi_i:
                mid = (lo_i + hi_i) // 2
                if knots_flat[start + mid] < x:
                    lo_i = mid + 1
                else:
                    hi_i = mid
            cell_i = lo_i - 1
            if cell_i < 0:
                cell_i = 0
            elif cell_i > ncells - 1:
                cell_i = ncells - 1
            cid += cell_i * strides[d]
        out[p] = cid if inside else -1


def _warmup_numba_functions() -> None:
    """Trigger compilation of the locate kernel on a tiny single-cell grid.

    Provided for explicit, on-demand warmup (for example in benchmarks). It is
    deliberately *not* called from :mod:`pantr`'s import-time warmup: the grid
    kernels compile lazily on first use and are cached to disk by Numba's
    ``cache=True``.
    """
    points = np.array([[0.5]], dtype=np.float64)
    knots_flat = np.array([0.0, 1.0], dtype=np.float64)
    knot_starts = np.zeros(1, dtype=np.int64)
    cells_per_axis = np.ones(1, dtype=np.int64)
    strides = np.ones(1, dtype=np.int64)
    out = np.empty(1, dtype=np.int64)
    _locate_points_core(points, knots_flat, knot_starts, cells_per_axis, strides, out)


__all__ = ["_locate_points_core"]
