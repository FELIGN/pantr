"""Tensor-product grid with per-axis breakpoint vectors.

:class:`TensorProductGrid` is the first-class concrete :class:`pantr.grid.Grid`:
a tensor product of axis-aligned boxes, stored as one strictly-increasing
breakpoint array per axis. A cell is addressed by a flat row-major (C-order)
identifier; see :mod:`pantr.grid._cell_index` for the convention, which matches
:class:`pantr.bspline.SpanwiseElementExtraction` so a grid and an extraction
operator built on the same :class:`pantr.bspline.BsplineSpace` agree on cell ids.

Footprint
---------

The grid stores only references to the per-axis breakpoint arrays (total size
``sum_k (cells_per_axis[k] + 1)``) plus a handful of small metadata arrays. It
never materializes per-cell bounds or connectivity -- those are computed on
demand. The only ``O(num_cells)`` structure, the :class:`pantr.grid.BVH` behind
:meth:`~pantr.grid.Grid.query_aabb`, is built lazily on first query, so a grid
used purely for geometry (the common case for a B-spline knot grid) stays
proportional to the breakpoints, not to the cell count.

Construction
------------

- ``TensorProductGrid(breakpoints)`` -- from explicit per-axis breakpoint arrays.
- :func:`uniform_grid` -- a uniform grid on a bounding box with given per-axis
  cell counts.
- :func:`tensor_product_grid` -- the knot-span grid of a
  :class:`pantr.bspline.BsplineSpace` (its per-axis unique in-domain knots).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import numpy as np

from ._cell_index import c_order_strides, flat_to_multi, multi_to_flat
from ._grid import Grid
from ._grid_utils import _as_float64
from ._locate_core import _locate_points_core

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from ..bspline import BsplineSpace

# Relative tolerance for detecting uniform per-axis spacing. Applied as
# ``ptp(diff) < _UNIFORM_SPACING_RTOL * mean(diff)`` so the check scales with
# the actual breakpoint spacing and works correctly for grids at any physical
# scale (e.g. millimetre meshes or kilometre-scale domains).
_UNIFORM_SPACING_RTOL: Final[float] = 1e-10
# Smallest useful grid: at least one cell per axis (two breakpoints).
_MIN_BREAKPOINTS_PER_AXIS: Final[int] = 2


def _is_axis_uniform(bp: npt.NDArray[np.float64]) -> bool:
    """Return whether the breakpoint vector ``bp`` has uniform spacing.

    A single-interval axis (two breakpoints) is vacuously uniform. Longer
    axes use a relative tolerance: ``ptp(diff) < RTOL * mean(diff)``.

    Args:
        bp (npt.NDArray[np.float64]): Strictly increasing breakpoint vector,
            length ``>= 2``.

    Returns:
        bool: ``True`` iff the spacing is uniform to within
        ``_UNIFORM_SPACING_RTOL``.
    """
    if bp.shape[0] <= 2:  # noqa: PLR2004
        return True
    diff = np.diff(bp)
    return bool(float(np.ptp(diff)) < _UNIFORM_SPACING_RTOL * float(diff.mean()))


class TensorProductGrid(Grid):
    """Tensor-product grid of axis-aligned boxes with per-axis breakpoints.

    Cells are numbered in row-major (C) order over :attr:`cells_per_axis` (last
    axis varies fastest). See the module docstring for the footprint and
    construction notes. Size and geometry metadata are exposed through the
    :attr:`ndim`, :attr:`num_cells`, :attr:`cells_per_axis`, :attr:`breakpoints`,
    and :attr:`bounds` properties.
    """

    __slots__ = (
        "_bounds",
        "_breakpoints",
        "_cells_per_axis",
        "_is_uniform",
        "_ndim",
        "_num_cells",
        "_strides",
    )

    def __init__(self, breakpoints: Sequence[npt.ArrayLike]) -> None:
        """Build a tensor-product grid from per-axis breakpoint vectors.

        Args:
            breakpoints (Sequence[npt.ArrayLike]): One strictly increasing
                ``float64`` array-like per axis, each of length
                ``cells_per_axis[d] + 1 >= 2``.

        Raises:
            ValueError: If ``breakpoints`` is empty, any axis has fewer than two
                entries, or any axis is non-finite or not strictly increasing.
            TypeError: If a breakpoint array cannot be cast to ``float64``.
        """
        super().__init__()
        ndim = len(breakpoints)
        if ndim < 1:
            raise ValueError(f"TensorProductGrid needs at least one axis; got {ndim}.")
        validated: list[npt.NDArray[np.float64]] = []
        cells_per_axis: list[int] = []
        for d, bp in enumerate(breakpoints):
            arr = _as_float64(bp, name=f"breakpoints[{d}]").ravel()
            if arr.shape[0] < _MIN_BREAKPOINTS_PER_AXIS:
                raise ValueError(
                    f"breakpoints[{d}] must have at least {_MIN_BREAKPOINTS_PER_AXIS} entries "
                    f"(>= 1 cell); got shape {arr.shape}."
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"breakpoints[{d}] must contain only finite values.")
            if not np.all(np.diff(arr) > 0.0):
                raise ValueError(f"breakpoints[{d}] must be strictly increasing.")
            frozen = np.ascontiguousarray(arr.copy(), dtype=np.float64)
            frozen.flags.writeable = False
            validated.append(frozen)
            cells_per_axis.append(int(frozen.shape[0]) - 1)
        self._ndim = ndim
        self._breakpoints = tuple(validated)
        self._cells_per_axis = tuple(cells_per_axis)
        self._num_cells = math.prod(self._cells_per_axis)
        bounds = np.empty((ndim, 2), dtype=np.float64)
        for d in range(ndim):
            bounds[d, 0] = self._breakpoints[d][0]
            bounds[d, 1] = self._breakpoints[d][-1]
        bounds.flags.writeable = False
        self._bounds = bounds
        self._strides = c_order_strides(self._cells_per_axis)
        self._strides.flags.writeable = False
        self._is_uniform = all(_is_axis_uniform(bp) for bp in self._breakpoints)

    # ------------------------------------------------------------------
    # Read-only attributes
    # ------------------------------------------------------------------

    @property
    def ndim(self) -> int:
        """Get the spatial dimension of the grid.

        Returns:
            int: Number of axes (``>= 1``).
        """
        return self._ndim

    @property
    def num_cells(self) -> int:
        """Get the total number of cells.

        Returns:
            int: Product of the per-axis cell counts.
        """
        return self._num_cells

    @property
    def cells_per_axis(self) -> tuple[int, ...]:
        """Get the per-axis cell counts.

        Returns:
            tuple[int, ...]: Length-``ndim`` tuple of per-axis counts.
        """
        return self._cells_per_axis

    @property
    def breakpoints(self) -> tuple[npt.NDArray[np.float64], ...]:
        """Get the per-axis strictly increasing breakpoint arrays.

        Returns:
            tuple[npt.NDArray[np.float64], ...]: Read-only ``float64`` arrays;
            ``breakpoints[d]`` has length ``cells_per_axis[d] + 1``.
        """
        return self._breakpoints

    @property
    def bounds(self) -> npt.NDArray[np.float64]:
        """Get the per-axis ``[lo, hi]`` extremes.

        Returns:
            npt.NDArray[np.float64]: Read-only ``(ndim, 2)`` array.
        """
        return self._bounds

    @property
    def is_uniform(self) -> bool:
        """Get whether every axis has uniform breakpoint spacing.

        Returns:
            bool: ``True`` iff each axis's spacing is constant to within a
            relative tolerance (``_UNIFORM_SPACING_RTOL``).
        """
        return self._is_uniform

    # ------------------------------------------------------------------
    # Index helpers (row-major C-order)
    # ------------------------------------------------------------------

    def cell_multi_index(self, cid: int) -> tuple[int, ...]:
        """Return the per-axis indices ``(i_0, ..., i_{ndim-1})`` of cell ``cid``.

        Args:
            cid (int): Flat cell identifier.

        Returns:
            tuple[int, ...]: Length-``ndim`` per-axis index tuple (C-order).

        Raises:
            IndexError: If ``cid`` is out of range.
        """
        return flat_to_multi(int(cid), self._cells_per_axis)

    def flat_cell_index(self, multi: Sequence[int]) -> int:
        """Map per-axis cell indices to the flat cell identifier (C-order).

        Args:
            multi (Sequence[int]): Length-``ndim`` per-axis indices; each entry
                must satisfy ``0 <= i_k < cells_per_axis[k]``.

        Returns:
            int: Flat cell identifier.

        Raises:
            ValueError: If ``len(multi) != ndim``.
            IndexError: If any per-axis index is out of range.
        """
        return multi_to_flat(multi, self._cells_per_axis)

    # ------------------------------------------------------------------
    # Grid contract overrides
    # ------------------------------------------------------------------

    def cell_bounds(
        self,
        cid: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return cell ``cid``'s axis-aligned ``(lo, hi)`` corners.

        Args:
            cid (int): Cell identifier.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Fresh,
            writeable length-``ndim`` ``float64`` arrays.

        Raises:
            IndexError: If ``cid`` is out of range.
        """
        multi = self.cell_multi_index(cid)
        lo = np.empty(self._ndim, dtype=np.float64)
        hi = np.empty(self._ndim, dtype=np.float64)
        for d, i in enumerate(multi):
            lo[d] = self._breakpoints[d][i]
            hi[d] = self._breakpoints[d][i + 1]
        return lo, hi

    def locate(self, pt: npt.ArrayLike) -> int | None:
        """Return the cell containing ``pt``, or ``None`` if ``pt`` is outside.

        A point exactly on an interior breakpoint is assigned to the
        lower-indexed cell sharing that face; a point on the outer boundary is
        assigned to the adjacent boundary cell.

        Args:
            pt (npt.ArrayLike): Length-``ndim`` point.

        Returns:
            int | None: Containing cell id, or ``None`` when ``pt`` lies outside
            the grid domain.

        Raises:
            ValueError: If ``pt`` does not have length ``ndim``.
        """
        arr = _as_float64(pt, name="pt").ravel()
        if arr.shape != (self._ndim,):
            raise ValueError(f"pt must have shape ({self._ndim},); got {arr.shape}.")
        cid = 0
        for d in range(self._ndim):
            bp = self._breakpoints[d]
            x = float(arr[d])
            if x < bp[0] or x > bp[-1]:
                return None
            idx = int(np.searchsorted(bp, x, side="left")) - 1
            idx = min(max(idx, 0), self._cells_per_axis[d] - 1)
            cid += idx * int(self._strides[d])
        return cid

    def locate_many(self, points: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """Locate a batch of points via the Numba per-axis search kernel.

        Args:
            points (npt.ArrayLike): ``(npts, ndim)`` array-like of points, or a
                single length-``ndim`` point.

        Returns:
            npt.NDArray[np.int64]: Shape ``(npts,)`` cell ids; ``-1`` for points
            outside the grid.

        Raises:
            ValueError: If the trailing axis of ``points`` is not ``ndim``.
        """
        pts = self._normalize_points(points)
        counts = np.array([bp.shape[0] for bp in self._breakpoints], dtype=np.int64)
        knot_starts = np.zeros(self._ndim, dtype=np.int64)
        knot_starts[1:] = np.cumsum(counts[:-1])
        knots_flat = np.concatenate(self._breakpoints).astype(np.float64, copy=False)
        cells_per_axis = np.array(self._cells_per_axis, dtype=np.int64)
        out = np.empty(pts.shape[0], dtype=np.int64)
        _locate_points_core(pts, knots_flat, knot_starts, cells_per_axis, self._strides, out)
        return out

    def neighbor_across_facet(self, cid: int, lfid: int) -> int | None:
        """Return the cell across local facet ``lfid`` of ``cid``, or ``None``.

        Uses the ``lfid = 2 * axis + side`` encoding and per-axis arithmetic.

        Args:
            cid (int): Cell identifier.
            lfid (int): Local facet identifier in ``[0, 2 * ndim)``.

        Returns:
            int | None: Neighbouring cell id, or ``None`` on a boundary facet.

        Raises:
            IndexError: If ``cid`` or ``lfid`` is out of range.
        """
        self._check_lfid(cid, lfid)
        axis, side = divmod(int(lfid), 2)
        multi = list(self.cell_multi_index(cid))
        i = multi[axis]
        if side == 0:
            if i == 0:
                return None
            multi[axis] = i - 1
        else:
            if i == self._cells_per_axis[axis] - 1:
                return None
            multi[axis] = i + 1
        return self.flat_cell_index(multi)

    def _collect_cell_bounds(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Materialize per-cell ``(lo, hi)`` in C-order via meshgrid and fancy indexing.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            ``(cell_lo, cell_hi)`` of shape ``(num_cells, ndim)`` in cell-id
            order.
        """
        lo_axes = [bp[:-1] for bp in self._breakpoints]
        hi_axes = [bp[1:] for bp in self._breakpoints]
        idx_grids = np.meshgrid(*[np.arange(n) for n in self._cells_per_axis], indexing="ij")
        cell_lo = np.empty((self._num_cells, self._ndim), dtype=np.float64)
        cell_hi = np.empty((self._num_cells, self._ndim), dtype=np.float64)
        for d in range(self._ndim):
            flat_idx = idx_grids[d].ravel()
            cell_lo[:, d] = lo_axes[d][flat_idx]
            cell_hi[:, d] = hi_axes[d][flat_idx]
        return cell_lo, cell_hi

    def __repr__(self) -> str:
        """Return a compact representation useful for debugging.

        Returns:
            str: ``"TensorProductGrid(ndim=..., cells_per_axis=..., uniform=...)"``
        """
        return (
            f"TensorProductGrid(ndim={self._ndim}, cells_per_axis={self._cells_per_axis}, "
            f"uniform={self._is_uniform})"
        )


def uniform_grid(
    bounds: npt.ArrayLike,
    cells: int | Sequence[int],
) -> TensorProductGrid:
    """Build a uniform :class:`TensorProductGrid` on a bounding box.

    Args:
        bounds (npt.ArrayLike): ``(ndim, 2)`` array-like of per-axis
            ``[lo, hi]`` pairs with ``ndim >= 1``.
        cells (int | Sequence[int]): Number of cells per axis. A scalar is
            broadcast to every axis; a length-``ndim`` sequence gives per-axis
            counts. Each count must be ``>= 1``.

    Returns:
        TensorProductGrid: A uniform tensor-product grid.

    Raises:
        ValueError: If ``bounds`` does not have shape ``(ndim, 2)``, any axis has
            ``lo >= hi``, ``cells`` has the wrong length, or any count is ``< 1``.
        TypeError: If ``bounds`` cannot be cast to ``float64``.
    """
    arr = _as_float64(bounds, name="bounds")
    if arr.ndim != 2 or arr.shape[-1] != 2:  # noqa: PLR2004
        raise ValueError(f"uniform_grid: bounds must have shape (ndim, 2); got {arr.shape}.")
    ndim = int(arr.shape[0])
    if ndim < 1:
        raise ValueError(f"uniform_grid: ndim must be >= 1; got {ndim}.")
    if not np.all(arr[:, 0] < arr[:, 1]):
        raise ValueError(
            f"uniform_grid: bounds must satisfy lo < hi on every axis; got {arr.tolist()!r}."
        )
    cells_tuple = (int(cells),) * ndim if isinstance(cells, int) else tuple(int(c) for c in cells)
    if len(cells_tuple) != ndim:
        raise ValueError(
            f"uniform_grid: cells must be a scalar or a length-{ndim} sequence; "
            f"got length {len(cells_tuple)}."
        )
    if any(c < 1 for c in cells_tuple):
        raise ValueError(f"uniform_grid: every cells entry must be >= 1; got {cells_tuple!r}.")
    breakpoints = [
        np.linspace(arr[d, 0], arr[d, 1], cells_tuple[d] + 1, dtype=np.float64) for d in range(ndim)
    ]
    return TensorProductGrid(breakpoints)


def tensor_product_grid(space: BsplineSpace) -> TensorProductGrid:
    """Build the knot-span grid of a :class:`pantr.bspline.BsplineSpace`.

    The grid's per-axis breakpoints are the unique in-domain knots of each
    1-D sub-space, so its cells are exactly the space's knot spans and its cell
    ids match :class:`pantr.bspline.SpanwiseElementExtraction` on the same space.

    Args:
        space (BsplineSpace): A tensor-product B-spline space. Periodic
            directions are rejected: a periodic knot vector does not map cleanly
            to a bounded grid.

    Returns:
        TensorProductGrid: A grid whose cells are the knot spans of ``space``.

    Raises:
        ValueError: If any direction of ``space`` is periodic.
    """
    breakpoints: list[npt.NDArray[np.float64]] = []
    for d, sub in enumerate(space.spaces):
        if sub.periodic:
            raise ValueError(
                f"tensor_product_grid: axis {d} is periodic; periodic B-spline spaces "
                "do not map to a bounded tensor-product grid."
            )
        unique, _ = sub.get_unique_knots_and_multiplicity(in_domain=True)
        breakpoints.append(_as_float64(unique, name=f"space.spaces[{d}] unique knots").ravel())
    return TensorProductGrid(breakpoints)


__all__ = ["TensorProductGrid", "tensor_product_grid", "uniform_grid"]
