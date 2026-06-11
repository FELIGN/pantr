"""Hierarchical structured grid with uniform per-direction subdivision.

A :class:`HierarchicalGrid` wraps a root :class:`TensorProductGrid` and adds a
hierarchy of refinement levels.  Each ``refine`` call promotes a rectangular
region of currently-active cells to the next level, splitting every cell in that
region into ``m_0 * ... * m_{d-1}`` equal children (the per-direction *factor*).

Design highlights
-----------------

*No per-cell storage.*  Only **rectangular blocks** are stored — at most one
small list of ``(lo, hi)`` index pairs per level.  Memory is therefore
``O(total_blocks * ndim)``, independent of the total cell count.

*Union semantics.*  Calling :meth:`refine` with a region that overlaps an
already-refined area is silently correct: only the currently-active portion of
the region is refined.  Since the children of newly-active cells are always
disjoint from existing level blocks, no deduplication is needed.

*No balance constraint.*  :meth:`refine` imposes no 2:1 grading: cells of any
two levels may share a facet.  Facet adjacency (:meth:`neighbor_across_facet`,
:meth:`hanging_neighbors`) walks as many levels up or down as the interface
requires.

Main exports:

- :class:`HierarchicalGrid`: hierarchical grid built on a
  :class:`TensorProductGrid`.
- :func:`hierarchical_grid`: standalone factory.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

from ._grid import Grid, GridRestriction
from ._grid_utils import _as_float64
from ._hier_core import (
    _decode_flat_id_core,
    _encode_midx_core,
    _hier_collect_cell_bounds_core,
    _hier_locate_points_core,
)
from ._tensor_product_grid import TensorProductGrid

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

# A Block is a pair (lo, hi) of per-direction inclusive-start / exclusive-end
# integer index tuples, all at the coordinate system of a specific level.
_Block = tuple[tuple[int, ...], tuple[int, ...]]


# ---------------------------------------------------------------------------
# Pure rectangle helpers (module-level for reuse in tests)
# ---------------------------------------------------------------------------


def _block_size(lo: tuple[int, ...], hi: tuple[int, ...]) -> int:
    """Return the number of cells in the integer rectangle ``[lo, hi)``.

    Args:
        lo (tuple[int, ...]): Per-direction start indices (inclusive).
        hi (tuple[int, ...]): Per-direction end indices (exclusive).

    Returns:
        int: Product of ``hi[k] - lo[k]`` for each axis ``k``.
    """
    s = 1
    for lo_k, h in zip(lo, hi, strict=False):
        s *= h - lo_k
    return s


def _in_block(
    midx: tuple[int, ...],
    lo: tuple[int, ...],
    hi: tuple[int, ...],
) -> bool:
    """Return ``True`` iff ``midx`` is inside ``[lo, hi)``.

    Args:
        midx (tuple[int, ...]): Per-direction index to test.
        lo (tuple[int, ...]): Block lower bound (inclusive).
        hi (tuple[int, ...]): Block upper bound (exclusive).

    Returns:
        bool: ``True`` iff ``lo[k] <= midx[k] < hi[k]`` for every ``k``.
    """
    return all(lo_k <= m < h for m, lo_k, h in zip(midx, lo, hi, strict=False))


def _rect_intersect(
    lo1: tuple[int, ...],
    hi1: tuple[int, ...],
    lo2: tuple[int, ...],
    hi2: tuple[int, ...],
) -> _Block | None:
    """Return the intersection of two rectangles, or ``None`` if disjoint.

    Args:
        lo1 (tuple[int, ...]): Lower bound of the first rectangle.
        hi1 (tuple[int, ...]): Upper bound of the first rectangle.
        lo2 (tuple[int, ...]): Lower bound of the second rectangle.
        hi2 (tuple[int, ...]): Upper bound of the second rectangle.

    Returns:
        _Block | None: ``(lo, hi)`` of the intersection, or ``None`` when
        the rectangles do not overlap.
    """
    lo = tuple(max(a, b) for a, b in zip(lo1, lo2, strict=False))
    hi = tuple(min(a, b) for a, b in zip(hi1, hi2, strict=False))
    if all(lo_k < h for lo_k, h in zip(lo, hi, strict=False)):
        return lo, hi
    return None


def _peel(
    outer_lo: tuple[int, ...],
    outer_hi: tuple[int, ...],
    inner_lo: tuple[int, ...],
    inner_hi: tuple[int, ...],
) -> list[_Block]:
    r"""Subtract ``[inner_lo, inner_hi)`` from ``[outer_lo, outer_hi)``.

    Assumes ``inner`` is fully contained in ``outer``.  Returns at most
    ``2 * ndim`` non-overlapping axis-aligned rectangular slabs that cover
    ``outer`` minus the ``inner`` region.  Empty slabs are omitted.

    Args:
        outer_lo (tuple[int, ...]): Outer rectangle lower bound.
        outer_hi (tuple[int, ...]): Outer rectangle upper bound.
        inner_lo (tuple[int, ...]): Inner rectangle lower bound.
        inner_hi (tuple[int, ...]): Inner rectangle upper bound.

    Returns:
        list[_Block]: Non-overlapping slabs covering ``outer \\ inner``.
    """
    slabs: list[_Block] = []
    a = list(outer_lo)
    b = list(outer_hi)
    for k in range(len(a)):
        if a[k] < inner_lo[k]:
            hi_k = list(b)
            hi_k[k] = inner_lo[k]
            slab: _Block = (tuple(a), tuple(hi_k))
            if _block_size(*slab) > 0:
                slabs.append(slab)
            a[k] = inner_lo[k]
        if inner_hi[k] < b[k]:
            lo_k = list(a)
            lo_k[k] = inner_hi[k]
            slab = (tuple(lo_k), tuple(b))
            if _block_size(*slab) > 0:
                slabs.append(slab)
            b[k] = inner_hi[k]
    return slabs


def _try_merge(
    lo1: tuple[int, ...],
    hi1: tuple[int, ...],
    lo2: tuple[int, ...],
    hi2: tuple[int, ...],
) -> _Block | None:
    """Merge two blocks that differ in exactly one axis and are adjacent there.

    Args:
        lo1 (tuple[int, ...]): First block lower bound.
        hi1 (tuple[int, ...]): First block upper bound.
        lo2 (tuple[int, ...]): Second block lower bound.
        hi2 (tuple[int, ...]): Second block upper bound.

    Returns:
        _Block | None: Merged block if the two are adjacent and aligned in all
        other axes, else ``None``.
    """
    merge_axis = -1
    for k in range(len(lo1)):
        if lo1[k] == lo2[k] and hi1[k] == hi2[k]:
            continue
        if merge_axis >= 0:
            return None
        if hi1[k] == lo2[k] or hi2[k] == lo1[k]:
            merge_axis = k
        else:
            return None
    if merge_axis < 0:
        return None
    lo = list(lo1)
    hi = list(hi1)
    lo[merge_axis] = min(lo1[merge_axis], lo2[merge_axis])
    hi[merge_axis] = max(hi1[merge_axis], hi2[merge_axis])
    return (tuple(lo), tuple(hi))


def _normalize_blocks(blocks: list[_Block]) -> list[_Block]:
    """Sort a block list and greedily merge adjacent aligned pairs.

    Args:
        blocks (list[_Block]): List of non-overlapping blocks (modified in
            place and returned sorted).

    Returns:
        list[_Block]: Sorted, compacted list of non-overlapping blocks.
    """
    if len(blocks) <= 1:
        return sorted(blocks)
    changed = True
    while changed:
        changed = False
        used = [False] * len(blocks)
        new_blocks: list[_Block] = []
        for i in range(len(blocks)):
            if used[i]:
                continue
            merged = blocks[i]
            for j in range(i + 1, len(blocks)):
                if used[j]:
                    continue
                result = _try_merge(*merged, *blocks[j])
                if result is not None:
                    merged = result
                    used[j] = True
                    changed = True
            new_blocks.append(merged)
        blocks = new_blocks
    return sorted(blocks)


# ---------------------------------------------------------------------------
# HierarchicalGrid
# ---------------------------------------------------------------------------


class HierarchicalGrid(Grid):
    """Hierarchical grid with a fixed per-direction uniform subdivision factor.

    Built on a root :class:`TensorProductGrid`.  Active cells are stored as
    non-overlapping rectangular *blocks* at each level; no per-cell data is
    kept.  A new level is created by :meth:`refine`; the grid starts with all
    root cells active at level 0.

    Flat cell ids are assigned level-by-level (level 0 first), block-by-block
    within a level (sorted by ``lo`` tuple), and C-order within each block.

    Attributes:
        _root (TensorProductGrid): The level-0 root grid.
        _factor (tuple[int, ...]): Per-direction subdivision factor (``>= 1``).
        _blocks (list[list[tuple[tuple[int, ...], tuple[int, ...]]]]): ``_blocks[l]``
            is a sorted list of non-overlapping ``(lo, hi)`` blocks of active
            cells at level ``l``.
        _level_base (list[int]): ``_level_base[l]`` is the flat-id base of
            level ``l``; length ``max_level + 2`` (includes sentinel).
        _num_cells (int): Cached total active cell count.
        _version (int): Monotonic mutation counter, incremented on every
            structural change (see :attr:`version`).
        _packed_block_lo (npt.NDArray[np.int64]): Packed block lower bounds for
            the Numba kernels, shape ``(n_blocks_total, ndim)``, concatenated
            level by level in flat-id order (see :mod:`pantr.grid._hier_core`).
        _packed_block_hi (npt.NDArray[np.int64]): Packed block upper bounds,
            same shape.
        _packed_block_base (npt.NDArray[np.int64]): Flat cell id of each
            block's first cell, shape ``(n_blocks_total,)``.
        _packed_level_start (npt.NDArray[np.int64]): Block index range of each
            level, shape ``(max_level + 2,)``.
        _root_knots_flat (npt.NDArray[np.float64]): Root per-axis breakpoints
            concatenated end to end (kernel descriptor).
        _root_knot_starts (npt.NDArray[np.int64]): Per-axis start offset into
            ``_root_knots_flat``, shape ``(ndim,)``.
    """

    __slots__ = (
        "_blocks",
        "_factor",
        "_level_base",
        "_num_cells",
        "_packed_block_base",
        "_packed_block_hi",
        "_packed_block_lo",
        "_packed_level_start",
        "_root",
        "_root_knot_starts",
        "_root_knots_flat",
        "_version",
    )

    def __init__(
        self,
        root: TensorProductGrid,
        factor: int | Sequence[int],
    ) -> None:
        """Create a hierarchical grid from a root and a subdivision factor.

        Args:
            root (TensorProductGrid): The level-0 grid.
            factor (int | Sequence[int]): Per-direction subdivision factor.
                A scalar is broadcast to every axis.  Each entry must be
                ``>= 1``; a factor of ``1`` on an axis prevents subdivision in
                that direction.

        Raises:
            TypeError: If ``root`` is not a :class:`TensorProductGrid`.
            ValueError: If ``factor`` has the wrong length or any entry is
                ``< 1``.
        """
        super().__init__()
        if not isinstance(root, TensorProductGrid):
            raise TypeError(f"root must be a TensorProductGrid; got {type(root).__name__!r}.")
        ndim = root.ndim
        if isinstance(factor, int):
            fac: tuple[int, ...] = (int(factor),) * ndim
        else:
            fac = tuple(int(f) for f in factor)
        if len(fac) != ndim:
            raise ValueError(
                f"factor must be a scalar or length-{ndim} sequence; got length {len(fac)}."
            )
        if any(f < 1 for f in fac):
            raise ValueError(
                f"every factor entry must be >= 1 (1 = no subdivision on that axis); got {fac!r}."
            )
        self._root = root
        self._factor = fac
        # Level 0: one block covering all root cells.
        lo0 = tuple(0 for _ in range(ndim))
        hi0 = root.cells_per_axis
        self._blocks: list[list[_Block]] = [[(lo0, hi0)]]
        self._level_base: list[int] = []
        self._num_cells: int = 0
        self._version: int = 0
        self._rebuild()

    @classmethod
    def _from_blocks(
        cls,
        root: TensorProductGrid,
        factor: tuple[int, ...],
        blocks: list[list[_Block]],
    ) -> HierarchicalGrid:
        """Build a grid directly from per-level block lists (internal constructor).

        Bypasses the public ``__init__`` (which starts with a single level-0 block
        spanning the whole root) to assemble an arbitrary, already-consistent
        active-leaf decomposition. Used by :meth:`restrict` to produce a windowed
        sub-grid. The per-level block lists are normalized (sorted and merged) and
        empty trailing levels are dropped.

        Args:
            root (TensorProductGrid): The level-0 root grid of the sub-hierarchy.
            factor (tuple[int, ...]): Per-direction subdivision factor.
            blocks (list[list[_Block]]): ``blocks[l]`` lists the active-leaf
                ``(lo, hi)`` rectangles at level ``l`` in level-``l`` coordinates.

        Returns:
            HierarchicalGrid: A grid whose active leaves span the same cells as
            ``blocks``, after greedy merging of adjacent aligned pairs.

        Note:
            Callers must supply a valid active-leaf decomposition: blocks at each
            level are non-overlapping, and the per-level sets collectively partition
            the root's cells consistently with ``factor`` (no gaps or overlaps across
            levels).
        """
        self = cls.__new__(cls)
        Grid.__init__(self)
        self._root = root
        self._factor = factor
        normalized = [_normalize_blocks(list(level_blocks)) for level_blocks in blocks]
        while len(normalized) > 1 and not normalized[-1]:
            normalized.pop()
        self._blocks = normalized
        self._level_base = []
        self._num_cells = 0
        self._version = 0
        self._rebuild()
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ndim(self) -> int:
        """Get the spatial dimension.

        Returns:
            int: Number of axes (``>= 1``).
        """
        return self._root.ndim

    @property
    def num_cells(self) -> int:
        """Get the total number of active cells across all levels.

        Returns:
            int: Total active cell count.
        """
        return self._num_cells

    @property
    def root(self) -> TensorProductGrid:
        """Get the level-0 root grid.

        Returns:
            TensorProductGrid: The root grid used to construct this hierarchy.
        """
        return self._root

    @property
    def factor(self) -> tuple[int, ...]:
        """Get the per-direction subdivision factor.

        Returns:
            tuple[int, ...]: Length-``ndim`` tuple; ``factor[k] >= 1``.
        """
        return self._factor

    @property
    def max_level(self) -> int:
        """Get the index of the deepest non-empty level.

        Returns:
            int: ``0`` before any refinement; increases with each :meth:`refine`
            call that adds a new level.
        """
        return len(self._blocks) - 1

    @property
    def version(self) -> int:
        """Get the monotonic mutation counter of this grid.

        Incremented on every structural change (:meth:`refine`, :meth:`coarsen`).
        Snapshot consumers (e.g. :class:`~pantr.bspline.THBSplineSpace`) compare
        it to detect *any* post-construction mutation -- ``max_level`` and
        ``num_cells`` alone cannot distinguish compensating refine/coarsen pairs.

        Returns:
            int: The current mutation count (``>= 1``).
        """
        return self._version

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        """Recompute ``_level_base``, ``_num_cells``, and reset the BVH/tags.

        Called after every structural change (construction or refinement).
        Bumps :attr:`version` so snapshot consumers can detect any mutation,
        including compensating refine/coarsen pairs that leave ``max_level``
        and ``num_cells`` unchanged.  Also repacks the block lists into the
        flat ``int64`` arrays consumed by the :mod:`pantr.grid._hier_core`
        kernels (``O(total_blocks)`` work).
        """
        base = 0
        level_base: list[int] = []
        for blocks_at_level in self._blocks:
            level_base.append(base)
            base += sum(_block_size(lo, hi) for lo, hi in blocks_at_level)
        level_base.append(base)  # sentinel for bisect_right
        self._level_base = level_base
        self._num_cells = base
        self._version += 1
        self._bvh = None
        self._cell_tags = None
        self._facet_tags = None

        # Pack the per-level block lists for the Numba kernels, in the same
        # order flat ids are assigned (level by level, block by block, C-order
        # within a block).
        ndim = self._root.ndim
        n_blocks = sum(len(blocks_at_level) for blocks_at_level in self._blocks)
        packed_lo = np.empty((n_blocks, ndim), dtype=np.int64)
        packed_hi = np.empty((n_blocks, ndim), dtype=np.int64)
        packed_base = np.empty(n_blocks, dtype=np.int64)
        level_start = np.empty(len(self._blocks) + 1, dtype=np.int64)
        b = 0
        cell_base = 0
        for level, blocks_at_level in enumerate(self._blocks):
            level_start[level] = b
            for lo, hi in blocks_at_level:
                packed_lo[b] = lo
                packed_hi[b] = hi
                packed_base[b] = cell_base
                cell_base += _block_size(lo, hi)
                b += 1
        level_start[len(self._blocks)] = b
        self._packed_block_lo = packed_lo
        self._packed_block_hi = packed_hi
        self._packed_block_base = packed_base
        self._packed_level_start = level_start

        # Root breakpoint descriptor for the kernels (root is immutable, but
        # rebuilding here keeps a single construction path).
        breakpoints = self._root.breakpoints
        counts = np.array([bp.shape[0] for bp in breakpoints], dtype=np.int64)
        knot_starts = np.zeros(ndim, dtype=np.int64)
        knot_starts[1:] = np.cumsum(counts[:-1])
        self._root_knots_flat = np.concatenate(breakpoints).astype(np.float64, copy=False)
        self._root_knot_starts = knot_starts

    def _decode_flat_id(self, cid: int) -> tuple[int, tuple[int, ...]]:
        """Convert a flat cell id to ``(level, multi_index)``.

        Backed by the :func:`~pantr.grid._hier_core._decode_flat_id_core`
        kernel over the packed block arrays.

        Args:
            cid (int): Flat cell identifier.

        Returns:
            tuple[int, tuple[int, ...]]: ``(level, multi_index)`` where
            ``multi_index`` uses level-``l`` integer coordinates.

        Raises:
            IndexError: If ``cid`` is out of range.
        """
        if not 0 <= cid < self._num_cells:
            raise IndexError(f"cell id {cid!r} is out of range [0, {self._num_cells}).")
        midx = np.empty(self._root.ndim, dtype=np.int64)
        level = _decode_flat_id_core(
            int(cid),
            self._packed_block_lo,
            self._packed_block_hi,
            self._packed_block_base,
            self._packed_level_start,
            midx,
        )
        return int(level), tuple(int(i) for i in midx)

    def _encode_midx(
        self,
        level: int,
        midx: tuple[int, ...],
    ) -> int | None:
        """Convert ``(level, multi_index)`` to a flat cell id, or ``None``.

        Returns ``None`` when the cell is not an active (leaf) cell.  Backed by
        the :func:`~pantr.grid._hier_core._encode_midx_core` kernel over the
        packed block arrays.

        Args:
            level (int): Hierarchy level.
            midx (tuple[int, ...]): Per-direction index in level-``level``
                coordinates.

        Returns:
            int | None: Flat cell id if ``(level, midx)`` is active, else
            ``None``.
        """
        if level >= len(self._blocks):
            return None
        cid = _encode_midx_core(
            int(level),
            np.asarray(midx, dtype=np.int64),
            self._packed_block_lo,
            self._packed_block_hi,
            self._packed_block_base,
            self._packed_level_start,
        )
        return None if cid < 0 else int(cid)

    def _cell_bounds_from_level_midx(
        self,
        level: int,
        midx: tuple[int, ...],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute ``(lo, hi)`` float bounds for a cell at ``(level, midx)``.

        Args:
            level (int): Hierarchy level.
            midx (tuple[int, ...]): Per-direction integer index.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Fresh
            writeable length-``ndim`` ``float64`` arrays.
        """
        lo = np.empty(self.ndim, dtype=np.float64)
        hi = np.empty(self.ndim, dtype=np.float64)
        for k, (ik, fk, bp) in enumerate(
            zip(midx, self._factor, self._root.breakpoints, strict=False)
        ):
            m_pow = fk**level
            root_ik = ik // m_pow
            sub_ik = ik % m_pow
            root_lo_k = float(bp[root_ik])
            root_hi_k = float(bp[root_ik + 1])
            size_k = (root_hi_k - root_lo_k) / m_pow
            lo[k] = root_lo_k + sub_ik * size_k
            hi[k] = lo[k] + size_k
        return lo, hi

    def _n_cells_at_level_k(self, level: int, axis: int) -> int:
        """Return the number of cells in ``axis`` at ``level``.

        Args:
            level (int): Hierarchy level.
            axis (int): Axis index.

        Returns:
            int: ``root.cells_per_axis[axis] * factor[axis] ** level``.
        """
        return int(self._root.cells_per_axis[axis]) * int(self._factor[axis] ** level)

    def _check_level(self, level: int) -> None:
        """Validate that ``level`` is an existing hierarchy level.

        Args:
            level (int): Hierarchy level to validate.

        Raises:
            ValueError: If ``level`` is outside ``[0, max_level]``.
        """
        if not (0 <= level <= self.max_level):
            raise ValueError(f"level must be in [0, {self.max_level}]; got {level!r}.")

    # ------------------------------------------------------------------
    # Grid contract
    # ------------------------------------------------------------------

    def cell_bounds(
        self,
        cid: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the axis-aligned ``(lo, hi)`` corners of cell ``cid``.

        Args:
            cid (int): Flat cell identifier.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Fresh,
            writeable length-``ndim`` ``float64`` arrays.

        Raises:
            IndexError: If ``cid`` is out of range.
        """
        self._check_cid(cid)
        level, midx = self._decode_flat_id(cid)
        return self._cell_bounds_from_level_midx(level, midx)

    def locate(self, pt: npt.ArrayLike) -> int | None:
        """Return the active leaf cell containing ``pt``, or ``None``.

        Performs a top-down traversal from the root level.

        Args:
            pt (npt.ArrayLike): Length-``ndim`` point in parametric
                coordinates.

        Returns:
            int | None: Active cell flat id, or ``None`` when ``pt`` is
            outside the grid domain.

        Raises:
            ValueError: If ``pt`` does not have length ``ndim``.
        """
        arr = _as_float64(pt, name="pt").ravel()
        if arr.shape != (self.ndim,):
            raise ValueError(f"pt must have shape ({self.ndim},); got {arr.shape}.")

        # Locate in the root grid; returns None if pt is outside the domain.
        root_cid = self._root.locate(arr)
        if root_cid is None:
            return None

        root_midx = self._root.cell_multi_index(root_cid)
        level = 0
        midx: tuple[int, ...] = root_midx

        # Compute root-cell bounds.
        lo = np.empty(self.ndim, dtype=np.float64)
        hi = np.empty(self.ndim, dtype=np.float64)
        for k, bp in enumerate(self._root.breakpoints):
            lo[k] = float(bp[midx[k]])
            hi[k] = float(bp[midx[k] + 1])

        # Top-down: check if current cell is a leaf; if not, descend.
        for level in range(len(self._blocks)):
            cid = self._encode_midx(level, midx)
            if cid is not None:
                return cid
            if level >= len(self._blocks) - 1:
                return None  # unreachable in a consistent grid
            # Descend: find the child of (level, midx) containing pt.
            new_midx: list[int] = []
            for k, fk in enumerate(self._factor):
                size_k = (hi[k] - lo[k]) / fk
                j = int((float(arr[k]) - lo[k]) / size_k)
                j = min(max(j, 0), fk - 1)
                lo[k] = lo[k] + j * size_k
                hi[k] = lo[k] + size_k
                new_midx.append(midx[k] * fk + j)
            midx = tuple(new_midx)

        return None  # unreachable

    def locate_many(self, points: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """Locate a batch of points via the Numba top-down descent kernel.

        Args:
            points (npt.ArrayLike): ``(npts, ndim)`` array-like of points, or a
                single length-``ndim`` point.

        Returns:
            npt.NDArray[np.int64]: Shape ``(npts,)`` cell ids; ``-1`` for points
            outside the grid (including points with NaN or infinite coordinates).

        Raises:
            ValueError: If the trailing axis of ``points`` is not ``ndim``.
        """
        pts = self._normalize_points(points)
        out = np.empty(pts.shape[0], dtype=np.int64)
        _hier_locate_points_core(
            pts,
            self._root_knots_flat,
            self._root_knot_starts,
            np.asarray(self._root.cells_per_axis, dtype=np.int64),
            np.asarray(self._factor, dtype=np.int64),
            self._packed_block_lo,
            self._packed_block_hi,
            self._packed_block_base,
            self._packed_level_start,
            out,
        )
        # The kernel's binary search has no NaN handling (NaN comparisons are
        # all False, silently landing in the first cell); mask them out here.
        finite = np.isfinite(pts).all(axis=1)
        if not finite.all():
            out[~finite] = -1
        return out

    def _facet_neighbor_position(
        self, cid: int, lfid: int
    ) -> tuple[int, tuple[int, ...], int, int] | None:
        """Resolve facet ``lfid`` of ``cid`` to its neighbour position.

        Args:
            cid (int): Cell identifier.
            lfid (int): Local facet identifier in ``[0, 2 * ndim)``.

        Returns:
            tuple[int, tuple[int, ...], int, int] | None:
            ``(level, nbr_midx, axis, face_j)`` where ``(level, nbr_midx)`` is the
            same-level position across the facet and ``face_j`` is the per-axis
            child offset of descendants touching the shared facet plane, or
            ``None`` when the facet lies on the grid outer boundary.

        Raises:
            IndexError: If ``cid`` or ``lfid`` is out of range.
        """
        self._check_lfid(cid, lfid)
        axis, side = divmod(int(lfid), 2)
        level, midx = self._decode_flat_id(cid)

        delta = -1 if side == 0 else 1
        new_ik = midx[axis] + delta
        n_k = self._n_cells_at_level_k(level, axis)
        if new_ik < 0 or new_ik >= n_k:
            return None  # grid outer boundary

        nbr_midx = (*midx[:axis], new_ik, *midx[axis + 1 :])
        face_j = self._factor[axis] - 1 if side == 0 else 0
        return level, nbr_midx, axis, face_j

    def _nearest_active_ancestor(self, level: int, midx: tuple[int, ...]) -> int | None:
        """Return the active leaf covering ``(level, midx)`` at a strictly coarser level.

        Args:
            level (int): Level of the queried position.
            midx (tuple[int, ...]): Per-axis index in level-``level`` coordinates.

        Returns:
            int | None: Flat id of the covering active leaf at the nearest coarser
            level, or ``None`` when no ancestor is active (the position is either
            an active leaf itself or subdivided into finer leaves).
        """
        anc = midx
        for lvl in range(level - 1, -1, -1):
            anc = tuple(i // f for i, f in zip(anc, self._factor, strict=False))
            cid = self._encode_midx(lvl, anc)
            if cid is not None:
                return cid
        return None

    def _active_face_descendants(
        self,
        level: int,
        midx: tuple[int, ...],
        axis: int,
        face_j: int,
    ) -> list[int]:
        """Collect the active leaves inside ``(level, midx)`` touching one of its faces.

        Descends recursively through the subdivision tree, at each step keeping only
        the children on the facet plane (child offset ``face_j`` on ``axis``) and
        enumerating the remaining axes in C-order (depth-first), so the returned ids
        are ordered by their position along the face.

        Args:
            level (int): Level of the starting position.
            midx (tuple[int, ...]): Per-axis index in level-``level`` coordinates.
            axis (int): Axis normal to the face.
            face_j (int): Child offset on ``axis`` adjacent to the facet plane
                (``factor[axis] - 1`` for the low side, ``0`` for the high side).

        Returns:
            list[int]: Flat ids of the active leaf descendants touching the face;
            ``[cid]`` when ``(level, midx)`` is itself an active leaf.
        """
        cid = self._encode_midx(level, midx)
        if cid is not None:
            return [cid]
        if level + 1 >= len(self._blocks):
            return []
        child_ranges = [
            (face_j,) if k == axis else tuple(range(self._factor[k])) for k in range(self.ndim)
        ]
        result: list[int] = []
        for offsets in itertools.product(*child_ranges):
            child = tuple(m * f + o for m, f, o in zip(midx, self._factor, offsets, strict=False))
            result.extend(self._active_face_descendants(level + 1, child, axis, face_j))
        return result

    def neighbor_across_facet(self, cid: int, lfid: int) -> int | None:
        """Return the cell across local facet ``lfid`` of ``cid``, or ``None``.

        Handles hanging-node interfaces across **any** level difference (no 2:1
        balance is enforced by :meth:`refine`): when the neighbour is coarser, the
        active leaf covering the position -- however many levels up -- is returned.
        When the neighbour side is finer, the first active leaf descendant touching
        the face (lowest C-order along the face) is returned.  Use
        :meth:`hanging_neighbors` to retrieve *all* fine neighbours.

        Args:
            cid (int): Cell identifier.
            lfid (int): Local facet identifier in ``[0, 2 * ndim)``.

        Returns:
            int | None: Neighbouring cell id, or ``None`` on a boundary facet.

        Raises:
            IndexError: If ``cid`` or ``lfid`` is out of range.
        """
        position = self._facet_neighbor_position(cid, lfid)
        if position is None:
            return None  # grid outer boundary
        level, nbr_midx, axis, face_j = position

        # Case 1: same-level active neighbour (conforming).
        ncid = self._encode_midx(level, nbr_midx)
        if ncid is not None:
            return ncid

        # Case 2: coarser active neighbour (any number of levels up).
        pcid = self._nearest_active_ancestor(level, nbr_midx)
        if pcid is not None:
            return pcid

        # Case 3: finer active neighbours (any number of levels down) — return
        # the first leaf touching the face.
        descendants = self._active_face_descendants(level, nbr_midx, axis, face_j)
        return descendants[0] if descendants else None

    def hanging_neighbors(self, cid: int, lfid: int) -> tuple[int, ...]:
        """Return all active neighbours across facet ``lfid`` of ``cid``.

        Equivalent to :meth:`neighbor_across_facet` for conforming and coarser
        interfaces (a single neighbour, however many levels up).  For a hanging
        (fine-side) interface, returns *all* active leaves touching the face,
        descending as many levels as the interface requires, ordered depth-first
        along the face (C-order over the non-``axis`` directions at each level).

        Args:
            cid (int): Cell identifier.
            lfid (int): Local facet identifier in ``[0, 2 * ndim)``.

        Returns:
            tuple[int, ...]: Neighbouring cell ids; empty on a boundary facet.

        Raises:
            IndexError: If ``cid`` or ``lfid`` is out of range.
        """
        position = self._facet_neighbor_position(cid, lfid)
        if position is None:
            return ()  # grid outer boundary
        level, nbr_midx, axis, face_j = position

        # Conforming or coarser — at most one neighbour.
        ncid = self._encode_midx(level, nbr_midx)
        if ncid is not None:
            return (ncid,)
        pcid = self._nearest_active_ancestor(level, nbr_midx)
        if pcid is not None:
            return (pcid,)

        # Finer side — collect all active leaves touching the face.
        return tuple(self._active_face_descendants(level, nbr_midx, axis, face_j))

    # ------------------------------------------------------------------
    # Restriction / windowing
    # ------------------------------------------------------------------

    def restrict(self, cell_ids: npt.ArrayLike) -> GridRestriction:
        """Return the root-cell-aligned bounding-box sub-grid spanning ``cell_ids``.

        The window is the multi-index bounding box, **in root-cell coordinates**,
        of the root cells containing the requested leaves (a leaf at
        ``(level, midx)`` lives in root cell ``midx[k] // factor[k] ** level``).
        The sub-grid's root is the matching slice of this grid's root breakpoints
        (never re-clamped) and it keeps the same ``factor``; its active leaves are
        the per-level intersections of this grid's blocks with the window.

        Because the window is root-cell-aligned, restricting a single deep leaf
        returns the whole leaf-tiling of its root cell, with only the requested
        leaf flagged in :attr:`GridRestriction.in_subset`.

        Args:
            cell_ids (npt.ArrayLike): Flat cell identifiers to span; duplicates
                are ignored. Each must satisfy ``0 <= cid < num_cells``.

        Returns:
            GridRestriction: The windowed :class:`HierarchicalGrid`, its
            ``local_to_global_cell`` map of shape ``(sub.num_cells,)``, and the
            boolean ``in_subset`` mask flagging requested versus bounding-box-fill cells.

        Raises:
            ValueError: If ``cell_ids`` is empty.
            IndexError: If any cell id is out of range ``[0, num_cells)``.
            TypeError: If ``cell_ids`` is not integer-valued.
            RuntimeError: If an internal invariant is violated (should be unreachable).
        """
        ids = np.asarray(cell_ids).ravel()
        if ids.size == 0:
            raise ValueError("restrict: cell_ids must be non-empty.")
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError(f"restrict: cell_ids must be integer-valued; got dtype {ids.dtype}.")
        ids = ids.astype(np.int64, copy=False)
        lo_id, hi_id = int(ids.min()), int(ids.max())
        if lo_id < 0 or hi_id >= self._num_cells:
            raise IndexError(
                f"restrict: cell id out of range [0, {self._num_cells}); got [{lo_id}, {hi_id}]."
            )

        ndim = self.ndim
        # Root-cell bounding box over the requested leaves.
        r_lo = list(self._root.cells_per_axis)
        r_hi = [0] * ndim
        for cid in {int(c) for c in ids}:
            level, midx = self._decode_flat_id(cid)
            for k in range(ndim):
                root_ik = midx[k] // (self._factor[k] ** level)
                r_lo[k] = min(r_lo[k], root_ik)
                r_hi[k] = max(r_hi[k], root_ik + 1)

        # Sub-root: pure slice of the root breakpoints (never re-clamped).
        sub_root = TensorProductGrid(
            [self._root.breakpoints[k][r_lo[k] : r_hi[k] + 1] for k in range(ndim)]
        )

        # Per-level block intersection, translated into sub coordinates.
        sub_blocks: list[list[_Block]] = []
        for level in range(len(self._blocks)):
            w_lo = tuple(r_lo[k] * self._factor[k] ** level for k in range(ndim))
            w_hi = tuple(r_hi[k] * self._factor[k] ** level for k in range(ndim))
            level_sub: list[_Block] = []
            for blo, bhi in self._blocks[level]:
                inter = _rect_intersect(blo, bhi, w_lo, w_hi)
                if inter is None:
                    continue
                i_lo, i_hi = inter
                s_lo = tuple(i_lo[k] - w_lo[k] for k in range(ndim))
                s_hi = tuple(i_hi[k] - w_lo[k] for k in range(ndim))
                level_sub.append((s_lo, s_hi))
            sub_blocks.append(level_sub)

        sub = HierarchicalGrid._from_blocks(sub_root, self._factor, sub_blocks)

        # Local -> global cell map: translate each sub leaf back to global coords.
        local_to_global = np.empty(sub.num_cells, dtype=np.int64)
        for sub_cid in range(sub.num_cells):
            level = sub.cell_level(sub_cid)
            sub_midx = sub.cell_multi_index(sub_cid)
            g_midx = tuple(sub_midx[k] + r_lo[k] * self._factor[k] ** level for k in range(ndim))
            g_cid = self._encode_midx(level, g_midx)
            assert g_cid is not None  # invariant: every windowed leaf maps to an active global leaf
            local_to_global[sub_cid] = g_cid

        in_subset = np.isin(local_to_global, ids)
        local_to_global.flags.writeable = False
        in_subset.flags.writeable = False
        return GridRestriction(sub, local_to_global, in_subset)

    # ------------------------------------------------------------------
    # Hierarchy accessors
    # ------------------------------------------------------------------

    def cell_level(self, cid: int) -> int:
        """Return the refinement level of cell ``cid``.

        Args:
            cid (int): Cell identifier.

        Returns:
            int: Refinement level (``0`` for unrefined root cells).

        Raises:
            IndexError: If ``cid`` is out of range.
        """
        self._check_cid(cid)
        level, _ = self._decode_flat_id(cid)
        return level

    def cell_multi_index(self, cid: int) -> tuple[int, ...]:
        """Return the per-axis index of cell ``cid`` in its level's coordinates.

        Args:
            cid (int): Cell identifier.

        Returns:
            tuple[int, ...]: Length-``ndim`` per-axis index tuple at the cell's
            refinement level.

        Raises:
            IndexError: If ``cid`` is out of range.
        """
        self._check_cid(cid)
        _, midx = self._decode_flat_id(cid)
        return midx

    def is_active_leaf(self, level: int, midx: Sequence[int]) -> bool:
        """Return whether ``(level, midx)`` is an active (leaf) cell.

        Args:
            level (int): Hierarchy level.
            midx (Sequence[int]): Per-axis index in level-``level`` coordinates.

        Returns:
            bool: ``True`` iff a cell with this level and multi-index is currently
            active (a leaf); ``False`` if it is out of range, not yet created, or has
            been refined away.
        """
        if level < 0 or level >= len(self._blocks):
            return False
        midx_t = tuple(int(i) for i in midx)
        if len(midx_t) != self.ndim or any(i < 0 for i in midx_t):
            return False
        return self._encode_midx(level, midx_t) is not None

    # ------------------------------------------------------------------
    # Active-set accessors
    # ------------------------------------------------------------------

    def level_cells_per_axis(self, level: int) -> tuple[int, ...]:
        """Return the per-axis cell count of the level-``level`` grid.

        This is a pure formula — ``level`` need not be an existing hierarchy level.
        Values above ``max_level`` return the count for the hypothetical finer grid
        that would result from additional uniform subdivision.  This differs from
        :meth:`active_blocks`, :meth:`active_leaf_mask`, and :meth:`subdomain_mask`,
        which all require ``level <= max_level``.

        Args:
            level (int): Hierarchy level.  Must be ``>= 0``; values above
                ``max_level`` are accepted and return the geometrically valid count.

        Returns:
            tuple[int, ...]: ``root.cells_per_axis[k] * factor[k] ** level`` for
            every axis ``k``.

        Raises:
            ValueError: If ``level < 0``.
        """
        if level < 0:
            raise ValueError(f"level must be >= 0; got {level!r}.")
        return tuple(self._n_cells_at_level_k(level, k) for k in range(self.ndim))

    def active_blocks(self, level: int) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        """Return the active-leaf blocks at ``level``.

        Args:
            level (int): Hierarchy level.  Must be in ``[0, max_level]``.

        Returns:
            tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]: The sorted,
            non-overlapping ``(lo, hi)`` integer rectangles of active (leaf) cells
            at ``level``, in level-``level`` coordinates.

        Raises:
            ValueError: If ``level`` is outside ``[0, max_level]``.
        """
        self._check_level(level)
        return tuple(self._blocks[level])

    def active_leaf_mask(self, level: int) -> npt.NDArray[np.bool_]:
        r"""Return a boolean mask of the active-leaf cells at ``level``.

        Args:
            level (int): Hierarchy level.  Must be in ``[0, max_level]``.

        Returns:
            npt.NDArray[np.bool\_]: Fresh array of shape
            ``level_cells_per_axis(level)``; ``True`` where the level-``level``
            cell ``(level, midx)`` is an active (leaf) cell.

        Raises:
            ValueError: If ``level`` is outside ``[0, max_level]``.
        """
        self._check_level(level)
        mask = np.zeros(self.level_cells_per_axis(level), dtype=np.bool_)
        for lo, hi in self._blocks[level]:
            mask[tuple(slice(lo_k, hi_k) for lo_k, hi_k in zip(lo, hi, strict=False))] = True
        return mask

    def subdomain_mask(self, level: int) -> npt.NDArray[np.bool_]:
        r"""Return a boolean mask of the level-``level`` refined subdomain.

        A level-``level`` cell lies in the subdomain :math:`\Omega_{level}` (the
        region refined to at least ``level``) iff it is **not** covered by an
        active leaf of a coarser level.  The mask is computed at the level-``level``
        resolution by projecting every coarser active-leaf block up to ``level`` and
        clearing those cells.

        Args:
            level (int): Hierarchy level.  Must be in ``[0, max_level]``.

        Returns:
            npt.NDArray[np.bool\_]: Fresh array of shape
            ``level_cells_per_axis(level)``; ``True`` where the level-``level``
            cell lies in :math:`\Omega_{level}`.

        Raises:
            ValueError: If ``level`` is outside ``[0, max_level]``.

        Note:
            The mask is sized to the level-``level`` cell grid and computed on
            demand; it is never stored.
        """
        self._check_level(level)
        mask = np.ones(self.level_cells_per_axis(level), dtype=np.bool_)
        for m in range(level):
            scale = tuple(self._factor[k] ** (level - m) for k in range(self.ndim))
            for lo, hi in self._blocks[m]:
                mask[
                    tuple(
                        slice(lo_k * s, hi_k * s)
                        for lo_k, hi_k, s in zip(lo, hi, scale, strict=False)
                    )
                ] = False
        return mask

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------

    def refine(
        self,
        level: int,
        lo: Sequence[int],
        hi: Sequence[int],
    ) -> None:
        """Refine the rectangular region ``[lo, hi)`` at ``level`` to ``level+1``.

        Union semantics: only the currently-active portion of ``[lo, hi)`` is
        refined.  If the intersection with active blocks at ``level`` is empty,
        the call is a silent no-op.  Overlapping calls therefore safely extend
        the refined region.

        After the call all flat cell ids are **reassigned** (the BVH, cell tags,
        and facet tags are also invalidated).

        Args:
            level (int): Refinement level at which the region lives.  Must
                satisfy ``0 <= level <= max_level``.
            lo (Sequence[int]): Per-direction start index (inclusive), in
                level-``level`` coordinates.
            hi (Sequence[int]): Per-direction end index (exclusive), in
                level-``level`` coordinates.

        Raises:
            ValueError: If ``level`` is out of range, ``lo``/``hi`` have the
                wrong length, any ``lo[k] >= hi[k]``, or ``[lo, hi)`` falls
                entirely outside the level-``level`` domain.
        """
        ndim = self.ndim
        if not (0 <= int(level) <= self.max_level):
            raise ValueError(f"level must be in [0, {self.max_level}]; got {level!r}.")
        lo_t = tuple(int(x) for x in lo)
        hi_t = tuple(int(x) for x in hi)
        if len(lo_t) != ndim or len(hi_t) != ndim:
            raise ValueError(f"lo and hi must have length {ndim}; got {len(lo_t)} and {len(hi_t)}.")
        if any(lo_k >= h for lo_k, h in zip(lo_t, hi_t, strict=False)):
            raise ValueError(
                f"lo must be strictly less than hi in every dimension; "
                f"got lo={lo_t!r}, hi={hi_t!r}."
            )
        # Validate against the level domain.
        for k in range(ndim):
            n_k = self._n_cells_at_level_k(level, k)
            if lo_t[k] < 0 or hi_t[k] > n_k:
                raise ValueError(
                    f"[lo, hi) out of bounds at level {level}: "
                    f"axis {k} needs [0, {n_k}), got [{lo_t[k]}, {hi_t[k]})."
                )

        new_blocks_at_level: list[_Block] = []
        new_children: list[_Block] = []

        for block_lo, block_hi in self._blocks[level]:
            inter = _rect_intersect(block_lo, block_hi, lo_t, hi_t)
            if inter is None:
                new_blocks_at_level.append((block_lo, block_hi))
                continue
            i_lo, i_hi = inter
            # Subtract intersection from the current block.
            new_blocks_at_level.extend(_peel(block_lo, block_hi, i_lo, i_hi))
            # Add children of the intersection at level+1.
            child_lo = tuple(i * f for i, f in zip(i_lo, self._factor, strict=False))
            child_hi = tuple(i * f for i, f in zip(i_hi, self._factor, strict=False))
            new_children.append((child_lo, child_hi))

        if not new_children:
            return  # no active cells in the requested region — no-op

        self._blocks[level] = _normalize_blocks(new_blocks_at_level)

        # Extend _blocks if needed and add children at level+1.
        while len(self._blocks) <= level + 1:
            self._blocks.append([])
        self._blocks[level + 1] = _normalize_blocks(self._blocks[level + 1] + new_children)

        self._rebuild()

    def refine_cells(self, cell_ids: Sequence[int]) -> None:
        """Refine a set of active cells using per-level bounding-box aggregation.

        Groups ``cell_ids`` by their level, computes the bounding box of all
        cells at each level (the smallest axis-aligned rectangle containing
        them), and calls :meth:`refine` once per level.

        Args:
            cell_ids (Sequence[int]): Flat cell ids to refine.  Cells from
                multiple levels are handled; repeated ids are silently ignored.

        Raises:
            IndexError: If any id in ``cell_ids`` is out of range.
        """
        if not cell_ids:
            return
        # Group by level.
        level_lo: dict[int, list[int]] = {}
        level_hi: dict[int, list[int]] = {}
        for cid in cell_ids:
            self._check_cid(int(cid))
            lv, midx = self._decode_flat_id(int(cid))
            if lv not in level_lo:
                level_lo[lv] = list(midx)
                level_hi[lv] = [m + 1 for m in midx]
            else:
                for k, m in enumerate(midx):
                    level_lo[lv][k] = min(level_lo[lv][k], m)
                    level_hi[lv][k] = max(level_hi[lv][k], m + 1)
        for lv in sorted(level_lo):
            self.refine(lv, level_lo[lv], level_hi[lv])

    def coarsen(
        self,
        level: int,
        lo: Sequence[int],
        hi: Sequence[int],
    ) -> None:
        """Coarsen the rectangular region ``[lo, hi)`` at ``level`` (inverse of refine).

        Reactivates the level-``level`` cells in ``[lo, hi)`` and removes their
        level-``(level+1)`` children.  The region must be **fully refined to exactly
        level ``level+1``**: every child cell in ``[lo*factor, hi*factor)`` must be an
        active leaf at ``level+1`` (none further refined, none still a leaf at
        ``level``).  Calling :meth:`coarsen` with the same arguments as a preceding
        :meth:`refine` exactly restores the grid.

        After the call all flat cell ids are **reassigned** (the BVH, cell tags, and
        facet tags are invalidated).

        Args:
            level (int): Level whose cells are reactivated.  Must satisfy
                ``0 <= level < max_level``.
            lo (Sequence[int]): Per-direction start index (inclusive), in level-``level``
                coordinates.
            hi (Sequence[int]): Per-direction end index (exclusive), in level-``level``
                coordinates.

        Raises:
            ValueError: If ``level`` is out of range, ``lo``/``hi`` have the wrong
                length, any ``lo[k] >= hi[k]``, ``[lo, hi)`` is out of bounds, or the
                region is not fully refined to exactly level ``level+1``.
        """
        ndim = self.ndim
        if not (0 <= int(level) < self.max_level):
            raise ValueError(f"level must be in [0, {self.max_level}); got {level!r}.")
        lo_t = tuple(int(x) for x in lo)
        hi_t = tuple(int(x) for x in hi)
        if len(lo_t) != ndim or len(hi_t) != ndim:
            raise ValueError(f"lo and hi must have length {ndim}; got {len(lo_t)} and {len(hi_t)}.")
        if any(lo_k >= h for lo_k, h in zip(lo_t, hi_t, strict=False)):
            raise ValueError(
                f"lo must be strictly less than hi in every dimension; "
                f"got lo={lo_t!r}, hi={hi_t!r}."
            )
        for k in range(ndim):
            n_k = self._n_cells_at_level_k(level, k)
            if lo_t[k] < 0 or hi_t[k] > n_k:
                raise ValueError(
                    f"[lo, hi) out of bounds at level {level}: "
                    f"axis {k} needs [0, {n_k}), got [{lo_t[k]}, {hi_t[k]})."
                )

        child_lo = tuple(lo_t[k] * self._factor[k] for k in range(ndim))
        child_hi = tuple(hi_t[k] * self._factor[k] for k in range(ndim))
        child_size = _block_size(child_lo, child_hi)

        # The children region must be fully tiled by active leaves at level+1.
        covered = 0
        new_finer: list[_Block] = []
        for block_lo, block_hi in self._blocks[level + 1]:
            inter = _rect_intersect(block_lo, block_hi, child_lo, child_hi)
            if inter is None:
                new_finer.append((block_lo, block_hi))
                continue
            covered += _block_size(*inter)
            new_finer.extend(_peel(block_lo, block_hi, *inter))
        if covered != child_size:
            raise ValueError(
                f"cannot coarsen [{lo_t}, {hi_t}) at level {level}: the region is not "
                f"fully refined to exactly level {level + 1}."
            )

        self._blocks[level + 1] = _normalize_blocks(new_finer)
        self._blocks[level] = _normalize_blocks([*self._blocks[level], (lo_t, hi_t)])
        while len(self._blocks) > 1 and not self._blocks[-1]:
            self._blocks.pop()
        self._rebuild()

    # ------------------------------------------------------------------
    # Overrides for BVH efficiency
    # ------------------------------------------------------------------

    def collect_cell_bounds(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Materialize ``(cell_lo, cell_hi)`` in flat-id order for the BVH.

        Backed by a Numba kernel parallelizing over the active blocks; results
        are identical to calling :meth:`cell_bounds` per cell.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            ``(cell_lo, cell_hi)`` of shape ``(num_cells, ndim)``.
        """
        n = self._num_cells
        nd = self.ndim
        cell_lo = np.empty((n, nd), dtype=np.float64)
        cell_hi = np.empty((n, nd), dtype=np.float64)
        _hier_collect_cell_bounds_core(
            self._root_knots_flat,
            self._root_knot_starts,
            np.asarray(self._factor, dtype=np.int64),
            self._packed_block_lo,
            self._packed_block_hi,
            self._packed_block_base,
            self._packed_level_start,
            cell_lo,
            cell_hi,
        )
        return cell_lo, cell_hi

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a compact developer-friendly representation.

        Returns:
            str: Shows dimension, root cell counts, factor, active cells,
            and max level.
        """
        return (
            f"HierarchicalGrid(ndim={self.ndim}, "
            f"root_cells={self._root.cells_per_axis}, "
            f"factor={self._factor}, "
            f"num_cells={self._num_cells}, "
            f"max_level={self.max_level})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def hierarchical_grid(
    root: TensorProductGrid,
    factor: int | Sequence[int],
) -> HierarchicalGrid:
    """Build a :class:`HierarchicalGrid` from a root grid and a subdivision factor.

    Args:
        root (TensorProductGrid): The level-0 grid.
        factor (int | Sequence[int]): Per-direction subdivision factor.  A
            scalar is broadcast to all axes.  Each entry must be ``>= 1``.

    Returns:
        HierarchicalGrid: A new hierarchical grid starting with all root cells
        active at level 0.

    Raises:
        TypeError: If ``root`` is not a :class:`TensorProductGrid`.
        ValueError: If ``factor`` has the wrong length or any entry is ``< 1``.
    """
    return HierarchicalGrid(root, factor)


__all__ = ["HierarchicalGrid", "hierarchical_grid"]
