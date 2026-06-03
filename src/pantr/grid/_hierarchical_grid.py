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

*Automatic single-level balance.*  A level-``l`` frame cell is always adjacent
to level-``(l+1)`` cells (diff = 1).  No explicit balancing pass is needed.

Main exports:

- :class:`HierarchicalGrid`: hierarchical grid built on a
  :class:`TensorProductGrid`.
- :func:`hierarchical_grid`: standalone factory.
"""

from __future__ import annotations

import bisect
from typing import TYPE_CHECKING

import numpy as np

from ._cell_index import flat_to_multi, multi_to_flat
from ._grid import Grid
from ._grid_utils import _as_float64
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
    """

    __slots__ = ("_blocks", "_factor", "_level_base", "_num_cells", "_root")

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
        self._rebuild()

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        """Recompute ``_level_base``, ``_num_cells``, and reset the BVH/tags.

        Called after every structural change (construction or refinement).
        """
        base = 0
        level_base: list[int] = []
        for blocks_at_level in self._blocks:
            level_base.append(base)
            base += sum(_block_size(lo, hi) for lo, hi in blocks_at_level)
        level_base.append(base)  # sentinel for bisect_right
        self._level_base = level_base
        self._num_cells = base
        self._bvh = None
        self._cell_tags = None
        self._facet_tags = None

    def _decode_flat_id(self, cid: int) -> tuple[int, tuple[int, ...]]:
        """Convert a flat cell id to ``(level, multi_index)``.

        Args:
            cid (int): Flat cell identifier.

        Returns:
            tuple[int, tuple[int, ...]]: ``(level, multi_index)`` where
            ``multi_index`` uses level-``l`` integer coordinates.

        Raises:
            IndexError: If ``cid`` is out of range.
        """
        level = bisect.bisect_right(self._level_base, cid) - 1
        if level < 0 or level >= len(self._blocks):
            raise IndexError(f"cell id {cid!r} is out of range [0, {self._num_cells}).")
        offset = cid - self._level_base[level]
        cum = 0
        for lo, hi in self._blocks[level]:
            size = _block_size(lo, hi)
            if cum + size > offset:
                local = offset - cum
                shape = tuple(h - lo_k for h, lo_k in zip(hi, lo, strict=False))
                local_midx = flat_to_multi(local, shape)
                midx = tuple(lo_k + lm for lo_k, lm in zip(lo, local_midx, strict=False))
                return level, midx
            cum += size
        raise IndexError(f"cell id {cid!r} is out of range [0, {self._num_cells}).")

    def _encode_midx(
        self,
        level: int,
        midx: tuple[int, ...],
    ) -> int | None:
        """Convert ``(level, multi_index)`` to a flat cell id, or ``None``.

        Returns ``None`` when the cell is not an active (leaf) cell.

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
        base = self._level_base[level]
        cum = 0
        for lo, hi in self._blocks[level]:
            if _in_block(midx, lo, hi):
                shape = tuple(h - lo_k for h, lo_k in zip(hi, lo, strict=False))
                local = tuple(m - lo_k for m, lo_k in zip(midx, lo, strict=False))
                return base + cum + multi_to_flat(local, shape)
            cum += _block_size(lo, hi)
        return None

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

    def neighbor_across_facet(self, cid: int, lfid: int) -> int | None:
        """Return the cell across local facet ``lfid`` of ``cid``, or ``None``.

        Handles hanging-node interfaces: when the neighbour is coarser (the
        current cell is at a finer level), the coarse neighbour is returned.
        When the neighbour is finer (the current cell is at a coarser level),
        the first fine child touching the face (lowest C-order) is returned.
        Use :meth:`hanging_neighbors` to retrieve *all* fine neighbours.

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
        level, midx = self._decode_flat_id(cid)

        delta = -1 if side == 0 else 1
        new_ik = midx[axis] + delta
        n_k = self._n_cells_at_level_k(level, axis)
        if new_ik < 0 or new_ik >= n_k:
            return None  # grid outer boundary

        nbr_midx = (*midx[:axis], new_ik, *midx[axis + 1 :])

        # Case 1: same-level active neighbour (conforming).
        ncid = self._encode_midx(level, nbr_midx)
        if ncid is not None:
            return ncid

        # Case 2: coarser active neighbour — current cell is finer.
        if level > 0:
            parent_midx = tuple(i // f for i, f in zip(nbr_midx, self._factor, strict=False))
            pcid = self._encode_midx(level - 1, parent_midx)
            if pcid is not None:
                return pcid

        # Case 3: finer active neighbour — current cell is coarser.
        # (level, nbr_midx) was created and then refined.  Return the first
        # child of (level, nbr_midx) at level+1 that touches the face.
        if level + 1 < len(self._blocks):
            face_j = self._factor[axis] - 1 if side == 0 else 0
            child_midx: list[int] = []
            for k, (nk, fk) in enumerate(zip(nbr_midx, self._factor, strict=False)):
                child_midx.append(nk * fk + (face_j if k == axis else 0))
            ccid = self._encode_midx(level + 1, tuple(child_midx))
            if ccid is not None:
                return ccid

        return None

    def hanging_neighbors(self, cid: int, lfid: int) -> tuple[int, ...]:
        """Return all active neighbours across facet ``lfid`` of ``cid``.

        Equivalent to :meth:`neighbor_across_facet` for conforming interfaces.
        For a hanging (fine-to-coarse) interface, returns all
        ``factor^(ndim-1)`` fine children that touch the face in C-order.

        Args:
            cid (int): Cell identifier.
            lfid (int): Local facet identifier in ``[0, 2 * ndim)``.

        Returns:
            tuple[int, ...]: Neighbouring cell ids; empty on a boundary facet.

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
            return ()

        nbr_midx = (*midx[:axis], new_ik, *midx[axis + 1 :])

        # Conforming or coarser — at most one neighbour.
        ncid = self._encode_midx(level, nbr_midx)
        if ncid is not None:
            return (ncid,)
        if level > 0:
            parent_midx = tuple(i // f for i, f in zip(nbr_midx, self._factor, strict=False))
            pcid = self._encode_midx(level - 1, parent_midx)
            if pcid is not None:
                return (pcid,)

        # Finer side — collect all factor^(ndim-1) children touching the face.
        if level + 1 >= len(self._blocks):
            return ()

        face_j = self._factor[axis] - 1 if side == 0 else 0
        result: list[int] = []

        import itertools  # noqa: PLC0415

        other_ranges = [range(self._factor[k]) for k in range(self.ndim) if k != axis]
        for other_js in itertools.product(*other_ranges):
            it = iter(other_js)
            child_midx: list[int] = []
            for k, (nk, fk) in enumerate(zip(nbr_midx, self._factor, strict=False)):
                j = face_j if k == axis else next(it)
                child_midx.append(nk * fk + j)
            ccid = self._encode_midx(level + 1, tuple(child_midx))
            if ccid is not None:
                result.append(ccid)
        return tuple(result)

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

    # ------------------------------------------------------------------
    # Overrides for BVH efficiency
    # ------------------------------------------------------------------

    def collect_cell_bounds(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Materialize ``(cell_lo, cell_hi)`` in flat-id order for the BVH.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            ``(cell_lo, cell_hi)`` of shape ``(num_cells, ndim)``.
        """
        n = self._num_cells
        nd = self.ndim
        cell_lo = np.empty((n, nd), dtype=np.float64)
        cell_hi = np.empty((n, nd), dtype=np.float64)
        flat_id = 0
        for level, blocks_at_level in enumerate(self._blocks):
            for blo, bhi in blocks_at_level:
                shape = tuple(h - lo_k for h, lo_k in zip(bhi, blo, strict=False))
                n_block = _block_size(blo, bhi)
                for offset in range(n_block):
                    local = flat_to_multi(offset, shape)
                    midx = tuple(lo_k + lm for lo_k, lm in zip(blo, local, strict=False))
                    lo, hi = self._cell_bounds_from_level_midx(level, midx)
                    cell_lo[flat_id] = lo
                    cell_hi[flat_id] = hi
                    flat_id += 1
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
