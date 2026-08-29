"""Grid kernels and grid types of the compiled extension.

The free functions are bound by ``cpp/bindings/grid.cpp``; the classes are bound
by the per-type files beside it (``grid_tags.cpp`` and its siblings). See
``__init__.pyi`` for what this package promises and who has to keep it.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

class CellTags:
    """Sparse named integer tags over a grid's cells, owned by the C++ core.

    Wrapped by :class:`pantr.grid.CellTags`, which is the class a caller holds;
    this one is reached only through it. The wrapper is what raises ``KeyError``
    for an unregistered name and ``TypeError`` for a non-integer argument, because
    nanobind can produce neither.

    ``scatter`` is registered once per numpy integer dtype and writes only the
    tagged entries: the wrapper allocates and fills the destination.

    Attributes:
        num_cells (int): Number of cells in the owning grid.
        names (list[str]): Registered tag names, in insertion order.
    """

    def __init__(self, num_cells: int) -> None: ...
    @property
    def num_cells(self) -> int: ...
    @property
    def names(self) -> list[str]: ...
    def __len__(self) -> int: ...
    def __contains__(self, name: str) -> bool: ...
    def set(
        self,
        name: str,
        ids: npt.NDArray[np.int64],
        values: npt.NDArray[np.int64],
    ) -> None: ...
    def get(self, name: str) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: ...
    def remove(self, name: str) -> None: ...
    def scatter(self, name: str, out: npt.NDArray[Any]) -> None: ...

class BVH:
    """Bounding-volume hierarchy over cell AABBs, owned by the C++ core.

    Wrapped by :class:`pantr.grid.BVH`, which is the class a caller holds; this one
    is reached only through it. The wrapper owns the ``TypeError`` dtype checks and
    the numpy-shaped messages, and translates ``CapacityError`` back to the
    ``ValueError`` the pre-port class raised for a tree deeper than the traversal
    stack.

    The five node arrays are zero-copy read-only views owned by the instance;
    ``query_aabb`` returns a fresh writeable array.

    Attributes:
        ndim (int): Spatial dimension of the indexed AABBs.
        n_cells (int): Number of cells indexed.
        n_nodes (int): Total number of nodes.
        node_lo (npt.NDArray[np.float64]): Per-node lo corners, read-only.
        node_hi (npt.NDArray[np.float64]): Per-node hi corners, read-only.
        node_left (npt.NDArray[np.int64]): Left-child indices, read-only.
        node_right (npt.NDArray[np.int64]): Right-child indices, read-only.
        node_cell (npt.NDArray[np.int64]): Leaf cell ids, read-only.
    """

    def __init__(
        self,
        node_lo: npt.NDArray[np.float64],
        node_hi: npt.NDArray[np.float64],
        node_left: npt.NDArray[np.int64],
        node_right: npt.NDArray[np.int64],
        node_cell: npt.NDArray[np.int64],
        n_cells: int,
    ) -> None: ...
    @staticmethod
    def from_cell_bounds(
        cell_lo: npt.NDArray[np.float64], cell_hi: npt.NDArray[np.float64]
    ) -> BVH: ...
    @property
    def ndim(self) -> int: ...
    @property
    def n_cells(self) -> int: ...
    @property
    def n_nodes(self) -> int: ...
    @property
    def node_lo(self) -> npt.NDArray[np.float64]: ...
    @property
    def node_hi(self) -> npt.NDArray[np.float64]: ...
    @property
    def node_left(self) -> npt.NDArray[np.int64]: ...
    @property
    def node_right(self) -> npt.NDArray[np.int64]: ...
    @property
    def node_cell(self) -> npt.NDArray[np.int64]: ...
    def query_aabb(
        self, qlo: npt.NDArray[np.float64], qhi: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.int64]: ...

class Partition:
    """A per-cell owner assignment over a grid's cells, owned by the C++ core.

    Wrapped by :class:`pantr.grid.Partition`, which is the class a caller holds;
    this one is reached only through it. The wrapper is what checks that
    ``cell_owner`` is a 1-D integer array and performs the narrowing cast to
    ``int32``, because both facts are gone by the time this receives a span.

    ``cell_owner`` is a zero-copy read-only view owned by the instance;
    ``owned_cells`` returns a fresh writeable array, matching what the pre-port
    class returned from ``numpy.flatnonzero``.

    Attributes:
        cell_owner (npt.NDArray[np.int32]): Per-cell owners, read-only.
        n_parts (int): Number of parts (ranks).
        n_cells (int): Number of cells.
    """

    def __init__(self, cell_owner: npt.NDArray[np.int32], n_parts: int) -> None: ...
    @property
    def cell_owner(self) -> npt.NDArray[np.int32]: ...
    @property
    def n_parts(self) -> int: ...
    @property
    def n_cells(self) -> int: ...
    def owned_cells(self, rank: int) -> npt.NDArray[np.int64]: ...

class FacetTags:
    """Sparse named integer tags over a grid's local facets, owned by the C++ core.

    The facet counterpart of :class:`CellTags`; see that class. Keys are
    ``(cell_id, local_facet_id)`` rows and ``scatter``'s destination is
    ``(num_cells, facets_per_cell)``.

    Attributes:
        num_cells (int): Number of cells in the owning grid.
        facets_per_cell (int): Number of local facets per cell.
        names (list[str]): Registered tag names, in insertion order.
    """

    def __init__(self, num_cells: int, facets_per_cell: int) -> None: ...
    @property
    def num_cells(self) -> int: ...
    @property
    def facets_per_cell(self) -> int: ...
    @property
    def names(self) -> list[str]: ...
    def __len__(self) -> int: ...
    def __contains__(self, name: str) -> bool: ...
    def set(
        self,
        name: str,
        keys: npt.NDArray[np.int64],
        values: npt.NDArray[np.int64],
    ) -> None: ...
    def get(self, name: str) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: ...
    def remove(self, name: str) -> None: ...
    def scatter(self, name: str, out: npt.NDArray[Any]) -> None: ...

def locate_points(
    points: npt.NDArray[np.float64],
    *,
    knots_flat: npt.NDArray[np.float64],
    knot_starts: npt.NDArray[np.int64],
    cells_per_axis: npt.NDArray[np.int64],
    strides: npt.NDArray[np.int64],
    out: npt.NDArray[np.int64],
) -> None:
    """Locate a batch of points on an axis-aligned tensor-product grid.

    Args:
        points (npt.NDArray[np.float64]): Query points of shape ``(npts, ndim)``,
            C-contiguous.
        knots_flat (npt.NDArray[np.float64]): All per-axis knot vectors concatenated
            end to end, strictly increasing within each axis. Keyword-only.
        knot_starts (npt.NDArray[np.int64]): Per-axis offset into ``knots_flat``,
            shape ``(ndim,)``. Keyword-only.
        cells_per_axis (npt.NDArray[np.int64]): Per-axis cell counts, shape
            ``(ndim,)``, each at least one. Keyword-only.
        strides (npt.NDArray[np.int64]): Per-axis C-order flat strides, shape
            ``(ndim,)``. Keyword-only.
        out (npt.NDArray[np.int64]): Shape ``(npts,)``, C-contiguous and writable.
            Receives the flat cell id, or ``-1`` outside the domain. Keyword-only.

    Raises:
        TypeError: If any array has the wrong dtype or rank, is not C-contiguous,
            or if a keyword-only argument is passed positionally.
        ValueError: If the descriptor arrays disagree on ``ndim``, if an axis
            declares fewer than one cell or spans past the end of ``knots_flat``,
            or if ``out`` is shorter than ``npts``.
    """

def bvh_build(
    *,
    cell_lo: npt.NDArray[np.float64],
    cell_hi: npt.NDArray[np.float64],
    node_lo: npt.NDArray[np.float64],
    node_hi: npt.NDArray[np.float64],
    node_left: npt.NDArray[np.int64],
    node_right: npt.NDArray[np.int64],
    node_cell: npt.NDArray[np.int64],
) -> None:
    """Build a BVH over per-cell AABBs by median-of-longest-axis splits.

    Every argument is keyword-only: ``cell_lo``/``cell_hi`` and
    ``node_left``/``node_right`` are same-typed neighbours that would type-check
    and run if transposed.

    Args:
        cell_lo (npt.NDArray[np.float64]): Per-cell lo corners, shape
            ``(n_cells, ndim)`` with ``n_cells >= 1``. Keyword-only.
        cell_hi (npt.NDArray[np.float64]): Per-cell hi corners, same shape, with
            ``hi >= lo`` everywhere. Keyword-only.
        node_lo (npt.NDArray[np.float64]): Output per-node lo corners, at least
            ``2 * n_cells - 1`` rows of ``ndim`` columns. Keyword-only.
        node_hi (npt.NDArray[np.float64]): Output per-node hi corners, same shape.
            Keyword-only.
        node_left (npt.NDArray[np.int64]): Output left-child indices, at least
            ``2 * n_cells - 1`` entries, pre-filled with ``-1``. Keyword-only.
        node_right (npt.NDArray[np.int64]): Output right-child indices, likewise.
            Keyword-only.
        node_cell (npt.NDArray[np.int64]): Output per-leaf cell ids, likewise.
            Keyword-only.

    Raises:
        TypeError: If any array has the wrong dtype or rank or is not C-contiguous.
        ValueError: If ``cell_lo`` is empty, if the shapes disagree, if any cell has
            ``hi < lo`` or a non-finite corner, if an output cannot hold
            ``2 * n_cells - 1`` entries, or if the cell count would produce a tree
            deeper than the traversal stack allows.
    """

def bvh_query_count(
    *,
    qlo: npt.NDArray[np.float64],
    qhi: npt.NDArray[np.float64],
    node_lo: npt.NDArray[np.float64],
    node_hi: npt.NDArray[np.float64],
    node_left: npt.NDArray[np.int64],
    node_right: npt.NDArray[np.int64],
    node_cell: npt.NDArray[np.int64],
) -> int:
    """Count the leaves whose AABB overlaps the query box, inclusive on every face.

    Called before :func:`bvh_query_emit` so the caller can size its output exactly.

    Args:
        qlo (npt.NDArray[np.float64]): Query box lo corner, shape ``(ndim,)``.
            Keyword-only.
        qhi (npt.NDArray[np.float64]): Query box hi corner, same shape. Keyword-only.
        node_lo (npt.NDArray[np.float64]): Per-node lo corners, shape
            ``(n_nodes, ndim)``. Keyword-only.
        node_hi (npt.NDArray[np.float64]): Per-node hi corners, same shape.
            Keyword-only.
        node_left (npt.NDArray[np.int64]): Left-child indices, ``-1`` at a leaf.
            Keyword-only.
        node_right (npt.NDArray[np.int64]): Right-child indices. Keyword-only.
        node_cell (npt.NDArray[np.int64]): Per-leaf cell ids, ``-1`` at an internal
            node. Keyword-only.

    Returns:
        int: The number of overlapping leaves.

    Raises:
        TypeError: If any array has the wrong dtype or rank or is not C-contiguous.
        ValueError: If the arrays disagree on ``ndim`` or on the node count, or if
            the node arrays are empty.

    Note:
        The tree's depth is **not** validated here, matching
        :class:`pantr.grid.BVH`, which establishes it once at construction rather
        than on every query. A caller assembling node arrays by hand owns that
        contract; an unbalanced tree overflows a fixed-size traversal stack.
    """

def bvh_query_emit(
    *,
    qlo: npt.NDArray[np.float64],
    qhi: npt.NDArray[np.float64],
    node_lo: npt.NDArray[np.float64],
    node_hi: npt.NDArray[np.float64],
    node_left: npt.NDArray[np.int64],
    node_right: npt.NDArray[np.int64],
    node_cell: npt.NDArray[np.int64],
    out: npt.NDArray[np.int64],
) -> int:
    """Write the cell ids of the leaves whose AABB overlaps the query box.

    Visits nodes in the same order as :func:`bvh_query_count`, so the two agree on
    the size.

    Args:
        qlo (npt.NDArray[np.float64]): Query box lo corner, shape ``(ndim,)``.
            Keyword-only.
        qhi (npt.NDArray[np.float64]): Query box hi corner, same shape. Keyword-only.
        node_lo (npt.NDArray[np.float64]): Per-node lo corners. Keyword-only.
        node_hi (npt.NDArray[np.float64]): Per-node hi corners. Keyword-only.
        node_left (npt.NDArray[np.int64]): Left-child indices. Keyword-only.
        node_right (npt.NDArray[np.int64]): Right-child indices. Keyword-only.
        node_cell (npt.NDArray[np.int64]): Per-leaf cell ids. Keyword-only.
        out (npt.NDArray[np.int64]): C-contiguous and writable, sized from a prior
            :func:`bvh_query_count`. Keyword-only.

    Returns:
        int: The number of overlapping leaves, which equals what
            :func:`bvh_query_count` returns for the same arguments.

    Raises:
        TypeError: If any array has the wrong dtype or rank or is not C-contiguous.
        ValueError: If the arrays disagree on ``ndim`` or on the node count, or if
            ``out`` is too small to hold the matches.

    Note:
        The kernel counts unconditionally and writes only within ``out``'s capacity,
        so an undersized buffer raises rather than corrupting memory. The oracle
        writes unconditionally; this is a deliberate difference, and the only input
        on which the two disagree is one Layer 2 never produces.
    """

def encode_midx(
    level: int,
    *,
    midx: npt.NDArray[np.int64],
    block_lo: npt.NDArray[np.int64],
    block_hi: npt.NDArray[np.int64],
    block_base: npt.NDArray[np.int64],
    level_block_start: npt.NDArray[np.int64],
) -> int:
    """Return the flat cell id of ``(level, midx)``, or ``-1`` when not active.

    Args:
        level (int): Hierarchy level of the queried position, in
            ``[0, n_levels)``.
        midx (npt.NDArray[np.int64]): Per-axis index in level coordinates, shape
            ``(ndim,)``. Keyword-only.
        block_lo (npt.NDArray[np.int64]): Packed block lower bounds, shape
            ``(n_blocks, ndim)``. Keyword-only.
        block_hi (npt.NDArray[np.int64]): Packed block upper bounds, same shape,
            strictly greater than ``block_lo`` on every axis. Keyword-only.
        block_base (npt.NDArray[np.int64]): Flat-id base per block, shape
            ``(n_blocks,)``. Keyword-only.
        level_block_start (npt.NDArray[np.int64]): Block index range per level,
            shape ``(n_levels + 1,)``, non-decreasing, starting at 0 and ending at
            ``n_blocks``. Keyword-only.

    Returns:
        int: The flat cell id, or ``-1`` when the position is not an active leaf.

    Raises:
        TypeError: If any array has the wrong dtype or rank or is not C-contiguous.
        ValueError: If the block descriptor is inconsistent, if ``midx`` disagrees
            with ``block_lo`` on ``ndim``, or if ``level`` is out of range.
    """

def decode_flat_id(
    cid: int,
    *,
    block_lo: npt.NDArray[np.int64],
    block_hi: npt.NDArray[np.int64],
    block_base: npt.NDArray[np.int64],
    level_block_start: npt.NDArray[np.int64],
    out_midx: npt.NDArray[np.int64],
) -> int:
    """Recover a cell's level and level-coordinate index from its flat id.

    Args:
        cid (int): Flat cell id, at least the first block's base.
        block_lo (npt.NDArray[np.int64]): Packed block lower bounds, shape
            ``(n_blocks, ndim)``. Keyword-only.
        block_hi (npt.NDArray[np.int64]): Packed block upper bounds, same shape.
            Keyword-only.
        block_base (npt.NDArray[np.int64]): Flat-id base per block, ascending in
            flat-id order. Keyword-only.
        level_block_start (npt.NDArray[np.int64]): Block index range per level.
            Keyword-only.
        out_midx (npt.NDArray[np.int64]): Shape ``(ndim,)``, C-contiguous and
            writable. Receives the per-axis index. Keyword-only.

    Returns:
        int: The cell's level.

    Raises:
        TypeError: If any array has the wrong dtype or rank or is not C-contiguous.
        ValueError: If the block descriptor is inconsistent, if ``out_midx`` is
            shorter than ``ndim``, or if ``cid`` is below the first block's base.
    """

def hier_locate_points(
    points: npt.NDArray[np.float64],
    *,
    knots_flat: npt.NDArray[np.float64],
    knot_starts: npt.NDArray[np.int64],
    root_cells_per_axis: npt.NDArray[np.int64],
    factor: npt.NDArray[np.int64],
    block_lo: npt.NDArray[np.int64],
    block_hi: npt.NDArray[np.int64],
    block_base: npt.NDArray[np.int64],
    level_block_start: npt.NDArray[np.int64],
    out: npt.NDArray[np.int64],
) -> None:
    """Locate a batch of points on a hierarchical grid.

    Args:
        points (npt.NDArray[np.float64]): Query points, shape ``(npts, ndim)``.
        knots_flat (npt.NDArray[np.float64]): Root per-axis breakpoints concatenated
            end to end. Keyword-only.
        knot_starts (npt.NDArray[np.int64]): Per-axis offset into ``knots_flat``.
            Keyword-only.
        root_cells_per_axis (npt.NDArray[np.int64]): Per-axis root cell counts.
            Keyword-only.
        factor (npt.NDArray[np.int64]): Per-axis subdivision factor, at least one.
            A factor of one prevents subdivision on that axis, which is what an
            anisotropic grid needs; the oracle documents and tests it.
            Keyword-only.
        block_lo (npt.NDArray[np.int64]): Packed block lower bounds. Keyword-only.
        block_hi (npt.NDArray[np.int64]): Packed block upper bounds. Keyword-only.
        block_base (npt.NDArray[np.int64]): Flat-id base per block. Keyword-only.
        level_block_start (npt.NDArray[np.int64]): Block index range per level.
            Keyword-only.
        out (npt.NDArray[np.int64]): Shape ``(npts,)``, C-contiguous and writable.
            Receives the flat cell id, or ``-1`` outside the root domain.
            Keyword-only.

    Raises:
        TypeError: If any array has the wrong dtype or rank or is not C-contiguous.
        ValueError: If the descriptor arrays disagree on ``ndim``, if the block
            descriptor is inconsistent, if an axis spans past the end of
            ``knots_flat``, if a subdivision factor is below one, or if ``out`` is
            shorter than ``npts``.
    """

def hier_collect_cell_bounds(
    *,
    knots_flat: npt.NDArray[np.float64],
    knot_starts: npt.NDArray[np.int64],
    factor: npt.NDArray[np.int64],
    block_lo: npt.NDArray[np.int64],
    block_hi: npt.NDArray[np.int64],
    block_base: npt.NDArray[np.int64],
    level_block_start: npt.NDArray[np.int64],
    out_lo: npt.NDArray[np.float64],
    out_hi: npt.NDArray[np.float64],
) -> None:
    """Materialize per-cell ``(lo, hi)`` bounds in flat-id order.

    Args:
        knots_flat (npt.NDArray[np.float64]): Root per-axis breakpoints concatenated
            end to end. Keyword-only.
        knot_starts (npt.NDArray[np.int64]): Per-axis offset into ``knots_flat``.
            Keyword-only.
        factor (npt.NDArray[np.int64]): Per-axis subdivision factor, at least one.
            A factor of one prevents subdivision on that axis, which is what an
            anisotropic grid needs; the oracle documents and tests it.
            Keyword-only.
        block_lo (npt.NDArray[np.int64]): Packed block lower bounds. Keyword-only.
        block_hi (npt.NDArray[np.int64]): Packed block upper bounds. Keyword-only.
        block_base (npt.NDArray[np.int64]): Flat-id base per block. Keyword-only.
        level_block_start (npt.NDArray[np.int64]): Block index range per level.
            Keyword-only.
        out_lo (npt.NDArray[np.float64]): Output lower corners, shape
            ``(num_cells, ndim)``, C-contiguous and writable. Keyword-only.
        out_hi (npt.NDArray[np.float64]): Output upper corners, same shape.
            Keyword-only.

    Raises:
        TypeError: If any array has the wrong dtype or rank or is not C-contiguous.
        ValueError: If the block descriptor is inconsistent, if the descriptor
            arrays disagree on ``ndim``, if a subdivision factor is below one, if a
            block reaches a root cell past the end of ``knots_flat``, or if the
            outputs cannot hold every block's flat-id range.
    """
