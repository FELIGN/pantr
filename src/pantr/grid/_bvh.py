"""Bounding-volume hierarchy over grid cells.

A simple but efficient BVH that indexes a fixed collection of axis-aligned
bounding boxes (one per grid cell). The tree is built once (Layer 2, Python) by
recursive median-of-longest-axis splits; queries descend iteratively via the
Layer-3 kernels in :mod:`pantr.grid._bvh_core`.

Layout
------

The BVH is held as five parallel arrays, matching the representation consumed by
the kernels:

- ``node_lo`` / ``node_hi``: per-node AABB corners, shape ``(n_nodes, ndim)``.
- ``node_left`` / ``node_right``: child indices; ``-1`` on leaves.
- ``node_cell``: cell identifier on leaves; ``-1`` on internal nodes.

The root is always node ``0`` and covers every cell. For ``N`` cells the tree
has exactly ``2 * N - 1`` nodes (``N`` leaves, ``N - 1`` internal nodes).
Internal-node AABBs are tight: the union of the children's AABBs. Construction
stops at one cell per leaf, so leaves and cells are in one-to-one correspondence.
The construction order (preorder) is deterministic, which keeps query results
reproducible.

Queries
-------

:meth:`BVH.query_aabb` returns the ids of cells whose AABB overlaps the query
box. A touching-face pair counts as overlapping (inclusive comparison) to match
:meth:`pantr.geometry.AABB.overlaps`. Queries run in two passes: a count-only
descent sizes the output, then an emit descent writes the cell ids. Both passes
visit the same nodes in the same order, so the result is a fresh, compact
``int64`` array with no Python-side list growth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._bvh_core import (
    _BVH_STACK_DEPTH,
    _bvh_query_count_core,
    _bvh_query_emit_core,
)
from ._grid_utils import _as_float64

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..geometry import AABB


class BVH:
    """Bounding-volume hierarchy indexing a fixed collection of AABBs.

    Instances are immutable: once built, the internal arrays are flagged
    read-only. Queries allocate fresh ``int64`` output arrays per call.

    Build by passing per-cell AABBs to :meth:`from_cell_bounds`; direct
    construction from the raw array representation is supported via the default
    constructor but is mostly intended for tests.
    """

    __slots__ = (
        "_n_cells",
        "_n_nodes",
        "_ndim",
        "_node_cell",
        "_node_hi",
        "_node_left",
        "_node_lo",
        "_node_right",
    )

    def __init__(
        self,
        node_lo: npt.NDArray[np.float64],
        node_hi: npt.NDArray[np.float64],
        node_left: npt.NDArray[np.int64],
        node_right: npt.NDArray[np.int64],
        node_cell: npt.NDArray[np.int64],
    ) -> None:
        """Store the raw BVH arrays after validating their shapes.

        ``n_cells`` is derived from ``node_lo.shape[0]`` as
        ``(n_nodes + 1) // 2`` (or ``0`` when ``n_nodes == 0``); callers do not
        pass it explicitly.  Callers should prefer :meth:`from_cell_bounds`;
        this constructor is useful for tests that need to poke specific tree
        shapes.

        Args:
            node_lo (npt.NDArray[np.float64]): Per-node AABB lo corners, shape
                ``(n_nodes, ndim)``.
            node_hi (npt.NDArray[np.float64]): Per-node AABB hi corners, shape
                ``(n_nodes, ndim)``.
            node_left (npt.NDArray[np.int64]): Left-child indices; ``-1`` on
                leaves. Shape ``(n_nodes,)``.
            node_right (npt.NDArray[np.int64]): Right-child indices; ``-1`` on
                leaves.
            node_cell (npt.NDArray[np.int64]): Leaf cell identifiers; ``-1`` on
                internal nodes.

        Raises:
            TypeError: If any array has the wrong dtype.
            ValueError: If shapes are inconsistent, ``ndim`` is ``< 1``, or
                ``n_nodes`` is not ``0`` or ``2 * n_cells - 1`` for some
                ``n_cells >= 1``.
        """
        if node_lo.dtype != np.float64 or node_hi.dtype != np.float64:
            raise TypeError(
                f"node_lo / node_hi must be float64; got {node_lo.dtype!r} / {node_hi.dtype!r}."
            )
        if node_lo.ndim != 2:  # noqa: PLR2004
            raise ValueError(f"node_lo must be 2-D (n_nodes, ndim); got shape {node_lo.shape}.")
        if node_hi.shape != node_lo.shape:
            raise ValueError(
                f"node_hi shape {node_hi.shape} must match node_lo shape {node_lo.shape}."
            )
        n_nodes, ndim = int(node_lo.shape[0]), int(node_lo.shape[1])
        if ndim < 1:
            raise ValueError(f"BVH ndim must be >= 1; got {ndim}.")
        for arr, name in (
            (node_left, "node_left"),
            (node_right, "node_right"),
            (node_cell, "node_cell"),
        ):
            if arr.dtype != np.int64:
                raise TypeError(f"{name} must be int64; got {arr.dtype!r}.")
            if arr.shape != (n_nodes,):
                raise ValueError(f"{name} must have shape ({n_nodes},); got {arr.shape}.")
        # Derive n_cells from n_nodes; validate the implied node-count formula.
        n_cells_int = (n_nodes + 1) // 2 if n_nodes > 0 else 0
        expected_nodes = 2 * n_cells_int - 1 if n_cells_int > 0 else 0
        if n_nodes != expected_nodes:
            raise ValueError(
                f"BVH: n_nodes={n_nodes} is not a valid BVH node count "
                f"(must be 0 or 2*n_cells-1 for some n_cells >= 1); "
                f"implied n_cells={n_cells_int} implies n_nodes={expected_nodes}."
            )
        self._node_lo = np.ascontiguousarray(node_lo, dtype=np.float64)
        self._node_hi = np.ascontiguousarray(node_hi, dtype=np.float64)
        self._node_left = np.ascontiguousarray(node_left, dtype=np.int64)
        self._node_right = np.ascontiguousarray(node_right, dtype=np.int64)
        self._node_cell = np.ascontiguousarray(node_cell, dtype=np.int64)
        for arr_ro in (
            self._node_lo,
            self._node_hi,
            self._node_left,
            self._node_right,
            self._node_cell,
        ):
            arr_ro.flags.writeable = False
        self._ndim = ndim
        self._n_cells = n_cells_int
        self._n_nodes = n_nodes

    @classmethod
    def from_cell_bounds(
        cls,
        cell_lo: npt.ArrayLike,
        cell_hi: npt.ArrayLike,
    ) -> BVH:
        """Build a BVH over ``n_cells`` axis-aligned cell AABBs.

        Uses a top-down recursive median-of-longest-axis split. Cells are sorted
        by centroid on the longest axis; the median splits the list into two
        halves of equal size (``+/- 1``). Each leaf indexes exactly one cell.

        Args:
            cell_lo (npt.ArrayLike): Per-cell lo corners; shape
                ``(n_cells, ndim)`` with ``ndim >= 1``. Validated, not mutated.
            cell_hi (npt.ArrayLike): Per-cell hi corners; same shape and
                conventions as ``cell_lo``. Each entry must satisfy
                ``cell_hi >= cell_lo``.

        Returns:
            BVH: The constructed hierarchy.

        Raises:
            TypeError: If inputs cannot be cast to ``float64``.
            ValueError: If shapes are inconsistent, ``ndim`` is ``< 1``, any cell
                has ``hi < lo``, or the implied tree exceeds the internal stack
                depth.
        """
        lo = _as_float64(cell_lo, name="cell_lo")
        hi = _as_float64(cell_hi, name="cell_hi")
        if lo.ndim != 2:  # noqa: PLR2004
            raise ValueError(f"cell_lo must be 2-D (n_cells, ndim); got shape {lo.shape}.")
        if hi.shape != lo.shape:
            raise ValueError(f"cell_hi shape {hi.shape} must match cell_lo shape {lo.shape}.")
        n_cells, ndim = int(lo.shape[0]), int(lo.shape[1])
        if ndim < 1:
            raise ValueError(f"BVH ndim must be >= 1; got {ndim}.")
        if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
            raise ValueError(
                "BVH.from_cell_bounds: cell_lo and cell_hi must contain only finite "
                "values; got NaN or Inf."
            )
        if np.any(hi < lo):
            raise ValueError(
                "Every cell must satisfy cell_hi >= cell_lo on every axis; "
                "at least one cell violates this."
            )
        if n_cells == 0:
            empty_lo = np.zeros((0, ndim), dtype=np.float64)
            empty_hi = np.zeros((0, ndim), dtype=np.float64)
            empty_i = np.zeros(0, dtype=np.int64)
            return cls(empty_lo, empty_hi, empty_i, empty_i, empty_i)
        # Guard against the fixed-depth Numba stack in :mod:`pantr.grid._bvh_core`.
        # Median-of-longest-axis splits produce a balanced tree of height
        # ``ceil(log2(n_cells)) + 1``; the ``+ 1`` accounts for the root push.
        max_depth = int(np.ceil(np.log2(n_cells))) + 1 if n_cells > 1 else 1
        if max_depth > _BVH_STACK_DEPTH:
            raise ValueError(
                f"BVH.from_cell_bounds: {n_cells} cells would produce a tree of depth "
                f">= {max_depth}, exceeding the internal stack depth {_BVH_STACK_DEPTH}. "
                f"This is a library limit; please report this as an issue."
            )
        node_lo, node_hi, node_left, node_right, node_cell = _build_tree(lo, hi)
        return cls(node_lo, node_hi, node_left, node_right, node_cell)

    def query_aabb(self, aabb: AABB) -> npt.NDArray[np.int64]:
        """Return the ids of every leaf cell whose AABB overlaps ``aabb``.

        Args:
            aabb (AABB): Query box; must match :attr:`ndim`.

        Returns:
            npt.NDArray[np.int64]: Overlapping cell ids. Order matches the
            internal preorder traversal; callers that need a particular order
            should sort the result.

        Raises:
            ValueError: If ``aabb.ndim != self.ndim``.
        """
        if aabb.ndim != self._ndim:
            raise ValueError(
                f"BVH.query_aabb: aabb.ndim ({aabb.ndim}) must match self.ndim ({self._ndim})."
            )
        if self._n_cells == 0:
            return np.zeros(0, dtype=np.int64)
        count = int(
            _bvh_query_count_core(
                aabb.lo,
                aabb.hi,
                self._node_lo,
                self._node_hi,
                self._node_left,
                self._node_right,
                self._node_cell,
            )
        )
        out = np.empty(count, dtype=np.int64)
        if count == 0:
            return out
        written = int(
            _bvh_query_emit_core(
                aabb.lo,
                aabb.hi,
                self._node_lo,
                self._node_hi,
                self._node_left,
                self._node_right,
                self._node_cell,
                out,
            )
        )
        if written != count:
            raise RuntimeError(
                f"BVH.query_aabb: internal count/emit mismatch "
                f"(count={count}, written={written}, "
                f"n_cells={self._n_cells}, ndim={self._ndim}, "
                f"query=[{aabb.lo.tolist()}, {aabb.hi.tolist()}]). "
                f"This is a bug in the BVH kernel; please report it."
            )
        return out

    @property
    def ndim(self) -> int:
        """Get the spatial dimension of the indexed AABBs.

        Returns:
            int: Number of axes (``>= 1``).
        """
        return self._ndim

    @property
    def n_cells(self) -> int:
        """Get the number of cells indexed (equal to the number of leaves).

        Returns:
            int: ``(n_nodes + 1) // 2``, or ``0`` for an empty BVH.
        """
        return self._n_cells

    @property
    def n_nodes(self) -> int:
        """Get the total number of nodes.

        Returns:
            int: ``2 * n_cells - 1`` for ``n_cells > 0``, else ``0``.
        """
        return self._n_nodes

    def __repr__(self) -> str:
        """Return a concise representation useful for debugging.

        Returns:
            str: ``"BVH(ndim=..., n_cells=..., n_nodes=...)"``
        """
        return f"BVH(ndim={self._ndim}, n_cells={self._n_cells}, n_nodes={self._n_nodes})"

    @property
    def node_lo(self) -> npt.NDArray[np.float64]:
        """Get the read-only view of per-node AABB lo corners.

        Returns:
            npt.NDArray[np.float64]: Shape ``(n_nodes, ndim)``.
        """
        return self._node_lo

    @property
    def node_hi(self) -> npt.NDArray[np.float64]:
        """Get the read-only view of per-node AABB hi corners.

        Returns:
            npt.NDArray[np.float64]: Shape ``(n_nodes, ndim)``.
        """
        return self._node_hi

    @property
    def node_left(self) -> npt.NDArray[np.int64]:
        """Get the read-only view of per-node left-child indices.

        Returns:
            npt.NDArray[np.int64]: Shape ``(n_nodes,)``; ``-1`` on leaves.
        """
        return self._node_left

    @property
    def node_right(self) -> npt.NDArray[np.int64]:
        """Get the read-only view of per-node right-child indices.

        Returns:
            npt.NDArray[np.int64]: Shape ``(n_nodes,)``; ``-1`` on leaves.
        """
        return self._node_right

    @property
    def node_cell(self) -> npt.NDArray[np.int64]:
        """Get the read-only view of per-leaf cell identifiers.

        Returns:
            npt.NDArray[np.int64]: Shape ``(n_nodes,)``; ``-1`` on internal
            nodes, ``0 <= id < n_cells`` on leaves.
        """
        return self._node_cell


def _build_tree(
    cell_lo: npt.NDArray[np.float64],
    cell_hi: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]:
    """Build the BVH arrays from validated per-cell AABBs.

    Iterative top-down construction with a work stack. The root claims node index
    ``0``; subsequent splits claim indices in preorder.

    Args:
        cell_lo (npt.NDArray[np.float64]): Per-cell lo corners, shape
            ``(n_cells, ndim)``, ``n_cells >= 1``.
        cell_hi (npt.NDArray[np.float64]): Per-cell hi corners, same shape;
            ``hi >= lo`` is assumed.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64],
        npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        ``(node_lo, node_hi, node_left, node_right, node_cell)`` as defined in the
        module docstring.

    Note:
        Inputs are assumed validated by the caller. This helper never raises.
    """
    n_cells, ndim = cell_lo.shape
    max_nodes = 2 * n_cells - 1
    node_lo = np.empty((max_nodes, ndim), dtype=np.float64)
    node_hi = np.empty((max_nodes, ndim), dtype=np.float64)
    node_left = np.full(max_nodes, -1, dtype=np.int64)
    node_right = np.full(max_nodes, -1, dtype=np.int64)
    node_cell = np.full(max_nodes, -1, dtype=np.int64)
    centroid = 0.5 * (cell_lo + cell_hi)
    perm = np.arange(n_cells, dtype=np.int64)
    # Work stack: (node_idx, cell_start, cell_end).
    work: list[tuple[int, int, int]] = [(0, 0, n_cells)]
    next_idx = 1
    while work:
        idx, start, end = work.pop()
        sub = perm[start:end]
        node_lo[idx] = cell_lo[sub].min(axis=0)
        node_hi[idx] = cell_hi[sub].max(axis=0)
        count = end - start
        if count == 1:
            node_cell[idx] = int(sub[0])
            continue
        extent = node_hi[idx] - node_lo[idx]
        axis = int(np.argmax(extent))
        order = np.argsort(centroid[sub, axis], kind="stable")
        perm[start:end] = sub[order]
        mid = start + count // 2
        left_idx = next_idx
        right_idx = next_idx + 1
        next_idx += 2
        node_left[idx] = left_idx
        node_right[idx] = right_idx
        # Push right first so left pops first (preorder left-to-right construction
        # order, matching the _bvh.py module docstring). Note: the query kernels
        # in _bvh_core.py push left first and visit right first — opposite
        # direction — but count and emit use the same order, so results agree.
        work.append((right_idx, mid, end))
        work.append((left_idx, start, mid))
    return node_lo, node_hi, node_left, node_right, node_cell


__all__ = ["BVH"]
