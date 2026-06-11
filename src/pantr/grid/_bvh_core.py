"""Layer-3 Numba kernels for the bounding-volume hierarchy over grid cells.

The BVH is stored as five parallel arrays (see :mod:`pantr.grid._bvh` for the
Layer-2 wrapper that owns them). Queries use a single-threaded iterative descent
with an explicit fixed-size stack so the code stays fast and allocation-free
under Numba ``nopython=True``.

Three kernels are exposed to Layer 2:

- :func:`_bvh_build_core` -- top-down median-of-longest-axis construction,
  filling the five node arrays in preorder (left to right).
- :func:`_bvh_query_count_core` -- counts the leaves whose AABB overlaps the
  query box. Called first so Layer 2 can allocate an exact-size output array.
- :func:`_bvh_query_emit_core` -- fills a preallocated output array with the
  cell identifiers of overlapping leaves.

Both kernels share the same traversal structure and visit nodes in an identical
order so that ``count`` and ``emit`` agree on the output size. The traversal is
deterministic: the left child is pushed first and the right child last, so the
stack pops right first and the tree is visited in preorder, right to left. This
is the opposite of the left-to-right construction order in
:func:`_bvh_build_core`; both are preorder, but only the count/emit consistency
matters for correctness.

The stack is a fixed-size ``int64`` array. For ``N`` cells a balanced
median-split BVH has height ``ceil(log2(N)) + 1``; depth 128 covers any
realistic cell count on a single process. Layer 2 validates the depth bound at
construction time so the kernels never overflow their stack.

The AABB overlap test is inclusive on every face so that a query box sharing a
face with a cell is reported as overlapping, matching
:meth:`pantr.geometry.AABB.overlaps`.

Note:
    Inputs are assumed to be correct (no validation performed).
    For general use, call :class:`pantr.grid.BVH` instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np

from .._numba_compat import nb_jit

if TYPE_CHECKING:
    import numpy.typing as npt


# Maximum depth of the iterative-descent stack. 128 is far larger than any
# balanced BVH over an ``int64``-indexed cell set but small enough to stay
# comfortably on the Numba-allocated function stack.
_BVH_STACK_DEPTH: Final[int] = 128


@nb_jit(nopython=True, cache=True)
def _bvh_build_core(  # noqa: PLR0913, PLR0915 -- kernel fills a flat BVH struct
    cell_lo: npt.NDArray[np.float64],
    cell_hi: npt.NDArray[np.float64],
    node_lo: npt.NDArray[np.float64],
    node_hi: npt.NDArray[np.float64],
    node_left: npt.NDArray[np.int64],
    node_right: npt.NDArray[np.int64],
    node_cell: npt.NDArray[np.int64],
) -> None:
    """Build the BVH arrays from per-cell AABBs (median-of-longest-axis splits).

    Iterative top-down construction with an explicit work stack; the root
    claims node ``0`` and subsequent splits claim indices in preorder, left to
    right.  At each node the cells are sorted by centroid along the longest
    axis with a *stable* sort and split at the median, which makes the tree a
    deterministic function of the input (and identical to the previous
    pure-Python builder).

    The work stack holds at most one pending right-sibling range per level of
    the current descent path, so its peak size is bounded by the tree height;
    Layer 2 validates the ``_BVH_STACK_DEPTH`` bound before calling.

    Args:
        cell_lo (npt.NDArray[np.float64]): Per-cell lo corners, shape
            ``(n_cells, ndim)``, ``n_cells >= 1``.
        cell_hi (npt.NDArray[np.float64]): Per-cell hi corners, same shape;
            ``hi >= lo`` is assumed.
        node_lo (npt.NDArray[np.float64]): Output per-node lo corners, shape
            ``(2 * n_cells - 1, ndim)``.
        node_hi (npt.NDArray[np.float64]): Output per-node hi corners, same
            shape.
        node_left (npt.NDArray[np.int64]): Output left-child indices, shape
            ``(2 * n_cells - 1,)``; must be pre-filled with ``-1``.
        node_right (npt.NDArray[np.int64]): Output right-child indices, same
            shape and pre-fill.
        node_cell (npt.NDArray[np.int64]): Output per-leaf cell ids, same
            shape and pre-fill.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :meth:`pantr.grid.BVH.from_cell_bounds` instead.
    """
    n_cells = cell_lo.shape[0]
    ndim = cell_lo.shape[1]

    centroid = 0.5 * (cell_lo + cell_hi)
    perm = np.arange(n_cells, dtype=np.int64)

    # Work stack of (node_idx, cell_start, cell_end) triples.
    stack_idx = np.empty(_BVH_STACK_DEPTH, dtype=np.int64)
    stack_start = np.empty(_BVH_STACK_DEPTH, dtype=np.int64)
    stack_end = np.empty(_BVH_STACK_DEPTH, dtype=np.int64)
    stack_idx[0] = 0
    stack_start[0] = 0
    stack_end[0] = n_cells
    sp = 1
    next_idx = 1

    while sp > 0:
        sp -= 1
        idx = stack_idx[sp]
        start = stack_start[sp]
        end = stack_end[sp]

        # Tight AABB of the node: min/max over its cells.
        for k in range(ndim):
            lo_k = cell_lo[perm[start], k]
            hi_k = cell_hi[perm[start], k]
            for c in range(start + 1, end):
                v_lo = cell_lo[perm[c], k]
                v_hi = cell_hi[perm[c], k]
                lo_k = min(lo_k, v_lo)
                hi_k = max(hi_k, v_hi)
            node_lo[idx, k] = lo_k
            node_hi[idx, k] = hi_k

        count = end - start
        if count == 1:
            node_cell[idx] = perm[start]
            continue

        # Longest axis of the node AABB (first maximum, matching np.argmax).
        axis = 0
        best = node_hi[idx, 0] - node_lo[idx, 0]
        for k in range(1, ndim):
            extent_k = node_hi[idx, k] - node_lo[idx, k]
            if extent_k > best:
                best = extent_k
                axis = k
        # Stable sort of the node's cells by centroid along the split axis.
        vals = np.empty(count, dtype=np.float64)
        for c in range(count):
            vals[c] = centroid[perm[start + c], axis]
        order = np.argsort(vals, kind="mergesort")
        sub = perm[start:end].copy()
        for c in range(count):
            perm[start + c] = sub[order[c]]

        mid = start + count // 2
        left_idx = next_idx
        right_idx = next_idx + 1
        next_idx += 2
        node_left[idx] = left_idx
        node_right[idx] = right_idx
        # Push right first so left pops first (preorder left-to-right
        # construction order, matching the _bvh.py module docstring). Note: the
        # query kernels below push left first and visit right first — opposite
        # direction — but count and emit use the same order, so results agree.
        stack_idx[sp] = right_idx
        stack_start[sp] = mid
        stack_end[sp] = end
        stack_idx[sp + 1] = left_idx
        stack_start[sp + 1] = start
        stack_end[sp + 1] = mid
        sp += 2


@nb_jit(nopython=True, cache=True)
def _bvh_query_count_core(  # noqa: PLR0913 -- kernel consumes a flat BVH struct
    qlo: npt.NDArray[np.float64],
    qhi: npt.NDArray[np.float64],
    node_lo: npt.NDArray[np.float64],
    node_hi: npt.NDArray[np.float64],
    node_left: npt.NDArray[np.int64],
    node_right: npt.NDArray[np.int64],
    node_cell: npt.NDArray[np.int64],
) -> int:
    """Count leaves whose AABB overlaps the query box.

    Args:
        qlo (npt.NDArray[np.float64]): Query AABB lower corner, ``float64``,
            shape ``(ndim,)``.
        qhi (npt.NDArray[np.float64]): Query AABB upper corner, ``float64``,
            shape ``(ndim,)``.
        node_lo (npt.NDArray[np.float64]): Per-node AABB lower corners, shape
            ``(n_nodes, ndim)``.
        node_hi (npt.NDArray[np.float64]): Per-node AABB upper corners, shape
            ``(n_nodes, ndim)``.
        node_left (npt.NDArray[np.int64]): Left-child indices; ``-1`` marks a
            leaf. Shape ``(n_nodes,)``.
        node_right (npt.NDArray[np.int64]): Right-child indices; ``-1`` marks a
            leaf. Shape ``(n_nodes,)``.
        node_cell (npt.NDArray[np.int64]): Cell identifier per leaf; ``-1`` on
            internal nodes. Shape ``(n_nodes,)``.

    Returns:
        int: Number of overlapping leaves.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :meth:`pantr.grid.BVH.query_aabb` instead.
    """
    ndim = qlo.shape[0]
    stack = np.empty(_BVH_STACK_DEPTH, dtype=np.int64)
    stack[0] = 0
    top = 1
    count = 0
    while top > 0:
        top -= 1
        idx = stack[top]
        overlaps = True
        for d in range(ndim):
            if qhi[d] < node_lo[idx, d] or qlo[d] > node_hi[idx, d]:
                overlaps = False
                break
        if not overlaps:
            continue
        cell = node_cell[idx]
        if cell >= 0:
            count += 1
        else:
            stack[top] = node_left[idx]
            top += 1
            stack[top] = node_right[idx]
            top += 1
    return count


@nb_jit(nopython=True, cache=True)
def _bvh_query_emit_core(  # noqa: PLR0913 -- kernel consumes a flat BVH struct
    qlo: npt.NDArray[np.float64],
    qhi: npt.NDArray[np.float64],
    node_lo: npt.NDArray[np.float64],
    node_hi: npt.NDArray[np.float64],
    node_left: npt.NDArray[np.int64],
    node_right: npt.NDArray[np.int64],
    node_cell: npt.NDArray[np.int64],
    out: npt.NDArray[np.int64],
) -> int:
    """Emit overlapping leaf cell ids into ``out`` and return the count.

    Uses the same traversal order as :func:`_bvh_query_count_core`, so
    ``out[:count]`` holds exactly the same cells that would have been counted.
    ``out`` must be large enough to hold every overlapping cell; Layer 2 sizes
    it using the count pass.

    Args:
        qlo (npt.NDArray[np.float64]): Query AABB lower corner.
        qhi (npt.NDArray[np.float64]): Query AABB upper corner.
        node_lo (npt.NDArray[np.float64]): Per-node AABB lower corners.
        node_hi (npt.NDArray[np.float64]): Per-node AABB upper corners.
        node_left (npt.NDArray[np.int64]): Left-child indices.
        node_right (npt.NDArray[np.int64]): Right-child indices.
        node_cell (npt.NDArray[np.int64]): Leaf cell ids (or ``-1``).
        out (npt.NDArray[np.int64]): Output buffer, shape ``(n_overlaps_or_more,)``.

    Returns:
        int: Number of cell ids written to ``out``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :meth:`pantr.grid.BVH.query_aabb` instead.
    """
    ndim = qlo.shape[0]
    stack = np.empty(_BVH_STACK_DEPTH, dtype=np.int64)
    stack[0] = 0
    top = 1
    count = 0
    while top > 0:
        top -= 1
        idx = stack[top]
        overlaps = True
        for d in range(ndim):
            if qhi[d] < node_lo[idx, d] or qlo[d] > node_hi[idx, d]:
                overlaps = False
                break
        if not overlaps:
            continue
        cell = node_cell[idx]
        if cell >= 0:
            out[count] = cell
            count += 1
        else:
            stack[top] = node_left[idx]
            top += 1
            stack[top] = node_right[idx]
            top += 1
    return count


def _warmup_numba_functions() -> None:
    """Trigger compilation of the BVH kernels on a tiny single-cell tree.

    Provided for explicit, on-demand warmup (for example in benchmarks). It is
    deliberately *not* called from :mod:`pantr`'s import-time warmup: the grid
    kernels compile lazily on first use and are cached to disk by Numba's
    ``cache=True``.
    """
    node_lo = np.zeros((1, 1), dtype=np.float64)
    node_hi = np.ones((1, 1), dtype=np.float64)
    leaf = np.zeros(1, dtype=np.int64)
    internal = np.full(1, -1, dtype=np.int64)
    qlo = np.zeros(1, dtype=np.float64)
    qhi = np.ones(1, dtype=np.float64)
    _bvh_query_count_core(qlo, qhi, node_lo, node_hi, internal, internal, leaf)
    out = np.empty(1, dtype=np.int64)
    _bvh_query_emit_core(qlo, qhi, node_lo, node_hi, internal, internal, leaf, out)
    cell_lo = np.zeros((1, 1), dtype=np.float64)
    cell_hi = np.ones((1, 1), dtype=np.float64)
    b_lo = np.empty((1, 1), dtype=np.float64)
    b_hi = np.empty((1, 1), dtype=np.float64)
    b_left = np.full(1, -1, dtype=np.int64)
    b_right = np.full(1, -1, dtype=np.int64)
    b_cell = np.full(1, -1, dtype=np.int64)
    _bvh_build_core(cell_lo, cell_hi, b_lo, b_hi, b_left, b_right, b_cell)


__all__ = [
    "_bvh_build_core",
    "_bvh_query_count_core",
    "_bvh_query_emit_core",
]
