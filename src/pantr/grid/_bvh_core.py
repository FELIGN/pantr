"""Layer-3 Numba kernels for the bounding-volume hierarchy over grid cells.

The BVH is stored as five parallel arrays (see :mod:`pantr.grid._bvh` for the
Layer-2 wrapper that owns them). Queries use a single-threaded iterative descent
with an explicit fixed-size stack so the code stays fast and allocation-free
under Numba ``nopython=True``.

Two kernels are exposed to Layer 2:

- :func:`_bvh_query_count_core` -- counts the leaves whose AABB overlaps the
  query box. Called first so Layer 2 can allocate an exact-size output array.
- :func:`_bvh_query_emit_core` -- fills a preallocated output array with the
  cell identifiers of overlapping leaves.

Both kernels share the same traversal structure and visit nodes in an identical
order so that ``count`` and ``emit`` agree on the output size. The traversal is
deterministic: the left child is pushed first and the right child last, so the
stack pops right first and the tree is visited in preorder, right to left. This
is the opposite of the left-to-right construction order in ``_build_tree``
(:mod:`pantr.grid._bvh`); both are preorder, but only the count/emit consistency
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


__all__ = [
    "_bvh_query_count_core",
    "_bvh_query_emit_core",
]
