"""Layer-3 Numba kernels for batch operations on a hierarchical grid.

A :class:`~pantr.grid.HierarchicalGrid` stores its active cells as rectangular
blocks per level.  For Numba consumption the per-level block lists are packed
into flat ``int64`` arrays (rebuilt on every structural change):

- ``block_lo`` / ``block_hi``: ``(n_blocks_total, ndim)`` block bounds, in the
  coordinates of their level, concatenated level by level in flat-id order.
- ``block_base``: ``(n_blocks_total,)`` flat cell id of each block's first cell.
- ``level_block_start``: ``(n_levels + 1,)`` block index range of each level.

The root grid's per-axis breakpoints are passed as a single concatenated
``float64`` array plus per-axis start offsets (the same flat descriptor used by
:mod:`pantr.grid._locate_core`).

Note:
    Inputs are assumed to be correct (no validation performed).
    For general use, call the :class:`~pantr.grid.HierarchicalGrid` methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._numba_compat import nb_jit, nb_prange

if TYPE_CHECKING:
    import numpy.typing as npt


@nb_jit(nopython=True, cache=True, inline="always")
def _block_of_midx(  # noqa: PLR0913
    level: int,
    midx: npt.NDArray[np.int64],
    block_lo: npt.NDArray[np.int64],
    block_hi: npt.NDArray[np.int64],
    block_base: npt.NDArray[np.int64],
    level_block_start: npt.NDArray[np.int64],
) -> int:
    """Return the flat cell id of ``(level, midx)``, or ``-1`` when not active.

    Scans the level's blocks (small lists in practice) and, on a hit, combines
    the block's flat-id base with the C-order offset of ``midx`` inside it.

    Args:
        level (int): Hierarchy level of the queried position.
        midx (npt.NDArray[np.int64]): Per-axis index, shape ``(ndim,)``, in
            level-``level`` coordinates.
        block_lo (npt.NDArray[np.int64]): Packed block lower bounds.
        block_hi (npt.NDArray[np.int64]): Packed block upper bounds.
        block_base (npt.NDArray[np.int64]): Flat-id base per block.
        level_block_start (npt.NDArray[np.int64]): Block index range per level.

    Returns:
        int: Flat cell id, or ``-1`` when ``(level, midx)`` is not an active leaf.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    ndim = midx.shape[0]
    for b in range(level_block_start[level], level_block_start[level + 1]):
        inside = True
        for k in range(ndim):
            if midx[k] < block_lo[b, k] or midx[k] >= block_hi[b, k]:
                inside = False
                break
        if not inside:
            continue
        offset = 0
        for k in range(ndim):
            offset = offset * int(block_hi[b, k] - block_lo[b, k]) + int(midx[k] - block_lo[b, k])
        return int(block_base[b]) + offset
    return -1


@nb_jit(nopython=True, cache=True)
def _encode_midx_core(  # noqa: PLR0913
    level: int,
    midx: npt.NDArray[np.int64],
    block_lo: npt.NDArray[np.int64],
    block_hi: npt.NDArray[np.int64],
    block_base: npt.NDArray[np.int64],
    level_block_start: npt.NDArray[np.int64],
) -> int:
    """Return the flat cell id of ``(level, midx)``, or ``-1`` when not active.

    Serial entry point wrapping :func:`_block_of_midx` for scalar Python
    callers (:meth:`pantr.grid.HierarchicalGrid._encode_midx`).

    Args:
        level (int): Hierarchy level of the queried position.
        midx (npt.NDArray[np.int64]): Per-axis index, shape ``(ndim,)``.
        block_lo (npt.NDArray[np.int64]): Packed block lower bounds.
        block_hi (npt.NDArray[np.int64]): Packed block upper bounds.
        block_base (npt.NDArray[np.int64]): Flat-id base per block.
        level_block_start (npt.NDArray[np.int64]): Block index range per level.

    Returns:
        int: Flat cell id, or ``-1`` when ``(level, midx)`` is not an active leaf.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    return _block_of_midx(level, midx, block_lo, block_hi, block_base, level_block_start)


@nb_jit(nopython=True, cache=True)
def _decode_flat_id_core(  # noqa: PLR0913
    cid: int,
    block_lo: npt.NDArray[np.int64],
    block_hi: npt.NDArray[np.int64],
    block_base: npt.NDArray[np.int64],
    level_block_start: npt.NDArray[np.int64],
    out_midx: npt.NDArray[np.int64],
) -> int:
    """Decode a flat cell id into its level and per-axis multi-index.

    Binary-searches ``block_base`` (globally ascending in flat-id order) for
    the containing block, recovers the block's level from
    ``level_block_start``, and expands the in-block C-order offset.

    Args:
        cid (int): Flat cell id in ``[0, num_cells)``.
        block_lo (npt.NDArray[np.int64]): Packed block lower bounds.
        block_hi (npt.NDArray[np.int64]): Packed block upper bounds.
        block_base (npt.NDArray[np.int64]): Flat-id base per block.
        level_block_start (npt.NDArray[np.int64]): Block index range per level.
        out_midx (npt.NDArray[np.int64]): Output per-axis index, shape
            ``(ndim,)``, in level coordinates.

    Returns:
        int: The cell's level.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    ndim = out_midx.shape[0]
    n_blocks = block_base.shape[0]

    # upper_bound: first block whose base exceeds cid, minus one.
    lo_b = 0
    hi_b = n_blocks
    while lo_b < hi_b:
        mid = (lo_b + hi_b) // 2
        if block_base[mid] <= cid:
            lo_b = mid + 1
        else:
            hi_b = mid
    b = lo_b - 1

    # Level of block b.
    n_levels = level_block_start.shape[0] - 1
    level = 0
    for lev in range(n_levels):
        if level_block_start[lev] <= b < level_block_start[lev + 1]:
            level = lev
            break

    # Expand the C-order offset inside the block.
    offset = cid - block_base[b]
    for k in range(ndim - 1, -1, -1):
        extent = block_hi[b, k] - block_lo[b, k]
        out_midx[k] = block_lo[b, k] + offset % extent
        offset //= extent
    return level


@nb_jit(nopython=True, cache=True, parallel=True)
def _hier_locate_points_core(  # noqa: PLR0912, PLR0913, PLR0915 -- flat grid descriptor
    points: npt.NDArray[np.float64],
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
    """Locate a batch of points on a hierarchical grid (top-down descent).

    Mirrors :meth:`pantr.grid.HierarchicalGrid.locate`: each point is located in
    the root grid by per-axis binary search, then descended level by level —
    at each level the position is looked up in the active blocks, and on a miss
    the containing child cell is computed from the floating-point cell bounds
    (identical arithmetic to the scalar method, so results agree exactly).

    Args:
        points (npt.NDArray[np.float64]): Query points, shape ``(npts, ndim)``.
        knots_flat (npt.NDArray[np.float64]): Root per-axis breakpoints
            concatenated end to end; axis ``d`` occupies
            ``knots_flat[knot_starts[d] : knot_starts[d] + root_cells_per_axis[d] + 1]``.
        knot_starts (npt.NDArray[np.int64]): Per-axis start offset into
            ``knots_flat``. Shape ``(ndim,)``.
        root_cells_per_axis (npt.NDArray[np.int64]): Root per-axis cell counts.
        factor (npt.NDArray[np.int64]): Per-axis subdivision factor.
        block_lo (npt.NDArray[np.int64]): Packed block lower bounds.
        block_hi (npt.NDArray[np.int64]): Packed block upper bounds.
        block_base (npt.NDArray[np.int64]): Flat-id base per block.
        level_block_start (npt.NDArray[np.int64]): Block index range per level.
        out (npt.NDArray[np.int64]): Output flat cell ids, shape ``(npts,)``;
            ``-1`` for points outside the grid domain.

    Note:
        Inputs are assumed to be correct (no validation performed).  NaN
        coordinates are not handled here (all comparisons are False); the
        caller masks them out.
        For general use, call :meth:`pantr.grid.HierarchicalGrid.locate_many`.
    """
    npts = points.shape[0]
    ndim = points.shape[1]
    n_levels = level_block_start.shape[0] - 1

    for p in nb_prange(npts):
        midx = np.empty(ndim, dtype=np.int64)
        lo = np.empty(ndim, dtype=np.float64)
        hi = np.empty(ndim, dtype=np.float64)

        # --- Root-level location: per-axis lower_bound binary search. ---
        inside = True
        for d in range(ndim):
            start = knot_starts[d]
            ncells = root_cells_per_axis[d]
            x = points[p, d]
            if x < knots_flat[start] or x > knots_flat[start + ncells]:
                inside = False
                break
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
            midx[d] = cell_i
            lo[d] = knots_flat[start + cell_i]
            hi[d] = knots_flat[start + cell_i + 1]
        if not inside:
            out[p] = -1
            continue

        # --- Top-down descent through the levels. ---
        result = -1
        for level in range(n_levels):
            cid = _block_of_midx(level, midx, block_lo, block_hi, block_base, level_block_start)
            if cid >= 0:
                result = cid
                break
            if level >= n_levels - 1:
                break  # unreachable in a consistent grid
            # Descend: find the child of (level, midx) containing the point.
            for k in range(ndim):
                fk = factor[k]
                size_k = (hi[k] - lo[k]) / fk
                j = int((points[p, k] - lo[k]) / size_k)
                if j < 0:
                    j = 0
                elif j > fk - 1:
                    j = fk - 1
                lo[k] = lo[k] + j * size_k
                hi[k] = lo[k] + size_k
                midx[k] = midx[k] * fk + j
        out[p] = result


@nb_jit(nopython=True, cache=True, parallel=True)
def _hier_collect_cell_bounds_core(  # noqa: PLR0913 -- flat hierarchical grid descriptor
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

    Parallelizes over blocks; each block writes the contiguous flat-id range
    ``[block_base[b], block_base[b] + size)`` enumerating its cells in C-order.
    Per-cell bounds use the same arithmetic as
    :meth:`pantr.grid.HierarchicalGrid.cell_bounds` (root breakpoint looked up
    by integer division, child size as the root span divided by
    ``factor ** level``), so results agree exactly with the scalar method.

    Args:
        knots_flat (npt.NDArray[np.float64]): Root per-axis breakpoints
            concatenated end to end (see :func:`_hier_locate_points_core`).
        knot_starts (npt.NDArray[np.int64]): Per-axis start offset into
            ``knots_flat``. Shape ``(ndim,)``.
        factor (npt.NDArray[np.int64]): Per-axis subdivision factor.
        block_lo (npt.NDArray[np.int64]): Packed block lower bounds.
        block_hi (npt.NDArray[np.int64]): Packed block upper bounds.
        block_base (npt.NDArray[np.int64]): Flat-id base per block.
        level_block_start (npt.NDArray[np.int64]): Block index range per level.
        out_lo (npt.NDArray[np.float64]): Output lower corners,
            shape ``(num_cells, ndim)``.
        out_hi (npt.NDArray[np.float64]): Output upper corners, same shape.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call
        :meth:`pantr.grid.HierarchicalGrid.collect_cell_bounds`.
    """
    ndim = block_lo.shape[1]
    n_levels = level_block_start.shape[0] - 1
    n_blocks = block_base.shape[0]

    for b in nb_prange(n_blocks):
        # Recover the block's level (blocks are packed level by level).
        level = 0
        for lev in range(n_levels):
            if level_block_start[lev] <= b < level_block_start[lev + 1]:
                level = lev
                break

        # Per-axis subdivision m_pow = factor[k] ** level.
        m_pow = np.empty(ndim, dtype=np.int64)
        for k in range(ndim):
            mp = np.int64(1)
            for _ in range(level):
                mp *= factor[k]
            m_pow[k] = mp

        # Odometer over the block's cells in C-order.
        midx = np.empty(ndim, dtype=np.int64)
        n_block = np.int64(1)
        for k in range(ndim):
            midx[k] = block_lo[b, k]
            n_block *= block_hi[b, k] - block_lo[b, k]

        flat_id = block_base[b]
        for _ in range(n_block):
            for k in range(ndim):
                ik = midx[k]
                root_ik = ik // m_pow[k]
                sub_ik = ik % m_pow[k]
                root_lo_k = knots_flat[knot_starts[k] + root_ik]
                root_hi_k = knots_flat[knot_starts[k] + root_ik + 1]
                size_k = (root_hi_k - root_lo_k) / m_pow[k]
                out_lo[flat_id, k] = root_lo_k + sub_ik * size_k
                out_hi[flat_id, k] = out_lo[flat_id, k] + size_k
            flat_id += 1
            # Increment the odometer (last axis fastest).
            for k in range(ndim - 1, -1, -1):
                midx[k] += 1
                if midx[k] < block_hi[b, k]:
                    break
                midx[k] = block_lo[b, k]


__all__ = [
    "_decode_flat_id_core",
    "_encode_midx_core",
    "_hier_collect_cell_bounds_core",
    "_hier_locate_points_core",
]
