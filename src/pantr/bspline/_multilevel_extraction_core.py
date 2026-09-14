r"""Numba kernels for the windowed multi-level extraction operator (Layer 3).

On an active cell ``e`` of level ``L`` every level-``m`` function that is non-zero on
``e`` lies in the window ``W_m`` of the ``prod(p_k + 1)`` level-``m`` functions supported
on ``e``'s level-``m`` ancestor, because supports are unions of level-``m`` cells.  The
multi-level extraction operator :math:`M^e` is therefore built without leaving those
windows: each candidate function starts as a unit vector in its own level's window and is
pushed one level at a time through the two-scale block restricted to the two windows (a
Kronecker product of ``(p_k + 1) x (p_k + 1)`` 1D blocks), with the truncation mask of
the finer level applied after each step.  Restricting before truncating is exact:
functions outside ``W_{m+1}`` vanish on ``e``, and so do all their refinements.

Which functions vanish on ``e`` is decided before any row is pushed.  Every two-scale
coefficient is nonnegative and truncation only zeroes entries, so no sum cancels, and a
coefficient at window position ``b`` of level ``m`` reaches the cell iff some unmasked
position of level ``m + 1`` it refines into does.  That set is one backward sweep of the
transposed blocks per level, after which only the rows that survive are pushed.

Working memory is ``(L + 1) * prod(p_k + 1)`` rows of ``prod(p_k + 1)`` doubles, however
deep the hierarchy.

Main exports:

- :func:`_windowed_multilevel_rows`: rows of :math:`M^e` for one cell.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._numba_compat import nb_jit


@nb_jit(nopython=True, cache=True)
def _refine_axis(
    block: npt.NDArray[np.float64],
    size: int,
    inner: int,
    src: npt.NDArray[np.float64],
    dst: npt.NDArray[np.float64],
) -> None:
    """Apply one direction's two-scale block to a window vector: ``dst = (I x B x I) src``.

    Each output entry accumulates its terms in increasing ``b``, and the innermost loop
    runs over contiguous runs of both arrays.

    Args:
        block (npt.NDArray[np.float64]): Block of at least ``(size, size)``; entry
            ``[a, b]`` maps input index ``b`` to output index ``a`` along the axis.
        size (int): Extent of the axis.
        inner (int): Product of the extents of the later axes (the C-order stride).
        src (npt.NDArray[np.float64]): Input window vector, shape ``(n,)``.
        dst (npt.NDArray[np.float64]): Output window vector, shape ``(n,)``; must not
            alias ``src``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :meth:`~pantr.bspline.MultiLevelExtraction.multilevel_operator`
        instead.
    """
    n = src.shape[0]
    outer = n // (size * inner)
    dst[:] = 0.0
    for o in range(outer):
        for a in range(size):
            out_start = (o * size + a) * inner
            for b in range(size):
                coef = block[a, b]
                if coef == 0.0:
                    continue
                in_start = (o * size + b) * inner
                for i in range(inner):
                    dst[out_start + i] += coef * src[in_start + i]


@nb_jit(nopython=True, cache=True)
def _reach_axis(
    block: npt.NDArray[np.float64],
    size: int,
    inner: int,
    src: npt.NDArray[np.bool_],
    dst: npt.NDArray[np.bool_],
) -> None:
    """Propagate a support pattern backwards through one direction's two-scale block.

    ``dst[b]`` is ``True`` iff some ``a`` with ``block[a, b] != 0`` along the axis has
    ``src[a]`` set: the positions of the coarser window whose refinement touches the set.

    Args:
        block (npt.NDArray[np.float64]): Block of at least ``(size, size)``, as in
            :func:`_refine_axis`.
        size (int): Extent of the axis.
        inner (int): Product of the extents of the later axes.
        src (npt.NDArray[np.bool_]): Pattern over the finer window, shape ``(n,)``.
        dst (npt.NDArray[np.bool_]): Pattern over the coarser window, shape ``(n,)``;
            must not alias ``src``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :meth:`~pantr.bspline.MultiLevelExtraction.multilevel_operator`
        instead.
    """
    n = src.shape[0]
    outer = n // (size * inner)
    dst[:] = False
    for o in range(outer):
        for a in range(size):
            in_start = (o * size + a) * inner
            for b in range(size):
                if block[a, b] == 0.0:
                    continue
                out_start = (o * size + b) * inner
                for i in range(inner):
                    dst[out_start + i] = dst[out_start + i] or src[in_start + i]


@nb_jit(nopython=True, cache=True)
def _windowed_multilevel_rows(  # noqa: PLR0912, PLR0913, PLR0915
    cell_level: int,
    cell_multi: npt.NDArray[np.int64],
    factor: npt.NDArray[np.int64],
    degrees: npt.NDArray[np.int64],
    num_basis: npt.NDArray[np.int64],
    first_basis: npt.NDArray[np.int64],
    first_basis_offset: npt.NDArray[np.int64],
    two_scale: npt.NDArray[np.float64],
    two_scale_offset: npt.NDArray[np.int64],
    active: npt.NDArray[np.int64],
    func_offset: npt.NDArray[np.int64],
    truncate: bool,
    rows: npt.NDArray[np.float64],
    dofs: npt.NDArray[np.int64],
    nonzero: npt.NDArray[np.bool_],
) -> int:
    """Write the rows of the multi-level extraction operator of one cell.

    Rows come out in increasing global dof: levels are visited coarse to fine, and within
    a level the window is walked in C-order, which is increasing flat index.  Every active
    function whose tensor-product support covers the cell gets a row, including those the
    truncation annihilates on the cell; ``nonzero`` tells them apart, and a row flagged
    ``False`` is all zeros.

    Args:
        cell_level (int): Level ``L`` of the cell.
        cell_multi (npt.NDArray[np.int64]): Per-direction cell index at level ``L``,
            shape ``(dim,)``.
        factor (npt.NDArray[np.int64]): Per-direction subdivision factor, shape
            ``(dim,)``.
        degrees (npt.NDArray[np.int64]): Per-direction degree, shape ``(dim,)``.
        num_basis (npt.NDArray[np.int64]): Per-level, per-direction function count,
            shape ``(num_levels, dim)``.
        first_basis (npt.NDArray[np.int64]): Concatenated first non-zero function index
            per cell, for every level and direction.
        first_basis_offset (npt.NDArray[np.int64]): Start of level ``m``, direction
            ``k`` in ``first_basis``, shape ``(num_levels, dim)``.
        two_scale (npt.NDArray[np.float64]): Concatenated two-scale blocks, shape
            ``(total, p_max + 1, p_max + 1)``.  The block for the transition from level
            ``m`` to ``m + 1`` in direction ``k`` on the level-``m + 1`` cell ``j`` is at
            ``two_scale_offset[m, k] + j``; entry ``[a, b]`` is the coefficient of window
            function ``b`` of level ``m`` on window function ``a`` of level ``m + 1``.
        two_scale_offset (npt.NDArray[np.int64]): Start of transition ``m``, direction
            ``k`` in ``two_scale``, shape ``(max(num_levels - 1, 1), dim)``.
        active (npt.NDArray[np.int64]): Concatenated sorted flat indices of the active
            functions of every level; position ``func_offset[m] + i`` is the global dof
            of the ``i``-th active function of level ``m``.
        func_offset (npt.NDArray[np.int64]): Per-level global dof base, shape
            ``(num_levels + 1,)``.
        truncate (bool): Whether to apply the truncation mask (THB) or not (HB).
        rows (npt.NDArray[np.float64]): Output rows, shape ``(capacity, n)`` with
            ``n = prod(degrees + 1)`` and ``capacity >= (cell_level + 1) * n``.
        dofs (npt.NDArray[np.int64]): Output global dof per row, shape ``(capacity,)``.
        nonzero (npt.NDArray[np.bool_]): Output flag per row, ``True`` iff the function
            does not vanish on the cell, shape ``(capacity,)``.

    Returns:
        int: Number of rows written.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :meth:`~pantr.bspline.MultiLevelExtraction.multilevel_operator`
        instead.

        ``nonzero`` is decided on the sparsity pattern, which is exact because every
        two-scale coefficient is nonnegative and truncation only zeroes entries: a row
        flagged ``True`` has a structurally non-zero entry.  It is also non-zero in
        floating point unless a product of two-scale coefficients along the chain
        underflows, which needs a chain far deeper than any representable hierarchy of
        moderate degree.
    """
    dim = cell_multi.shape[0]
    num_windows = cell_level + 1
    n = 1
    for k in range(dim):
        n *= degrees[k] + 1
    # C-order strides of the window.
    stride = np.empty(dim, dtype=np.int64)
    acc = 1
    for k in range(dim - 1, -1, -1):
        stride[k] = acc
        acc *= degrees[k] + 1

    # Pass 1: each level's ancestor cell and the global dof of each window function
    # (-1 when it is not active at that level).
    level_cell = np.empty((num_windows, dim), dtype=np.int64)
    window_dof = np.empty((num_windows, n), dtype=np.int64)
    window_first = np.empty(dim, dtype=np.int64)
    for m in range(num_windows):
        for k in range(dim):
            div = 1
            for _ in range(cell_level - m):
                div *= factor[k]
            level_cell[m, k] = cell_multi[k] // div
            window_first[k] = first_basis[first_basis_offset[m, k] + level_cell[m, k]]
        lo = func_offset[m]
        hi = func_offset[m + 1]
        for w in range(n):
            flat = 0
            rem = w
            for k in range(dim):
                local = rem // stride[k]
                rem -= local * stride[k]
                flat = flat * num_basis[m, k] + window_first[k] + local
            pos = lo + np.searchsorted(active[lo:hi], flat)
            window_dof[m, w] = pos if pos < hi and active[pos] == flat else -1

    # Pass 2, backwards: reaches[m, b] iff a coefficient at window position b of level m,
    # after that level's mask, contributes to the cell's level-L coefficients.
    reaches = np.empty((num_windows, n), dtype=np.bool_)
    reaches[cell_level, :] = True
    pattern = np.empty(n, dtype=np.bool_)
    pattern_tmp = np.empty(n, dtype=np.bool_)
    for m in range(cell_level - 1, -1, -1):
        for w in range(n):
            pattern[w] = reaches[m + 1, w] and not (truncate and window_dof[m + 1, w] >= 0)
        for k in range(dim):
            block = two_scale[two_scale_offset[m, k] + level_cell[m + 1, k]]
            _reach_axis(block, degrees[k] + 1, stride[k], pattern, pattern_tmp)
            pattern, pattern_tmp = pattern_tmp, pattern
        reaches[m, :] = pattern

    # Pass 3, forwards: push only the rows that reach the cell.
    scratch = np.empty(n, dtype=np.float64)
    birth_position = np.empty(rows.shape[0], dtype=np.int64)
    count = 0
    born_start = 0
    for m in range(num_windows):
        if m > 0:
            for r in range(count):
                if not nonzero[r]:
                    continue
                row = rows[r]
                if r >= born_start:
                    # Born one level up as a unit vector: its refinement is the tensor
                    # product of the blocks' columns, built in the order the per-axis
                    # contraction would multiply the factors.
                    row[0] = 1.0
                    filled = 1
                    rem = birth_position[r]
                    for k in range(dim):
                        size = degrees[k] + 1
                        column = rem // stride[k]
                        rem -= column * stride[k]
                        block = two_scale[two_scale_offset[m - 1, k] + level_cell[m, k]]
                        for j in range(filled - 1, -1, -1):
                            base = row[j]
                            for a in range(size - 1, -1, -1):
                                row[j * size + a] = base * block[a, column]
                        filled *= size
                else:
                    for k in range(dim):
                        block = two_scale[two_scale_offset[m - 1, k] + level_cell[m, k]]
                        scratch[:] = row
                        _refine_axis(block, degrees[k] + 1, stride[k], scratch, row)
                if truncate:
                    for w in range(n):
                        if window_dof[m, w] >= 0:
                            row[w] = 0.0
        born_start = count
        for w in range(n):
            if window_dof[m, w] >= 0:
                row = rows[count]
                row[:] = 0.0
                if reaches[m, w]:
                    row[w] = 1.0
                    nonzero[count] = True
                else:
                    nonzero[count] = False
                dofs[count] = window_dof[m, w]
                birth_position[count] = w
                count += 1

    return count
