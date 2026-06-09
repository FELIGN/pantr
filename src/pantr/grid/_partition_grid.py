"""Native, dependency-free partitioning of a structured grid into rank subdomains.

Produces a :class:`Partition` (per-cell owner assignment) without any external
dependency or MPI -- the zero-dependency default for distributing a grid. The only
backend here is ``"block"``: a Cartesian split of a :class:`TensorProductGrid` into
contiguous, aspect-ratio-aware box subdomains, one per rank. It is ideal for uniform
tensor-product grids whose part count factors reasonably across the axes. Weight- and
activity-aware partitioning (the ``"rcb"`` backend) and external graph partitioners
(ParMETIS / PT-Scotch) arrive in later work; ``"block"`` raises rather than silently
producing empty ranks when a part count does not factor onto the grid.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np

from ._partition import Partition
from ._tensor_product_grid import TensorProductGrid

if TYPE_CHECKING:
    import numpy.typing as npt

    from ._grid import Grid

_VALID_BACKENDS = ("block",)
"""Partitioning backends recognized by :func:`partition_grid`."""


def partition_grid(grid: Grid, n_parts: int, *, backend: str = "block") -> Partition:
    """Partition a grid's cells into ``n_parts`` contiguous rank subdomains.

    The ``"block"`` backend splits a :class:`TensorProductGrid` into ``n_parts``
    axis-aligned box subdomains: ``n_parts`` is factored across the axes
    proportionally to the cell counts (so subdomains are as cube-like as possible),
    and each axis is cut into contiguous, balanced cell ranges. Every cell is owned,
    each rank owns one box, and every rank receives roughly ``num_cells / n_parts``
    cells.

    This is the serial, communication-free partitioner: it returns a
    :class:`Partition` (a plain per-cell owner array) that the distributed-space
    machinery consumes. No MPI is involved.

    Args:
        grid (Grid): The grid to partition. The ``"block"`` backend requires a
            :class:`TensorProductGrid`.
        n_parts (int): Number of parts (ranks); must be ``>= 1``.
        backend (str): Partitioning strategy. Currently only ``"block"`` is
            available. Defaults to ``"block"``.

    Returns:
        Partition: A per-cell owner assignment with ``n_parts`` parts. Every cell is
        active, so :attr:`Partition.active_mask` is all ``True`` and the owner ids
        cover ``range(n_parts)``.

    Raises:
        ValueError: If ``n_parts < 1``; if ``backend`` is not a recognized backend;
            if the ``"block"`` backend is used on a non-:class:`TensorProductGrid`;
            or if ``n_parts`` cannot be factored across the axes without leaving a
            rank empty (e.g. a prime ``n_parts`` larger than every axis).

    Example:
        >>> from pantr.grid import partition_grid, uniform_grid
        >>> grid = uniform_grid([[0.0, 1.0], [0.0, 1.0]], [4, 4])
        >>> part = partition_grid(grid, 4)
        >>> part.n_parts
        4
        >>> int(part.cell_owner.min()), int(part.cell_owner.max())
        (0, 3)
    """
    if n_parts < 1:
        raise ValueError(f"n_parts must be >= 1; got {n_parts}.")
    if backend not in _VALID_BACKENDS:
        valid = ", ".join(repr(b) for b in _VALID_BACKENDS)
        raise ValueError(f"unknown backend {backend!r}; valid backends: {valid}.")
    if not isinstance(grid, TensorProductGrid):
        raise ValueError(
            f"the {backend!r} backend requires a TensorProductGrid; got {type(grid).__name__}."
        )
    owner = _block_partition(grid, int(n_parts))
    return Partition(owner, int(n_parts))


def _block_partition(grid: TensorProductGrid, n_parts: int) -> npt.NDArray[np.int32]:
    """Compute the per-cell owner array for the Cartesian ``"block"`` backend.

    Factors ``n_parts`` into a per-axis block count (:func:`_factor_blocks`), then
    maps each cell's multi-index to its block via a balanced contiguous split
    ``block_d = cell_d * blocks_d // cells_d`` and flattens the block multi-index
    (C-order, matching :class:`TensorProductGrid` cell ids) to the owner rank.

    Args:
        grid (TensorProductGrid): Grid to partition.
        n_parts (int): Number of parts (``>= 1``).

    Returns:
        npt.NDArray[np.int32]: Shape ``(num_cells,)`` owner ranks in
        ``range(n_parts)``, in C-order cell-id order.

    Raises:
        ValueError: If ``n_parts`` cannot be factored onto the grid's axes (see
            :func:`_factor_blocks`).
    """
    cells_per_axis = grid.cells_per_axis
    blocks_per_axis = _factor_blocks(cells_per_axis, n_parts)
    cpa = np.asarray(cells_per_axis, dtype=np.int64)
    bpa = np.asarray(blocks_per_axis, dtype=np.int64)
    multi = np.stack(np.unravel_index(np.arange(grid.num_cells), cells_per_axis), axis=0)
    block_multi = (multi * bpa[:, None]) // cpa[:, None]
    owner = np.ravel_multi_index(block_multi, blocks_per_axis).astype(np.int32)
    return cast("npt.NDArray[np.int32]", owner)


def _factor_blocks(cells_per_axis: tuple[int, ...], n_parts: int) -> tuple[int, ...]:
    """Factor ``n_parts`` into a per-axis block count proportional to cell counts.

    Searches all factorizations of ``n_parts`` into ``len(cells_per_axis)`` ordered
    factors with ``blocks[d] <= cells_per_axis[d]`` and returns the one whose block
    extents ``cells_per_axis[d] / blocks[d]`` are the most uniform (minimizing the
    variance of their logarithms), i.e. the most cube-like subdomains. The search is
    exact, so it finds a valid factorization whenever one exists; the divisor count of
    ``n_parts`` is small, so the enumeration is cheap.

    Args:
        cells_per_axis (tuple[int, ...]): Cell count along each axis (each ``>= 1``).
        n_parts (int): Number of parts (``>= 1``).

    Returns:
        tuple[int, ...]: Block count per axis; the product equals ``n_parts`` and
        every entry satisfies ``blocks[d] <= cells_per_axis[d]``.

    Raises:
        ValueError: If no such factorization exists -- ``n_parts`` is too large or
            factors too coarsely for this grid, so some rank would get no cells.
            Use a part count that divides the grid better, or the ``"rcb"`` backend.
    """
    ndim = len(cells_per_axis)
    best: tuple[int, ...] | None = None
    best_score = math.inf

    def recurse(axis: int, remaining: int, acc: tuple[int, ...]) -> None:
        nonlocal best, best_score
        if axis == ndim - 1:
            if remaining <= cells_per_axis[axis]:
                blocks = (*acc, remaining)
                score = _extent_imbalance(cells_per_axis, blocks)
                if score < best_score:
                    best_score, best = score, blocks
            return
        for divisor in _divisors(remaining):
            if divisor <= cells_per_axis[axis]:
                recurse(axis + 1, remaining // divisor, (*acc, divisor))

    recurse(0, n_parts, ())
    if best is None:
        raise ValueError(
            f"cannot factor n_parts={n_parts} across axes with "
            f"cells_per_axis={cells_per_axis} without leaving a rank empty. "
            f"Use a part count that divides the grid better, or the 'rcb' backend."
        )
    return best


def _divisors(n: int) -> list[int]:
    """Return all positive divisors of ``n``, ascending.

    Args:
        n (int): A positive integer.

    Returns:
        list[int]: Sorted divisors of ``n`` (includes ``1`` and ``n``).
    """
    small: list[int] = []
    large: list[int] = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            small.append(i)
            if i != n // i:
                large.append(n // i)
        i += 1
    return small + large[::-1]


def _extent_imbalance(cells_per_axis: tuple[int, ...], blocks: tuple[int, ...]) -> float:
    """Return the variance of the log block extents (lower is more cube-like).

    The block extent along an axis is ``cells_per_axis[d] / blocks[d]`` (mean cells
    per block along that axis). Using log extents makes the metric scale-invariant,
    so the minimizer favors subdomains that are as cube-like as possible.

    Args:
        cells_per_axis (tuple[int, ...]): Cell count along each axis.
        blocks (tuple[int, ...]): Candidate block count along each axis.

    Returns:
        float: Variance of ``log(cells_per_axis[d] / blocks[d])`` over the axes.
    """
    logs = [math.log(c / b) for c, b in zip(cells_per_axis, blocks, strict=True)]
    mean = math.fsum(logs) / len(logs)
    return math.fsum((x - mean) ** 2 for x in logs) / len(logs)


__all__ = ["partition_grid"]
