"""Native, dependency-free partitioning of a structured grid into rank subdomains.

Produces a :class:`Partition` (per-cell owner assignment) without any external
dependency or MPI -- the zero-dependency default for distributing a grid. Two
backends are provided, with an ``"auto"`` dispatch:

- ``"block"`` -- a Cartesian split of a :class:`TensorProductGrid` into contiguous,
  aspect-ratio-aware box subdomains. Ideal for tensor-product grids whose part count
  factors reasonably across the axes; raises rather than producing empty ranks when
  it does not. Ignores cell weights and activity.
- ``"rcb"`` -- recursive coordinate bisection on cell centroids. Geometric and
  grid-agnostic (works on any :class:`Grid`, hierarchical or immersed), weight-aware
  (balances total cell cost, not cell count) and activity-aware (inactive cells get
  owner ``-1`` and are excluded). Handles arbitrary part counts, including prime ones.
- ``"auto"`` -- ``"block"`` when the grid is tensor-product, no weights/activity are
  given, and the part count factors onto the axes; otherwise ``"rcb"``.

External graph partitioners (ParMETIS / PT-Scotch) arrive in later work.
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

_VALID_BACKENDS = ("auto", "block", "rcb")
"""Partitioning backends recognized by :func:`partition_grid`."""


def partition_grid(
    grid: Grid,
    n_parts: int,
    *,
    backend: str = "auto",
    cell_weights: npt.ArrayLike | None = None,
    cell_active: npt.ArrayLike | None = None,
) -> Partition:
    """Partition a grid's cells into ``n_parts`` rank subdomains.

    Returns a :class:`Partition` (a plain per-cell owner array) for the
    serial, communication-free distribution of a grid -- no MPI is involved.
    See the module docstring for the ``"block"``, ``"rcb"``, and ``"auto"``
    backends.

    The ``cell_weights`` and ``cell_active`` hooks let a consumer (e.g. an
    immersed/unfitted code) drive the partition without pantr storing any
    classification: per-cell assembly cost via ``cell_weights`` (the ``"rcb"``
    backend balances total weight), and an active subset via ``cell_active``
    (inactive cells get owner ``-1``). The ``"block"`` backend supports neither.

    Args:
        grid (Grid): The grid to partition. The ``"block"`` backend requires a
            :class:`TensorProductGrid`; ``"rcb"`` accepts any grid.
        n_parts (int): Number of parts (ranks); must be ``>= 1``.
        backend (str): ``"auto"`` (default), ``"block"``, or ``"rcb"``.
        cell_weights (npt.ArrayLike | None): Optional per-cell cost, shape
            ``(num_cells,)``, finite and non-negative. ``None`` means uniform.
            Used by ``"rcb"``; rejected by ``"block"``.
        cell_active (npt.ArrayLike | None): Optional boolean mask, shape
            ``(num_cells,)``; inactive cells get owner ``-1`` and are excluded
            from partitioning. ``None`` means all active. Used by ``"rcb"``;
            rejected by ``"block"``.

    Returns:
        Partition: A per-cell owner assignment with ``n_parts`` parts. Cells
        excluded by ``cell_active`` have owner ``-1``; every active cell is
        assigned a rank in ``range(n_parts)`` and no rank is left empty.

    Raises:
        ValueError: If ``n_parts < 1``; if ``backend`` is unknown; if ``"block"``
            is used on a non-:class:`TensorProductGrid` or with weights/activity;
            if ``n_parts`` cannot be factored onto the axes (``"block"``); if
            ``cell_weights`` / ``cell_active`` have the wrong shape or invalid
            values; or if ``n_parts`` exceeds the number of active cells
            (``"rcb"``).

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

    n_parts = int(n_parts)
    weights = _validate_weights(cell_weights, grid.num_cells)
    active = _validate_active(cell_active, grid.num_cells)

    if backend == "block":
        owner = _block_backend(grid, n_parts, weights, active)
    elif backend == "rcb":
        owner = _rcb_partition(grid, n_parts, weights, active)
    elif isinstance(grid, TensorProductGrid) and weights is None and active is None:
        # "auto": prefer the cheap Cartesian split; fall back to rcb when n_parts
        # does not factor onto the axes (awkward / prime part counts).
        try:
            owner = _block_partition(grid, n_parts)
        except ValueError:
            owner = _rcb_partition(grid, n_parts, None, None)
    else:
        owner = _rcb_partition(grid, n_parts, weights, active)

    return Partition(owner, n_parts)


def _validate_weights(
    cell_weights: npt.ArrayLike | None, n_cells: int
) -> npt.NDArray[np.float64] | None:
    """Validate and coerce ``cell_weights`` to a ``(n_cells,)`` ``float64`` array.

    Args:
        cell_weights (npt.ArrayLike | None): Candidate per-cell weights, or ``None``.
        n_cells (int): Expected length.

    Returns:
        npt.NDArray[np.float64] | None: The coerced weights, or ``None`` if the
        input was ``None``.

    Raises:
        ValueError: If the shape is not ``(n_cells,)`` or any entry is negative or
            non-finite.
    """
    if cell_weights is None:
        return None
    weights = np.asarray(cell_weights, dtype=np.float64)
    if weights.shape != (n_cells,):
        raise ValueError(f"cell_weights must have shape ({n_cells},); got {weights.shape}.")
    if not bool(np.all(np.isfinite(weights))) or bool(np.any(weights < 0.0)):
        raise ValueError("cell_weights must be finite and non-negative.")
    return weights


def _validate_active(
    cell_active: npt.ArrayLike | None, n_cells: int
) -> npt.NDArray[np.bool_] | None:
    """Validate and coerce ``cell_active`` to a ``(n_cells,)`` boolean array.

    Args:
        cell_active (npt.ArrayLike | None): Candidate activity mask, or ``None``.
        n_cells (int): Expected length.

    Returns:
        npt.NDArray[np.bool_] | None: The coerced mask, or ``None`` if the input
        was ``None``.

    Raises:
        ValueError: If the shape is not ``(n_cells,)`` or no cell is active.
    """
    if cell_active is None:
        return None
    active = np.asarray(cell_active)
    if active.shape != (n_cells,):
        raise ValueError(f"cell_active must have shape ({n_cells},); got {active.shape}.")
    active = active.astype(bool)
    if not bool(active.any()):
        raise ValueError("cell_active must mark at least one cell active.")
    return cast("npt.NDArray[np.bool_]", active)


def _block_backend(
    grid: Grid,
    n_parts: int,
    weights: npt.NDArray[np.float64] | None,
    active: npt.NDArray[np.bool_] | None,
) -> npt.NDArray[np.int32]:
    """Run the ``"block"`` backend, rejecting unsupported grids and hooks.

    Args:
        grid (Grid): Grid to partition; must be a :class:`TensorProductGrid`.
        n_parts (int): Number of parts (``>= 1``).
        weights (npt.NDArray[np.float64] | None): Must be ``None`` (block ignores
            weights).
        active (npt.NDArray[np.bool_] | None): Must be ``None`` (block ignores
            activity).

    Returns:
        npt.NDArray[np.int32]: Per-cell owner array (see :func:`_block_partition`).

    Raises:
        ValueError: If ``grid`` is not a :class:`TensorProductGrid`, or if weights
            or activity are supplied.
    """
    if not isinstance(grid, TensorProductGrid):
        raise ValueError(
            f"the 'block' backend requires a TensorProductGrid; got {type(grid).__name__}."
        )
    if weights is not None or active is not None:
        raise ValueError(
            "the 'block' backend does not support cell_weights or cell_active; "
            "use backend='rcb' (or 'auto')."
        )
    return _block_partition(grid, n_parts)


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
    owner = np.ravel_multi_index(
        [block_multi[d] for d in range(len(cells_per_axis))], blocks_per_axis
    ).astype(np.int32)
    return cast("npt.NDArray[np.int32]", owner)


def _rcb_partition(
    grid: Grid,
    n_parts: int,
    weights: npt.NDArray[np.float64] | None,
    active: npt.NDArray[np.bool_] | None,
) -> npt.NDArray[np.int32]:
    """Partition a grid by recursive coordinate bisection of its active cells.

    Operates on the centroids of the active cells (the midpoints of
    :meth:`Grid.collect_cell_bounds`). It is geometric, so it works on any grid,
    and balances total weight (uniform when ``weights is None``). Inactive cells
    get owner ``-1``.

    Args:
        grid (Grid): Grid to partition.
        n_parts (int): Number of parts (``>= 1``).
        weights (npt.NDArray[np.float64] | None): Per-cell weights, or ``None`` for
            uniform.
        active (npt.NDArray[np.bool_] | None): Activity mask, or ``None`` for all
            active.

    Returns:
        npt.NDArray[np.int32]: Shape ``(num_cells,)`` owner ranks; ``-1`` for
        inactive cells, otherwise in ``range(n_parts)``.

    Raises:
        ValueError: If ``n_parts`` exceeds the number of active cells.
    """
    n_cells = grid.num_cells
    active_idx = np.arange(n_cells) if active is None else np.flatnonzero(active)
    n_active = int(active_idx.size)
    if n_parts > n_active:
        raise ValueError(
            f"n_parts={n_parts} exceeds the number of active cells ({n_active}); "
            f"cannot assign every rank a cell."
        )

    cell_lo, cell_hi = grid.collect_cell_bounds()
    centroids = (0.5 * (cell_lo + cell_hi))[active_idx]
    w_active = np.ones(n_active) if weights is None else weights[active_idx]

    owner_active = np.empty(n_active, dtype=np.int32)

    def bisect(idx: npt.NDArray[np.intp], part_lo: int, part_hi: int) -> None:
        # Split cells `idx` into parts [part_lo, part_hi) by weight, cutting the
        # longest-spread axis at the (clamped) weighted split point so that each
        # side keeps at least as many cells as parts (no rank is left empty).
        k = part_hi - part_lo
        if k == 1:
            owner_active[idx] = part_lo
            return
        coords = centroids[idx]
        axis = int(np.argmax(coords.max(axis=0) - coords.min(axis=0)))
        order = idx[np.argsort(coords[:, axis], kind="stable")]
        cumw = np.cumsum(w_active[order])
        k_left = k // 2
        target = float(cumw[-1]) * k_left / k
        split = int(np.searchsorted(cumw, target, side="left")) + 1
        split = max(k_left, min(split, int(order.size) - (k - k_left)))
        bisect(order[:split], part_lo, part_lo + k_left)
        bisect(order[split:], part_lo + k_left, part_hi)

    bisect(np.arange(n_active), 0, n_parts)

    owner = np.full(n_cells, -1, dtype=np.int32)
    owner[active_idx] = owner_active
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
