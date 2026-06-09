"""Tests for the native block partitioner :func:`pantr.grid.partition_grid`."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pantr.grid import (
    Partition,
    hierarchical_grid,
    partition_grid,
    uniform_grid,
)
from pantr.grid._partition_grid import _divisors, _factor_blocks


def _cell_multi_indices(cells_per_axis: tuple[int, ...]) -> npt.NDArray[np.intp]:
    """Return the (num_cells, ndim) C-order multi-index of every cell."""
    n = int(np.prod(cells_per_axis))
    multi: npt.NDArray[np.intp] = np.array(
        np.unravel_index(np.arange(n), cells_per_axis), dtype=np.intp
    ).T
    return multi


def _assert_owner_boxes(owner: npt.NDArray[Any], cells_per_axis: tuple[int, ...]) -> None:
    """Assert each owner's cells form a solid axis-aligned box, tiling the grid."""
    multi = _cell_multi_indices(cells_per_axis)
    for rank in np.unique(owner):
        cells = multi[owner == rank]
        lo = cells.min(axis=0)
        hi = cells.max(axis=0)
        box_count = int(np.prod(hi - lo + 1))
        assert cells.shape[0] == box_count, f"rank {rank} cells are not a full box"
        in_box = np.all((multi >= lo) & (multi <= hi), axis=1)
        assert np.all(owner[in_box] == rank), f"rank {rank} box overlaps another rank"


# --------------------------------------------------------------------------- #
# Invariants over a sweep of grids and part counts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "cells_per_axis",
    [(8,), (4, 4), (8, 2), (6, 4), (4, 3, 2), (3, 3, 3)],
)
@pytest.mark.parametrize("n_parts", [1, 2, 3, 4, 6, 8])
def test_block_partition_invariants(cells_per_axis: tuple[int, ...], n_parts: int) -> None:
    grid = uniform_grid([[0.0, 1.0]] * len(cells_per_axis), list(cells_per_axis))
    num_cells = grid.num_cells
    try:
        part = partition_grid(grid, n_parts)
    except ValueError:
        # Infeasible factorization for this grid/part count; covered separately.
        return

    assert isinstance(part, Partition)
    assert part.n_parts == n_parts
    assert part.n_cells == num_cells
    owner = part.cell_owner
    assert owner.shape == (num_cells,)
    assert np.all(part.active_mask), "block partition leaves every cell active"
    assert owner.min() >= 0 and owner.max() < n_parts
    # Every rank is used (no empty subdomain) and counts sum to num_cells.
    assert set(owner.tolist()) == set(range(n_parts))
    counts = np.bincount(owner, minlength=n_parts)
    assert counts.sum() == num_cells
    assert counts.min() >= 1
    _assert_owner_boxes(owner, cells_per_axis)


def test_block_partition_is_deterministic() -> None:
    grid = uniform_grid([[0.0, 1.0], [0.0, 1.0]], [6, 4])
    a = partition_grid(grid, 6).cell_owner
    b = partition_grid(grid, 6).cell_owner
    np.testing.assert_array_equal(a, b)


def test_block_load_is_balanced() -> None:
    # 1D uneven split: each rank gets floor or ceil of num_cells/n_parts.
    grid = uniform_grid([[0.0, 1.0]], 7)
    owner = partition_grid(grid, 3).cell_owner
    counts = np.bincount(owner, minlength=3)
    assert int(counts.min()) == 7 // 3
    assert int(counts.max()) == math.ceil(7 / 3)


# --------------------------------------------------------------------------- #
# Exact, hand-computed cases
# --------------------------------------------------------------------------- #


def test_block_1d_even_split() -> None:
    grid = uniform_grid([[0.0, 1.0]], 8)
    owner = partition_grid(grid, 4).cell_owner
    np.testing.assert_array_equal(owner, [0, 0, 1, 1, 2, 2, 3, 3])


def test_block_2d_four_parts() -> None:
    # (4, 4) into 4 -> blocks (2, 2); owner = (i0 // 2) * 2 + (i1 // 2).
    grid = uniform_grid([[0.0, 1.0], [0.0, 1.0]], [4, 4])
    owner = partition_grid(grid, 4).cell_owner
    i0, i1 = np.unravel_index(np.arange(16), (4, 4))
    expected = (i0 // 2) * 2 + (i1 // 2)
    np.testing.assert_array_equal(owner, expected)


def test_block_is_aspect_ratio_aware() -> None:
    # (8, 2) into 4: cube-like blocks are (4, 1), so owner depends only on axis 0.
    grid = uniform_grid([[0.0, 1.0], [0.0, 1.0]], [8, 2])
    owner = partition_grid(grid, 4).cell_owner.reshape(8, 2)
    # Constant across axis 1 (the short axis was not split)...
    assert np.all(owner[:, 0] == owner[:, 1])
    # ...and varies along axis 0 (the long axis was split into 4).
    assert np.array_equal(owner[:, 0], np.repeat(np.arange(4), 2))


def test_block_n_parts_one() -> None:
    grid = uniform_grid([[0.0, 1.0], [0.0, 1.0]], [5, 3])
    part = partition_grid(grid, 1)
    assert part.n_parts == 1
    assert np.all(part.cell_owner == 0)


def test_block_n_parts_equals_num_cells_is_bijection() -> None:
    grid = uniform_grid([[0.0, 1.0], [0.0, 1.0]], [3, 3])
    owner = partition_grid(grid, 9).cell_owner
    np.testing.assert_array_equal(np.sort(owner), np.arange(9))


def test_block_single_cell_axis_concentrates_blocks() -> None:
    # (1, 8) into 4: axis 0 has 1 cell so all blocks must go on axis 1.
    grid = uniform_grid([[0.0, 1.0], [0.0, 1.0]], [1, 8])
    owner = partition_grid(grid, 4).cell_owner
    np.testing.assert_array_equal(owner, np.repeat(np.arange(4), 2))


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_parts", [0, -1, -5])
def test_invalid_n_parts_raises(n_parts: int) -> None:
    grid = uniform_grid([[0.0, 1.0]], 4)
    with pytest.raises(ValueError, match="n_parts must be >= 1"):
        partition_grid(grid, n_parts)


def test_unknown_backend_raises() -> None:
    grid = uniform_grid([[0.0, 1.0]], 4)
    with pytest.raises(ValueError, match="unknown backend"):
        partition_grid(grid, 2, backend="rcb")


def test_block_on_non_tensor_product_grid_raises() -> None:
    hgrid = hierarchical_grid(uniform_grid([[0.0, 1.0]], 4), 2)
    with pytest.raises(ValueError, match="requires a TensorProductGrid"):
        partition_grid(hgrid, 2)


def test_infeasible_factorization_raises() -> None:
    # 7 is prime and larger than every axis (4), so no non-empty box split exists.
    grid = uniform_grid([[0.0, 1.0], [0.0, 1.0]], [4, 4])
    with pytest.raises(ValueError, match="cannot factor n_parts"):
        partition_grid(grid, 7)


def test_infeasible_1d_n_parts_exceeds_cells_raises() -> None:
    # 1D: n_parts=7 > num_cells=4; the final-axis check rejects it.
    grid = uniform_grid([[0.0, 1.0]], 4)
    with pytest.raises(ValueError, match="cannot factor n_parts"):
        partition_grid(grid, 7)


# --------------------------------------------------------------------------- #
# Factoring helpers
# --------------------------------------------------------------------------- #


def test_divisors() -> None:
    assert _divisors(1) == [1]
    assert _divisors(4) == [1, 2, 4]  # perfect square: i == n//i branch
    assert _divisors(9) == [1, 3, 9]  # perfect square: i == n//i branch
    assert _divisors(12) == [1, 2, 3, 4, 6, 12]
    assert _divisors(13) == [1, 13]
    assert _divisors(36) == [1, 2, 3, 4, 6, 9, 12, 18, 36]


@pytest.mark.parametrize(
    ("cells_per_axis", "n_parts", "expected"),
    [
        ((8,), 4, (4,)),
        ((4, 4), 4, (2, 2)),
        ((8, 2), 4, (4, 1)),  # aspect-aware: cube-like blocks
        ((4, 3), 12, (4, 3)),  # greedy-by-ratio would fail; exact search succeeds
        ((3, 3, 3), 27, (3, 3, 3)),
        ((4, 4, 4), 8, (2, 2, 2)),  # 3D symmetric
        ((6, 4, 2), 6, (3, 2, 1)),  # 3D asymmetric: extents (2,2,2) → var=0
        ((10, 1), 2, (2, 1)),
        ((6, 4), 1, (1, 1)),
    ],
)
def test_factor_blocks(
    cells_per_axis: tuple[int, ...], n_parts: int, expected: tuple[int, ...]
) -> None:
    blocks = _factor_blocks(cells_per_axis, n_parts)
    assert blocks == expected
    assert np.prod(blocks) == n_parts
    assert all(b <= c for b, c in zip(blocks, cells_per_axis, strict=True))


def test_factor_blocks_tie_only_contract() -> None:
    # (3,3,3) into 9 has three equally optimal factorizations: (1,3,3), (3,1,3),
    # (3,3,1). Test only the contract, not the specific tuple.
    blocks = _factor_blocks((3, 3, 3), 9)
    assert math.prod(blocks) == 9
    assert all(b <= c for b, c in zip(blocks, (3, 3, 3), strict=True))


def test_factor_blocks_infeasible_raises() -> None:
    with pytest.raises(ValueError, match="cannot factor"):
        _factor_blocks((4, 4), 7)
