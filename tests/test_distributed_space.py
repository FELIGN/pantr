"""Tests for :class:`pantr.mpi.DistributedSpace`.

MPI is not available in the test environment, so the communicator is duck-typed by
``_FakeComm`` (exposing only ``rank`` and ``size``, which is all DistributedSpace reads).
"""

from __future__ import annotations

import numpy as np
import pytest

from pantr.bspline import (
    BsplineSpace,
    LocalSpace,
    THBSplineSpace,
    build_local,
    create_uniform_space,
)
from pantr.grid import (
    Partition,
    hierarchical_grid,
    partition_grid,
    tensor_product_grid,
    uniform_grid,
)
from pantr.mpi import DistributedSpace


class _FakeComm:
    """Minimal stand-in for an mpi4py communicator (rank/size only)."""

    def __init__(self, rank: int, size: int) -> None:
        self._rank = rank
        self._size = size

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def size(self) -> int:
        return self._size


def _assert_local_equal(a: LocalSpace, b: LocalSpace) -> None:
    np.testing.assert_array_equal(a.local_to_global_cell, b.local_to_global_cell)
    np.testing.assert_array_equal(a.local_to_global_dof, b.local_to_global_dof)
    np.testing.assert_array_equal(a.owned_cell_mask, b.owned_cell_mask)
    np.testing.assert_array_equal(a.owned_dof_mask, b.owned_dof_mask)
    assert a.n_global_cells == b.n_global_cells
    assert a.n_global_dofs == b.n_global_dofs
    assert a.space.num_total_basis == b.space.num_total_basis


def _tp_space_and_partition(n_parts: int) -> tuple[BsplineSpace, Partition]:
    space = create_uniform_space([2, 2], [4, 4])
    part = partition_grid(tensor_product_grid(space), n_parts)
    return space, part


# --------------------------------------------------------------------------- #
# Local space delegation
# --------------------------------------------------------------------------- #


def test_local_matches_build_local_tp() -> None:
    space, part = _tp_space_and_partition(4)
    for rank in range(4):
        ds = DistributedSpace(space, part, _FakeComm(rank, 4))
        assert ds.local is not None
        _assert_local_equal(ds.local, build_local(space, part, rank))


def test_local_matches_build_local_thb() -> None:
    root = create_uniform_space(2, 4)
    grid = hierarchical_grid(uniform_grid([[0.0, 1.0]], 4), 2)
    grid.refine(0, [0], [2])
    thb = THBSplineSpace(root, grid)
    part = partition_grid(thb.grid, 2)
    for rank in range(2):
        ds = DistributedSpace(thb, part, _FakeComm(rank, 2))
        assert ds.local is not None
        _assert_local_equal(ds.local, build_local(thb, part, rank))


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #


def test_metadata() -> None:
    space, part = _tp_space_and_partition(4)
    comm = _FakeComm(2, 4)
    ds = DistributedSpace(space, part, comm)
    assert ds.comm is comm
    assert ds.rank == 2
    assert ds.n_parts == 4
    assert ds.global_space is space
    assert ds.partition is part
    assert ds.owns_cells is True
    np.testing.assert_array_equal(ds.owned_cells, part.owned_cells(2))
    assert not ds.owned_cells.flags.writeable


# --------------------------------------------------------------------------- #
# Distributed correctness: owned cells / DOFs partition the globals exactly
# --------------------------------------------------------------------------- #


def test_owned_cells_partition_the_grid() -> None:
    space, part = _tp_space_and_partition(4)
    owned = np.concatenate(
        [DistributedSpace(space, part, _FakeComm(r, 4)).owned_cells for r in range(4)]
    )
    np.testing.assert_array_equal(np.sort(owned), np.arange(space.num_total_intervals))


def test_owned_dofs_partition_the_global_space() -> None:
    space, part = _tp_space_and_partition(4)
    owned_dofs = []
    for rank in range(4):
        local = DistributedSpace(space, part, _FakeComm(rank, 4)).local
        assert local is not None
        owned_dofs.append(local.local_to_global_dof[local.owned_dof_mask])
    allotted = np.concatenate(owned_dofs)
    np.testing.assert_array_equal(np.sort(allotted), np.arange(space.num_total_basis))


# --------------------------------------------------------------------------- #
# Empty ranks
# --------------------------------------------------------------------------- #


def test_empty_rank_has_no_local() -> None:
    space = create_uniform_space([2, 2], [4, 4])
    # Rank 0 owns every cell; rank 1 owns none.
    part = Partition(np.zeros(space.num_total_intervals, dtype=np.int64), 2)
    full = DistributedSpace(space, part, _FakeComm(0, 2))
    empty = DistributedSpace(space, part, _FakeComm(1, 2))
    assert full.owns_cells is True
    assert full.local is not None
    assert empty.owns_cells is False
    assert empty.local is None
    assert empty.owned_cells.size == 0


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_comm_size_mismatch_raises() -> None:
    space, part = _tp_space_and_partition(4)
    with pytest.raises(ValueError, match="must equal partition.n_parts"):
        DistributedSpace(space, part, _FakeComm(0, 3))


def test_partition_cell_count_mismatch_raises() -> None:
    space = create_uniform_space([2, 2], [4, 4])  # 16 cells
    part = Partition(np.zeros(10, dtype=np.int64), 1)  # wrong cell count
    with pytest.raises(ValueError, match="the global space's cell count"):
        DistributedSpace(space, part, _FakeComm(0, 1))


def test_rank_out_of_range_raises() -> None:
    space, part = _tp_space_and_partition(4)
    # comm.size matches n_parts, but rank 4 is out of [0, 4).
    with pytest.raises(ValueError, match="rank"):
        DistributedSpace(space, part, _FakeComm(4, 4))
