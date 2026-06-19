"""Tests for :class:`pantr.mpi.DistributedSpace`.

MPI is not available in the test environment, so the communicator is duck-typed by
``_FakeComm`` (exposing only ``rank`` and ``size``, which is all DistributedSpace reads).
"""

from __future__ import annotations

import numpy as np
import pytest

from pantr.bspline import (
    BsplineSpace,
    BsplineSpace1D,
    LocalSpace,
    THBSplineSpace,
    build_local,
    coupling_graph,
    create_uniform_space,
    partition_graph,
)
from pantr.grid import (
    Partition,
    hierarchical_grid,
    partition_grid,
    tensor_product_grid,
    uniform_grid,
)
from pantr.mpi import DistributedSpace, create_distributed_space


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


def _thb_space_and_partition(n_parts: int) -> tuple[THBSplineSpace, Partition]:
    root = create_uniform_space(2, 4)
    grid = hierarchical_grid(uniform_grid([[0.0, 1.0]], 4), 2)
    grid.refine(0, [0], [2])
    thb = THBSplineSpace(root, grid)
    part = partition_grid(thb.grid, n_parts)
    return thb, part


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
    thb, part = _thb_space_and_partition(2)
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
    assert ds.n_parts == comm.size == ds.partition.n_parts
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


def test_owned_cells_partition_the_grid_thb() -> None:
    thb, part = _thb_space_and_partition(2)
    owned = np.concatenate(
        [DistributedSpace(thb, part, _FakeComm(r, 2)).owned_cells for r in range(2)]
    )
    np.testing.assert_array_equal(np.sort(owned), np.arange(thb.grid.num_cells))


def test_owned_dofs_partition_the_global_space() -> None:
    space, part = _tp_space_and_partition(4)
    owned_dofs = []
    for rank in range(4):
        local = DistributedSpace(space, part, _FakeComm(rank, 4)).local
        assert local is not None
        owned_dofs.append(local.local_to_global_dof[local.owned_dof_mask])
    allotted = np.concatenate(owned_dofs)
    np.testing.assert_array_equal(np.sort(allotted), np.arange(space.num_total_basis))


def test_owned_dofs_partition_the_global_space_thb() -> None:
    thb, part = _thb_space_and_partition(2)
    owned_dofs = []
    for rank in range(2):
        local = DistributedSpace(thb, part, _FakeComm(rank, 2)).local
        assert local is not None
        owned_dofs.append(local.local_to_global_dof[local.owned_dof_mask])
    allotted = np.concatenate(owned_dofs)
    np.testing.assert_array_equal(np.sort(allotted), np.arange(thb.num_total_basis))


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
    assert not empty.owned_cells.flags.writeable


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
    with pytest.raises(ValueError, match=r"rank must be in \[0,"):
        DistributedSpace(space, part, _FakeComm(4, 4))


def test_periodic_space_raises() -> None:
    from pantr.bspline import create_uniform_periodic_knots  # noqa: PLC0415

    # Periodic check must fire eagerly on every rank, including empty ones.
    knots_p = create_uniform_periodic_knots(num_intervals=4, degree=2)
    sp1d_periodic = BsplineSpace1D(knots_p, 2, periodic=True)
    periodic_space = BsplineSpace([sp1d_periodic])
    n_cells = periodic_space.num_total_intervals
    part = Partition(np.zeros(n_cells, dtype=np.int64), 1)
    with pytest.raises(ValueError, match="periodic"):
        DistributedSpace(periodic_space, part, _FakeComm(0, 1))
    # Empty rank (rank 1 owns nothing) must also raise without reaching build_local.
    part2 = Partition(np.zeros(n_cells, dtype=np.int64), 2)
    with pytest.raises(ValueError, match="periodic"):
        DistributedSpace(periodic_space, part2, _FakeComm(1, 2))


# --------------------------------------------------------------------------- #
# create_distributed_space convenience factory
# --------------------------------------------------------------------------- #


class TestCreateDistributedSpace:
    """create_distributed_space mirrors the explicit grid/graph -> DistributedSpace flow."""

    def test_grid_method_matches_explicit_tp(self) -> None:
        space = create_uniform_space([2, 2], [4, 4])
        part = partition_grid(tensor_product_grid(space), 4)
        for rank in range(4):
            comm = _FakeComm(rank, 4)
            got = create_distributed_space(space, comm)  # method="grid" default
            ref = DistributedSpace(space, part, comm)
            if got.local is not None:
                assert ref.local is not None
                _assert_local_equal(got.local, ref.local)
            np.testing.assert_array_equal(got.owned_cells, ref.owned_cells)

    def test_grid_method_matches_explicit_thb(self) -> None:
        root = create_uniform_space(2, 4)
        grid = hierarchical_grid(uniform_grid([[0.0, 1.0]], 4), 2)
        grid.refine(0, [0], [2])
        thb = THBSplineSpace(root, grid)
        part = partition_grid(thb.grid, 2)
        for rank in range(2):
            comm = _FakeComm(rank, 2)
            got = create_distributed_space(thb, comm)
            ref = DistributedSpace(thb, part, comm)
            if got.local is not None:
                assert ref.local is not None
                _assert_local_equal(got.local, ref.local)
            np.testing.assert_array_equal(got.owned_cells, ref.owned_cells)

    def test_graph_method_matches_explicit(self) -> None:
        space = create_uniform_space([2, 2], [4, 4])
        part = partition_graph(coupling_graph(space), 3)
        for rank in range(3):
            comm = _FakeComm(rank, 3)
            got = create_distributed_space(space, comm, method="graph")
            ref = DistributedSpace(space, part, comm)
            if got.local is not None:
                assert ref.local is not None
                _assert_local_equal(got.local, ref.local)
            np.testing.assert_array_equal(got.owned_cells, ref.owned_cells)

    def test_grid_backend_passthrough(self) -> None:
        space = create_uniform_space([2, 2], [4, 4])
        part = partition_grid(tensor_product_grid(space), 4, backend="rcb")
        got = create_distributed_space(space, _FakeComm(0, 4), backend="rcb")
        np.testing.assert_array_equal(got.partition.cell_owner, part.cell_owner)

    def test_cell_active_passthrough_grid(self) -> None:
        space = create_uniform_space([2, 2], [4, 4])
        active = np.ones(space.num_total_intervals, dtype=bool)
        active[:2] = False  # exclude two cells
        part = partition_grid(tensor_product_grid(space), 2, cell_active=active)
        got = create_distributed_space(space, _FakeComm(0, 2), cell_active=active)
        np.testing.assert_array_equal(got.partition.cell_owner, part.cell_owner)

    def test_cell_weights_passthrough_graph(self) -> None:
        space = create_uniform_space([2, 2], [4, 4])
        weights = np.arange(1, space.num_total_intervals + 1, dtype=np.float64)
        part = partition_graph(coupling_graph(space, cell_weights=weights), 3)
        got = create_distributed_space(space, _FakeComm(0, 3), method="graph", cell_weights=weights)
        np.testing.assert_array_equal(got.partition.cell_owner, part.cell_owner)

    def test_invalid_method_raises(self) -> None:
        space = create_uniform_space([2, 2], [4, 4])
        with pytest.raises(ValueError, match="method must be"):
            create_distributed_space(space, _FakeComm(0, 4), method="bogus")

    def test_grid_method_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="BsplineSpace or THBSplineSpace"):
            create_distributed_space(object(), _FakeComm(0, 2))  # type: ignore[arg-type]
