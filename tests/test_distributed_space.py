"""Tests for :class:`pantr.mpi.DistributedSpace`.

MPI is not available in the test environment, so the communicator is duck-typed by
``_FakeComm`` (exposing only ``rank`` and ``size``, which is all DistributedSpace reads).
"""

from __future__ import annotations

import numpy as np
import pytest

from pantr.bspline import (
    Bspline,
    BsplineSpace,
    BsplineSpace1D,
    LocalSpace,
    THBSpline,
    THBSplineSpace,
    build_local,
    coupling_graph,
    create_thb_space,
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
from pantr.mpi import (
    DistributedFunction,
    DistributedSpace,
    create_distributed_function,
    create_distributed_space,
)


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


# --------------------------------------------------------------------------- #
# DistributedSpace.localize and DistributedFunction
# --------------------------------------------------------------------------- #


def _assert_local_field_matches_global(
    local_fn: Bspline | THBSpline,
    global_fn: Bspline | THBSpline,
    local: LocalSpace,
    grid: object,
) -> None:
    """Local function reproduces the global one at every owned cell's midpoint."""
    for lc in np.flatnonzero(local.owned_cell_mask):
        lo, hi = grid.cell_bounds(int(lc))  # type: ignore[attr-defined]
        mid = (0.5 * (lo + hi))[None]
        np.testing.assert_allclose(local_fn.evaluate(mid), global_fn.evaluate(mid), atol=1e-10)


class TestLocalize:
    """DistributedSpace.localize slices a global field to this rank's local function."""

    def test_matches_manual_slice_thb(self) -> None:
        # THBSpline keeps control points flat per DOF, so the slice is direct.
        thb = create_thb_space(create_uniform_space([2, 2], [8, 8])).refine_region(
            0, [0, 0], [4, 4]
        )
        cp = np.random.default_rng(0).standard_normal(thb.num_total_basis)
        part = partition_grid(thb.grid, 3)
        for rank in range(3):
            ds = DistributedSpace(thb, part, _FakeComm(rank, 3))
            local_fn = ds.localize(cp)
            if local_fn is None:
                continue
            assert ds.local is not None
            np.testing.assert_array_equal(local_fn.control_points, cp[ds.local.local_to_global_dof])

    def test_value_preserving_bspline(self) -> None:
        space = create_uniform_space([2, 2], [4, 4])
        cp = np.random.default_rng(1).standard_normal((space.num_total_basis, 2))  # vector
        gfn = Bspline(space, cp)
        part = partition_grid(tensor_product_grid(space), 4)
        for rank in range(4):
            ds = DistributedSpace(space, part, _FakeComm(rank, 4))
            lf = ds.localize(cp)
            if lf is None:
                continue
            assert ds.local is not None
            assert isinstance(lf, Bspline)
            _assert_local_field_matches_global(lf, gfn, ds.local, tensor_product_grid(lf.space))

    def test_value_preserving_thb(self) -> None:
        thb = create_thb_space(create_uniform_space([2, 2], [8, 8])).refine_region(
            0, [0, 0], [4, 4]
        )
        cp = np.random.default_rng(2).standard_normal(thb.num_total_basis)
        gfn = THBSpline(thb, cp)
        part = partition_grid(thb.grid, 3)
        for rank in range(3):
            ds = DistributedSpace(thb, part, _FakeComm(rank, 3))
            lf = ds.localize(cp)
            if lf is None:
                continue
            assert ds.local is not None
            assert isinstance(lf, THBSpline)
            _assert_local_field_matches_global(lf, gfn, ds.local, lf.space.grid)

    def test_scalar_thb_stays_scalar(self) -> None:
        thb = create_thb_space(create_uniform_space([2, 2], [8, 8])).refine_region(
            0, [0, 0], [4, 4]
        )
        cp = np.random.default_rng(3).standard_normal(thb.num_total_basis)  # scalar (1-D)
        ds = DistributedSpace(thb, partition_grid(thb.grid, 2), _FakeComm(0, 2))
        lf = ds.localize(cp)
        assert lf is not None and lf.control_points.ndim == 1

    def test_empty_rank_returns_none(self) -> None:
        space = create_uniform_space([2, 2], [4, 4])
        n_cells = space.num_total_intervals
        part = Partition(np.zeros(n_cells, dtype=np.int64), 2)  # rank 1 owns nothing
        cp = np.zeros(space.num_total_basis)
        assert DistributedSpace(space, part, _FakeComm(1, 2)).localize(cp) is None

    def test_wrong_size_raises(self) -> None:
        space = create_uniform_space([2, 2], [4, 4])
        ds = DistributedSpace(space, partition_grid(tensor_product_grid(space), 2), _FakeComm(0, 2))
        with pytest.raises(ValueError, match="leading dimension must be"):
            ds.localize(np.zeros(space.num_total_basis + 1))


class TestDistributedFunction:
    """create_distributed_function / DistributedFunction wrap a distributed field."""

    def test_local_matches_localize(self) -> None:
        space = create_uniform_space([2, 2], [4, 4])
        cp = np.random.default_rng(4).standard_normal(space.num_total_basis)
        fn = Bspline(space, cp)
        for rank in range(4):
            comm = _FakeComm(rank, 4)
            dfn = create_distributed_function(fn, comm)
            ref = create_distributed_space(space, comm).localize(cp)
            assert dfn.rank == rank and dfn.n_parts == 4
            assert dfn.owns_cells == (dfn.local is not None)
            if dfn.local is None:
                assert ref is None
            else:
                assert ref is not None
                np.testing.assert_array_equal(dfn.local.control_points, ref.control_points)

    def test_vector_bspline_value_preserving(self) -> None:
        # Exercises _dof_coeffs flattening of a vector Bspline (*num_basis, rank).
        space = create_uniform_space([2, 2], [4, 4])
        cp = np.random.default_rng(10).standard_normal((space.num_total_basis, 2))
        gfn = Bspline(space, cp)
        for rank in range(4):
            dfn = create_distributed_function(gfn, _FakeComm(rank, 4))
            if dfn.local is None:
                continue
            assert isinstance(dfn.local, Bspline) and dfn.local.control_points.shape[-1] == 2
            local = dfn.distributed_space.local
            assert local is not None
            grid = tensor_product_grid(dfn.local.space)
            for lc in np.flatnonzero(local.owned_cell_mask):
                lo, hi = grid.cell_bounds(int(lc))
                mid = (0.5 * (lo + hi))[None]
                np.testing.assert_allclose(dfn.local.evaluate(mid), gfn.evaluate(mid), atol=1e-10)

    def test_empty_rank_owns_nothing(self) -> None:
        space = create_uniform_space([2, 2], [4, 4])
        part = Partition(np.zeros(space.num_total_intervals, dtype=np.int64), 2)  # rank 1 empty
        fn = Bspline(space, np.zeros(space.num_total_basis))
        dfn = DistributedFunction(fn, DistributedSpace(space, part, _FakeComm(1, 2)))
        assert dfn.local is None
        assert dfn.owns_cells is False

    def test_float32_bspline(self) -> None:
        space = create_uniform_space([2, 2], [4, 4], dtype=np.float32)
        cp = np.random.default_rng(11).standard_normal(space.num_total_basis).astype(np.float32)
        dfn = create_distributed_function(Bspline(space, cp), _FakeComm(0, 2))
        assert dfn.local is not None
        assert dfn.local.control_points.dtype == np.float32

    def test_graph_method_thb(self) -> None:
        thb = create_thb_space(create_uniform_space([2, 2], [8, 8])).refine_region(
            0, [0, 0], [4, 4]
        )
        fn = THBSpline(thb, np.random.default_rng(5).standard_normal(thb.num_total_basis))
        dfn = create_distributed_function(fn, _FakeComm(0, 2), method="graph")
        assert isinstance(dfn.local, THBSpline)
        assert dfn.global_function is fn
        assert dfn.distributed_space.global_space is thb

    def test_space_mismatch_raises(self) -> None:
        space = create_uniform_space([2, 2], [4, 4])
        other = create_uniform_space([2, 2], [4, 4])  # equal but a different object
        fn = Bspline(space, np.zeros(space.num_total_basis))
        ds = create_distributed_space(other, _FakeComm(0, 2))
        with pytest.raises(ValueError, match="global_space"):
            DistributedFunction(fn, ds)
