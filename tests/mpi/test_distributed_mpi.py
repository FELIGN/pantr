"""Real-MPI smoke tests for the distributed-space stack (run under ``mpiexec``).

Skipped unless ``PANTR_RUN_MPI`` is set (and ``mpi4py`` is importable). The dedicated CI
job runs them as ``PANTR_RUN_MPI=1 mpiexec -n {2,3} python -m pytest tests/mpi/``. Each
test is collective: every rank executes the same code on ``MPI.COMM_WORLD``, so the
partition and the global space are identical across ranks and the cross-rank invariants
are checked with real ``allgather`` collectives.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pantr.bspline import (
    Bspline,
    BsplineSpace,
    THBSpline,
    build_local,
    create_thb_space,
    create_uniform_space,
    l2_project_bspline,
    quasi_interpolate_bspline,
    quasi_interpolate_thb_spline,
)
from pantr.grid import partition_grid, tensor_product_grid
from pantr.mpi import (
    DistributedSpace,
    create_distributed_function,
    create_distributed_space,
    from_dolfinx,
    l2_project_bspline_distributed,
    quasi_interpolate_bspline_distributed,
    quasi_interpolate_thb_spline_distributed,
)

MPI = pytest.importorskip("mpi4py.MPI")

pytestmark = pytest.mark.skipif(
    not os.environ.get("PANTR_RUN_MPI"),
    reason="MPI test: set PANTR_RUN_MPI=1 and run under mpiexec",
)


def test_distributed_space_partitions_globals_across_ranks() -> None:
    comm = MPI.COMM_WORLD
    space = create_uniform_space([2, 2], [6, 6])  # identical on every rank
    partition = partition_grid(tensor_product_grid(space), comm.size)  # deterministic
    ds = DistributedSpace(space, partition, comm)

    assert ds.rank == comm.rank
    assert ds.n_parts == comm.size
    assert ds.local is not None  # the block backend never leaves a rank empty

    # This rank's local space equals the serial build for its rank.
    ref = build_local(space, partition, comm.rank)
    np.testing.assert_array_equal(ds.local.local_to_global_cell, ref.local_to_global_cell)
    np.testing.assert_array_equal(ds.local.local_to_global_dof, ref.local_to_global_dof)

    # Collective: owned DOFs and cells across ranks partition the globals exactly.
    all_dofs = np.concatenate(comm.allgather(ds.local.local_to_global_dof[ds.local.owned_dof_mask]))
    np.testing.assert_array_equal(np.sort(all_dofs), np.arange(space.num_total_basis))

    all_cells = np.concatenate(comm.allgather(ds.owned_cells))
    np.testing.assert_array_equal(np.sort(all_cells), np.arange(space.num_total_intervals))


def test_create_distributed_space_matches_explicit_across_ranks() -> None:
    comm = MPI.COMM_WORLD
    space = create_uniform_space([2, 2], [6, 6])

    ds = create_distributed_space(space, comm)  # one-call factory
    ref = DistributedSpace(space, partition_grid(tensor_product_grid(space), comm.size), comm)

    assert ds.rank == comm.rank and ds.n_parts == comm.size
    np.testing.assert_array_equal(ds.owned_cells, ref.owned_cells)
    if ds.local is not None:
        assert ref.local is not None
        np.testing.assert_array_equal(ds.local.local_to_global_dof, ref.local.local_to_global_dof)

    # Collective: the factory's partition still tiles the globals exactly.
    owned = (
        ds.local.local_to_global_dof[ds.local.owned_dof_mask] if ds.local else np.empty(0, np.int64)
    )
    all_dofs = np.concatenate(comm.allgather(owned))
    np.testing.assert_array_equal(np.sort(all_dofs), np.arange(space.num_total_basis))


def test_create_distributed_function_reproduces_serial_field() -> None:
    comm = MPI.COMM_WORLD
    space = create_uniform_space([2, 2], [6, 6])
    # Identical global field on every rank (same seed), then distribute it.
    cp = np.arange(space.num_total_basis, dtype=np.float64)  # deterministic, same on all ranks
    global_fn = Bspline(space, cp)
    dfn = create_distributed_function(global_fn, comm)

    assert dfn.rank == comm.rank and dfn.n_parts == comm.size
    assert dfn.owns_cells == (dfn.local is not None)

    # Each rank evaluates its local field at the midpoints of its owned cells; the
    # gathered (cell, value) pairs must reproduce the serial field over every cell.
    cells: list[int] = []
    vals: list[float] = []
    if dfn.local is not None:
        assert isinstance(dfn.local, Bspline)
        local = dfn.distributed_space.local
        assert local is not None
        lgrid = tensor_product_grid(dfn.local.space)
        for lc in np.flatnonzero(local.owned_cell_mask):
            lo, hi = lgrid.cell_bounds(int(lc))
            mid = (0.5 * (lo + hi))[None]
            cells.append(int(local.local_to_global_cell[lc]))
            vals.append(float(np.asarray(dfn.local.evaluate(mid)).reshape(-1)[0]))

    all_cells = np.concatenate(comm.allgather(np.asarray(cells, dtype=np.int64)))
    all_vals = np.concatenate(comm.allgather(np.asarray(vals, dtype=np.float64)))
    order = np.argsort(all_cells)
    np.testing.assert_array_equal(all_cells[order], np.arange(space.num_total_intervals))

    grid = tensor_product_grid(space)
    lo, hi = grid.collect_cell_bounds()
    serial = np.asarray(global_fn.evaluate(0.5 * (lo + hi))).reshape(-1)
    np.testing.assert_allclose(all_vals[order], serial, atol=1e-10)


def _slice_mesh(comm: Any, owned: list[int]) -> SimpleNamespace:
    """A dolfinx-mesh stand-in over the real communicator: this rank owns ``owned``."""
    return SimpleNamespace(
        comm=comm,
        topology=SimpleNamespace(
            dim=1,
            original_cell_index=np.asarray(owned, dtype=np.int64),
            index_map=lambda _dim: SimpleNamespace(size_local=len(owned)),
        ),
    )


def test_from_dolfinx_assembles_global_partition_across_ranks() -> None:
    comm = MPI.COMM_WORLD
    per_rank = 4
    n_cells = per_rank * comm.size
    owned = list(range(comm.rank * per_rank, (comm.rank + 1) * per_rank))
    partition = from_dolfinx(_slice_mesh(comm, owned), n_cells)  # real allgather collective

    assert partition.n_parts == comm.size
    expected = np.repeat(np.arange(comm.size), per_rank)
    np.testing.assert_array_equal(partition.cell_owner, expected)


def test_quasi_interpolate_bspline_distributed_matches_serial() -> None:
    """Distributed QI reproduces the serial quasi-interpolant over every cell.

    Each rank evaluates func only on its owned-DOF interior points; a single
    allgather assembles the global field.  The test verifies:
    1. The global function is identical on every rank (allgather correctness).
    2. Each rank's local function agrees with the serial QI at its owned cell midpoints.
    """
    comm = MPI.COMM_WORLD
    space = create_uniform_space([2, 2], [8, 8])
    func = lambda p: np.sin(np.pi * p[:, 0]) * np.cos(np.pi * p[:, 1])  # noqa: E731

    ds = create_distributed_space(space, comm)
    dfn = quasi_interpolate_bspline_distributed(func, ds)
    serial = quasi_interpolate_bspline(func, space)

    # 1. Global function must be identical on all ranks.
    all_global_cp = comm.allgather(np.asarray(dfn.global_function.control_points))
    for rank_cp in all_global_cp:
        np.testing.assert_allclose(rank_cp, serial.control_points, atol=1e-10)

    # 2. Local function reproduces serial QI at owned cell midpoints.
    if dfn.local is None:
        return
    assert ds.local is not None
    assert isinstance(dfn.local.space, BsplineSpace)
    grid = tensor_product_grid(dfn.local.space)
    for lc in np.flatnonzero(ds.local.owned_cell_mask):
        lo, hi = grid.cell_bounds(int(lc))
        mid = (0.5 * (lo + hi))[None]
        np.testing.assert_allclose(
            dfn.local.evaluate(mid),
            serial.evaluate(mid),
            atol=1e-10,
        )


def test_quasi_interpolate_bspline_distributed_vector_func() -> None:
    """Vector-valued func (rank=2) distributes correctly."""
    comm = MPI.COMM_WORLD
    space = create_uniform_space([2, 2], [6, 6])
    func = lambda p: np.stack([p[:, 0], 1.0 - p[:, 1]], axis=-1)  # noqa: E731

    ds = create_distributed_space(space, comm)
    dfn = quasi_interpolate_bspline_distributed(func, ds)
    serial = quasi_interpolate_bspline(func, space)

    np.testing.assert_allclose(
        dfn.global_function.control_points, serial.control_points, atol=1e-10
    )


def test_quasi_interpolate_thb_spline_distributed_matches_serial() -> None:
    """Distributed THB QI reproduces the serial hierarchical quasi-interpolant.

    On a two-level THB space (lower-left quadrant refined), the test verifies:
    1. The global function is identical on every rank (allgather correctness).
    2. Each rank's local function agrees with the serial QI at its owned cell midpoints.
    """
    comm = MPI.COMM_WORLD
    space = create_thb_space(create_uniform_space([2, 2], [8, 8])).refine_region(0, [0, 0], [4, 4])
    func = lambda p: np.sin(np.pi * p[:, 0]) * np.cos(np.pi * p[:, 1])  # noqa: E731

    ds = create_distributed_space(space, comm)
    dfn = quasi_interpolate_thb_spline_distributed(func, ds)
    serial = quasi_interpolate_thb_spline(func, space)

    # 1. Global function must be identical on all ranks.
    all_global_cp = comm.allgather(np.asarray(dfn.global_function.control_points))
    for rank_cp in all_global_cp:
        np.testing.assert_allclose(rank_cp, serial.control_points, atol=1e-10)

    # 2. Local function reproduces serial QI at owned cell midpoints.  Evaluate at all
    # midpoints in a single call (one parallel kernel launch, not one per cell).
    if dfn.local is None:
        return
    assert ds.local is not None
    assert isinstance(dfn.local, THBSpline)
    grid = dfn.local.space.grid
    mids = np.empty((0, space.dim))
    for lc in np.flatnonzero(ds.local.owned_cell_mask):
        lo, hi = grid.cell_bounds(int(lc))
        mids = np.vstack([mids, 0.5 * (lo + hi)])
    np.testing.assert_allclose(
        dfn.local.evaluate(mids),
        serial.evaluate(mids),
        atol=1e-10,
    )


def test_quasi_interpolate_thb_spline_distributed_vector_func() -> None:
    """Vector-valued func (rank=2) distributes correctly over a THB space."""
    comm = MPI.COMM_WORLD
    space = create_thb_space(create_uniform_space([2, 2], [8, 8])).refine_region(0, [0, 0], [4, 4])
    func = lambda p: np.stack([p[:, 0], 1.0 - p[:, 1]], axis=-1)  # noqa: E731

    ds = create_distributed_space(space, comm)
    dfn = quasi_interpolate_thb_spline_distributed(func, ds)
    serial = quasi_interpolate_thb_spline(func, space)

    np.testing.assert_allclose(
        dfn.global_function.control_points, serial.control_points, atol=1e-10
    )


def _grid_mesh(lat: Any) -> tuple[np.ndarray, ...]:
    """Return the C-order, flattened meshgrid of a PointsLattice's per-direction nodes.

    Mirrors the serial ``func(lattice)`` convention: each returned array is the flat
    ``(n_total,)`` coordinate vector for one parametric direction.
    """
    return tuple(g.reshape(-1) for g in np.meshgrid(*list(lat.pts_per_dir), indexing="ij"))


def test_l2_project_bspline_distributed_matches_serial() -> None:
    """Distributed L2 projection reproduces the serial result over every cell.

    Each rank assembles the load over its owned cells only; a single allreduce sums the
    global load and the replicated Kronecker solve recovers the coefficients.  The test
    verifies:
    1. The global function is identical on every rank (allreduce correctness).
    2. Each rank's local function agrees with the serial L2 at its owned cell midpoints.
    """
    comm = MPI.COMM_WORLD
    space = create_uniform_space([2, 2], [8, 8])
    func = lambda lat: np.sin(np.pi * _grid_mesh(lat)[0]) * np.cos(np.pi * _grid_mesh(lat)[1])  # noqa: E731

    ds = create_distributed_space(space, comm)
    dfn = l2_project_bspline_distributed(func, ds)
    serial = l2_project_bspline(func, space)

    # 1. Global function must be identical on all ranks.
    all_global_cp = comm.allgather(np.asarray(dfn.global_function.control_points))
    for rank_cp in all_global_cp:
        np.testing.assert_allclose(rank_cp, serial.control_points, atol=1e-10)

    # 2. Local function reproduces serial L2 at owned cell midpoints.
    if dfn.local is None:
        return
    assert ds.local is not None
    assert isinstance(dfn.local.space, BsplineSpace)
    grid = tensor_product_grid(dfn.local.space)
    for lc in np.flatnonzero(ds.local.owned_cell_mask):
        lo, hi = grid.cell_bounds(int(lc))
        mid = (0.5 * (lo + hi))[None]
        np.testing.assert_allclose(
            dfn.local.evaluate(mid),
            serial.evaluate(mid),
            atol=1e-10,
        )


def test_l2_project_bspline_distributed_boundary_interpolation() -> None:
    """Distributed L2 with boundary_interpolation matches the serial result."""
    comm = MPI.COMM_WORLD
    space = create_uniform_space([3, 3], [6, 6])
    func = lambda lat: np.cos(_grid_mesh(lat)[0]) * _grid_mesh(lat)[1] ** 2  # noqa: E731

    ds = create_distributed_space(space, comm)
    dfn = l2_project_bspline_distributed(func, ds, boundary_interpolation=True)
    serial = l2_project_bspline(func, space, boundary_interpolation=True)

    np.testing.assert_allclose(
        dfn.global_function.control_points, serial.control_points, atol=1e-10
    )
