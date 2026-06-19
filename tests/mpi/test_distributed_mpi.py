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

from pantr.bspline import build_local, create_uniform_space
from pantr.grid import partition_grid, tensor_product_grid
from pantr.mpi import DistributedSpace, create_distributed_space, from_dolfinx

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
