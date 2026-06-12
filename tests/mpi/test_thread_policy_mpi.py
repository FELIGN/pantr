"""Real-MPI smoke tests for the per-rank thread policy (run under ``mpiexec``).

Skipped unless ``PANTR_RUN_MPI`` is set (and ``mpi4py`` is importable); see
``tests/mpi/test_distributed_mpi.py`` for the launch convention. Each test is
collective: every rank executes the same code on ``MPI.COMM_WORLD`` and the
per-rank thread counts are cross-checked with a real ``allgather``.
"""

from __future__ import annotations

import os

import numba as nb
import pytest

import pantr
import pantr.mpi
from pantr.bspline import create_uniform_space
from pantr.grid import partition_grid, tensor_product_grid
from pantr.mpi import DistributedSpace

MPI = pytest.importorskip("mpi4py.MPI")

pytestmark = pytest.mark.skipif(
    not os.environ.get("PANTR_RUN_MPI"),
    reason="MPI test: set PANTR_RUN_MPI=1 and run under mpiexec",
)


def _build_distributed_space(comm: object) -> DistributedSpace:
    """Construct the same small distributed space on every rank."""
    space = create_uniform_space([2, 2], [6, 6])
    partition = partition_grid(tensor_product_grid(space), comm.size)  # type: ignore[attr-defined]
    return DistributedSpace(space, partition, comm)


def test_default_policy_one_thread_per_rank() -> None:
    """Every rank ends up with exactly one Numba thread after construction."""
    if "NUMBA_NUM_THREADS" in os.environ:
        pytest.skip("explicit NUMBA_NUM_THREADS disables the default policy")
    comm = MPI.COMM_WORLD
    _build_distributed_space(comm)
    counts = comm.allgather(pantr.get_num_threads())
    assert counts == [1] * comm.size


def test_configure_threads_wins_across_ranks() -> None:
    """An explicit configure_threads value survives construction on every rank."""
    n = min(2, int(nb.config.NUMBA_NUM_THREADS))
    if n < 2:
        pytest.skip("needs >= 2 available threads to observe a non-default count")
    comm = MPI.COMM_WORLD
    pantr.mpi.configure_threads(n)
    _build_distributed_space(comm)
    counts = comm.allgather(pantr.get_num_threads())
    assert counts == [n] * comm.size
