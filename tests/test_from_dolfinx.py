"""Tests for :func:`pantr.mpi.from_dolfinx` (dolfinx mesh -> Partition bridge).

dolfinx and MPI are not available in the test environment, so the dolfinx ``Mesh``
and its communicator are duck-typed by the fakes below. ``_FakeComm.allgather``
models ``mpi4py``'s collective: the calling rank contributes its own ``sendobj`` and
the other ranks contribute preset owned-cell lists.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from pantr.grid import Partition
from pantr.mpi import from_dolfinx


class _FakeComm:
    """Minimal stand-in for an mpi4py communicator."""

    def __init__(
        self, rank: int, size: int, others: dict[int, Sequence[int]] | None = None
    ) -> None:
        self._rank = rank
        self._size = size
        self._others = others or {}

    @property
    def size(self) -> int:
        return self._size

    def allgather(self, sendobj: object) -> list[object]:
        result: list[object] = [np.array([], dtype=np.int64)] * self._size
        result[self._rank] = sendobj
        for rank, ids in self._others.items():
            result[rank] = np.asarray(ids, dtype=np.int64)
        return result


class _FakeIndexMap:
    def __init__(self, size_local: int) -> None:
        self.size_local = size_local


class _FakeTopology:
    def __init__(self, dim: int, original: Sequence[int], size_local: int) -> None:
        self.dim = dim
        self.original_cell_index = np.asarray(original, dtype=np.int64)
        self._index_map = _FakeIndexMap(size_local)

    def index_map(self, dim: int) -> _FakeIndexMap:
        return self._index_map


class _FakeMesh:
    def __init__(self, topology: _FakeTopology, comm: _FakeComm) -> None:
        self.topology = topology
        self.comm = comm


def _mesh(
    rank: int,
    size: int,
    owned: Sequence[int],
    *,
    others: dict[int, Sequence[int]] | None = None,
    ghosts: Sequence[int] = (),
) -> _FakeMesh:
    """Build a fake dolfinx mesh: this rank owns ``owned`` (original global indices)."""
    original = list(owned) + list(ghosts)
    topology = _FakeTopology(2, original, len(owned))
    return _FakeMesh(topology, _FakeComm(rank, size, others))


# --------------------------------------------------------------------------- #
# Core assembly
# --------------------------------------------------------------------------- #


def test_two_ranks_full_mesh() -> None:
    mesh = _mesh(0, 2, [0, 1, 2], others={1: [3, 4, 5]})
    part = from_dolfinx(mesh, 6)
    assert isinstance(part, Partition)
    assert part.n_parts == 2
    np.testing.assert_array_equal(part.cell_owner, [0, 0, 0, 1, 1, 1])
    assert np.all(part.active_mask)


def test_ghost_cells_are_ignored() -> None:
    # Owned cells are [0, 1, 2]; 3 and 4 are ghosts (beyond size_local) and dropped.
    mesh = _mesh(0, 2, [0, 1, 2], others={1: [3, 4, 5]}, ghosts=[3, 4])
    part = from_dolfinx(mesh, 6)
    np.testing.assert_array_equal(part.cell_owner, [0, 0, 0, 1, 1, 1])


def test_cells_absent_from_mesh_are_inactive() -> None:
    mesh = _mesh(0, 2, [0, 1, 2], others={1: [3, 4, 5]})
    part = from_dolfinx(mesh, 8)
    np.testing.assert_array_equal(part.cell_owner, [0, 0, 0, 1, 1, 1, -1, -1])
    np.testing.assert_array_equal(part.active_mask, [True] * 6 + [False] * 2)


def test_serial_single_rank() -> None:
    mesh = _mesh(0, 1, [0, 1, 2, 3])
    part = from_dolfinx(mesh, 4)
    assert part.n_parts == 1
    assert np.all(part.cell_owner == 0)


def test_empty_rank_is_allowed() -> None:
    mesh = _mesh(0, 2, [0, 1, 2, 3], others={1: []})
    part = from_dolfinx(mesh, 4)
    assert part.n_parts == 2
    np.testing.assert_array_equal(part.cell_owner, [0, 0, 0, 0])


def test_consistent_across_ranks() -> None:
    # Every rank assembles the same global partition.
    rank0 = from_dolfinx(_mesh(0, 2, [0, 1, 2], others={1: [3, 4, 5]}), 6)
    rank1 = from_dolfinx(_mesh(1, 2, [3, 4, 5], others={0: [0, 1, 2]}), 6)
    np.testing.assert_array_equal(rank0.cell_owner, rank1.cell_owner)
    np.testing.assert_array_equal(rank0.cell_owner, [0, 0, 0, 1, 1, 1])


# --------------------------------------------------------------------------- #
# Cell-index mapping
# --------------------------------------------------------------------------- #


def test_explicit_map_reorders_ownership() -> None:
    # dolfinx global -> pantr id reversal: rank0 owns dolfinx {0,1} -> pantr {3,2},
    # rank1 owns dolfinx {2,3} -> pantr {1,0}.
    # others carries rank 1's post-mapping pantr ids (allgather sends local_ids, not originals).
    mapping = [3, 2, 1, 0]
    mesh = _mesh(0, 2, [0, 1], others={1: [1, 0]})
    part = from_dolfinx(mesh, 4, dolfinx_to_pantr=mapping)
    np.testing.assert_array_equal(part.cell_owner, [1, 1, 0, 0])


def test_identity_map_matches_none() -> None:
    mesh_none = _mesh(0, 2, [0, 1, 2], others={1: [3, 4, 5]})
    mesh_id = _mesh(0, 2, [0, 1, 2], others={1: [3, 4, 5]})
    a = from_dolfinx(mesh_none, 6)
    b = from_dolfinx(mesh_id, 6, dolfinx_to_pantr=[0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(a.cell_owner, b.cell_owner)


def test_map_numpy_array_works() -> None:
    mesh = _mesh(0, 2, [0, 1, 2], others={1: [3, 4, 5]})
    part = from_dolfinx(mesh, 6, dolfinx_to_pantr=np.array([0, 1, 2, 3, 4, 5], dtype=np.int32))
    np.testing.assert_array_equal(part.cell_owner, [0, 0, 0, 1, 1, 1])


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_cells", [0, -1])
def test_invalid_n_cells_raises(n_cells: int) -> None:
    mesh = _mesh(0, 1, [0])
    with pytest.raises(ValueError, match="n_cells must be >= 1"):
        from_dolfinx(mesh, n_cells)


def test_double_ownership_raises() -> None:
    mesh = _mesh(0, 2, [0, 1], others={1: [1, 2]})  # cell 1 claimed by both ranks
    with pytest.raises(ValueError, match="claimed by both"):
        from_dolfinx(mesh, 3)


def test_pantr_id_out_of_range_raises() -> None:
    mesh = _mesh(0, 1, [0, 5])  # cell id 5 with n_cells=4
    with pytest.raises(ValueError, match="mapped pantr cell ids must lie"):
        from_dolfinx(mesh, 4)


def test_map_negative_pantr_id_raises() -> None:
    mesh = _mesh(0, 1, [0, 1])
    with pytest.raises(ValueError, match="mapped pantr cell ids must lie"):
        from_dolfinx(mesh, 4, dolfinx_to_pantr=[-1, 0, 1, 2])


def test_map_not_1d_integer_raises() -> None:
    mesh = _mesh(0, 1, [0, 1])
    with pytest.raises(ValueError, match="1D integer array"):
        from_dolfinx(mesh, 4, dolfinx_to_pantr=[[0, 1], [2, 3]])


def test_map_float_dtype_raises() -> None:
    mesh = _mesh(0, 1, [0, 1])
    with pytest.raises(ValueError, match="1D integer array"):
        from_dolfinx(mesh, 4, dolfinx_to_pantr=[0.0, 1.0, 2.0, 3.0])


def test_map_index_out_of_bounds_raises() -> None:
    mesh = _mesh(0, 1, [0, 5])  # original index 5 but map has length 4
    with pytest.raises(ValueError, match="dolfinx_to_pantr has length"):
        from_dolfinx(mesh, 6, dolfinx_to_pantr=[0, 1, 2, 3])


def test_negative_original_index_raises() -> None:
    mesh = _mesh(0, 1, [-1, 0])  # original_cell_index contains -1
    with pytest.raises(ValueError, match="non-negative"):
        from_dolfinx(mesh, 4, dolfinx_to_pantr=[0, 1, 2, 3])


def test_non_injective_map_raises() -> None:
    mesh = _mesh(0, 1, [0, 1])  # two dolfinx cells both mapping to pantr id 0
    with pytest.raises(ValueError, match="not injective"):
        from_dolfinx(mesh, 4, dolfinx_to_pantr=[0, 0, 2, 3])
