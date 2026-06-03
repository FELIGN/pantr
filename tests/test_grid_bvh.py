"""Tests for the grid bounding-volume hierarchy (``pantr.grid.BVH``)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pantr.geometry import AABB
from pantr.grid import BVH


def _grid_cells(nx: int, ny: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return per-cell ``(lo, hi)`` for an ``nx`` x ``ny`` unit-cell grid."""
    lo = []
    hi = []
    for i in range(nx):
        for j in range(ny):
            lo.append([float(i), float(j)])
            hi.append([float(i + 1), float(j + 1)])
    return np.array(lo, dtype=np.float64), np.array(hi, dtype=np.float64)


def test_build_node_count() -> None:
    """A BVH over N cells has 2N-1 nodes and N leaves."""
    lo, hi = _grid_cells(4, 4)
    bvh = BVH.from_cell_bounds(lo, hi)
    assert bvh.n_cells == 16  # noqa: PLR2004
    assert bvh.n_nodes == 2 * 16 - 1
    assert bvh.ndim == 2  # noqa: PLR2004
    n_leaves = int(np.sum(bvh.node_cell >= 0))
    assert n_leaves == 16  # noqa: PLR2004


def test_query_returns_all_overlapping() -> None:
    """A query box returns exactly the overlapping cells (touching faces count)."""
    lo, hi = _grid_cells(3, 3)
    bvh = BVH.from_cell_bounds(lo, hi)
    # Box covering the lower-left 2x2 block of unit cells. With the axis-0-major
    # ordering of _grid_cells, cells (i, j) map to id i*3 + j.
    result = sorted(int(c) for c in bvh.query_aabb(AABB([0.0, 0.0], [1.5, 1.5])))
    expected = sorted([0, 1, 3, 4])  # (0,0),(0,1),(1,0),(1,1)
    assert result == expected


def test_query_touching_face_is_inclusive() -> None:
    """A box touching a cell only on a shared face still reports that cell."""
    lo, hi = _grid_cells(3, 3)
    bvh = BVH.from_cell_bounds(lo, hi)
    # The box [0,2]x[0,2] touches the line x=2 / y=2, so the i==2 / j==2 cells
    # (spanning [2,3]) are included: every cell overlaps.
    result = sorted(int(c) for c in bvh.query_aabb(AABB([0.0, 0.0], [2.0, 2.0])))
    assert result == list(range(9))


def test_query_single_cell() -> None:
    """A tiny query box inside one cell returns just that cell."""
    lo, hi = _grid_cells(5, 5)
    bvh = BVH.from_cell_bounds(lo, hi)
    result = bvh.query_aabb(AABB([2.4, 3.4], [2.6, 3.6]))
    assert result.tolist() == [2 * 5 + 3]


def test_query_disjoint_empty() -> None:
    """A query box outside the grid returns no cells."""
    lo, hi = _grid_cells(2, 2)
    bvh = BVH.from_cell_bounds(lo, hi)
    assert bvh.query_aabb(AABB([10.0, 10.0], [11.0, 11.0])).shape == (0,)


def test_single_cell_tree() -> None:
    """A one-cell BVH has a single leaf node and answers queries."""
    bvh = BVH.from_cell_bounds(np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]]))
    assert bvh.n_cells == 1
    assert bvh.n_nodes == 1
    assert bvh.query_aabb(AABB([0.5, 0.5], [0.5, 0.5])).tolist() == [0]
    assert bvh.query_aabb(AABB([2.0, 2.0], [3.0, 3.0])).shape == (0,)


def test_empty_tree() -> None:
    """A zero-cell BVH is valid and returns nothing."""
    bvh = BVH.from_cell_bounds(np.zeros((0, 3)), np.zeros((0, 3)))
    assert bvh.n_cells == 0
    assert bvh.n_nodes == 0
    assert bvh.ndim == 3  # noqa: PLR2004
    assert bvh.query_aabb(AABB([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])).shape == (0,)


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
def test_general_dimension(ndim: int) -> None:
    """The BVH works for any spatial dimension >= 1."""
    rng = np.random.default_rng(ndim)
    n = 20
    lo = rng.uniform(0.0, 5.0, size=(n, ndim))
    hi = lo + rng.uniform(0.1, 1.0, size=(n, ndim))
    bvh = BVH.from_cell_bounds(lo, hi)
    assert bvh.ndim == ndim
    # Query the whole domain: every cell overlaps.
    big = AABB(np.full(ndim, -1.0), np.full(ndim, 10.0))
    assert sorted(int(c) for c in bvh.query_aabb(big)) == list(range(n))


def test_query_matches_brute_force() -> None:
    """BVH query agrees with an O(n) brute-force overlap scan."""
    rng = np.random.default_rng(0)
    n = 200
    lo = rng.uniform(0.0, 10.0, size=(n, 3))
    hi = lo + rng.uniform(0.1, 2.0, size=(n, 3))
    bvh = BVH.from_cell_bounds(lo, hi)
    q = AABB([3.0, 3.0, 3.0], [6.0, 6.0, 6.0])
    got = sorted(int(c) for c in bvh.query_aabb(q))
    brute = sorted(i for i in range(n) if AABB(lo[i], hi[i]).overlaps(q))
    assert got == brute


def test_query_ndim_mismatch_raises() -> None:
    """Querying with a wrong-dimension AABB raises ValueError."""
    bvh = BVH.from_cell_bounds(np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]]))
    with pytest.raises(ValueError, match="must match"):
        bvh.query_aabb(AABB([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))


def test_hi_below_lo_raises() -> None:
    """A cell with hi < lo is rejected."""
    with pytest.raises(ValueError, match="cell_hi >= cell_lo"):
        BVH.from_cell_bounds(np.array([[1.0, 1.0]]), np.array([[0.0, 0.0]]))


def test_inconsistent_shapes_raise() -> None:
    """Mismatched lo/hi shapes are rejected."""
    with pytest.raises(ValueError, match="must match"):
        BVH.from_cell_bounds(np.zeros((3, 2)), np.zeros((3, 3)))


def test_ctor_node_count_validation() -> None:
    """The raw constructor checks n_nodes == 2*n_cells - 1."""
    node_lo = np.zeros((2, 2), dtype=np.float64)
    node_hi = np.ones((2, 2), dtype=np.float64)
    idx = np.zeros(2, dtype=np.int64)
    with pytest.raises(ValueError, match="implies n_nodes"):
        BVH(node_lo, node_hi, idx, idx, idx, n_cells=2)


def test_ctor_dtype_validation() -> None:
    """The raw constructor rejects non-int64 child arrays."""
    node_lo = np.zeros((1, 2), dtype=np.float64)
    node_hi = np.ones((1, 2), dtype=np.float64)
    bad = np.zeros(1, dtype=np.int32)
    with pytest.raises(TypeError, match="int64"):
        BVH(node_lo, node_hi, bad, bad, bad, n_cells=1)  # type: ignore[arg-type]


def test_nodes_are_read_only() -> None:
    """The stored node arrays are flagged read-only."""
    lo, hi = _grid_cells(2, 2)
    bvh = BVH.from_cell_bounds(lo, hi)
    assert not bvh.node_lo.flags.writeable
    assert not bvh.node_cell.flags.writeable
    with pytest.raises(ValueError, match="read-only|assignment"):
        bvh.node_lo[0, 0] = 5.0
