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
    assert bvh.n_cells == 16
    assert bvh.n_nodes == 2 * 16 - 1
    assert bvh.ndim == 2
    n_leaves = int(np.sum(bvh.node_cell >= 0))
    assert n_leaves == 16


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
    assert bvh.ndim == 3
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
    with pytest.raises(ValueError, match=r"read-only|assignment"):
        bvh.node_lo[0, 0] = 5.0


def test_query_1d_partial() -> None:
    """A 1-D BVH returns the cells that partially overlap the query range."""
    lo = np.arange(5, dtype=np.float64).reshape(-1, 1)
    hi = lo + 1.0  # cells [0,1],[1,2],[2,3],[3,4],[4,5]
    bvh = BVH.from_cell_bounds(lo, hi)
    result = sorted(int(c) for c in bvh.query_aabb(AABB([1.5], [3.5])))
    assert result == [1, 2, 3]  # cells [1,2],[2,3],[3,4] overlap [1.5, 3.5]


def test_build_tree_3_cells_structure() -> None:
    """A 3-cell BVH has 5 nodes and each cell is individually queryable."""
    lo = np.array([[0.0], [1.0], [2.0]])
    hi = lo + 1.0
    bvh = BVH.from_cell_bounds(lo, hi)
    assert bvh.n_nodes == 5
    assert bvh.n_cells == 3
    for c in range(3):
        mid = float(lo[c, 0]) + 0.5
        result = bvh.query_aabb(AABB([mid - 0.1], [mid + 0.1]))
        assert c in result.tolist()


def test_from_cell_bounds_1d_array_raises() -> None:
    """from_cell_bounds rejects a flat (1-D) input array."""
    with pytest.raises(ValueError, match="2-D"):
        BVH.from_cell_bounds(np.array([0.0, 1.0]), np.array([1.0, 2.0]))


def test_from_cell_bounds_rejects_nan_inf() -> None:
    """Non-finite cell bounds raise ValueError before building the BVH."""
    lo = np.array([[0.0, 0.0]])
    hi_nan = np.array([[np.nan, 1.0]])
    hi_inf = np.array([[np.inf, 1.0]])
    with pytest.raises(ValueError, match="finite"):
        BVH.from_cell_bounds(lo, hi_nan)
    with pytest.raises(ValueError, match="finite"):
        BVH.from_cell_bounds(lo, hi_inf)


def test_stack_overflow_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """from_cell_bounds raises when the tree depth would overflow the kernel stack."""
    import pantr.grid._bvh as _bvh_mod  # noqa: PLC0415

    monkeypatch.setattr(_bvh_mod, "_BVH_STACK_DEPTH", 1)
    lo, hi = _grid_cells(2, 2)  # 4 cells → depth 3 > 1
    with pytest.raises(ValueError, match="stack depth"):
        BVH.from_cell_bounds(lo, hi)


# ---------------------------------------------------------------------------
# Numba build kernel (PR 3 of #197)
# ---------------------------------------------------------------------------


def test_build_is_deterministic() -> None:
    """Two builds over the same input produce identical arrays."""
    rng = np.random.default_rng(7)
    lo = rng.random((500, 3))
    hi = lo + rng.random((500, 3))
    a = BVH.from_cell_bounds(lo, hi)
    b = BVH.from_cell_bounds(lo, hi)
    np.testing.assert_array_equal(a.node_lo, b.node_lo)
    np.testing.assert_array_equal(a.node_hi, b.node_hi)
    np.testing.assert_array_equal(a.node_left, b.node_left)
    np.testing.assert_array_equal(a.node_right, b.node_right)
    np.testing.assert_array_equal(a.node_cell, b.node_cell)


def test_build_invariants_random() -> None:
    """Internal AABBs are the union of their children; leaves partition the cells."""
    rng = np.random.default_rng(11)
    n = 2000
    lo = rng.random((n, 2))
    hi = lo + rng.random((n, 2)) * 0.1
    bvh = BVH.from_cell_bounds(lo, hi)
    assert bvh.n_nodes == 2 * n - 1

    leaves = []
    for i in range(bvh.n_nodes):
        left, right = bvh.node_left[i], bvh.node_right[i]
        if left == -1:
            assert right == -1
            cell = bvh.node_cell[i]
            assert cell >= 0
            leaves.append(cell)
            np.testing.assert_array_equal(bvh.node_lo[i], lo[cell])
            np.testing.assert_array_equal(bvh.node_hi[i], hi[cell])
        else:
            assert bvh.node_cell[i] == -1
            np.testing.assert_array_equal(
                bvh.node_lo[i], np.minimum(bvh.node_lo[left], bvh.node_lo[right])
            )
            np.testing.assert_array_equal(
                bvh.node_hi[i], np.maximum(bvh.node_hi[left], bvh.node_hi[right])
            )
    assert sorted(leaves) == list(range(n))


def test_build_identical_cells_stable() -> None:
    """All-identical cells (fully tied centroids) build a valid, full tree."""
    n = 33
    lo = np.zeros((n, 2))
    hi = np.ones((n, 2))
    bvh = BVH.from_cell_bounds(lo, hi)
    assert bvh.n_nodes == 2 * n - 1
    cells = sorted(bvh.node_cell[bvh.node_cell >= 0].tolist())
    assert cells == list(range(n))
    result = sorted(bvh.query_aabb(AABB([0.5, 0.5], [0.6, 0.6])).tolist())
    assert result == list(range(n))
