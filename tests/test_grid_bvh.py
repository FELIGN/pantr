"""Tests for the grid bounding-volume hierarchy (``pantr.grid.BVH``)."""

from __future__ import annotations

import pickle

import numpy as np
import numpy.typing as npt
import pytest

from pantr._backend import Backend, use_backend
from pantr.geometry import AABB
from pantr.grid import BVH
from pantr.grid._bvh_core import _BVH_STACK_DEPTH
from tests._parity_harness import demand_cpp_backend


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
    # Exact preorder node-index layout (regression guard for next_idx / push order).
    # Root splits [0,3) at median 1 → left=node1 (leaf, cell 0), right=node2.
    # Node2 splits [1,3) at median 1 → left=node3 (leaf, cell 1), right=node4 (leaf, cell 2).
    assert bvh.node_left[0] == 1 and bvh.node_right[0] == 2
    assert bvh.node_cell[0] == -1
    assert bvh.node_cell[1] == 0 and bvh.node_left[1] == -1
    assert bvh.node_left[2] == 3 and bvh.node_right[2] == 4
    assert bvh.node_cell[2] == -1
    assert bvh.node_cell[3] == 1 and bvh.node_cell[4] == 2


def test_build_tree_2_cells_structure() -> None:
    """A 2-cell BVH (minimal internal node) has 3 nodes with correct structure."""
    lo = np.array([[0.0], [1.0]])
    hi = lo + 1.0
    bvh = BVH.from_cell_bounds(lo, hi)
    assert bvh.n_nodes == 3
    assert bvh.n_cells == 2
    assert bvh.node_cell[0] == -1
    assert bvh.node_left[0] == 1 and bvh.node_right[0] == 2
    assert sorted(bvh.node_cell[1:].tolist()) == [0, 1]


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


def _chain_bvh_arrays(
    n_leaves: int,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]:
    """Build raw node arrays for a maximally unbalanced (linear-chain) 1-D BVH.

    Leaf ``k`` covers the unit cell ``[k, k+1]``. Node ``i`` for
    ``i < n_leaves - 1`` is internal, with left child = leaf ``i`` and right
    child = the next internal node (the last internal node's right child is
    leaf ``n_leaves - 1`` instead). Every internal node's AABB is the honest
    union of its subtree, so a correctly-pruning query still answers right.
    This chain is the deepest possible tree over ``n_leaves`` leaves: the
    longest root-to-leaf path has exactly ``n_leaves`` nodes.
    """
    n_internal = n_leaves - 1
    n_nodes = 2 * n_leaves - 1
    node_lo = np.zeros((n_nodes, 1), dtype=np.float64)
    node_hi = np.zeros((n_nodes, 1), dtype=np.float64)
    node_left = np.full(n_nodes, -1, dtype=np.int64)
    node_right = np.full(n_nodes, -1, dtype=np.int64)
    node_cell = np.full(n_nodes, -1, dtype=np.int64)
    for i in range(n_internal):
        node_lo[i, 0] = float(i)
        node_hi[i, 0] = float(n_leaves)
        node_left[i] = n_internal + i
        node_right[i] = i + 1 if i < n_internal - 1 else n_internal + i + 1
    for k in range(n_leaves):
        leaf = n_internal + k
        node_lo[leaf, 0] = float(k)
        node_hi[leaf, 0] = float(k + 1)
        node_cell[leaf] = k
    return node_lo, node_hi, node_left, node_right, node_cell


def test_ctor_unbalanced_chain_raises_at_construction() -> None:
    """A hand-built 148-leaf linear-chain tree exceeds the kernel stack depth.

    148 leaves force a chain of depth 148, past the traversal kernels' fixed
    stack depth (``_BVH_STACK_DEPTH == 128`` in ``_bvh_core.py``). Unlike
    ``from_cell_bounds``, the raw constructor accepts arbitrary node arrays, so
    this depth cannot be inferred from ``n_cells`` alone; ``BVH.__init__`` must
    reject it before any query touches the fixed-size kernel stack.
    """
    node_lo, node_hi, node_left, node_right, node_cell = _chain_bvh_arrays(148)
    with pytest.raises(ValueError, match="stack depth"):
        BVH(node_lo, node_hi, node_left, node_right, node_cell, n_cells=148)


def test_ctor_unbalanced_chain_at_the_limit_constructs_and_queries() -> None:
    """A 128-leaf chain sits exactly on the limit and must still be accepted.

    A depth-``d`` DFS occupies at most ``d`` stack slots: the root takes one, then
    each of the ``d - 1`` internal expansions along the deepest path pops one entry
    and pushes two, a net ``+1``. So ``d == _BVH_STACK_DEPTH == 128`` fits exactly
    while ``d == 129`` does not, which is what makes the guard's strict ``>`` the
    right comparison. Confirmed against the kernel under ``NUMBA_BOUNDSCHECK=1``
    with the guard lifted: a depth-128 chain queries correctly, and a depth-129
    chain raises ``IndexError`` from inside the compiled traversal.

    So this is the exact boundary, and it also shows the guard cannot pass merely
    by rejecting everything.
    """
    node_lo, node_hi, node_left, node_right, node_cell = _chain_bvh_arrays(128)
    bvh = BVH(node_lo, node_hi, node_left, node_right, node_cell, n_cells=128)
    result = sorted(int(c) for c in bvh.query_aabb(AABB([50.5], [52.5])))
    assert result == [50, 51, 52]


def test_ctor_unbalanced_chain_one_past_the_limit_raises() -> None:
    """A 129-leaf chain is the smallest chain that overflows, and must be rejected.

    Pins the other side of the boundary above: without the guard this constructs
    cleanly and then writes one slot past the kernel's 128-entry stack on the very
    first query.
    """
    node_lo, node_hi, node_left, node_right, node_cell = _chain_bvh_arrays(129)
    with pytest.raises(ValueError, match="stack depth"):
        BVH(node_lo, node_hi, node_left, node_right, node_cell, n_cells=129)


def test_ctor_rejects_a_cyclic_node_graph_without_hanging() -> None:
    """A node array that is not a tree must be rejected, not walked forever.

    The depth walk stops as soon as the bound is exceeded, so a self-referential
    or cyclic child pointer terminates instead of growing the traversal stack
    without limit. A validator that hangs would be worse than the out-of-bounds
    write it replaces.
    """
    node_lo, node_hi, node_left, node_right, node_cell = _chain_bvh_arrays(4)
    node_right[0] = 0  # root points back at itself

    with pytest.raises(ValueError, match="stack depth"):
        BVH(node_lo, node_hi, node_left, node_right, node_cell, n_cells=4)


def _three_leaf_arrays() -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]:
    """Raw node arrays of a valid 3-leaf BVH, taken from ``from_cell_bounds`` itself."""
    lo = np.array([[0.0], [1.0], [2.0]])
    hi = np.array([[1.0], [2.0], [3.0]])
    ref = BVH.from_cell_bounds(lo, hi)
    return (
        np.array(ref.node_lo, dtype=np.float64),
        np.array(ref.node_hi, dtype=np.float64),
        np.array(ref.node_left, dtype=np.int64),
        np.array(ref.node_right, dtype=np.int64),
        np.array(ref.node_cell, dtype=np.int64),
    )


def test_ctor_accepts_a_well_formed_hand_built_tree() -> None:
    """The consistency checks must not reject a tree the library itself produced."""
    node_lo, node_hi, node_left, node_right, node_cell = _three_leaf_arrays()

    bvh = BVH(node_lo, node_hi, node_left, node_right, node_cell, n_cells=3)

    assert sorted(int(c) for c in bvh.query_aabb(AABB([0.5], [0.6]))) == [0]


def test_ctor_rejects_internal_node_with_a_missing_child() -> None:
    """An "internal" node whose child is -1 must be rejected at construction.

    The traversal kernels decide leaf-vs-internal from ``node_cell`` alone and then
    push both children without checking either against -1, so a -1 child index wraps
    to the last row of the node array. Before this check the tree constructed
    cleanly and the query silently returned the wrong cells: this arrangement gave
    ``[]`` where the correct answer is ``[0]``.

    With only *one* child cleared the node still does not look like a leaf, so the
    leaf-consistency check passes it on and the child-range check is what rejects
    it. Either is fine; what matters is that it no longer constructs.
    """
    node_lo, node_hi, node_left, node_right, node_cell = _three_leaf_arrays()
    internal = int(np.where(node_cell < 0)[0][0])
    node_left[internal] = -1

    with pytest.raises(ValueError, match=r"node_left contains values outside"):
        BVH(node_lo, node_hi, node_left, node_right, node_cell, n_cells=3)


def test_ctor_rejects_leaf_marked_internal() -> None:
    """The mirror case: a node with children but a non-negative ``node_cell``."""
    node_lo, node_hi, node_left, node_right, node_cell = _three_leaf_arrays()
    leaf = int(np.where(node_cell >= 0)[0][0])
    node_cell[leaf] = -1

    with pytest.raises(ValueError, match="disagree about which nodes are leaves"):
        BVH(node_lo, node_hi, node_left, node_right, node_cell, n_cells=3)


def test_ctor_rejects_out_of_range_child_index() -> None:
    """A child index outside ``[0, n_nodes)`` must be rejected, negatives included.

    A negative other than -1 passes the leaf-consistency check (the node still does
    not look like a leaf) but still wraps when the kernel indexes with it.
    """
    node_lo, node_hi, node_left, node_right, node_cell = _three_leaf_arrays()
    internal = int(np.where(node_cell < 0)[0][0])
    node_right[internal] = -5

    with pytest.raises(ValueError, match=r"node_right contains values outside"):
        BVH(node_lo, node_hi, node_left, node_right, node_cell, n_cells=3)


def test_ctor_rejects_out_of_range_leaf_cell_id() -> None:
    """A leaf naming a cell id beyond ``n_cells`` must be rejected."""
    node_lo, node_hi, node_left, node_right, node_cell = _three_leaf_arrays()
    leaf = int(np.where(node_cell >= 0)[0][0])
    node_cell[leaf] = 42

    with pytest.raises(ValueError, match=r"node_cell contains values outside"):
        BVH(node_lo, node_hi, node_left, node_right, node_cell, n_cells=3)


def test_from_cell_bounds_always_satisfies_the_consistency_invariant() -> None:
    """Every tree the builder produces must pass the new checks, at every size.

    Guards the checks against being stricter than the library's own producer.
    """
    for n in range(1, 60):
        lo = np.arange(n, dtype=np.float64).reshape(-1, 1)
        tree = BVH.from_cell_bounds(lo, lo + 1.0)
        # Round-tripping the raw arrays back through __init__ runs every check.
        BVH(
            np.array(tree.node_lo, dtype=np.float64),
            np.array(tree.node_hi, dtype=np.float64),
            np.array(tree.node_left, dtype=np.int64),
            np.array(tree.node_right, dtype=np.int64),
            np.array(tree.node_cell, dtype=np.int64),
            n_cells=n,
        )


def test_the_depth_guard_follows_the_stack_depth_constant() -> None:
    """The refused depth is exactly one past ``_BVH_STACK_DEPTH``, derived not hardcoded.

    **This replaces a monkeypatching test, and the replacement is the point.** The
    old ``test_stack_overflow_guard`` set ``pantr.grid._bvh._BVH_STACK_DEPTH`` to 1
    and built four cells. That worked while the module global was the only thing
    enforcing the limit. Under ``PANTR_BACKEND=cpp`` the enforcing constant is
    ``kBvhStackDepth``, a C++ ``constexpr`` the monkeypatch cannot reach, so the
    test passed while testing nothing -- which is worse than not having it.

    So the boundary is exercised for real, on both sides, with the leaf counts
    **derived from the constant** rather than written as 128 and 129. Its three
    siblings above pin the same boundary with literal counts; this one is what fails
    if the Python constant moves and the guard does not move with it. The other
    drift -- Python against the C++ mirror -- is not visible from here at all, and
    ``scripts/ci_local.sh`` carries a guard that extracts both by regex.
    """
    at_limit = _chain_bvh_arrays(_BVH_STACK_DEPTH)
    tree = BVH(*at_limit, n_cells=_BVH_STACK_DEPTH)
    assert tree.n_cells == _BVH_STACK_DEPTH

    past_limit = _chain_bvh_arrays(_BVH_STACK_DEPTH + 1)
    with pytest.raises(ValueError, match="stack depth"):
        BVH(*past_limit, n_cells=_BVH_STACK_DEPTH + 1)


# ---------------------------------------------------------------------------
# The port to C++ ownership (FELIGN/pantr#384)
# ---------------------------------------------------------------------------


def _backend_pairs() -> list[tuple[Backend, Backend]]:
    """Every ordered pair of backends, for the serialization round trips.

    Returns:
        list[tuple[Backend, Backend]]: The four ``(writer, reader)`` pairs.
    """
    return [(writer, reader) for writer in Backend for reader in Backend]


@pytest.fixture
def both_backends() -> None:
    """Require the compiled extension for a test that uses both backends at once.

    Routed through the parity harness rather than a bare ``skipif``: a bare skip is
    silent, and a suite that skips its way to green has let real failures through in
    this repository.
    """
    demand_cpp_backend()


@pytest.mark.parametrize("name", ["node_lo", "node_hi", "node_left", "node_right", "node_cell"])
def test_the_node_arrays_are_read_only_views_that_own_their_storage(
    both_backends: None, name: str
) -> None:
    """Each of the five arrays is read-only, and under C++ it aliases rather than copies.

    ``base is not None`` is the half that catches the ownerless binding: an
    ``nb::ndarray`` returned with no owner silently *copies* and comes back
    **writable**, which would pass every value assertion in this file while dropping
    the read-only contract and turning the backend switch into a performance switch
    on the one surface ``design/bvh.md`` calls public API.

    Asserted for the C++ backend only, because the Python one hands back the array
    it stores and ``numpy.ascontiguousarray`` returns that array itself, base and
    all.
    """
    del both_backends
    lo, hi = _grid_cells(2, 2)
    with use_backend(Backend.PYTHON):
        py = BVH.from_cell_bounds(lo, hi)
    with use_backend(Backend.CPP):
        cpp = BVH.from_cell_bounds(lo, hi)

    # Only the flags are asserted here. Whether the two backends built the same tree
    # is `tests/parity/test_grid.py`'s claim, and stating it twice would put the
    # parity failure message under a test named for read-only-ness.
    for tree in (py, cpp):
        assert not getattr(tree, name).flags.writeable

    assert getattr(cpp, name).base is not None, (
        f"the C++ view for {name} must carry an owner: without one nanobind copies, "
        f"and the copy comes back writable"
    )


def test_an_empty_tree_has_no_nodes_and_short_circuits_its_query() -> None:
    """Zero cells builds, reports ``n_nodes == 0``, and answers without touching a node.

    The short-circuit is what makes this safe rather than merely correct: with no
    nodes there is no node ``0`` for the traversal to read, so a query that did not
    return early would index an empty array.
    """
    tree = BVH.from_cell_bounds(np.zeros((0, 3)), np.zeros((0, 3)))
    assert tree.n_cells == 0
    assert tree.n_nodes == 0
    assert tree.ndim == 3
    assert tree.node_lo.shape == (0, 3)
    assert tree.node_left.shape == (0,)

    matched = tree.query_aabb(AABB([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
    assert matched.shape == (0,)
    assert matched.dtype == np.int64


@pytest.mark.parametrize("name", ["ndim", "n_cells", "n_nodes"])
def test_the_three_counts_are_not_writable(name: str) -> None:
    """``bvh.ndim = 7`` raises ``AttributeError`` rather than corrupting the tree.

    The three were plain public writable attributes before the port, while the class
    docstring said instances were immutable. ``bvh.ndim = 7`` defeated the dimension
    check in ``query_aabb`` and the kernels then indexed a mis-shaped array. The
    2026-08-28 census of the downstream consumer found no writes to any of them.
    """
    tree = BVH.from_cell_bounds(np.zeros((2, 1)), np.ones((2, 1)))
    with pytest.raises(AttributeError):
        setattr(tree, name, 7)
    assert tree.ndim == 1


def test_a_reversed_query_interval_is_reported_by_both_backends(both_backends: None) -> None:
    """An empty query box matches a cell whose interval contains its reversed one.

    ``BVH.query_aabb`` is a separating-axis test with no emptiness branch, so it is
    **not** :meth:`pantr.geometry.AABB.overlaps`, which reports that an empty box
    overlaps nothing. Both backends behave this way and agree with each other; the
    divergence is between the predicate and the box, and reconciling them has to
    move both backends at once.

    Pinned so that a later reconciliation has to change a test that says why, and
    with the box's own answer asserted beside it so the disagreement is on the
    record rather than implied.
    """
    del both_backends
    cell = AABB([0.0], [10.0])
    query = AABB([5.0], [3.0])
    assert query.is_empty()
    assert cell.overlaps(query) is False

    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            tree = BVH.from_cell_bounds(np.array([[0.0]]), np.array([[10.0]]))
            assert tree.query_aabb(query).tolist() == [0], backend


def test_the_capacity_error_reaches_python_as_its_own_type(both_backends: None) -> None:
    """The C++ type's traversal-stack refusal arrives as ``CapacityError``, not ``RuntimeError``.

    ``pantr::CapacityError`` derives publicly from ``std::runtime_error``, and
    nanobind's default translator maps that base to ``RuntimeError``. **That the
    specific ``nb::exception<CapacityError>`` wins over the generic catch was an
    assumption until this test**, and it is the first site in the port to throw one.
    Asserted on the exact type rather than with ``isinstance``, which the generic
    mapping would also satisfy.

    Reached through the raw handle deliberately: :class:`BVH` translates it back to
    the ``ValueError`` the pre-port class raised, so the wrapper is exactly where
    this cannot be observed.
    """
    del both_backends
    from pantr import _pantr_cpp  # noqa: PLC0415

    arrays = _chain_bvh_arrays(_BVH_STACK_DEPTH + 1)
    with pytest.raises(_pantr_cpp.CapacityError) as excinfo:
        _pantr_cpp.BVH(*arrays, _BVH_STACK_DEPTH + 1)

    assert type(excinfo.value) is _pantr_cpp.CapacityError, (
        "nanobind fell back to the generic std::runtime_error translator, so the "
        "specific exception this port registered is not being preferred"
    )
    assert issubclass(_pantr_cpp.CapacityError, RuntimeError)
    assert "stack depth" in str(excinfo.value)

    # And the wrapper presents the class a caller has always caught.
    with use_backend(Backend.CPP), pytest.raises(ValueError, match="stack depth"):
        BVH(*arrays, n_cells=_BVH_STACK_DEPTH + 1)


@pytest.mark.parametrize(("writer", "reader"), _backend_pairs())
def test_pickle_round_trips_across_every_backend_pair(
    both_backends: None, writer: Backend, reader: Backend
) -> None:
    """A pickled tree written under one backend loads under the other.

    The pre-port class had no ``__reduce__`` and pickled through its ``__slots__``.
    A C++ handle cannot pickle at all, so the port had to add one; this is what says
    the wire format did not quietly become backend-specific.
    """
    del both_backends
    lo, hi = _grid_cells(3, 3)
    with use_backend(writer):
        original = BVH.from_cell_bounds(lo, hi)
        blob = pickle.dumps(original)

    with use_backend(reader):
        loaded = pickle.loads(blob)
        assert loaded.n_cells == 9
        assert loaded.n_nodes == 17
        assert loaded.ndim == 2
        for name in ("node_lo", "node_hi", "node_left", "node_right", "node_cell"):
            np.testing.assert_array_equal(getattr(loaded, name), getattr(original, name))
        assert sorted(loaded.query_aabb(AABB([0.0, 0.0], [1.5, 1.5])).tolist()) == [0, 1, 3, 4]


def test_repr_is_byte_identical_under_both_backends(both_backends: None) -> None:
    """``repr`` is computed by the wrapper, so ``PANTR_BACKEND`` cannot move it."""
    del both_backends
    lo, hi = _grid_cells(2, 2)
    with use_backend(Backend.PYTHON):
        from_python = repr(BVH.from_cell_bounds(lo, hi))
    with use_backend(Backend.CPP):
        from_cpp = repr(BVH.from_cell_bounds(lo, hi))

    assert from_python == from_cpp
    assert from_python == "BVH(n_cells=4, n_nodes=7, ndim=2)"


def test_both_backends_agree_on_the_messages_a_caller_reads(both_backends: None) -> None:
    """Every rejection carries the same class and the same text under either backend.

    Includes the traversal-stack refusal, which is the one the C++ type reports as a
    different exception class and the wrapper translates back.
    """
    del both_backends
    node_lo, node_hi, node_left, node_right, node_cell = _three_leaf_arrays()
    deep = _chain_bvh_arrays(_BVH_STACK_DEPTH + 1)

    def messages() -> list[str]:
        out: list[str] = []
        for call in (
            lambda: BVH.from_cell_bounds(np.array([[1.0, 1.0]]), np.array([[0.0, 0.0]])),
            lambda: BVH.from_cell_bounds(np.zeros((3, 2)), np.zeros((3, 3))),
            lambda: BVH.from_cell_bounds(np.array([[np.nan, 0.0]]), np.array([[1.0, 1.0]])),
            lambda: BVH(node_lo, node_hi, node_left, node_right, node_cell, n_cells=2),
            lambda: BVH(*deep, n_cells=_BVH_STACK_DEPTH + 1),
            lambda: BVH.from_cell_bounds(np.zeros((2, 1)), np.ones((2, 1))).query_aabb(
                AABB([0.0, 0.0], [1.0, 1.0])
            ),
        ):
            with pytest.raises(ValueError) as excinfo:
                call()
            out.append(f"{type(excinfo.value).__name__}: {excinfo.value}")
        return out

    with use_backend(Backend.PYTHON):
        from_python = messages()
    with use_backend(Backend.CPP):
        from_cpp = messages()

    assert from_cpp == from_python


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
            assert 0 <= left < bvh.n_nodes
            assert 0 <= right < bvh.n_nodes
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
