"""Tests for :func:`pantr.bspline.partition_graph` (spectral and METIS backends)."""

from __future__ import annotations

import importlib.util

import numpy as np
import numpy.typing as npt
import pytest
from scipy import sparse
from scipy.sparse.linalg import ArpackError

from pantr.bspline import (
    CouplingGraph,
    THBSplineSpace,
    coupling_graph,
    create_uniform_space,
    partition_graph,
)
from pantr.grid import hierarchical_grid, uniform_grid

requires_pymetis = pytest.mark.skipif(
    importlib.util.find_spec("pymetis") is None, reason="pymetis is not installed"
)


def _make_graph(
    adjacency: npt.NDArray[np.int64], vertex_weights: npt.ArrayLike | None = None
) -> CouplingGraph:
    """Build a CouplingGraph from a dense symmetric integer adjacency matrix."""
    n = adjacency.shape[0]
    csr = sparse.csr_matrix(np.asarray(adjacency, dtype=np.int64))
    csr.sort_indices()
    vweights = (
        np.ones(n) if vertex_weights is None else np.asarray(vertex_weights, dtype=np.float64)
    )
    return CouplingGraph(
        n,
        csr.indptr.astype(np.int64),
        csr.indices.astype(np.int64),
        csr.data.astype(np.int64),
        vweights,
    )


def _cliques(clusters: list[list[int]], n: int, *, intra: int = 10) -> npt.NDArray[np.int64]:
    """Dense adjacency with a strong clique on each cluster."""
    adjacency = np.zeros((n, n), dtype=np.int64)
    for cluster in clusters:
        for i in cluster:
            for j in cluster:
                if i != j:
                    adjacency[i, j] = intra
    return adjacency


# --------------------------------------------------------------------------- #
# Hand-built graphs: the optimal cut is obvious (space-free correctness)
# --------------------------------------------------------------------------- #


def test_two_clusters_separated_by_weak_bridge() -> None:
    adjacency = _cliques([[0, 1, 2], [3, 4, 5]], 6)
    adjacency[2, 3] = adjacency[3, 2] = 1
    owner = partition_graph(_make_graph(adjacency), 2).cell_owner
    assert owner[0] == owner[1] == owner[2]
    assert owner[3] == owner[4] == owner[5]
    assert owner[0] != owner[3]


def test_disconnected_components_separated() -> None:
    # No bridge -> two connected components; exercises the component branch.
    adjacency = _cliques([[0, 1, 2], [3, 4, 5]], 6)
    owner = partition_graph(_make_graph(adjacency), 2).cell_owner
    assert owner[0] == owner[1] == owner[2]
    assert owner[3] == owner[4] == owner[5]
    assert owner[0] != owner[3]


def test_clique_chain_into_four_parts() -> None:
    clusters = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    adjacency = _cliques(clusters, 12)
    for i, j in [(2, 3), (5, 6), (8, 9)]:  # weak chain bridges
        adjacency[i, j] = adjacency[j, i] = 1
    owner = partition_graph(_make_graph(adjacency), 4).cell_owner
    for cluster in clusters:
        assert len(set(owner[cluster].tolist())) == 1, "each clique should be one part"
    assert len(set(owner.tolist())) == 4


def test_large_path_graph_single_cut() -> None:
    # n > dense threshold exercises the sparse eigsh path; a path bisects with one cut.
    n = 600
    diag = np.ones(n - 1, dtype=np.int64)
    csr = sparse.diags([diag, diag], [1, -1], format="csr", dtype=np.int64)
    graph = CouplingGraph(
        n,
        csr.indptr.astype(np.int64),
        csr.indices.astype(np.int64),
        csr.data.astype(np.int64),
        np.ones(n),
    )
    owner = partition_graph(graph, 2).cell_owner
    assert set(owner.tolist()) == {0, 1}
    assert int(np.sum(owner[:-1] != owner[1:])) == 1


def test_eigsh_failure_falls_back_to_dense(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the sparse eigensolver to fail; the dense fallback must still partition.
    def _boom(*args: object, **kwargs: object) -> object:
        raise ArpackError(-1)

    monkeypatch.setattr("pantr.bspline._partition_graph.eigsh", _boom)
    n = 600
    diag = np.ones(n - 1, dtype=np.int64)
    csr = sparse.diags([diag, diag], [1, -1], format="csr", dtype=np.int64)
    graph = CouplingGraph(
        n,
        csr.indptr.astype(np.int64),
        csr.indices.astype(np.int64),
        csr.data.astype(np.int64),
        np.ones(n),
    )
    owner = partition_graph(graph, 2).cell_owner
    assert set(owner.tolist()) == {0, 1}
    assert int(np.sum(owner[:-1] != owner[1:])) == 1


def test_clique_chain_into_three_parts() -> None:
    clusters = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    adjacency = _cliques(clusters, 9)
    for i, j in [(2, 3), (5, 6)]:
        adjacency[i, j] = adjacency[j, i] = 1
    owner = partition_graph(_make_graph(adjacency), 3).cell_owner
    for cluster in clusters:
        assert len(set(owner[cluster].tolist())) == 1, "each clique should be one part"
    assert len(set(owner.tolist())) == 3


def test_dense_threshold_boundary() -> None:
    # m=512 must use dense eigh (<=); m=513 must use sparse eigsh (>).
    for n in (512, 513):
        diag = np.ones(n - 1, dtype=np.int64)
        csr = sparse.diags([diag, diag], [1, -1], format="csr", dtype=np.int64)
        graph = CouplingGraph(
            n,
            csr.indptr.astype(np.int64),
            csr.indices.astype(np.int64),
            csr.data.astype(np.int64),
            np.ones(n),
        )
        owner = partition_graph(graph, 2).cell_owner
        assert int(np.sum(owner[:-1] != owner[1:])) == 1


# --------------------------------------------------------------------------- #
# Real spaces
# --------------------------------------------------------------------------- #


def _assert_full_partition(owner: npt.NDArray[np.int32], n_parts: int, n_active: int) -> None:
    assert int(np.count_nonzero(owner >= 0)) == n_active
    assert set(owner[owner >= 0].tolist()) == set(range(n_parts))


def test_real_space_partition_is_valid() -> None:
    space = create_uniform_space([2, 2], [6, 6])
    part = partition_graph(coupling_graph(space), 4)
    assert part.n_parts == 4
    _assert_full_partition(part.cell_owner, 4, 36)
    assert np.bincount(part.cell_owner, minlength=4).min() >= 1


def test_thb_space_partition_is_valid() -> None:
    root = create_uniform_space(2, 4)
    grid = hierarchical_grid(uniform_grid([[0.0, 1.0]], 4), 2)
    grid.refine(0, [0], [2])
    thb = THBSplineSpace(root, grid)
    part = partition_graph(coupling_graph(thb), 2)
    _assert_full_partition(part.cell_owner, 2, thb.grid.num_cells)


def test_weight_balanced_not_count_balanced() -> None:
    space = create_uniform_space(2, 8)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0])
    part = partition_graph(coupling_graph(space, cell_weights=weights), 2)
    owner = part.cell_owner
    part_weights = np.array([weights[owner == r].sum() for r in range(2)])
    np.testing.assert_allclose(np.sort(part_weights), [7.0, 7.0])


def test_is_deterministic() -> None:
    graph = coupling_graph(create_uniform_space([2, 2], [5, 5]))
    a = partition_graph(graph, 4).cell_owner
    b = partition_graph(graph, 4).cell_owner
    np.testing.assert_array_equal(a, b)


def test_n_parts_one() -> None:
    owner = partition_graph(coupling_graph(create_uniform_space(2, 4)), 1).cell_owner
    assert np.all(owner == 0)


def test_n_parts_equals_vertices_is_bijection() -> None:
    owner = partition_graph(coupling_graph(create_uniform_space(2, 5)), 5).cell_owner
    np.testing.assert_array_equal(np.sort(owner), np.arange(5))


def test_respects_cell_active() -> None:
    space = create_uniform_space(2, 8)
    active = np.array([True] * 4 + [False] * 4)
    part = partition_graph(coupling_graph(space), 2, cell_active=active)
    owner = part.cell_owner
    assert np.all(owner[4:] == -1)
    assert set(owner[:4].tolist()) == {0, 1}
    np.testing.assert_array_equal(part.active_mask, active)


def test_cell_active_int_coerced_to_bool() -> None:
    graph = coupling_graph(create_uniform_space(2, 4))
    part = partition_graph(graph, 2, cell_active=[1, 1, 0, 0])
    assert np.all(part.cell_owner[2:] == -1)


def test_cell_owner_is_read_only() -> None:
    part = partition_graph(coupling_graph(create_uniform_space(2, 4)), 2)
    assert not part.cell_owner.flags.writeable


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_parts", [0, -1])
def test_invalid_n_parts_raises(n_parts: int) -> None:
    graph = coupling_graph(create_uniform_space(2, 4))
    with pytest.raises(ValueError, match="n_parts must be >= 1"):
        partition_graph(graph, n_parts)


def test_n_parts_exceeds_active_raises() -> None:
    graph = coupling_graph(create_uniform_space(2, 4))
    with pytest.raises(ValueError, match="exceeds the number of active cells"):
        partition_graph(graph, 5)


def test_n_parts_exceeds_active_with_mask_raises() -> None:
    graph = coupling_graph(create_uniform_space(2, 8))
    active = np.array([True] * 2 + [False] * 6)
    with pytest.raises(ValueError, match="exceeds the number of active cells"):
        partition_graph(graph, 3, cell_active=active)


def test_non_coupling_graph_raises() -> None:
    with pytest.raises(TypeError, match="CouplingGraph"):
        partition_graph(np.zeros(3), 2)  # type: ignore[arg-type]


def test_cell_active_wrong_shape_raises() -> None:
    graph = coupling_graph(create_uniform_space(2, 4))
    with pytest.raises(ValueError, match="cell_active must have shape"):
        partition_graph(graph, 2, cell_active=[True, False])


def test_cell_active_all_false_raises() -> None:
    graph = coupling_graph(create_uniform_space(2, 4))
    with pytest.raises(ValueError, match="at least one cell active"):
        partition_graph(graph, 2, cell_active=[False, False, False, False])


def test_unknown_backend_raises() -> None:
    graph = coupling_graph(create_uniform_space(2, 4))
    with pytest.raises(ValueError, match="unknown backend"):
        partition_graph(graph, 2, backend="nope")


def test_metis_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate pymetis being unavailable even if it is installed.
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, package: str | None = None) -> object:
        return None if name == "pymetis" else real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    graph = coupling_graph(create_uniform_space(2, 4))
    with pytest.raises(ImportError, match="requires 'pymetis'"):
        partition_graph(graph, 2, backend="metis")


# --------------------------------------------------------------------------- #
# METIS backend (optional pymetis)
# --------------------------------------------------------------------------- #


@requires_pymetis
def test_metis_separates_clusters() -> None:
    adjacency = _cliques([[0, 1, 2], [3, 4, 5]], 6)
    adjacency[2, 3] = adjacency[3, 2] = 1
    owner = partition_graph(_make_graph(adjacency), 2, backend="metis").cell_owner
    assert owner[0] == owner[1] == owner[2]
    assert owner[3] == owner[4] == owner[5]
    assert owner[0] != owner[3]


@requires_pymetis
def test_metis_real_space_is_valid() -> None:
    part = partition_graph(coupling_graph(create_uniform_space([2, 2], [6, 6])), 4, backend="metis")
    owner = part.cell_owner
    assert part.n_parts == 4
    assert int(np.count_nonzero(owner >= 0)) == 36  # every (active) cell assigned
    assert owner.min() >= 0 and owner.max() < 4


@requires_pymetis
def test_metis_thb_space_is_valid() -> None:
    root = create_uniform_space(2, 4)
    grid = hierarchical_grid(uniform_grid([[0.0, 1.0]], 4), 2)
    grid.refine(0, [0], [2])
    thb = THBSplineSpace(root, grid)
    owner = partition_graph(coupling_graph(thb), 2, backend="metis").cell_owner
    assert np.all(owner >= 0) and owner.max() < 2


@requires_pymetis
def test_metis_respects_cell_active() -> None:
    space = create_uniform_space(2, 8)
    active = np.array([True] * 4 + [False] * 4)
    part = partition_graph(coupling_graph(space), 2, backend="metis", cell_active=active)
    owner = part.cell_owner
    assert np.all(owner[4:] == -1)
    assert set(owner[:4].tolist()) <= {0, 1}
    np.testing.assert_array_equal(part.active_mask, active)


@requires_pymetis
def test_metis_n_parts_one() -> None:
    graph = coupling_graph(create_uniform_space(2, 4))
    owner = partition_graph(graph, 1, backend="metis").cell_owner
    assert np.all(owner == 0)


@requires_pymetis
def test_metis_is_deterministic() -> None:
    graph = coupling_graph(create_uniform_space([2, 2], [5, 5]))
    a = partition_graph(graph, 4, backend="metis").cell_owner
    b = partition_graph(graph, 4, backend="metis").cell_owner
    np.testing.assert_array_equal(a, b)
