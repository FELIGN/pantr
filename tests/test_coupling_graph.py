"""Tests for :func:`pantr.bspline.coupling_graph`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline import (
    CouplingGraph,
    THBSplineSpace,
    coupling_graph,
    create_uniform_space,
)
from pantr.grid import hierarchical_grid, uniform_grid


def _dense(graph: CouplingGraph) -> npt.NDArray[np.int64]:
    """Densify a CouplingGraph into a symmetric (n, n) edge-weight matrix."""
    n = graph.num_vertices
    mat: npt.NDArray[np.int64] = np.zeros((n, n), dtype=np.int64)
    for c in range(n):
        neighbors = graph.adjncy[graph.xadj[c] : graph.xadj[c + 1]]
        weights = graph.edge_weights[graph.xadj[c] : graph.xadj[c + 1]]
        mat[c, neighbors] = weights
    return mat


def _thb_two_level(*, truncate: bool = True) -> THBSplineSpace:
    """A 1D two-level THB space: degree-2, 4 root cells, left half refined once."""
    root = create_uniform_space(2, 4)
    grid = hierarchical_grid(uniform_grid([[0.0, 1.0]], 4), 2)
    grid.refine(0, [0], [2])
    return THBSplineSpace(root, grid, truncate=truncate)


# --------------------------------------------------------------------------- #
# Tensor-product: closed-form coupling weights (independent of the impl)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
@pytest.mark.parametrize("n_cells", [4, 6, 7])
def test_1d_coupling_closed_form(degree: int, n_cells: int) -> None:
    # Open-uniform 1D B-splines: cells i, j share max(0, p + 1 - |i - j|) functions.
    space = create_uniform_space(degree, n_cells)
    graph = coupling_graph(space)
    assert graph.num_vertices == n_cells

    expected = np.zeros((n_cells, n_cells), dtype=np.int64)
    for i in range(n_cells):
        for j in range(n_cells):
            if i != j:
                expected[i, j] = max(0, degree + 1 - abs(i - j))
    np.testing.assert_array_equal(_dense(graph), expected)


@pytest.mark.parametrize(
    ("degrees", "cells"),
    [((1, 1), (3, 3)), ((2, 1), (3, 4)), ((2, 2), (4, 3))],
)
def test_2d_coupling_closed_form(degrees: tuple[int, int], cells: tuple[int, int]) -> None:
    # Tensor product: shared functions = product of the per-axis shared counts.
    space = create_uniform_space(list(degrees), list(cells))
    graph = coupling_graph(space)
    n = int(np.prod(cells))
    assert graph.num_vertices == n

    multi = np.array(np.unravel_index(np.arange(n), cells)).T
    expected = np.zeros((n, n), dtype=np.int64)
    for a in range(n):
        for b in range(n):
            if a != b:
                weight = 1
                for d in range(2):
                    weight *= max(0, degrees[d] + 1 - abs(int(multi[a, d]) - int(multi[b, d])))
                expected[a, b] = weight
    np.testing.assert_array_equal(_dense(graph), expected)


# --------------------------------------------------------------------------- #
# THB: cross-check against active_basis intersections
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("truncate", [True, False])
def test_thb_matches_active_basis_intersection(truncate: bool) -> None:
    space = _thb_two_level(truncate=truncate)
    graph = coupling_graph(space)
    assert graph.num_vertices == space.grid.num_cells
    mat = _dense(graph)
    n = space.grid.num_cells
    dofs = [set(space.active_basis(c).tolist()) for c in range(n)]
    for a in range(n):
        for b in range(n):
            shared = 0 if a == b else len(dofs[a] & dofs[b])
            assert int(mat[a, b]) == shared


# --------------------------------------------------------------------------- #
# Structural invariants
# --------------------------------------------------------------------------- #

_INVARIANT_SPACES = [
    create_uniform_space(2, 5),
    create_uniform_space([2, 2], [3, 3]),
    create_uniform_space([1, 2, 1], [2, 3, 2]),
    _thb_two_level(),
]
_INVARIANT_IDS = ["tp_1d", "tp_2d", "tp_3d", "thb_2level"]


@pytest.mark.parametrize("space", _INVARIANT_SPACES, ids=_INVARIANT_IDS)
def test_graph_invariants(space: object) -> None:
    graph = coupling_graph(space)  # type: ignore[arg-type]
    n = graph.num_vertices

    assert graph.xadj.shape == (n + 1,)
    assert int(graph.xadj[0]) == 0
    assert int(graph.xadj[-1]) == graph.adjncy.size
    assert np.all(np.diff(graph.xadj) >= 0)
    assert graph.edge_weights.shape == graph.adjncy.shape
    if graph.adjncy.size:
        assert int(graph.adjncy.min()) >= 0 and int(graph.adjncy.max()) < n
        assert np.all(graph.edge_weights >= 1)

    mat = _dense(graph)
    assert np.array_equal(mat, mat.T), "coupling must be symmetric"
    assert np.all(np.diag(mat) == 0), "no self-loops"

    assert graph.vertex_weights.shape == (n,)
    np.testing.assert_array_equal(graph.vertex_weights, 1.0)
    for arr in (graph.xadj, graph.adjncy, graph.edge_weights, graph.vertex_weights):
        assert not arr.flags.writeable


def test_single_cell_has_no_edges() -> None:
    space = create_uniform_space(2, 1)
    graph = coupling_graph(space)
    assert graph.num_vertices == 1
    assert graph.adjncy.size == 0
    assert graph.edge_weights.size == 0
    np.testing.assert_array_equal(graph.xadj, [0, 0])


# --------------------------------------------------------------------------- #
# Vertex weights and error handling
# --------------------------------------------------------------------------- #


def test_custom_cell_weights_passthrough() -> None:
    graph = coupling_graph(create_uniform_space(2, 4), cell_weights=[1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(graph.vertex_weights, [1.0, 2.0, 3.0, 4.0])
    assert not graph.vertex_weights.flags.writeable


def test_cell_weights_input_not_frozen() -> None:
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    coupling_graph(create_uniform_space(2, 4), cell_weights=weights)
    assert weights.flags.writeable, "the caller's array must not be frozen"


@pytest.mark.parametrize("bad", [[1.0, 2.0], [1.0, 2.0, 3.0, 4.0, 5.0]])
def test_cell_weights_wrong_shape_raises(bad: list[float]) -> None:
    with pytest.raises(ValueError, match="cell_weights must have shape"):
        coupling_graph(create_uniform_space(2, 4), cell_weights=bad)


@pytest.mark.parametrize("bad", [[1.0, -1.0, 1.0, 1.0], [1.0, np.inf, 1.0, 1.0]])
def test_cell_weights_invalid_values_raise(bad: list[float]) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        coupling_graph(create_uniform_space(2, 4), cell_weights=bad)


def test_periodic_space_raises() -> None:
    space = create_uniform_space(2, 4, periodic=True)
    with pytest.raises(ValueError, match="periodic"):
        coupling_graph(space)


def test_cell_weights_all_zero_accepted() -> None:
    graph = coupling_graph(create_uniform_space(2, 4), cell_weights=[0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(graph.vertex_weights, 0.0)
    assert not graph.vertex_weights.flags.writeable


def test_non_space_input_raises() -> None:
    with pytest.raises(TypeError, match="BsplineSpace or THBSplineSpace"):
        coupling_graph(uniform_grid([[0.0, 1.0]], 4))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Reduced-continuity closed-form coupling
# --------------------------------------------------------------------------- #


def test_1d_coupling_reduced_continuity() -> None:
    # C^0 (continuity=0) degree-2 space: adjacent cells share exactly 1 DOF,
    # non-adjacent share 0 (vs max-continuity where adjacent cells share 2).
    degree, n_cells = 2, 5
    space = create_uniform_space(degree, n_cells, continuity=0)
    graph = coupling_graph(space)
    assert graph.num_vertices == n_cells

    expected = np.zeros((n_cells, n_cells), dtype=np.int64)
    for i in range(n_cells):
        for j in range(n_cells):
            if abs(i - j) == 1:
                expected[i, j] = 1
    np.testing.assert_array_equal(_dense(graph), expected)
