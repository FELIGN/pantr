"""Tests for pantr.bspline.THBSplineSpace (non-truncated / HB path)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from pantr.bspline import BsplineSpace, BsplineSpace1D, THBSplineSpace
from pantr.bspline._thb_spline_space import _func_support_1d
from pantr.grid import HierarchicalGrid, hierarchical_grid, uniform_grid

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_KNOTS_DEG2_4 = np.array([0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0])


def _root_1d() -> BsplineSpace:
    """Degree-2 open space with 4 intervals on [0, 1]."""
    return BsplineSpace([BsplineSpace1D(_KNOTS_DEG2_4, 2)])


def _root_2d() -> BsplineSpace:
    """Degree-(2, 2) open space with 4x4 intervals on [0, 1]^2."""
    sp = BsplineSpace1D(_KNOTS_DEG2_4, 2)
    return BsplineSpace([sp, sp])


def _grid_1d() -> HierarchicalGrid:
    return hierarchical_grid(uniform_grid([[0.0, 1.0]], 4), 2)


def _grid_2d() -> HierarchicalGrid:
    return hierarchical_grid(uniform_grid([[0.0, 1.0], [0.0, 1.0]], 4), 2)


def _collocation(
    thb: THBSplineSpace, n_per_axis: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Build a global collocation matrix and sample points.

    Returns ``(A, pts)`` where ``A[r, dof]`` is the value of active function ``dof``
    at sample point ``pts[r]`` (zero where the function is inactive on the point's
    cell), sampling interior points of every active cell.
    """
    grid = thb.grid
    dim = thb.dim
    n_active = thb.num_active_functions
    npa = (thb.degrees[0] + 3) if n_per_axis is None else n_per_axis
    u = np.linspace(0.0, 1.0, npa)[1:-1]
    rows: list[np.ndarray] = []
    pts: list[np.ndarray] = []
    for cid in range(grid.num_cells):
        lo, hi = grid.cell_bounds(cid)
        axes = [lo[k] + (hi[k] - lo[k]) * u for k in range(dim)]
        mesh = np.meshgrid(*axes, indexing="ij")
        cell_pts = np.stack([m.ravel() for m in mesh], axis=-1)
        active = thb.active_basis(cid)
        values = thb.tabulate_basis(cid, cell_pts)
        for i in range(cell_pts.shape[0]):
            row = np.zeros(n_active)
            row[active] = values[i]
            rows.append(row)
            pts.append(cell_pts[i])
    return np.asarray(rows), np.asarray(pts)


def _max_reproduction_residual(
    thb: THBSplineSpace, target: Callable[[np.ndarray], np.ndarray]
) -> float:
    """Least-squares fit ``target`` with the HB basis; return max residual."""
    mat, pts = _collocation(thb)
    rhs = target(pts)
    coef, _, _, _ = np.linalg.lstsq(mat, rhs, rcond=None)
    return float(np.abs(mat @ coef - rhs).max())


# ──────────────────────────────────────────────────────────────────────────────
# Construction / validation
# ──────────────────────────────────────────────────────────────────────────────


class TestConstruction:
    """Constructor validation and the deferred-truncation guard."""

    def test_truncate_true_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="truncate=False"):
            THBSplineSpace(_root_1d(), _grid_1d())

    def test_truncate_true_explicit_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            THBSplineSpace(_root_1d(), _grid_1d(), truncate=True)

    def test_grid_not_hierarchical_raises(self) -> None:
        flat = uniform_grid([[0.0, 1.0]], 4)
        with pytest.raises(TypeError, match="HierarchicalGrid"):
            THBSplineSpace(_root_1d(), flat, truncate=False)  # type: ignore[arg-type]

    def test_dim_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="ndim"):
            THBSplineSpace(_root_1d(), _grid_2d(), truncate=False)

    def test_root_grid_cell_count_mismatch_raises(self) -> None:
        grid = hierarchical_grid(uniform_grid([[0.0, 1.0]], 5), 2)
        with pytest.raises(ValueError, match="num_intervals"):
            THBSplineSpace(_root_1d(), grid, truncate=False)

    def test_root_grid_bounds_mismatch_raises(self) -> None:
        grid = hierarchical_grid(uniform_grid([[0.0, 2.0]], 4), 2)
        with pytest.raises(ValueError, match="bounds"):
            THBSplineSpace(_root_1d(), grid, truncate=False)

    def test_regularity_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="regularity"):
            THBSplineSpace(_root_1d(), _grid_1d(), truncate=False, regularity=[1, 1])


# ──────────────────────────────────────────────────────────────────────────────
# Properties / unrefined behaviour
# ──────────────────────────────────────────────────────────────────────────────


class TestUnrefined:
    """An unrefined hierarchy reduces to the root tensor-product basis."""

    def test_properties(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d(), truncate=False)
        assert thb.dim == 1
        assert thb.degrees == (2,)
        assert thb.num_levels == 1
        assert thb.truncate is False
        assert isinstance(thb.grid, HierarchicalGrid)
        assert thb.root_space is not None

    def test_active_count_equals_root_1d(self) -> None:
        root = _root_1d()
        thb = THBSplineSpace(root, _grid_1d(), truncate=False)
        assert thb.num_active_functions == root.num_total_basis
        assert thb.num_active_functions_per_level == (root.num_total_basis,)

    def test_active_count_equals_root_2d(self) -> None:
        root = _root_2d()
        thb = THBSplineSpace(root, _grid_2d(), truncate=False)
        assert thb.num_active_functions == root.num_total_basis

    def test_partition_of_unity_when_unrefined(self) -> None:
        thb = THBSplineSpace(_root_2d(), _grid_2d(), truncate=False)
        mat, _ = _collocation(thb)
        np.testing.assert_allclose(mat.sum(axis=1), 1.0, atol=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# Coarse-space reproduction (the HB correctness check)
# ──────────────────────────────────────────────────────────────────────────────


class TestReproduction:
    """V_0 ⊆ V_h: the HB basis reproduces coarse polynomials exactly."""

    def test_reproduce_1d_three_levels(self) -> None:
        root = _root_1d()
        grid = _grid_1d()
        grid.refine(0, [0], [2])
        grid.refine(1, [0], [2])
        thb = THBSplineSpace(root, grid, truncate=False)
        assert thb.num_levels == 3
        assert _max_reproduction_residual(thb, lambda p: np.ones(len(p))) < 1e-9
        assert _max_reproduction_residual(thb, lambda p: p[:, 0]) < 1e-9
        assert _max_reproduction_residual(thb, lambda p: p[:, 0] ** 2) < 1e-9

    def test_reproduce_2d_corner_refinement(self) -> None:
        root = _root_2d()
        grid = _grid_2d()
        grid.refine(0, [0, 0], [2, 2])
        thb = THBSplineSpace(root, grid, truncate=False)
        assert _max_reproduction_residual(thb, lambda p: np.ones(len(p))) < 1e-9
        assert _max_reproduction_residual(thb, lambda p: p[:, 0]) < 1e-9
        assert _max_reproduction_residual(thb, lambda p: p[:, 1]) < 1e-9
        assert _max_reproduction_residual(thb, lambda p: p[:, 0] * p[:, 1]) < 1e-9

    def test_reproduce_2d_two_levels(self) -> None:
        root = _root_2d()
        grid = _grid_2d()
        grid.refine(0, [1, 1], [3, 3])
        grid.refine(1, [2, 2], [6, 6])
        thb = THBSplineSpace(root, grid, truncate=False)
        assert thb.num_levels == 3
        assert _max_reproduction_residual(thb, lambda p: p[:, 0] * p[:, 1]) < 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# Kraft selection
# ──────────────────────────────────────────────────────────────────────────────


class TestSelection:
    """Active-function selection (the Kraft rule)."""

    def test_hand_example_1d(self) -> None:
        # Degree 2, 4 cells, refine the left half [0, 2) at level 0.
        grid = _grid_1d()
        grid.refine(0, [0], [2])
        thb = THBSplineSpace(_root_1d(), grid, truncate=False)
        assert thb.num_active_functions_per_level == (4, 4)
        np.testing.assert_array_equal(thb.active_function_indices(0), [2, 3, 4, 5])
        np.testing.assert_array_equal(thb.active_function_indices(1), [0, 1, 2, 3])

    def test_selection_invariant(self) -> None:
        # Every active function: support ⊆ Ω_l and ⊄ Ω_{l+1}.
        root = _root_2d()
        grid = _grid_2d()
        grid.refine(0, [0, 0], [2, 2])
        grid.refine(1, [0, 0], [2, 2])
        thb = THBSplineSpace(root, grid, truncate=False)
        for level in range(thb.num_levels):
            space = thb.level_space(level)
            num_basis = space.num_basis
            subdomain = grid.subdomain_mask(level)
            refined = subdomain & ~grid.active_leaf_mask(level)
            support = [_func_support_1d(sp1d) for sp1d in space.spaces]
            for flat in thb.active_function_indices(level):
                multi = np.unravel_index(int(flat), num_basis)
                box = tuple(
                    slice(int(support[k][1][multi[k]]), int(support[k][2][multi[k]]) + 1)
                    for k in range(thb.dim)
                )
                assert subdomain[box].all(), (level, multi)
                assert not refined[box].all(), (level, multi)

    def test_active_indices_returns_copy(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d(), truncate=False)
        idx = thb.active_function_indices(0)
        idx[0] = -999
        np.testing.assert_array_equal(thb.active_function_indices(0)[0], 0)

    def test_active_function_indices_out_of_range(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d(), truncate=False)
        with pytest.raises(IndexError):
            thb.active_function_indices(5)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation / active_basis
# ──────────────────────────────────────────────────────────────────────────────


class TestEvaluation:
    """Per-cell evaluation and the active-basis index set."""

    def test_active_basis_matches_nonzeros(self) -> None:
        root = _root_1d()
        grid = _grid_1d()
        grid.refine(0, [0], [2])
        thb = THBSplineSpace(root, grid, truncate=False)
        for cid in range(grid.num_cells):
            lo, hi = grid.cell_bounds(cid)
            mid = (0.5 * (lo + hi)).reshape(1, thb.dim)
            active = thb.active_basis(cid)
            values = thb.tabulate_basis(cid, mid)
            assert values.shape == (1, active.shape[0])
            # B-splines nonzero on the cell are strictly positive at its midpoint.
            assert np.all(values[0] > 0.0)
            # active_basis is sorted and unique.
            assert np.all(np.diff(active) > 0)

    def test_values_nonnegative(self) -> None:
        root = _root_2d()
        grid = _grid_2d()
        grid.refine(0, [0, 0], [2, 2])
        thb = THBSplineSpace(root, grid, truncate=False)
        mat, _ = _collocation(thb)
        assert mat.min() >= 0.0

    def test_tabulate_basis_out_argument(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d(), truncate=False)
        pts = np.array([[0.1], [0.2]])
        k = thb.active_basis(0).shape[0]
        out = np.empty((2, k), dtype=np.float64)
        ret = thb.tabulate_basis(0, pts, out=out)
        assert ret is out
        np.testing.assert_allclose(out, thb.tabulate_basis(0, pts))

    def test_tabulate_basis_bad_out_shape_raises(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d(), truncate=False)
        with pytest.raises(ValueError, match="shape"):
            thb.tabulate_basis(0, np.array([[0.1]]), out=np.empty((1, 99)))

    def test_tabulate_basis_bad_point_dim_raises(self) -> None:
        thb = THBSplineSpace(_root_2d(), _grid_2d(), truncate=False)
        with pytest.raises(ValueError, match="trailing dimension"):
            thb.tabulate_basis(0, np.array([[0.1, 0.2, 0.3]]))

    def test_not_partition_of_unity_when_refined(self) -> None:
        # HB (non-truncated) is NOT a partition of unity over refined regions.
        root = _root_1d()
        grid = _grid_1d()
        grid.refine(0, [0], [2])
        thb = THBSplineSpace(root, grid, truncate=False)
        cid = grid.locate([0.1])
        assert cid is not None
        total = thb.tabulate_basis(cid, np.array([[0.1]])).sum()
        assert total > 1.0 + 1e-3


# ──────────────────────────────────────────────────────────────────────────────
# Level spaces / nesting
# ──────────────────────────────────────────────────────────────────────────────


class TestLevelSpaces:
    """Nested per-level tensor-product spaces."""

    def test_level_space_out_of_range(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d(), truncate=False)
        with pytest.raises(IndexError):
            thb.level_space(1)

    def test_levels_are_nested_1d(self) -> None:
        root = _root_1d()
        grid = _grid_1d()
        grid.refine(0, [0], [2])
        grid.refine(1, [0], [2])
        thb = THBSplineSpace(root, grid, truncate=False)
        for level in range(thb.num_levels - 1):
            coarse = thb.level_space(level).spaces[0].knots
            fine = set(np.round(thb.level_space(level + 1).spaces[0].knots, 12))
            for knot in coarse:
                assert round(float(knot), 12) in fine

    def test_factor_one_axis_not_subdivided(self) -> None:
        # Anisotropic refinement: factor 1 on the second axis.
        root = _root_2d()
        grid = hierarchical_grid(uniform_grid([[0.0, 1.0], [0.0, 1.0]], 4), (2, 1))
        grid.refine(0, [0, 0], [2, 2])
        thb = THBSplineSpace(root, grid, truncate=False)
        level0 = thb.level_space(0)
        level1 = thb.level_space(1)
        # Axis 0 subdivided (8 intervals), axis 1 unchanged (4 intervals).
        assert level1.num_intervals == (8, 4)
        assert level0.num_intervals == (4, 4)
        assert _max_reproduction_residual(thb, lambda p: p[:, 0] * p[:, 1]) < 1e-9
