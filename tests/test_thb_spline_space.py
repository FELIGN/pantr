"""Tests for pantr.bspline.THBSplineSpace (HB and THB bases, values and derivatives)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline import (
    BsplineSpace,
    BsplineSpace1D,
    MultiLevelExtraction,
    THBSpline,
    THBSplineSpace,
    THBSplineSpaceRestriction,
    create_thb_space,
    create_uniform_space,
)
from pantr.bspline._thb_eval_core import _combine_tp_values
from pantr.bspline._thb_spline_space import _box_all_true, _func_support_1d
from pantr.grid import HierarchicalGrid, hierarchical_grid, tensor_product_grid, uniform_grid

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
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Build a global collocation matrix and sample points.

    Returns ``(A, pts)`` where ``A[r, dof]`` is the value of active function ``dof``
    at sample point ``pts[r]`` (zero where the function is inactive on the point's
    cell), sampling interior points of every active cell.
    """
    grid = thb.grid
    dim = thb.dim
    n_active = thb.num_total_basis
    npa = (thb.degrees[0] + 3) if n_per_axis is None else n_per_axis
    u = np.linspace(0.0, 1.0, npa)[1:-1]
    rows: list[npt.NDArray[np.float64]] = []
    pts: list[npt.NDArray[np.float64]] = []
    for cid in range(grid.num_cells):
        lo, hi = grid.cell_bounds(cid)
        axes = [lo[k] + (hi[k] - lo[k]) * u for k in range(dim)]
        mesh = np.meshgrid(*axes, indexing="ij")
        cell_pts = np.stack([m.ravel() for m in mesh], axis=-1)
        values, active = thb.tabulate_basis(cid, cell_pts)
        for i in range(cell_pts.shape[0]):
            row = np.zeros(n_active)
            row[active] = values[i]
            rows.append(row)
            pts.append(cell_pts[i])
    return np.asarray(rows), np.asarray(pts)


def _max_reproduction_residual(
    thb: THBSplineSpace, target: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
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
    """Constructor validation and the ``truncate`` flag."""

    def test_default_is_truncated(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d())
        assert thb.truncate is True

    def test_truncate_false_succeeds(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d(), truncate=False)
        assert thb.truncate is False

    def test_root_space_not_bspline_raises(self) -> None:
        sp1d = BsplineSpace1D(_KNOTS_DEG2_4, 2)
        with pytest.raises(TypeError, match="BsplineSpace"):
            THBSplineSpace(sp1d, _grid_1d(), truncate=False)  # type: ignore[arg-type]

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

    def test_regularity_value_too_high_raises(self) -> None:
        # degree=2 → max regularity is 1; regularity=2 is invalid.
        with pytest.raises(ValueError, match="regularity"):
            THBSplineSpace(_root_1d(), _grid_1d(), truncate=False, regularity=2)

    def test_regularity_value_too_low_raises(self) -> None:
        # -1 is the minimum (C^{-1}); -2 is invalid.
        with pytest.raises(ValueError, match="regularity"):
            THBSplineSpace(_root_1d(), _grid_1d(), truncate=False, regularity=-2)


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
        assert thb.num_total_basis == root.num_total_basis
        assert thb.num_basis_per_level == (root.num_total_basis,)

    def test_active_count_equals_root_2d(self) -> None:
        root = _root_2d()
        thb = THBSplineSpace(root, _grid_2d(), truncate=False)
        assert thb.num_total_basis == root.num_total_basis

    def test_partition_of_unity_when_unrefined(self) -> None:
        thb = THBSplineSpace(_root_2d(), _grid_2d(), truncate=False)
        mat, _ = _collocation(thb)
        np.testing.assert_allclose(mat.sum(axis=1), 1.0, atol=1e-12)

    def test_repr_smoke(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d(), truncate=False)
        r = repr(thb)
        assert "THBSplineSpace" in r
        assert "dim=1" in r
        assert "num_levels=1" in r
        assert "truncate=False" in r


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
        assert thb.num_basis_per_level == (4, 4)
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
        original_first = int(idx[0])
        idx[0] = -999
        assert int(thb.active_function_indices(0)[0]) == original_first

    def test_active_function_indices_out_of_range(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d(), truncate=False)
        with pytest.raises(ValueError):
            thb.active_function_indices(5)

    def test_fully_refined_level_has_no_active_functions(self) -> None:
        # Refining the entire domain at level 0 displaces all level-0 functions.
        root = _root_1d()
        grid = _grid_1d()
        grid.refine(0, [0], [4])
        thb = THBSplineSpace(root, grid, truncate=False)
        assert thb.num_basis_per_level[0] == 0
        assert thb.num_basis_per_level[1] > 0


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
            values, active = thb.tabulate_basis(cid, mid)
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

    def test_tabulate_basis_out_arguments(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d(), truncate=False)
        pts = np.array([[0.1], [0.2]])
        k = thb.active_basis(0).shape[0]
        out_basis = np.empty((2, k), dtype=np.float64)
        out_dofs = np.empty((k,), dtype=np.int64)
        vals, dofs = thb.tabulate_basis(0, pts, out_basis=out_basis, out_dofs=out_dofs)
        assert vals is out_basis
        assert dofs is out_dofs
        exp_vals, exp_dofs = thb.tabulate_basis(0, pts)
        np.testing.assert_allclose(out_basis, exp_vals)
        np.testing.assert_array_equal(out_dofs, exp_dofs)

    def test_tabulate_basis_bad_out_shape_raises(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d(), truncate=False)
        with pytest.raises(ValueError, match="shape"):
            thb.tabulate_basis(0, np.array([[0.1]]), out_basis=np.empty((1, 99)))

    def test_tabulate_basis_bad_out_dofs_shape_raises(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d(), truncate=False)
        with pytest.raises(ValueError, match="out_dofs must have shape"):
            thb.tabulate_basis(0, np.array([[0.1]]), out_dofs=np.empty((99,), dtype=np.int64))

    def test_tabulate_basis_bad_out_dtype_raises(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d(), truncate=False)
        k = thb.active_basis(0).shape[0]
        out_bad = np.empty((1, k), dtype=np.float32)
        with pytest.raises(ValueError, match="dtype"):
            thb.tabulate_basis(0, np.array([[0.1]]), out_basis=out_bad)  # type: ignore[arg-type]

    def test_tabulate_basis_readonly_out_raises(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d(), truncate=False)
        k = thb.active_basis(0).shape[0]
        out = np.empty((1, k), dtype=np.float64)
        out.flags.writeable = False
        with pytest.raises(ValueError, match="writeable"):
            thb.tabulate_basis(0, np.array([[0.1]]), out_basis=out)

    def test_tabulate_basis_bad_point_dim_raises(self) -> None:
        thb = THBSplineSpace(_root_2d(), _grid_2d(), truncate=False)
        with pytest.raises(ValueError, match="trailing dimension"):
            thb.tabulate_basis(0, np.array([[0.1, 0.2, 0.3]]))

    def test_active_basis_bad_cid_raises(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d(), truncate=False)
        with pytest.raises(IndexError):
            thb.active_basis(999)

    def test_tabulate_basis_bad_cid_raises(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d(), truncate=False)
        with pytest.raises(IndexError):
            thb.tabulate_basis(999, np.array([[0.5]]))

    def test_tabulate_basis_pts_outside_cell_raises(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d(), truncate=False)
        # Cell 0 covers [0.0, 0.25); point 0.5 is outside.
        with pytest.raises(ValueError, match="bounds"):
            thb.tabulate_basis(0, np.array([[0.5]]))

    def test_tabulate_basis_scalar_point(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d(), truncate=False)
        # Shape (dim,) — single point without a leading batch dimension.
        result, _ = thb.tabulate_basis(0, np.array([0.1]))
        assert result.ndim == 1
        assert result.shape[0] == thb.active_basis(0).shape[0]

    def test_not_partition_of_unity_when_refined(self) -> None:
        # HB (non-truncated) is NOT a partition of unity over refined regions.
        root = _root_1d()
        grid = _grid_1d()
        grid.refine(0, [0], [2])
        thb = THBSplineSpace(root, grid, truncate=False)
        cid = grid.locate([0.1])
        assert cid is not None
        lo, hi = grid.cell_bounds(cid)
        mid = float(0.5 * (lo[0] + hi[0]))
        total = thb.tabulate_basis(cid, np.array([[mid]]))[0].sum()
        # Functions from both level 0 and level 1 contribute; sum > 1.
        assert total > 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Stale-grid detection
# ──────────────────────────────────────────────────────────────────────────────


class TestStaleGrid:
    """THBSplineSpace raises RuntimeError if the grid is modified after construction."""

    def test_stale_grid_active_basis_raises(self) -> None:
        grid = _grid_1d()
        thb = THBSplineSpace(_root_1d(), grid, truncate=False)
        grid.refine(0, [0], [2])
        with pytest.raises(RuntimeError, match="stale"):
            thb.active_basis(0)

    def test_stale_grid_tabulate_basis_raises(self) -> None:
        grid = _grid_1d()
        thb = THBSplineSpace(_root_1d(), grid, truncate=False)
        grid.refine(0, [0], [2])
        with pytest.raises(RuntimeError, match="stale"):
            thb.tabulate_basis(0, np.array([[0.1]]))

    def test_unmodified_grid_does_not_raise(self) -> None:
        grid = _grid_1d()
        thb = THBSplineSpace(_root_1d(), grid, truncate=False)
        # No refine call — must not raise.
        _ = thb.active_basis(0)


# ──────────────────────────────────────────────────────────────────────────────
# Level spaces / nesting
# ──────────────────────────────────────────────────────────────────────────────


class TestLevelSpaces:
    """Nested per-level tensor-product spaces."""

    def test_level_space_out_of_range(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d(), truncate=False)
        with pytest.raises(ValueError):
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
        assert thb.num_total_basis == sum(thb.num_basis_per_level)
        assert _max_reproduction_residual(thb, lambda p: p[:, 0] * p[:, 1]) < 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# Truncation (THB)
# ──────────────────────────────────────────────────────────────────────────────


def _refined_1d_three_levels() -> THBSplineSpace:
    """Degree-2 1D space with the left region refined twice (truncated)."""
    grid = _grid_1d()
    grid.refine(0, [0], [2])
    grid.refine(1, [0], [2])
    return THBSplineSpace(_root_1d(), grid, truncate=True)


def _refined_2d_corner() -> THBSplineSpace:
    """Degree-(2, 2) space with the lower-left corner refined once (truncated)."""
    grid = _grid_2d()
    grid.refine(0, [0, 0], [2, 2])
    return THBSplineSpace(_root_2d(), grid, truncate=True)


class TestTruncation:
    """Truncated (THB) basis: partition of unity, nonnegativity, reproduction."""

    def test_partition_of_unity_1d(self) -> None:
        mat, _ = _collocation(_refined_1d_three_levels())
        np.testing.assert_allclose(mat.sum(axis=1), 1.0, atol=1e-10)

    def test_partition_of_unity_2d_corner(self) -> None:
        mat, _ = _collocation(_refined_2d_corner())
        np.testing.assert_allclose(mat.sum(axis=1), 1.0, atol=1e-10)

    def test_partition_of_unity_2d_three_levels(self) -> None:
        grid = _grid_2d()
        grid.refine(0, [1, 1], [3, 3])
        grid.refine(1, [2, 2], [6, 6])
        thb = THBSplineSpace(_root_2d(), grid, truncate=True)
        assert thb.num_levels == 3
        mat, _ = _collocation(thb)
        np.testing.assert_allclose(mat.sum(axis=1), 1.0, atol=1e-10)

    def test_values_nonnegative(self) -> None:
        mat, _ = _collocation(_refined_2d_corner())
        assert mat.min() >= -1e-14

    def test_reproduces_coarse_space_2d(self) -> None:
        # V_0 ⊆ V_h holds for THB as well (same space as HB, just truncated basis).
        thb = _refined_2d_corner()
        assert _max_reproduction_residual(thb, lambda p: np.ones(len(p))) < 1e-9
        assert _max_reproduction_residual(thb, lambda p: p[:, 0]) < 1e-9
        assert _max_reproduction_residual(thb, lambda p: p[:, 1]) < 1e-9
        assert _max_reproduction_residual(thb, lambda p: p[:, 0] * p[:, 1]) < 1e-9

    def test_some_functions_truncated(self) -> None:
        thb = _refined_2d_corner()
        n_truncated = len(thb._trunc)
        assert 0 < n_truncated < thb.num_total_basis

    def test_no_truncation_when_unrefined(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d(), truncate=True)
        assert len(thb._trunc) == 0
        assert thb.num_total_basis == _root_1d().num_total_basis

    def test_truncation_identity_outside_refinement(self) -> None:
        # On a cell entirely outside the refined region, truncation changes nothing:
        # THB and HB values coincide there (same selection, same values).
        root = _root_1d()
        grid = _grid_1d()
        grid.refine(0, [0], [2])  # refine left half [0, 0.5)
        thb = THBSplineSpace(root, grid, truncate=True)
        hb = THBSplineSpace(root, grid, truncate=False)
        cid = grid.locate([0.9])  # cell [0.75, 1.0], outside the refinement
        assert cid is not None
        lo, hi = grid.cell_bounds(cid)
        pts = lo + (hi - lo) * np.linspace(0.1, 0.9, 5)[:, None]
        np.testing.assert_array_equal(thb.active_basis(cid), hb.active_basis(cid))
        np.testing.assert_allclose(
            thb.tabulate_basis(cid, pts)[0], hb.tabulate_basis(cid, pts)[0], atol=1e-12
        )

    def test_truncation_restores_partition_of_unity_on_refined_cell(self) -> None:
        # Where HB over-sums (> 1), THB sums to exactly 1.
        root = _root_1d()
        grid = _grid_1d()
        grid.refine(0, [0], [2])
        thb = THBSplineSpace(root, grid, truncate=True)
        hb = THBSplineSpace(root, grid, truncate=False)
        cid = grid.locate([0.1])  # refined region
        assert cid is not None
        pt = np.array([[0.1]])
        assert thb.tabulate_basis(cid, pt)[0].sum() == pytest.approx(1.0, abs=1e-12)
        assert hb.tabulate_basis(cid, pt)[0].sum() > 1.0 + 1e-3

    def test_repr_truncate_true(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d(), truncate=True)
        assert "truncate=True" in repr(thb)

    def test_tabulate_basis_out_argument_truncated(self) -> None:
        thb = _refined_2d_corner()
        cid = thb.grid.locate([0.1, 0.1])
        assert cid is not None
        active = thb.active_basis(cid)
        assert any(dof in thb._trunc for dof in active), "no truncated dof on this cell"
        pts = np.array([[0.1, 0.1]])
        out_basis = np.empty((1, active.shape[0]), dtype=np.float64)
        vals, _ = thb.tabulate_basis(cid, pts, out_basis=out_basis)
        assert vals is out_basis
        exp, _ = thb.tabulate_basis(cid, pts)
        np.testing.assert_allclose(out_basis, exp)

    def test_deepest_level_functions_not_in_trunc(self) -> None:
        # Active functions at the deepest level have no finer level below;
        # _compute_truncated_coeffs iterates over level < num_levels-1 only.
        thb = _refined_1d_three_levels()
        deepest = thb.num_levels - 1
        offset = int(thb._func_offset[deepest])
        count = int(thb.num_basis_per_level[deepest])
        for i in range(count):
            assert offset + i not in thb._trunc

    def test_partition_of_unity_with_regularity_c0(self) -> None:
        root = _root_1d()
        grid = _grid_1d()
        grid.refine(0, [0], [2])
        thb = THBSplineSpace(root, grid, truncate=True, regularity=0)
        mat, _ = _collocation(thb)
        np.testing.assert_allclose(mat.sum(axis=1), 1.0, atol=1e-10)

    def test_truncated_function_column_strictly_less_than_hb(self) -> None:
        # A truncated function on a refined cell evaluates to a value strictly
        # less than its HB counterpart (truncation reduced it) and >= 0.
        root = _root_1d()
        grid = _grid_1d()
        grid.refine(0, [0], [2])
        thb = THBSplineSpace(root, grid, truncate=True)
        hb = THBSplineSpace(root, grid, truncate=False)
        cid = grid.locate([0.3])
        assert cid is not None
        lo, hi = grid.cell_bounds(cid)
        mid = (0.5 * (lo + hi)).reshape(1, -1)
        thb_active = thb.active_basis(cid)
        hb_active = hb.active_basis(cid)
        thb_vals = thb.tabulate_basis(cid, mid)[0][0]
        hb_vals = hb.tabulate_basis(cid, mid)[0][0]
        found_truncated = False
        for i, dof in enumerate(thb_active):
            if dof in thb._trunc:
                found_truncated = True
                thb_val = float(thb_vals[i])
                j = int(np.searchsorted(hb_active, dof))
                hb_val = float(hb_vals[j])
                assert thb_val >= 0.0
                assert thb_val < hb_val - 1e-10, (
                    f"dof {dof}: THB value {thb_val:.6f} not < HB value {hb_val:.6f}"
                )
        assert found_truncated, "no truncated dof found on refined cell"

    def test_partition_of_unity_anisotropic_truncated(self) -> None:
        root = _root_2d()
        grid = hierarchical_grid(uniform_grid([[0.0, 1.0], [0.0, 1.0]], 4), (2, 1))
        grid.refine(0, [0, 0], [2, 2])
        thb = THBSplineSpace(root, grid, truncate=True)
        mat, _ = _collocation(thb)
        np.testing.assert_allclose(mat.sum(axis=1), 1.0, atol=1e-10)

    def test_tabulate_basis_multidim_leading_shape_truncated(self) -> None:
        thb = _refined_2d_corner()
        cid = thb.grid.locate([0.1, 0.1])
        assert cid is not None
        lo, hi = thb.grid.cell_bounds(cid)
        ts = np.linspace(0.1, 0.9, 6)
        pts_flat = lo + (hi - lo) * ts[:, None]  # (6, 2)
        pts = pts_flat.reshape(2, 3, thb.dim)
        result, dofs = thb.tabulate_basis(cid, pts)
        n_active = thb.active_basis(cid).shape[0]
        assert result.shape == (2, 3, n_active)
        assert dofs.shape == (n_active,)
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-10)

    def test_thb_and_hb_same_active_count(self) -> None:
        # Truncation changes function values but not the active-function selection.
        root, grid = _root_2d(), _grid_2d()
        grid.refine(0, [0, 0], [2, 2])
        thb = THBSplineSpace(root, grid, truncate=True)
        hb = THBSplineSpace(root, grid, truncate=False)
        assert thb.num_total_basis == hb.num_total_basis
        assert thb.num_basis_per_level == hb.num_basis_per_level
        assert thb.num_total_basis == sum(thb.num_basis_per_level)

    def test_active_basis_union_covers_all_dofs(self) -> None:
        # The union of active_basis(cid) over all cells must equal {0, …, N-1}.
        thb = _refined_2d_corner()
        seen: set[int] = set()
        for cid in range(thb.grid.num_cells):
            seen.update(thb.active_basis(cid).tolist())
        assert seen == set(range(thb.num_total_basis))

    def test_partition_of_unity_factor3(self) -> None:
        # Verify that factor=3 (non-power-of-2) refinement still gives partition of unity.
        root = _root_1d()
        grid = hierarchical_grid(uniform_grid([[0.0, 1.0]], 4), 3)
        grid.refine(0, [0], [2])
        thb = THBSplineSpace(root, grid, truncate=True)
        mat, _ = _collocation(thb)
        np.testing.assert_allclose(mat.sum(axis=1), 1.0, atol=1e-10)

    def test_partition_of_unity_degree1(self) -> None:
        # Degree-1 (linear): support = 1 interval per function; different truncation geometry.
        knots = np.array([0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0])
        sp1d = BsplineSpace1D(knots, 1)
        root = BsplineSpace([sp1d])
        grid = hierarchical_grid(uniform_grid([[0.0, 1.0]], 4), 2)
        grid.refine(0, [0], [2])
        thb = THBSplineSpace(root, grid, truncate=True)
        mat, _ = _collocation(thb)
        np.testing.assert_allclose(mat.sum(axis=1), 1.0, atol=1e-10)

    def test_regularity_sequence_form(self) -> None:
        # Scalar regularity=0 and equivalent sequence form [0] produce the same space.
        root = _root_1d()
        grid1 = _grid_1d()
        grid1.refine(0, [0], [2])
        thb_scalar = THBSplineSpace(root, grid1, truncate=True, regularity=0)
        grid2 = _grid_1d()
        grid2.refine(0, [0], [2])
        thb_seq = THBSplineSpace(root, grid2, truncate=True, regularity=[0])
        assert thb_scalar.num_total_basis == thb_seq.num_total_basis
        mat, _ = _collocation(thb_seq)
        np.testing.assert_allclose(mat.sum(axis=1), 1.0, atol=1e-10)


# ──────────────────────────────────────────────────────────────────────────────
# Derivatives
# ──────────────────────────────────────────────────────────────────────────────


def _collocation_derivatives(
    thb: THBSplineSpace, orders: int | Sequence[int]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Like :func:`_collocation` but evaluates the ``orders`` mixed partial."""
    grid = thb.grid
    dim = thb.dim
    n_active = thb.num_total_basis
    u = np.linspace(0.0, 1.0, thb.degrees[0] + 3)[1:-1]
    rows: list[npt.NDArray[np.float64]] = []
    pts: list[npt.NDArray[np.float64]] = []
    for cid in range(grid.num_cells):
        lo, hi = grid.cell_bounds(cid)
        axes = [lo[k] + (hi[k] - lo[k]) * u for k in range(dim)]
        mesh = np.meshgrid(*axes, indexing="ij")
        cell_pts = np.stack([m.ravel() for m in mesh], axis=-1)
        values, active = thb.tabulate_basis_derivatives(cid, cell_pts, orders)
        for i in range(cell_pts.shape[0]):
            row = np.zeros(n_active)
            row[active] = values[i]
            rows.append(row)
            pts.append(cell_pts[i])
    return np.asarray(rows), np.asarray(pts)


class TestDerivatives:
    """Parametric derivatives via tabulate_basis_derivatives (HB and THB)."""

    def test_order_zero_equals_values_thb(self) -> None:
        thb = _refined_2d_corner()
        cid = thb.grid.locate([0.1, 0.1])
        assert cid is not None
        lo, hi = thb.grid.cell_bounds(cid)
        pts = (lo + (hi - lo) * 0.5).reshape(1, thb.dim)
        np.testing.assert_allclose(
            thb.tabulate_basis_derivatives(cid, pts, 0)[0], thb.tabulate_basis(cid, pts)[0]
        )

    def test_order_zero_equals_values_hb(self) -> None:
        root = _root_1d()
        grid = _grid_1d()
        grid.refine(0, [0], [2])
        hb = THBSplineSpace(root, grid, truncate=False)
        cid = grid.locate([0.1])
        assert cid is not None
        lo, hi = grid.cell_bounds(cid)
        pts = lo + (hi - lo) * np.array([0.25, 0.5, 0.75])[:, None]
        np.testing.assert_allclose(
            hb.tabulate_basis_derivatives(cid, pts, 0)[0], hb.tabulate_basis(cid, pts)[0]
        )

    def test_partition_of_unity_derivative_is_zero_1d(self) -> None:
        # d/dx of (sum of THB basis = 1) is 0 everywhere.
        mat, _ = _collocation_derivatives(_refined_1d_three_levels(), 1)
        np.testing.assert_allclose(mat.sum(axis=1), 0.0, atol=1e-10)

    def test_partition_of_unity_derivative_is_zero_2d(self) -> None:
        thb = _refined_2d_corner()
        for orders in ([1, 0], [0, 1]):
            mat, _ = _collocation_derivatives(thb, orders)
            np.testing.assert_allclose(mat.sum(axis=1), 0.0, atol=1e-10)

    def test_reproduce_first_derivative_1d(self) -> None:
        thb = _refined_1d_three_levels()
        a_val, pts = _collocation(thb)
        a_dx, _ = _collocation_derivatives(thb, 1)
        coef, _, _, _ = np.linalg.lstsq(a_val, pts[:, 0] ** 2, rcond=None)  # reproduce x^2
        assert np.abs(a_dx @ coef - 2.0 * pts[:, 0]).max() < 1e-9  # d/dx (x^2) = 2x

    def test_reproduce_second_derivative_1d(self) -> None:
        thb = _refined_1d_three_levels()
        a_val, pts = _collocation(thb)
        a_dxx, _ = _collocation_derivatives(thb, 2)
        coef, _, _, _ = np.linalg.lstsq(a_val, pts[:, 0] ** 2, rcond=None)
        assert np.abs(a_dxx @ coef - 2.0).max() < 1e-9  # d2/dx2 (x^2) = 2

    def test_reproduce_mixed_derivative_2d(self) -> None:
        thb = _refined_2d_corner()
        a_val, pts = _collocation(thb)
        coef, _, _, _ = np.linalg.lstsq(a_val, pts[:, 0] * pts[:, 1], rcond=None)  # reproduce xy
        a_dx, _ = _collocation_derivatives(thb, [1, 0])
        a_dxy, _ = _collocation_derivatives(thb, [1, 1])
        assert np.abs(a_dx @ coef - pts[:, 1]).max() < 1e-9  # d/dx (xy) = y
        assert np.abs(a_dxy @ coef - 1.0).max() < 1e-9  # d2/dxdy (xy) = 1

    def test_finite_difference_1d(self) -> None:
        thb = _refined_1d_three_levels()
        h = 1e-6
        for cid in range(thb.grid.num_cells):
            lo, hi = thb.grid.cell_bounds(cid)
            x = float(0.5 * (lo[0] + hi[0]))
            fd = (
                thb.tabulate_basis(cid, np.array([[x + h]]))[0]
                - thb.tabulate_basis(cid, np.array([[x - h]]))[0]
            ) / (2.0 * h)
            analytic = thb.tabulate_basis_derivatives(cid, np.array([[x]]), 1)[0]
            np.testing.assert_allclose(fd, analytic, atol=1e-6)

    def test_high_order_is_zero(self) -> None:
        # Degree 2: the third derivative is identically zero.
        thb = _refined_1d_three_levels()
        cid = thb.grid.locate([0.1])
        assert cid is not None
        result, _ = thb.tabulate_basis_derivatives(cid, np.array([[0.1]]), 3)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_orders_length_mismatch_raises(self) -> None:
        thb = _refined_2d_corner()
        with pytest.raises(ValueError, match="orders"):
            thb.tabulate_basis_derivatives(0, np.array([[0.1, 0.1]]), [1])

    def test_negative_order_raises(self) -> None:
        thb = _refined_1d_three_levels()
        with pytest.raises(ValueError, match="non-negative"):
            thb.tabulate_basis_derivatives(0, np.array([[0.1]]), -1)

    def test_out_argument(self) -> None:
        thb = _refined_1d_three_levels()
        cid = thb.grid.locate([0.1])
        assert cid is not None
        lo, hi = thb.grid.cell_bounds(cid)
        pts = lo + (hi - lo) * np.array([0.3, 0.6])[:, None]
        k = thb.active_basis(cid).shape[0]
        out_basis = np.empty((2, k), dtype=np.float64)
        vals, _ = thb.tabulate_basis_derivatives(cid, pts, 1, out_basis=out_basis)
        assert vals is out_basis
        exp, _ = thb.tabulate_basis_derivatives(cid, pts, 1)
        np.testing.assert_allclose(out_basis, exp)

    def test_hb_derivative_reproduction(self) -> None:
        root = _root_1d()
        grid = _grid_1d()
        grid.refine(0, [0], [2])
        grid.refine(1, [0], [2])
        hb = THBSplineSpace(root, grid, truncate=False)
        a_val, pts = _collocation(hb)
        a_dx, _ = _collocation_derivatives(hb, 1)
        coef, _, _, _ = np.linalg.lstsq(a_val, pts[:, 0] ** 2, rcond=None)
        assert np.abs(a_dx @ coef - 2.0 * pts[:, 0]).max() < 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# Refinement / admissibility / prolongation
# ──────────────────────────────────────────────────────────────────────────────


def _dof_level(thb: THBSplineSpace, dof: int) -> int:
    """Return the hierarchy level owning global dof ``dof``."""
    return int(np.searchsorted(thb._func_offset, dof, side="right")) - 1


def _nonzero_level_span(thb: THBSplineSpace) -> int:
    """Return the max number of successive levels of nonzero THB functions on any cell.

    Uses tabulated values to filter: fully-truncated functions evaluate to exactly
    zero on cells where they have no truncated support, and correctly don't count.
    """
    grid = thb.grid
    u = np.linspace(0.0, 1.0, thb.degrees[0] + 3)[1:-1]
    worst = 0
    for cid in range(grid.num_cells):
        lo, hi = grid.cell_bounds(cid)
        mesh = np.meshgrid(*[lo[k] + (hi[k] - lo[k]) * u for k in range(thb.dim)], indexing="ij")
        pts = np.stack([m.ravel() for m in mesh], axis=-1)
        vals, active = thb.tabulate_basis(cid, pts)
        nonzero = active[np.abs(vals).max(axis=0) > 1e-12]
        if nonzero.size == 0:
            continue
        levels = [_dof_level(thb, int(d)) for d in nonzero]
        worst = max(worst, max(levels) - min(levels) + 1)
    return worst


def _field_at(
    thb: THBSplineSpace, coeffs: npt.NDArray[np.float64], cid: int, pts: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Evaluate the field ``sum coeffs[i] Phi_i`` at ``pts`` on cell ``cid``."""
    values, active = thb.tabulate_basis(cid, pts)
    result: npt.NDArray[np.float64] = values @ coeffs[active]
    return result


def _locate(grid: HierarchicalGrid, point: list[float]) -> int:
    """Locate the active cell containing ``point`` (asserting it exists)."""
    cid = grid.locate(point)
    assert cid is not None
    return cid


class TestRefine:
    """THBSplineSpace.refine returns a new, refined, non-mutating space."""

    def test_returns_new_space_original_unchanged(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d())
        n_cells, n_active = coarse.grid.num_cells, coarse.num_total_basis
        fine = coarse.refine([0, 1], admissible_class=None)
        assert fine is not coarse
        assert coarse.grid.num_cells == n_cells
        assert coarse.num_total_basis == n_active
        assert fine.grid.num_cells > n_cells

    def test_refine_preserves_partition_of_unity(self) -> None:
        coarse = THBSplineSpace(_root_2d(), _grid_2d())
        fine = coarse.refine([0])
        mat, _ = _collocation(fine)
        np.testing.assert_allclose(mat.sum(axis=1), 1.0, atol=1e-10)

    def test_basic_refines_exactly_marked(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d())
        fine = coarse.refine([0], admissible_class=None)
        assert fine.grid.max_level == 1
        child = fine.grid.locate([0.06])  # inside marked cell 0 = [0, 0.25]
        assert child is not None
        assert fine.grid.cell_level(child) == 1

    def test_refine_empty_copies_grid(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d())
        fine = coarse.refine([], admissible_class=None)
        assert fine.grid.num_cells == coarse.grid.num_cells

    def test_ungraded_refines_exactly_marked_multi_cell(self) -> None:
        # Verifies the is_active_leaf guard: already-coarse cells not in [0, 1] are untouched.
        coarse = THBSplineSpace(_root_1d(), _grid_1d())
        fine = coarse.refine([0, 1], admissible_class=None)
        assert fine.grid.max_level == 1
        # Cells 2 and 3 (the unmarked level-0 leaves) must still be at level 0.
        for unmarked_pt in [0.6, 0.9]:
            cid = fine.grid.locate([unmarked_pt])
            assert cid is not None
            assert fine.grid.cell_level(cid) == 0
        # Cells 0 and 1 are split: point inside each should be at level 1.
        for marked_pt in [0.06, 0.18]:
            cid = fine.grid.locate([marked_pt])
            assert cid is not None
            assert fine.grid.cell_level(cid) == 1

    def test_refine_out_of_range_raises(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d())
        with pytest.raises(IndexError, match=r"\[0, 4\)"):
            coarse.refine([999])

    def test_refine_negative_id_raises(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d())
        with pytest.raises(IndexError, match=r"\[0, 4\)"):
            coarse.refine([-1])

    def test_admissible_class_below_two_raises(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d())
        with pytest.raises(ValueError, match="admissible_class"):
            coarse.refine([0], admissible_class=1)

    def test_admissible_class_zero_raises(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d())
        with pytest.raises(ValueError, match="admissible_class"):
            coarse.refine([0], admissible_class=0)

    def test_stale_grid_refine_raises(self) -> None:
        grid = _grid_1d()
        coarse = THBSplineSpace(_root_1d(), grid)
        grid.refine(0, [0], [2])
        with pytest.raises(RuntimeError, match="stale"):
            coarse.refine([0])


def _lower_left_level0_ids(thb: THBSplineSpace, threshold: float = 0.5) -> npt.NDArray[np.int64]:
    """Flat ids of the active (level-0) cells whose midpoint is below ``threshold`` per axis."""
    lo, hi = thb.grid.collect_cell_bounds()
    mid = 0.5 * (lo + hi)
    return np.flatnonzero(np.all(mid < threshold, axis=1)).astype(np.int64)


class TestCreateThbSpace:
    """create_thb_space builds a trivial single-level THB space from a B-spline space."""

    def test_unrefined_single_level(self) -> None:
        root = _root_2d()
        thb = create_thb_space(root)
        assert thb.num_levels == 1
        assert thb.num_total_basis == root.num_total_basis
        assert thb.grid.num_cells == root.num_total_intervals

    def test_equivalent_to_explicit_construction(self) -> None:
        root = _root_2d()
        explicit = THBSplineSpace(root, hierarchical_grid(tensor_product_grid(root), 2))
        factory = create_thb_space(root, 2)
        assert factory.num_total_basis == explicit.num_total_basis
        assert factory.grid.num_cells == explicit.grid.num_cells

    def test_factor_forwarded_to_grid(self) -> None:
        thb = create_thb_space(_root_2d(), [2, 3])
        assert thb.grid.factor == (2, 3)

    def test_truncate_flag_forwarded(self) -> None:
        assert create_thb_space(_root_2d(), truncate=False).truncate is False
        assert create_thb_space(_root_2d(), truncate=True).truncate is True

    def test_refinable_after_creation(self) -> None:
        thb = create_thb_space(_root_2d())
        fine = thb.refine_region(0, [0, 0], [2, 2])
        assert fine.num_levels == 2
        assert thb.num_levels == 1  # original untouched

    def test_regularity_forwarded(self) -> None:
        root = _root_1d()  # degree 2 -> regularity 0 is valid
        explicit = THBSplineSpace(
            root, hierarchical_grid(tensor_product_grid(root), 2), regularity=0
        )
        factory = create_thb_space(root, regularity=0)
        assert factory._regularity == explicit._regularity
        # Reduced regularity survives refinement (extra knots -> more basis at level 1).
        f1 = factory.refine_region(0, [0], [2], admissible_class=None)
        e1 = explicit.refine_region(0, [0], [2], admissible_class=None)
        assert f1._regularity == e1._regularity
        assert f1.num_basis_per_level == e1.num_basis_per_level

    def test_anisotropic_factor_refines(self) -> None:
        thb = create_thb_space(_root_2d(), [2, 3])
        fine = thb.refine_region(0, [0, 0], [2, 2], admissible_class=None)
        assert fine.grid.max_level == 1
        assert fine.grid.factor == (2, 3)


class TestRefineRegion:
    """THBSplineSpace.refine_region refines the active cells in a cell-index box."""

    def test_returns_new_space_original_unchanged(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d())
        n_cells, n_active = coarse.grid.num_cells, coarse.num_total_basis
        fine = coarse.refine_region(0, [0], [2], admissible_class=None)
        assert fine is not coarse
        assert coarse.grid.num_cells == n_cells
        assert coarse.num_total_basis == n_active
        assert fine.grid.num_cells > n_cells

    def test_refines_active_leaves_in_box(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d())
        fine = coarse.refine_region(0, [0], [2], admissible_class=None)  # cells 0,1 = [0, 0.5)
        assert fine.grid.max_level == 1
        for inside in [0.06, 0.18]:
            assert fine.grid.cell_level(_locate(fine.grid, [inside])) == 1
        for outside in [0.6, 0.9]:
            assert fine.grid.cell_level(_locate(fine.grid, [outside])) == 0

    def test_matches_grid_refine_ungraded(self) -> None:
        root = _root_2d()
        grid = hierarchical_grid(tensor_product_grid(root), 2)
        grid.refine(0, [0, 0], [2, 2])
        reference = THBSplineSpace(root, grid)
        region = create_thb_space(root).refine_region(0, [0, 0], [2, 2], admissible_class=None)
        assert region.num_basis_per_level == reference.num_basis_per_level
        assert region.grid.num_cells == reference.grid.num_cells

    def test_matches_refine_by_ids_graded(self) -> None:
        thb = create_thb_space(_root_2d())
        via_ids = thb.refine(_lower_left_level0_ids(thb))
        via_box = thb.refine_region(0, [0, 0], [2, 2])
        # Compare per-level structure, not just totals (same counts can hide topology).
        assert via_box.num_basis_per_level == via_ids.num_basis_per_level
        assert via_box.grid.num_cells == via_ids.grid.num_cells

    def test_graded_keeps_admissibility(self) -> None:
        # Graded (default) refine_region across two levels stays class-2 admissible.
        coarse = THBSplineSpace(_root_1d(), _grid_1d())
        f1 = coarse.refine_region(0, [0], [1])
        f2 = f1.refine_region(1, [0], [1])  # refine a level-1 child
        assert _nonzero_level_span(f2) <= 2

    def test_chains(self) -> None:
        thb = create_thb_space(_root_2d())
        fine = thb.refine_region(0, [0, 0], [4, 4]).refine_region(1, [0, 0], [4, 4])
        assert fine.num_levels == 3

    def test_empty_region_is_noop(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d())
        all_refined = coarse.refine_region(0, [0], [4], admissible_class=None)  # all level-0 cells
        # No level-0 leaves remain, so the box maps to an empty marked set: structural no-op.
        again = all_refined.refine_region(0, [0], [4], admissible_class=None)
        assert again.grid.num_cells == all_refined.grid.num_cells
        assert again.num_basis_per_level == all_refined.num_basis_per_level

    def test_level_out_of_range_raises(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d())
        with pytest.raises(ValueError, match="level must be in"):
            coarse.refine_region(9, [0], [1])

    def test_wrong_length_raises(self) -> None:
        coarse = THBSplineSpace(_root_2d(), _grid_2d())
        with pytest.raises(ValueError, match="length 2"):
            coarse.refine_region(0, [0], [1])

    def test_lo_not_less_than_hi_raises(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d())
        with pytest.raises(ValueError, match="strictly less"):
            coarse.refine_region(0, [2], [1])

    def test_out_of_bounds_high_raises(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d())
        with pytest.raises(ValueError, match="out of bounds"):
            coarse.refine_region(0, [0], [99])

    def test_out_of_bounds_negative_lo_raises(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d())
        with pytest.raises(ValueError, match="out of bounds"):
            coarse.refine_region(0, [-1], [2])

    def test_admissible_class_below_two_raises(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d())
        with pytest.raises(ValueError, match="admissible_class"):
            coarse.refine_region(0, [0], [1], admissible_class=1)

    def test_stale_grid_raises(self) -> None:
        grid = _grid_1d()
        coarse = THBSplineSpace(_root_1d(), grid)
        grid.refine(0, [0], [2])
        with pytest.raises(RuntimeError, match="stale"):
            coarse.refine_region(0, [0], [1])


class TestAdmissibility:
    """Graded refinement maintains class-m admissibility (Carraturo et al. 2019)."""

    def test_graded_class_two_2d(self) -> None:
        coarse = THBSplineSpace(_root_2d(), _grid_2d())
        fine = coarse.refine([0], admissible_class=2)
        deeper = fine.refine([_locate(fine.grid, [0.05, 0.05])], admissible_class=2)
        assert _nonzero_level_span(deeper) <= 2

    def test_graded_class_two_1d(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d())
        fine = coarse.refine([0], admissible_class=2)
        deeper = fine.refine([_locate(fine.grid, [0.03])], admissible_class=2)
        assert _nonzero_level_span(deeper) <= 2

    def test_ungraded_can_violate_class_two(self) -> None:
        coarse = THBSplineSpace(_root_2d(), _grid_2d())
        fine = coarse.refine([0], admissible_class=None)
        deeper = fine.refine([_locate(fine.grid, [0.05, 0.05])], admissible_class=None)
        assert _nonzero_level_span(deeper) > 2  # grading is what bounds the span

    def test_grading_refines_strictly_more(self) -> None:
        # A level-1 cell has a non-empty neighborhood at level 0 (k_nbr = 1 - 2 + 1 = 0).
        # Graded refinement must pre-refine those level-0 neighbors, producing strictly
        # more cells than the ungraded path which only splits the one marked cell.
        coarse = THBSplineSpace(_root_2d(), _grid_2d())
        f_g = coarse.refine([0], admissible_class=2)
        deep_g = f_g.refine([_locate(f_g.grid, [0.05, 0.05])], admissible_class=2)
        f_u = coarse.refine([0], admissible_class=None)
        deep_u = f_u.refine([_locate(f_u.grid, [0.05, 0.05])], admissible_class=None)
        assert deep_g.grid.num_cells > deep_u.grid.num_cells


class TestProlongation:
    """The coarse->fine prolongation transfers a coefficient field exactly."""

    def _check_reproduction(
        self, coarse: THBSplineSpace, fine: THBSplineSpace, rng: np.random.Generator
    ) -> None:
        p = coarse.prolongation_to(fine)
        assert p.shape == (fine.num_total_basis, coarse.num_total_basis)
        u_coarse = rng.random(coarse.num_total_basis)
        u_fine = p @ u_coarse
        u_axis = np.linspace(0.1, 0.9, 3)
        err = 0.0
        for cid in range(fine.grid.num_cells):
            lo, hi = fine.grid.cell_bounds(cid)
            mesh = np.meshgrid(
                *[lo[k] + (hi[k] - lo[k]) * u_axis for k in range(fine.dim)], indexing="ij"
            )
            pts = np.stack([m.ravel() for m in mesh], axis=-1)
            f_fine = _field_at(fine, u_fine, cid, pts)
            coarse_cid = coarse.grid.locate(pts[0])
            assert coarse_cid is not None
            f_coarse = _field_at(coarse, u_coarse, coarse_cid, pts)
            err = max(err, float(np.abs(f_fine - f_coarse).max()))
        assert err < 1e-9

    def test_reproduction_1d_thb(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d())
        fine = coarse.refine([0, 1])
        self._check_reproduction(coarse, fine, np.random.default_rng(0))

    def test_reproduction_2d_thb(self) -> None:
        coarse = THBSplineSpace(_root_2d(), _grid_2d())
        fine = coarse.refine([0])
        self._check_reproduction(coarse, fine, np.random.default_rng(1))

    def test_reproduction_hb(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d(), truncate=False)
        fine = coarse.refine([0, 1], admissible_class=None)
        self._check_reproduction(coarse, fine, np.random.default_rng(2))

    def test_reproduction_multi_level(self) -> None:
        coarse = THBSplineSpace(_root_2d(), _grid_2d())
        f1 = coarse.refine([0], admissible_class=None)
        f2 = f1.refine([_locate(f1.grid, [0.05, 0.05])], admissible_class=None)
        assert f2.num_levels >= 3
        self._check_reproduction(coarse, f2, np.random.default_rng(3))

    def test_identity_when_no_refinement(self) -> None:
        coarse = THBSplineSpace(_root_2d(), _grid_2d())
        p = coarse.prolongation_to(coarse)
        np.testing.assert_allclose(p, np.eye(coarse.num_total_basis), atol=1e-10)

    def test_columns_reproduce_basis_functions(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d())
        fine = coarse.refine([0])
        p = coarse.prolongation_to(fine)
        unit = np.zeros(coarse.num_total_basis)
        unit[2] = 1.0  # a single coarse basis function
        col = p[:, 2]
        for cid in range(fine.grid.num_cells):
            lo, hi = fine.grid.cell_bounds(cid)
            mid = (0.5 * (lo + hi)).reshape(1, 1)
            coarse_cid = coarse.grid.locate(mid[0])
            assert coarse_cid is not None
            assert (
                abs(_field_at(fine, col, cid, mid)[0] - _field_at(coarse, unit, coarse_cid, mid)[0])
                < 1e-9
            )

    def test_sparse_and_local_for_local_refinement(self) -> None:
        # A coarse function untouched by the refinement maps to a single fine
        # function (unit column); the operator is sparse, not dense. This guards
        # against a regression to a global dense least-squares construction.
        coarse = create_thb_space(create_uniform_space([2, 2], [16, 16]))
        fine = coarse.refine_region(0, [0, 0], [2, 2]).refine_region(1, [0, 0], [2, 2])
        p = coarse.prolongation_to(fine)
        density = np.count_nonzero(p) / p.size
        assert density < 0.05, f"prolongation not sparse: density {density:.3f}"
        # Far-from-refinement coarse functions are reproduced by exactly one fine dof.
        unit_columns = int(np.sum(np.count_nonzero(p, axis=0) == 1))
        assert unit_columns > coarse.num_total_basis // 2

    def test_non_refinement_raises(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d())
        other = THBSplineSpace(_root_2d(), _grid_2d())
        with pytest.raises(ValueError, match="refinement"):
            coarse.prolongation_to(other)

    def test_dim_mismatch_raises(self) -> None:
        # self.dim=2 > fine.dim=1: old code raised IndexError; now raises ValueError
        coarse_2d = THBSplineSpace(_root_2d(), _grid_2d())
        fine_1d = THBSplineSpace(_root_1d(), _grid_1d())
        with pytest.raises(ValueError):
            coarse_2d.prolongation_to(fine_1d)

    def test_truncate_mismatch_raises(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d())  # truncate=True (default)
        fine_hb = THBSplineSpace(_root_1d(), _grid_1d(), truncate=False)
        with pytest.raises(ValueError, match="truncate"):
            coarse.prolongation_to(fine_hb)

    def test_regularity_mismatch_raises(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d())  # regularity=None (default)
        fine_reg0 = THBSplineSpace(_root_1d(), _grid_1d(), regularity=0)
        with pytest.raises(ValueError, match="regularity"):
            coarse.prolongation_to(fine_reg0)

    def test_non_thb_argument_raises(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d())
        with pytest.raises(TypeError, match="THBSplineSpace"):
            coarse.prolongation_to(_grid_1d())  # type: ignore[arg-type]


# ──────────────────────────────────────────────────────────────────────────────
# Coarsening / restriction
# ──────────────────────────────────────────────────────────────────────────────


def _cells_at_level(thb: THBSplineSpace, level: int) -> list[int]:
    """Return the flat ids of active cells at ``level``."""
    return [c for c in range(thb.grid.num_cells) if thb.grid.cell_level(c) == level]


def _active_indices(thb: THBSplineSpace) -> tuple[tuple[int, ...], ...]:
    """Return the per-level active function indices as nested tuples."""
    return tuple(tuple(thb.active_function_indices(lv).tolist()) for lv in range(thb.num_levels))


class TestCoarsen:
    """THBSplineSpace.coarsen reverses refinement and grades for admissibility."""

    def test_coarsen_inverts_refine_1d(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d())
        fine = coarse.refine([0, 1])
        back = fine.coarsen(_cells_at_level(fine, 1))
        assert _active_indices(back) == _active_indices(coarse)

    def test_coarsen_inverts_refine_2d(self) -> None:
        coarse = THBSplineSpace(_root_2d(), _grid_2d())
        fine = coarse.refine([0], admissible_class=None)
        back = fine.coarsen(_cells_at_level(fine, 1), admissible_class=None)
        assert _active_indices(back) == _active_indices(coarse)

    def test_coarsen_partition_of_unity(self) -> None:
        coarse = THBSplineSpace(_root_2d(), _grid_2d())
        fine = coarse.refine([0])
        back = fine.coarsen(_cells_at_level(fine, 1))
        mat, _ = _collocation(back)
        np.testing.assert_allclose(mat.sum(axis=1), 1.0, atol=1e-10)

    def test_partial_children_not_coarsened(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d())
        fine = coarse.refine([0])  # cell 0 -> children at level 1
        children = _cells_at_level(fine, 1)
        back = fine.coarsen([children[0]])  # only one of the two children marked
        assert back.num_total_basis == fine.num_total_basis  # no-op

    def test_root_cells_not_coarsened(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d())
        back = coarse.coarsen([0])  # level-0 cell has no parent
        assert back.num_total_basis == coarse.num_total_basis

    def test_coarsen_out_of_range_raises(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d())
        with pytest.raises(IndexError):
            coarse.coarsen([999])

    def test_coarsen_admissible_class_below_two_raises(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d())
        with pytest.raises(ValueError, match="admissible_class"):
            coarse.coarsen([0], admissible_class=1)

    def test_admissibility_blocks_unsafe_coarsening(self) -> None:
        # Build a 3-level 1D mesh: refine [0, 0.5) to level 1, then [0, 0.25) to level 2.
        coarse = THBSplineSpace(_root_1d(), _grid_1d())
        f1 = coarse.refine([0, 1], admissible_class=None)  # level-1 over [0, 0.5)
        deep = f1.refine(
            _cells_at_level(f1, 1)[:2], admissible_class=None
        )  # level-2 over [0, 0.25)
        assert deep.num_levels == 3
        # Coarsening the level-1 leaves back to level 0 would make a level-0 function
        # span levels 0..2 (class 3); the class-2 guard must refuse what ungraded allows.
        marked = _cells_at_level(deep, 1)
        graded = deep.coarsen(marked, admissible_class=2)
        ungraded = deep.coarsen(marked, admissible_class=None)
        assert _nonzero_level_span(graded) <= 2
        assert graded.num_total_basis != ungraded.num_total_basis


class TestRestriction:
    """The fine->coarse restriction is the pseudo-inverse of the prolongation."""

    def _check(self, coarse: THBSplineSpace, fine: THBSplineSpace) -> None:
        prolong = coarse.prolongation_to(fine)
        restrict = fine.restriction_to(coarse)
        assert restrict.shape == (coarse.num_total_basis, fine.num_total_basis)
        np.testing.assert_allclose(restrict @ prolong, np.eye(coarse.num_total_basis), atol=1e-9)
        u_coarse = np.random.default_rng(0).random(coarse.num_total_basis)
        np.testing.assert_allclose(restrict @ (prolong @ u_coarse), u_coarse, atol=1e-9)

    def test_restriction_1d_thb(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d())
        self._check(coarse, coarse.refine([0, 1]))

    def test_restriction_2d_thb(self) -> None:
        coarse = THBSplineSpace(_root_2d(), _grid_2d())
        self._check(coarse, coarse.refine([0]))

    def test_restriction_hb(self) -> None:
        coarse = THBSplineSpace(_root_1d(), _grid_1d(), truncate=False)
        self._check(coarse, coarse.refine([0, 1], admissible_class=None))

    def test_restriction_non_refinement_raises(self) -> None:
        fine = THBSplineSpace(_root_1d(), _grid_1d())
        other = THBSplineSpace(_root_2d(), _grid_2d())
        with pytest.raises(ValueError, match="refinement"):
            fine.restriction_to(other)


# ──────────────────────────────────────────────────────────────────────────────
# _combine_tp_values kernel
# ──────────────────────────────────────────────────────────────────────────────


class TestCombineTPValues:
    """Direct unit tests for the _combine_tp_values Numba kernel."""

    def test_in_support_1d(self) -> None:
        # first_basis=1 → local indices [0,2] map to global [1,3].
        # multi[0]=2: local = 2-1 = 1 → vals[0, 0, 1] = 0.5.
        vals = np.array([[[0.3, 0.5, 0.2]]], dtype=np.float64)  # (1, 1, 3)
        first_basis = np.array([[1]], dtype=np.int64)
        multis = np.array([[2]], dtype=np.int64)
        degrees = np.array([2], dtype=np.int64)
        result = _combine_tp_values(vals, first_basis, multis, degrees)
        assert result.shape == (1, 1)
        np.testing.assert_allclose(result[0, 0], 0.5)

    def test_out_of_support_high_index_1d(self) -> None:
        # multi[0]=5, first_basis=1: local = 4 > degree 2 → product = 0 (early break).
        vals = np.array([[[0.3, 0.5, 0.2]]], dtype=np.float64)
        first_basis = np.array([[1]], dtype=np.int64)
        multis = np.array([[5]], dtype=np.int64)
        degrees = np.array([2], dtype=np.int64)
        result = _combine_tp_values(vals, first_basis, multis, degrees)
        np.testing.assert_allclose(result[0, 0], 0.0)

    def test_out_of_support_low_index_1d(self) -> None:
        # multi[0]=0, first_basis=2: local = -2 < 0 → product = 0 (early break).
        vals = np.array([[[0.3, 0.5, 0.2]]], dtype=np.float64)
        first_basis = np.array([[2]], dtype=np.int64)
        multis = np.array([[0]], dtype=np.int64)
        degrees = np.array([2], dtype=np.int64)
        result = _combine_tp_values(vals, first_basis, multis, degrees)
        np.testing.assert_allclose(result[0, 0], 0.0)

    def test_anisotropic_degrees_2d(self) -> None:
        # degrees=(2,3) → max_order=4; direction 0 has only 3 valid entries.
        # multi=(1,2), first_basis=(0,0): local=(1,2).
        # dir 0: degree=2, local=1 → vals[0,0,1]=0.4; dir 1: degree=3, local=2 → vals[1,0,2]=0.4.
        vals = np.zeros((2, 1, 4), dtype=np.float64)
        vals[0, 0, :3] = [0.1, 0.4, 0.5]
        vals[1, 0, :4] = [0.2, 0.3, 0.4, 0.1]
        first_basis = np.zeros((2, 1), dtype=np.int64)
        multis = np.array([[1, 2]], dtype=np.int64)
        degrees = np.array([2, 3], dtype=np.int64)
        result = _combine_tp_values(vals, first_basis, multis, degrees)
        np.testing.assert_allclose(result[0, 0], 0.4 * 0.4)

    def test_anisotropic_out_of_support_second_direction(self) -> None:
        # degrees=(2,3), multi=(1,5): dir 1 local=5 > 3 → early break → 0.
        vals = np.zeros((2, 1, 4), dtype=np.float64)
        vals[0, 0, :3] = [0.1, 0.4, 0.5]
        vals[1, 0, :4] = [0.2, 0.3, 0.4, 0.1]
        first_basis = np.zeros((2, 1), dtype=np.int64)
        multis = np.array([[1, 5]], dtype=np.int64)
        degrees = np.array([2, 3], dtype=np.int64)
        result = _combine_tp_values(vals, first_basis, multis, degrees)
        np.testing.assert_allclose(result[0, 0], 0.0)

    def test_anisotropic_in_support_dir0_out_dir1_differs_from_isotropic(self) -> None:
        # Confirms degrees[k] (not max(degrees)) gates the out-of-support check.
        # If the check used max(degrees)=3 instead of degrees[0]=2,
        # local=3 would pass for dir 0 and return a nonzero product.
        vals = np.zeros((2, 1, 4), dtype=np.float64)
        vals[0, 0, :4] = [0.1, 0.2, 0.3, 0.9]  # index 3 is nonzero
        vals[1, 0, :4] = [0.2, 0.3, 0.4, 0.1]
        first_basis = np.zeros((2, 1), dtype=np.int64)
        # local for dir 0 = 3; degree[0]=2 → out of support → 0.
        multis = np.array([[3, 1]], dtype=np.int64)
        degrees = np.array([2, 3], dtype=np.int64)
        result = _combine_tp_values(vals, first_basis, multis, degrees)
        np.testing.assert_allclose(result[0, 0], 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# _box_all_true SAT helper
# ──────────────────────────────────────────────────────────────────────────────


class TestBoxAllTrue:
    """Direct unit tests for the _box_all_true summed-area-table helper."""

    def test_all_true_2d(self) -> None:
        mask = np.ones((5, 5), dtype=np.bool_)
        lo = np.array([[1, 1]], dtype=np.int64)
        hi = np.array([[4, 4]], dtype=np.int64)
        assert bool(_box_all_true(mask, lo, hi)[0]) is True

    def test_false_inside_box_2d(self) -> None:
        mask = np.ones((5, 5), dtype=np.bool_)
        mask[2, 2] = False
        lo = np.array([[1, 1]], dtype=np.int64)
        hi = np.array([[4, 4]], dtype=np.int64)
        assert bool(_box_all_true(mask, lo, hi)[0]) is False

    def test_false_outside_box_ignored_2d(self) -> None:
        mask = np.ones((5, 5), dtype=np.bool_)
        mask[0, 0] = False  # outside [1,4)x[1,4)
        lo = np.array([[1, 1]], dtype=np.int64)
        hi = np.array([[4, 4]], dtype=np.int64)
        assert bool(_box_all_true(mask, lo, hi)[0]) is True

    def test_single_cell_1d(self) -> None:
        mask = np.array([True, False, True], dtype=np.bool_)
        lo = np.array([[0], [1], [2]], dtype=np.int64)
        hi = np.array([[1], [2], [3]], dtype=np.int64)
        result = _box_all_true(mask, lo, hi)
        assert [bool(r) for r in result] == [True, False, True]

    def test_batch_1d_mixed(self) -> None:
        # mask[3]=False; [0,3) is all-True; [2,4) contains the False.
        mask = np.array([True, True, True, False, True, True], dtype=np.bool_)
        lo = np.array([[0], [2]], dtype=np.int64)
        hi = np.array([[3], [4]], dtype=np.int64)
        result = _box_all_true(mask, lo, hi)
        assert bool(result[0]) is True
        assert bool(result[1]) is False

    def test_full_mask_3d(self) -> None:
        mask = np.ones((4, 4, 4), dtype=np.bool_)
        lo = np.array([[0, 0, 0]], dtype=np.int64)
        hi = np.array([[4, 4, 4]], dtype=np.int64)
        assert bool(_box_all_true(mask, lo, hi)[0]) is True

    def test_single_false_3d(self) -> None:
        mask = np.ones((4, 4, 4), dtype=np.bool_)
        mask[1, 2, 3] = False
        lo = np.array([[0, 0, 0]], dtype=np.int64)
        hi = np.array([[4, 4, 4]], dtype=np.int64)
        assert bool(_box_all_true(mask, lo, hi)[0]) is False


# ──────────────────────────────────────────────────────────────────────────────
# _contrib_cache
# ──────────────────────────────────────────────────────────────────────────────


class TestContribCache:
    """Tests for the lazy per-cell _cell_contributions cache."""

    def test_second_call_returns_same_object(self) -> None:
        grid = _grid_1d()
        grid.refine(0, [0], [2])
        thb = THBSplineSpace(_root_1d(), grid)
        cid = grid.locate([0.1])
        assert cid is not None
        first = thb._cell_contributions(cid)
        second = thb._cell_contributions(cid)
        assert second is first

    def test_cache_warm_stale_grid_still_raises(self) -> None:
        grid = _grid_1d()
        thb = THBSplineSpace(_root_1d(), grid)
        _ = thb._cell_contributions(0)  # warm cache
        assert 0 in thb._contrib_cache
        grid.refine(0, [0], [2])  # mutate grid after cache is warm
        with pytest.raises(RuntimeError, match="stale"):
            thb._cell_contributions(0)

    def test_different_cells_cached_independently(self) -> None:
        grid = _grid_1d()
        grid.refine(0, [0], [2])
        thb = THBSplineSpace(_root_1d(), grid)
        c0 = grid.locate([0.1])
        c1 = grid.locate([0.8])
        assert c0 is not None and c1 is not None
        r0 = thb._cell_contributions(c0)
        r1 = thb._cell_contributions(c1)
        assert thb._cell_contributions(c0) is r0
        assert thb._cell_contributions(c1) is r1


# ──────────────────────────────────────────────────────────────────────────────
# restrict (windowed THB sub-space)
# ──────────────────────────────────────────────────────────────────────────────


def _refined_thb_2d(
    n: int = 4, lo: tuple[int, int] = (1, 1), hi: tuple[int, int] = (3, 3), *, truncate: bool = True
) -> THBSplineSpace:
    """Degree-2 ``n x n`` THB space on [0, n]^2, refining ``[lo, hi)`` to level 1."""
    knots = [0.0] * 3 + [float(i) for i in range(1, n)] + [float(n)] * 3
    sp = BsplineSpace1D(knots, 2)
    grid = hierarchical_grid(uniform_grid([[0.0, float(n)], [0.0, float(n)]], n), 2)
    grid.refine(0, list(lo), list(hi))
    return THBSplineSpace(BsplineSpace([sp, sp]), grid, truncate=truncate)


def _cells_in_root_box(g: THBSplineSpace, lo: tuple[int, ...], hi: tuple[int, ...]) -> list[int]:
    """Active cell ids whose root cell lies in the box ``[lo, hi)``."""
    factor = g.grid.factor
    out = []
    for cid in range(g.grid.num_cells):
        lv = g.grid.cell_level(cid)
        m = g.grid.cell_multi_index(cid)
        rc = [m[k] // factor[k] ** lv for k in range(g.dim)]
        if all(lo[k] <= rc[k] < hi[k] for k in range(g.dim)):
            out.append(cid)
    return out


def _check_restrict_interior(
    g: THBSplineSpace, cell_ids: list[int]
) -> tuple[int, THBSplineSpaceRestriction]:
    """Assert windowed THB basis and extraction match the global ones on interior cells.

    An interior cell is one whose active functions all map to a global dof.
    """
    r = g.restrict(cell_ids)
    sub, l2g_dof = r.space, r.local_to_global_dof
    assert isinstance(r, THBSplineSpaceRestriction)
    assert isinstance(sub, THBSplineSpace)
    assert not l2g_dof.flags.writeable
    l2g_cell = g.grid.restrict(cell_ids).local_to_global_cell
    ext_g, ext_s = MultiLevelExtraction(g), MultiLevelExtraction(sub)
    n_interior = 0
    for lcid in range(sub.grid.num_cells):
        mapped = l2g_dof[sub.active_basis(lcid)]
        if np.any(mapped < 0):
            continue  # boundary cell: a touching function has no global counterpart
        gcid = int(l2g_cell[lcid])
        lo, hi = sub.grid.cell_bounds(lcid)
        pt = (0.5 * (lo + hi)).reshape(1, -1)
        vs, _ = sub.tabulate_basis(lcid, pt)
        vg, dg = g.tabulate_basis(gcid, pt)
        gval = {int(d): float(v) for d, v in zip(dg, vg[0], strict=True)}
        assert {int(x) for x in mapped} == {int(d) for d in dg}
        op_s, op_g = ext_s.operator(lcid), ext_g.operator(gcid)
        grow = {int(d): op_g[j] for j, d in enumerate(g.active_basis(gcid))}
        for i, gd in enumerate(mapped):
            np.testing.assert_allclose(vs[0, i], gval[int(gd)], atol=1e-11)  # basis value
            np.testing.assert_allclose(op_s[i], grow[int(gd)], atol=1e-10)  # extraction row
        n_interior += 1
    return n_interior, r


def test_restrict_full_grid_2d() -> None:
    g = _refined_thb_2d()
    n_interior, r = _check_restrict_interior(g, list(range(g.grid.num_cells)))
    assert n_interior == g.grid.num_cells  # full restrict: every cell is interior
    np.testing.assert_array_equal(r.local_to_global_dof, np.arange(g.num_total_basis))


def test_restrict_full_grid_1d() -> None:
    knots = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0]  # degree 2, 4 intervals
    grid = hierarchical_grid(uniform_grid([[0.0, 4.0]], 4), 2)
    grid.refine(0, [1], [3])
    g = THBSplineSpace(BsplineSpace([BsplineSpace1D(knots, 2)]), grid)
    n_interior, _ = _check_restrict_interior(g, list(range(g.grid.num_cells)))
    assert n_interior == g.grid.num_cells


def test_restrict_hb_full_grid() -> None:
    g = _refined_thb_2d(truncate=False)
    n_interior, r = _check_restrict_interior(g, list(range(g.grid.num_cells)))
    assert n_interior == g.grid.num_cells
    assert r.space.truncate is False


def test_restrict_subset_has_interior_2d() -> None:
    g = _refined_thb_2d(n=8, lo=(3, 3), hi=(5, 5))
    cell_ids = _cells_in_root_box(g, (1, 1), (7, 7))
    n_interior, _ = _check_restrict_interior(g, cell_ids)
    assert n_interior > 0


def test_restrict_dof_map_injective() -> None:
    g = _refined_thb_2d(n=6, lo=(2, 2), hi=(4, 4))
    l2g = g.restrict(_cells_in_root_box(g, (1, 1), (5, 5))).local_to_global_dof
    valid = l2g[l2g >= 0]
    assert np.all(valid < g.num_total_basis)
    assert len(set(valid.tolist())) == valid.size  # injective on the mapped dofs


def test_restrict_returns_namedtuple() -> None:
    g = _refined_thb_2d()
    r = g.restrict([0, 1, 2, 3])
    assert isinstance(r, THBSplineSpaceRestriction)
    assert isinstance(r.space, THBSplineSpace)
    assert not r.local_to_global_dof.flags.writeable


def test_restrict_errors() -> None:
    g = _refined_thb_2d()
    with pytest.raises(ValueError, match="non-empty"):
        g.restrict([])
    with pytest.raises(IndexError):
        g.restrict([g.grid.num_cells])
    with pytest.raises(TypeError):
        g.restrict([0.5, 1.5])


# ──────────────────────────────────────────────────────────────────────────────
# THBSpline.evaluate batched locate + argsort grouping (PR 4 of #197)
# ──────────────────────────────────────────────────────────────────────────────


class TestTHBSplineEvaluateGrouping:
    """The batched evaluate path matches per-point evaluation exactly."""

    @staticmethod
    def _spline(rank: int = 1) -> THBSpline:
        grid = _grid_2d()
        grid.refine(0, [0, 0], [2, 2])
        grid.refine(1, [0, 0], [2, 2])
        thb = THBSplineSpace(_root_2d(), grid)
        rng = np.random.default_rng(31)
        shape = (thb.num_total_basis,) if rank == 1 else (thb.num_total_basis, rank)
        return THBSpline(thb, rng.random(shape))

    def test_matches_per_point_evaluation(self) -> None:
        """Shuffled multi-cell points (with duplicates) match per-point results."""
        spline = self._spline()
        rng = np.random.default_rng(7)
        pts = rng.random((400, 2))
        pts[100:120] = pts[:20]  # duplicated points
        lo, hi = spline.space.grid.collect_cell_bounds()
        pts[200:210] = lo[:: max(1, lo.shape[0] // 10)][:10]  # cell corners
        got = spline.evaluate(pts)
        cp = np.asarray(spline.control_points)
        expected = []
        for p in pts:
            cid = spline.space.grid.locate(p)
            assert cid is not None
            values, dofs = spline.space.tabulate_basis(int(cid), p[None, :])
            expected.append(float(np.asarray(values)[0] @ cp[dofs]))
        # Grouped points share a BLAS matrix-vector product whose summation
        # order differs from a single-point dot at the last ulp.
        np.testing.assert_allclose(got, np.asarray(expected), rtol=1e-13, atol=0.0)

    def test_vector_field_grouping(self) -> None:
        """Vector-valued evaluation keeps point-to-result correspondence for all points."""
        spline = self._spline(rank=3)
        rng = np.random.default_rng(13)
        pts = rng.random((150, 2))
        got = spline.evaluate(pts)
        assert got.shape == (150, 3)
        cp = np.asarray(spline.control_points)
        expected = []
        for p in pts:
            cid = spline.space.grid.locate(p)
            assert cid is not None
            values, dofs = spline.space.tabulate_basis(int(cid), p[None, :])
            expected.append(np.asarray(values)[0] @ cp[dofs])
        np.testing.assert_allclose(got, np.asarray(expected), rtol=1e-13, atol=0.0)

    def test_outside_point_raises(self) -> None:
        """An outside point raises ValueError naming the first offending point."""
        spline = self._spline()
        # pts[0] is inside; pts[1] is the first outside point (x=1.5 > 1.0).
        pts = np.array([[0.5, 0.5], [1.5, 0.5], [2.5, 0.5]])
        with pytest.raises(ValueError, match=r"1\.5.*outside the grid domain"):
            spline.evaluate(pts)

    def test_empty_points(self) -> None:
        """An empty point batch returns an empty result of the right shape."""
        spline = self._spline()
        got = spline.evaluate(np.empty((0, 2)))
        assert got.shape == (0,)

    def test_derivatives_grouping(self) -> None:
        """evaluate_derivatives goes through the same grouping path."""
        spline = self._spline()
        rng = np.random.default_rng(19)
        pts = rng.random((60, 2))
        got = spline.evaluate_derivatives(pts, (1, 0))
        cp = np.asarray(spline.control_points)
        for i in (0, 30, 59):
            cid = spline.space.grid.locate(pts[i])
            assert cid is not None
            values, dofs = spline.space.tabulate_basis_derivatives(
                int(cid), pts[i][None, :], (1, 0)
            )
            np.testing.assert_allclose(
                got[i], np.asarray(values)[0] @ cp[dofs], rtol=1e-12, atol=1e-12
            )

    def test_all_points_same_cell(self) -> None:
        """All points in one cell form a single group (boundaries empty, one slice)."""
        spline = self._spline()
        # Level-0 cell [3,3] covers [0.75, 1.0]^2 and is never refined by _spline().
        rng = np.random.default_rng(41)
        pts = rng.random((30, 2)) * 0.2 + 0.78  # all within [0.78, 0.98]^2
        got = spline.evaluate(pts)
        cp = np.asarray(spline.control_points)
        expected = []
        for p in pts:
            cid = spline.space.grid.locate(p)
            assert cid is not None
            values, dofs = spline.space.tabulate_basis(int(cid), p[None, :])
            expected.append(float(np.asarray(values)[0] @ cp[dofs]))
        # Verify all points landed in the same cell (single group, no splits).
        cids = np.array([spline.space.grid.locate(p) for p in pts])
        assert len(np.unique(cids)) == 1
        np.testing.assert_allclose(got, np.asarray(expected), rtol=1e-13, atol=0.0)

    def test_single_point_batch(self) -> None:
        """A single-point batch (n_pts=1) forms one group and evaluates correctly."""
        spline = self._spline()
        pt = np.array([[0.6, 0.6]])
        got = spline.evaluate(pt)
        assert got.shape == (1,)
        cp = np.asarray(spline.control_points)
        cid = spline.space.grid.locate(pt[0])
        assert cid is not None
        values, dofs = spline.space.tabulate_basis(int(cid), pt)
        np.testing.assert_allclose(
            got[0], float(np.asarray(values)[0] @ cp[dofs]), rtol=1e-13, atol=0.0
        )

    @staticmethod
    def _spline_1d() -> THBSpline:
        """1D THBSpline fixture on [0, 1] with one level of refinement."""
        grid = _grid_1d()
        grid.refine(0, [0], [2])
        thb = THBSplineSpace(_root_1d(), grid)
        rng = np.random.default_rng(53)
        return THBSpline(thb, rng.random(thb.num_total_basis))

    def test_1d_evaluation(self) -> None:
        """1D grouping path matches per-point scalar evaluation."""
        spline = self._spline_1d()
        rng = np.random.default_rng(61)
        pts = rng.random((80, 1))
        got = spline.evaluate(pts)
        assert got.shape == (80,)
        cp = np.asarray(spline.control_points)
        expected = []
        for p in pts:
            cid = spline.space.grid.locate(p)
            assert cid is not None
            values, dofs = spline.space.tabulate_basis(int(cid), p[None, :])
            expected.append(float(np.asarray(values)[0] @ cp[dofs]))
        np.testing.assert_allclose(got, np.asarray(expected), rtol=1e-13, atol=0.0)
