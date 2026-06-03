"""Tests for pantr.grid.cell_quadrature (reference rule -> cell box bridge)."""

from __future__ import annotations

import numpy as np
import numpy.testing as nptest
import numpy.typing as npt
import pytest

from pantr.grid import (
    Grid,
    TensorProductGrid,
    cell_quadrature,
    hierarchical_grid,
    uniform_grid,
)
from pantr.quad import QuadratureRule, gauss_legendre_quadrature


def _grid_2d() -> TensorProductGrid:
    """Return a 2x3 uniform grid on ``[0, 2] x [0, 2]``."""
    return uniform_grid([[0.0, 2.0], [0.0, 2.0]], [2, 3])


def _cell_volumes(grid: Grid) -> npt.NDArray[np.float64]:
    """Return each cell's volume via cell_bounds, in id order."""
    vols = np.empty(grid.num_cells, dtype=np.float64)
    for cid in range(grid.num_cells):
        lo, hi = grid.cell_bounds(cid)
        vols[cid] = float(np.prod(hi - lo))
    return vols


class TestShapesAndDtype:
    """Output shape and dtype of cell_quadrature."""

    def test_all_cells_shapes(self) -> None:
        grid = _grid_2d()
        rule = gauss_legendre_quadrature(2, 3)
        pts, w = cell_quadrature(grid, rule)
        assert pts.shape == (grid.num_cells, rule.num_points, grid.ndim)
        assert w.shape == (grid.num_cells, rule.num_points)

    def test_outputs_are_float64_and_contiguous(self) -> None:
        grid = _grid_2d()
        pts, w = cell_quadrature(grid, gauss_legendre_quadrature(2, 2))
        assert pts.dtype == np.float64
        assert w.dtype == np.float64
        assert pts.flags.c_contiguous
        assert w.flags.c_contiguous

    def test_one_dimensional_grid(self) -> None:
        grid = uniform_grid([[0.0, 1.0]], 4)
        pts, w = cell_quadrature(grid, gauss_legendre_quadrature(1, 3))
        assert pts.shape == (4, 3, 1)
        assert w.shape == (4, 3)


class TestCorrectness:
    """Exact integration of polynomials and basic invariants."""

    def test_integrate_polynomial_1d(self) -> None:
        # int_0^3 x^5 dx = 3^6 / 6 = 121.5; GL with 3 pts is exact to degree 5.
        grid = uniform_grid([[0.0, 3.0]], 3)
        pts, w = cell_quadrature(grid, gauss_legendre_quadrature(1, 3))
        integral = float((w * pts[..., 0] ** 5).sum())
        nptest.assert_allclose(integral, 3.0**6 / 6.0, rtol=1e-12)

    def test_integrate_polynomial_2d(self) -> None:
        # int over [0,2]^2 of x^2 y^3 = (8/3) * (16/4) = 32/3.
        grid = _grid_2d()
        pts, w = cell_quadrature(grid, gauss_legendre_quadrature(2, 3))
        f = pts[..., 0] ** 2 * pts[..., 1] ** 3
        nptest.assert_allclose(float((w * f).sum()), 32.0 / 3.0, rtol=1e-12)

    def test_integrate_polynomial_3d(self) -> None:
        # int over [0,1]^3 of x y z = 1/8.
        grid = uniform_grid([[0.0, 1.0]] * 3, [2, 1, 3])
        pts, w = cell_quadrature(grid, gauss_legendre_quadrature(3, 2))
        f = pts[..., 0] * pts[..., 1] * pts[..., 2]
        nptest.assert_allclose(float((w * f).sum()), 1.0 / 8.0, rtol=1e-12)

    def test_total_weight_equals_domain_volume(self) -> None:
        grid = _grid_2d()
        _, w = cell_quadrature(grid, gauss_legendre_quadrature(2, 2))
        nptest.assert_allclose(float(w.sum()), 4.0, rtol=1e-12)

    def test_per_cell_weight_sum_equals_cell_volume(self) -> None:
        # Non-uniform grid so cells have different volumes.
        grid = TensorProductGrid([[0.0, 1.0, 3.0], [0.0, 0.5, 2.0, 2.5]])
        _, w = cell_quadrature(grid, gauss_legendre_quadrature(2, 2))
        nptest.assert_allclose(w.sum(axis=1), _cell_volumes(grid), rtol=1e-12)

    def test_points_lie_inside_their_cell(self) -> None:
        grid = TensorProductGrid([[0.0, 1.0, 3.0], [0.0, 0.5, 2.5]])
        rule = gauss_legendre_quadrature(2, 3)
        pts, _ = cell_quadrature(grid, rule)
        for cid in range(grid.num_cells):
            lo, hi = grid.cell_bounds(cid)
            assert np.all(pts[cid] >= lo - 1e-15)
            assert np.all(pts[cid] <= hi + 1e-15)

    def test_matches_reference_map(self) -> None:
        # Independent cross-check against Grid.reference_map (matrix form).
        grid = TensorProductGrid([[0.0, 1.0, 3.0], [0.0, 0.5, 2.5]])
        rule = gauss_legendre_quadrature(2, 3)
        pts, w = cell_quadrature(grid, rule)
        for cid in range(grid.num_cells):
            expected_pts = grid.reference_map(cid)(rule.points)
            nptest.assert_allclose(pts[cid], expected_pts, rtol=1e-12)
            lo, hi = grid.cell_bounds(cid)
            nptest.assert_allclose(w[cid], rule.weights * float(np.prod(hi - lo)), rtol=1e-12)


class TestCellSelection:
    """The optional ``cells`` selector."""

    def test_subset_matches_full(self) -> None:
        grid = _grid_2d()
        rule = gauss_legendre_quadrature(2, 2)
        pts_all, w_all = cell_quadrature(grid, rule)
        sel = [0, 5, 2]
        pts, w = cell_quadrature(grid, rule, cells=sel)
        assert pts.shape == (3, rule.num_points, 2)
        for k, cid in enumerate(sel):
            nptest.assert_array_equal(pts[k], pts_all[cid])
            nptest.assert_array_equal(w[k], w_all[cid])

    def test_scalar_cell_id(self) -> None:
        grid = _grid_2d()
        rule = gauss_legendre_quadrature(2, 2)
        pts, w = cell_quadrature(grid, rule, cells=3)
        assert pts.shape == (1, rule.num_points, 2)
        pts_all, _ = cell_quadrature(grid, rule)
        nptest.assert_array_equal(pts[0], pts_all[3])

    def test_duplicates_preserved(self) -> None:
        grid = _grid_2d()
        rule = gauss_legendre_quadrature(2, 2)
        pts, _ = cell_quadrature(grid, rule, cells=[1, 1, 1])
        assert pts.shape == (3, rule.num_points, 2)
        nptest.assert_array_equal(pts[0], pts[1])
        nptest.assert_array_equal(pts[1], pts[2])

    def test_empty_selection(self) -> None:
        grid = _grid_2d()
        rule = gauss_legendre_quadrature(2, 2)
        pts, w = cell_quadrature(grid, rule, cells=[])
        assert pts.shape == (0, rule.num_points, 2)
        assert w.shape == (0, rule.num_points)

    def test_numpy_int_array_selector(self) -> None:
        grid = _grid_2d()
        rule = gauss_legendre_quadrature(2, 2)
        pts, _ = cell_quadrature(grid, rule, cells=np.array([4, 0], dtype=np.int32))
        assert pts.shape == (2, rule.num_points, 2)


class TestValidation:
    """Error handling."""

    def test_ndim_mismatch(self) -> None:
        grid = _grid_2d()
        with pytest.raises(ValueError, match="must match grid.ndim"):
            cell_quadrature(grid, gauss_legendre_quadrature(3, 2))

    def test_cells_out_of_range(self) -> None:
        grid = _grid_2d()
        with pytest.raises(ValueError, match=r"must lie in \[0, 6\)"):
            cell_quadrature(grid, gauss_legendre_quadrature(2, 2), cells=[0, 6])

    def test_cells_negative(self) -> None:
        grid = _grid_2d()
        with pytest.raises(ValueError, match=r"must lie in \[0, 6\)"):
            cell_quadrature(grid, gauss_legendre_quadrature(2, 2), cells=[-1])

    def test_cells_non_integer_dtype(self) -> None:
        grid = _grid_2d()
        with pytest.raises(ValueError, match="integer array"):
            cell_quadrature(grid, gauss_legendre_quadrature(2, 2), cells=[0.0, 1.0])

    def test_cells_two_dimensional(self) -> None:
        grid = _grid_2d()
        with pytest.raises(ValueError, match="must be 1D"):
            cell_quadrature(grid, gauss_legendre_quadrature(2, 2), cells=[[0, 1], [2, 3]])


class TestGenericGrid:
    """cell_quadrature works on any Grid, e.g. a refined HierarchicalGrid."""

    def test_hierarchical_grid_constant_and_volume(self) -> None:
        root = uniform_grid([[0.0, 4.0], [0.0, 4.0]], [4, 4])
        hg = hierarchical_grid(root, 2)
        hg.refine(0, [0, 0], [2, 2])  # refine the lower-left quadrant
        rule = gauss_legendre_quadrature(2, 2)
        pts, w = cell_quadrature(hg, rule)
        assert pts.shape == (hg.num_cells, rule.num_points, 2)
        # Active cells tile the domain exactly -> total weight is the area.
        nptest.assert_allclose(float(w.sum()), 16.0, rtol=1e-12)
        # Per-cell weight sum equals each cell's volume.
        nptest.assert_allclose(w.sum(axis=1), _cell_volumes(hg), rtol=1e-12)


class TestRuleConstruction:
    """A directly-built QuadratureRule maps correctly."""

    def test_custom_rule(self) -> None:
        # Midpoint rule on the unit square: one point at the center, weight 1.
        rule = QuadratureRule(points=[[0.5, 0.5]], weights=[1.0])
        grid = uniform_grid([[0.0, 2.0], [0.0, 2.0]], 1)
        pts, w = cell_quadrature(grid, rule)
        nptest.assert_allclose(pts[0, 0], [1.0, 1.0])
        nptest.assert_allclose(w[0, 0], 4.0)
