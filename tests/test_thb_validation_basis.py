"""Validation suite — group 1: THB basis properties (Giannelli-Jüttler-Speleers 2012).

Reproduces the defining basis properties of the truncated hierarchical B-spline basis:
nonnegativity, partition of unity, linear independence + strong stability (the Gram
matrix is SPD with full rank and a condition number that stays bounded under refinement),
and preservation of coefficients / coarse reproduction.  Properties are reproduced, not
exact published magnitudes (#164, PR8).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline import (
    BsplineSpace,
    BsplineSpace1D,
    THBSplineSpace,
    create_uniform_space,
    quasi_interpolate_thb_spline,
)
from pantr.grid import HierarchicalGrid, hierarchical_grid, uniform_grid
from tests._thb_assembly import gram_matrix, l2_project_thb

_KNOTS_DEG2_4 = np.array([0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0])


def _root_1d() -> BsplineSpace:
    return BsplineSpace([BsplineSpace1D(_KNOTS_DEG2_4, 2)])


def _root_2d() -> BsplineSpace:
    sp = BsplineSpace1D(_KNOTS_DEG2_4, 2)
    return BsplineSpace([sp, sp])


def _grid_1d() -> HierarchicalGrid:
    return hierarchical_grid(uniform_grid([[0.0, 1.0]], 4), 2)


def _grid_2d() -> HierarchicalGrid:
    return hierarchical_grid(uniform_grid([[0.0, 1.0], [0.0, 1.0]], 4), 2)


def _cell_samples(thb: THBSplineSpace, n: int = 4) -> list[tuple[int, npt.NDArray[np.float64]]]:
    """Interior sample points per active cell as ``(cid, pts)`` pairs."""
    out: list[tuple[int, npt.NDArray[np.float64]]] = []
    u = np.linspace(0.0, 1.0, n + 2)[1:-1]
    for cid in range(thb.grid.num_cells):
        lo, hi = thb.grid.cell_bounds(cid)
        axes = [lo[k] + (hi[k] - lo[k]) * u for k in range(thb.dim)]
        mesh = np.meshgrid(*axes, indexing="ij")
        out.append((cid, np.stack([m.ravel() for m in mesh], axis=-1)))
    return out


class TestNonnegativityAndPartitionOfUnity:
    """THB functions are nonnegative and form a partition of unity."""

    @pytest.mark.parametrize("dim", [1, 2])
    def test_nonnegative(self, dim: int) -> None:
        root, grid = (_root_1d(), _grid_1d()) if dim == 1 else (_root_2d(), _grid_2d())
        grid.refine(0, [0] * dim, [2] * dim)
        thb = THBSplineSpace(root, grid)
        for cid, pts in _cell_samples(thb):
            assert thb.tabulate_basis(cid, pts).min() >= -1e-13

    @pytest.mark.parametrize("dim", [1, 2])
    def test_partition_of_unity(self, dim: int) -> None:
        root, grid = (_root_1d(), _grid_1d()) if dim == 1 else (_root_2d(), _grid_2d())
        grid.refine(0, [0] * dim, [2] * dim)
        thb = THBSplineSpace(root, grid)
        for cid, pts in _cell_samples(thb):
            np.testing.assert_allclose(thb.tabulate_basis(cid, pts).sum(axis=-1), 1.0, atol=1e-12)

    def test_hb_is_not_partition_of_unity(self) -> None:
        # Truncation is what restores PoU; the non-truncated basis over-counts.
        grid = _grid_1d()
        grid.refine(0, [0], [2])
        hb = THBSplineSpace(_root_1d(), grid, truncate=False)
        worst = 0.0
        for cid, pts in _cell_samples(hb):
            worst = max(worst, float(np.abs(hb.tabulate_basis(cid, pts).sum(axis=-1) - 1.0).max()))
        assert worst > 1e-3


class TestLinearIndependenceAndStability:
    """The THB basis is linearly independent and strongly stable."""

    @pytest.mark.parametrize("dim", [1, 2])
    def test_gram_is_spd_full_rank(self, dim: int) -> None:
        root, grid = (_root_1d(), _grid_1d()) if dim == 1 else (_root_2d(), _grid_2d())
        grid.refine(0, [0] * dim, [2] * dim)
        thb = THBSplineSpace(root, grid)
        mass = gram_matrix(thb)
        np.testing.assert_allclose(mass, mass.T, atol=1e-14)
        eig = np.linalg.eigvalsh(mass)
        # Strictly positive spectrum ⇒ linearly independent basis (full rank).
        assert eig.min() > 0.0
        assert np.linalg.matrix_rank(mass) == thb.num_active_functions

    def test_condition_number_bounded_under_refinement(self) -> None:
        # Strong stability: the mesh-normalized Gram condition number does not blow up
        # as the hierarchy deepens.  Normalize by the diagonal (mass scales like the
        # cell size) so the bound is level-independent.  Successively deeper central
        # refinement bands.
        grids = [_grid_1d(), _grid_1d(), _grid_1d()]
        grids[1].refine(0, [1], [3])
        grids[2].refine(0, [1], [3])
        grids[2].refine(1, [3], [5])
        conds: list[float] = []
        for grid in grids:
            mass = gram_matrix(THBSplineSpace(_root_1d(), grid))
            d = np.sqrt(np.diag(mass))
            conds.append(float(np.linalg.cond(mass / np.outer(d, d))))
        assert max(conds) < 1e3


class TestPreservationAndReproduction:
    """Coarse reproduction: V_0 functions/polynomials are reproduced exactly."""

    @pytest.mark.parametrize("degree", [2, 3])
    def test_l2_reproduces_polynomials_1d(self, degree: int) -> None:
        root = create_uniform_space([degree], [5])
        grid = hierarchical_grid(uniform_grid([[0.0, 1.0]], 5), 2)
        grid.refine(0, [1], [3])
        thb = THBSplineSpace(root, grid)
        pcoeffs = np.random.default_rng(degree).standard_normal(degree + 1)

        def poly(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.asarray(sum(pcoeffs[k] * p[:, 0] ** k for k in range(degree + 1)))

        proj = l2_project_thb(thb, poly)
        xs = np.linspace(0.02, 0.98, 60).reshape(-1, 1)
        np.testing.assert_allclose(proj.evaluate(xs).ravel(), poly(xs), atol=1e-9)

    def test_qi_reproduces_polynomial_2d(self) -> None:
        grid = _grid_2d()
        grid.refine(0, [0, 0], [2, 2])
        thb = THBSplineSpace(_root_2d(), grid)

        def poly(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.asarray(p[:, 0] ** 2 + p[:, 0] * p[:, 1] + 1.0)

        qi = quasi_interpolate_thb_spline(poly, thb)
        u = np.linspace(0.05, 0.95, 7)
        pts = np.stack([m.ravel() for m in np.meshgrid(u, u, indexing="ij")], axis=-1)
        np.testing.assert_allclose(qi.evaluate(pts).ravel(), poly(pts), atol=1e-10)
