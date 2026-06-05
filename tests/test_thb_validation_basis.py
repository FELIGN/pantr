"""Validation suite — group 1: THB basis properties.

Reproduces the defining basis properties of the truncated hierarchical B-spline basis:

- nonnegativity, partition of unity, and linear independence (Giannelli-Jüttler-Speleers
  2012, Thms 6 and 10 — the truncated basis is a convex partition of unity and linearly
  independent);
- strong stability — the mesh-normalized Gram condition number stays bounded under
  refinement.  GJS 2012 leaves a stability analysis to future work; the strong-stability
  result is Giannelli-Jüttler-Speleers 2014, *Strongly stable bases for adaptively
  refined multilevel spline spaces* (Adv. Comput. Math. 40);
- coarse reproduction — a polynomial in the coarse space is reproduced exactly.

Properties (orders/qualitative behaviour) are reproduced, not exact published magnitudes
(#164, PR8).
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
            assert thb.tabulate_basis(cid, pts)[0].min() >= -1e-13

    @pytest.mark.parametrize("dim", [1, 2])
    def test_partition_of_unity(self, dim: int) -> None:
        root, grid = (_root_1d(), _grid_1d()) if dim == 1 else (_root_2d(), _grid_2d())
        grid.refine(0, [0] * dim, [2] * dim)
        thb = THBSplineSpace(root, grid)
        for cid, pts in _cell_samples(thb):
            vals, _ = thb.tabulate_basis(cid, pts)
            np.testing.assert_allclose(vals.sum(axis=-1), 1.0, atol=1e-12)

    def test_hb_is_not_partition_of_unity(self) -> None:
        # Truncation is what restores PoU; the non-truncated basis over-counts.
        grid = _grid_1d()
        grid.refine(0, [0], [2])
        hb = THBSplineSpace(_root_1d(), grid, truncate=False)
        worst = 0.0
        for cid, pts in _cell_samples(hb):
            vals, _ = hb.tabulate_basis(cid, pts)
            worst = max(worst, float(np.abs(vals.sum(axis=-1) - 1.0).max()))
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
        assert np.linalg.matrix_rank(mass) == thb.num_total_basis

    def test_condition_number_bounded_under_refinement(self) -> None:
        # Strong stability (GJS 2014, not 2012): the mesh-normalized Gram condition
        # number stays bounded as the hierarchy deepens.  Normalize by the diagonal
        # (a diagonal mass entry scales like the cell volume, h**dim) so the bound is
        # level-independent.  Successively deeper nested central refinement bands.
        refine_chains: list[list[tuple[int, list[int], list[int]]]] = [
            [],
            [(0, [1], [3])],
            [(0, [1], [3]), (1, [3], [5])],
            [(0, [1], [3]), (1, [3], [5]), (2, [7], [9])],
            [(0, [1], [3]), (1, [3], [5]), (2, [7], [9]), (3, [15], [17])],
        ]
        conds: list[float] = []
        for chain in refine_chains:
            grid = _grid_1d()
            for level, lo, hi in chain:
                grid.refine(level, lo, hi)
            mass = gram_matrix(THBSplineSpace(_root_1d(), grid))
            d = np.sqrt(np.diag(mass))
            conds.append(float(np.linalg.cond(mass / np.outer(d, d))))
        # Bounded (not merely finite): a regression that doubled the condition number
        # per level would breach this within a few levels.
        assert max(conds) < 50.0
        # Plateauing: each added level contributes less than the previous one, i.e. the
        # condition number converges rather than growing without bound.
        increments = [conds[i + 1] - conds[i] for i in range(len(conds) - 1)]
        assert all(increments[i + 1] < increments[i] for i in range(len(increments) - 1))


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
        xs = np.linspace(0.02, 0.98, 60).reshape(-1, 1).astype(np.float64)
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


class TestDerivativeReproduction:
    """The truncated basis differentiates exactly.

    The derivative of a reproduced polynomial matches the analytic derivative (exercises
    ``tabulate_basis_derivatives`` on the truncated path).
    """

    @pytest.mark.parametrize("degree", [2, 3])
    def test_first_derivative_of_polynomial_1d(self, degree: int) -> None:
        root = create_uniform_space([degree], [6])
        grid = hierarchical_grid(uniform_grid([[0.0, 1.0]], 6), 2)
        grid.refine(0, [1], [4])  # partial refinement => truncated functions present
        thb = THBSplineSpace(root, grid)
        pcoeffs = np.random.default_rng(degree).standard_normal(degree + 1)

        def poly(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.asarray(sum(pcoeffs[k] * p[:, 0] ** k for k in range(degree + 1)))

        def dpoly(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.asarray(sum(k * pcoeffs[k] * x ** (k - 1) for k in range(1, degree + 1)))

        coeffs = l2_project_thb(thb, poly).control_points
        xi = np.linspace(0.1, 0.9, 5)
        for cid in range(thb.grid.num_cells):
            lo, hi = thb.grid.cell_bounds(cid)
            x = (lo[0] + (hi[0] - lo[0]) * xi).reshape(-1, 1)
            vals, dofs = thb.tabulate_basis_derivatives(cid, x, 1)
            got = vals @ coeffs[dofs]
            np.testing.assert_allclose(got, dpoly(x[:, 0]), atol=1e-11)
