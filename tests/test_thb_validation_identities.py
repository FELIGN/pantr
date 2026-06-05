"""Validation suite — group 4: cross-operator algebraic identities.

End-to-end identities that tie several THB operators together (the per-operator
identities themselves are covered in the feature-PR tests: ``test_thb_spline_space.py``
for refinement/prolongation/coarsening, ``test_multilevel_extraction.py`` for Bézier
extraction, ``test_quasi_interpolation.py`` for quasi-interpolation):

- refinement **nesting** ``V_H ⊂ V_h``: a coarse THB field prolonged to a finer THB
  space evaluates identically (via the public :class:`~pantr.bspline.THBSpline`);
- **coarsening = exact left-inverse** of refinement: ``restriction ∘ prolongation == I``;
- **Bézier reconstruction integrated with L2**: the per-cell multi-level extraction
  reconstructs an L2-projected field from its Bernstein control values.

(#164, PR8.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pantr.basis import tabulate_bernstein
from pantr.bspline import (
    BsplineSpace,
    BsplineSpace1D,
    MultiLevelExtraction,
    THBSpline,
    THBSplineSpace,
)
from pantr.grid import HierarchicalGrid, hierarchical_grid, uniform_grid

if TYPE_CHECKING:
    from collections.abc import Callable

from tests._thb_assembly import l2_project_thb

_KNOTS_DEG2_4 = np.array([0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0])


def _root_1d() -> BsplineSpace:
    return BsplineSpace([BsplineSpace1D(_KNOTS_DEG2_4, 2)])


def _root_2d() -> BsplineSpace:
    sp = BsplineSpace1D(_KNOTS_DEG2_4, 2)
    return BsplineSpace([sp, sp])


class TestRefinementNesting:
    """V_H ⊂ V_h: a coarse THB field is reproduced exactly on a finer THB space."""

    def test_prolongation_reproduces_field(self) -> None:
        coarse_grid = hierarchical_grid(uniform_grid([[0.0, 1.0]], 4), 2)
        coarse_grid.refine(0, [1], [3])
        coarse = THBSplineSpace(_root_1d(), coarse_grid)

        fine_grid = hierarchical_grid(uniform_grid([[0.0, 1.0]], 4), 2)
        fine_grid.refine(0, [1], [3])
        fine_grid.refine(1, [2], [6])
        fine = THBSplineSpace(_root_1d(), fine_grid)

        rng = np.random.default_rng(0)
        c_coarse = rng.standard_normal(coarse.num_active_functions)
        prolong = coarse.prolongation_to(fine)
        c_fine = prolong @ c_coarse

        xs = np.linspace(0.02, 0.98, 80).reshape(-1, 1)
        got = THBSpline(fine, c_fine).evaluate(xs)
        expected = THBSpline(coarse, c_coarse).evaluate(xs)
        np.testing.assert_allclose(got, expected, atol=1e-10)


class TestCoarseningExactInverse:
    """restriction ∘ prolongation == I over a refinement chain."""

    def test_restriction_left_inverse(self) -> None:
        coarse_grid = hierarchical_grid(uniform_grid([[0.0, 1.0], [0.0, 1.0]], 4), 2)
        coarse_grid.refine(0, [0, 0], [2, 2])
        coarse = THBSplineSpace(_root_2d(), coarse_grid)

        fine_grid = hierarchical_grid(uniform_grid([[0.0, 1.0], [0.0, 1.0]], 4), 2)
        fine_grid.refine(0, [0, 0], [2, 2])
        fine_grid.refine(1, [0, 0], [4, 4])
        fine = THBSplineSpace(_root_2d(), fine_grid)

        prolong = coarse.prolongation_to(fine)
        restrict = fine.restriction_to(coarse)
        identity = restrict @ prolong
        np.testing.assert_allclose(identity, np.eye(coarse.num_active_functions), atol=1e-9)


class TestBezierReconstruction:
    """Per-cell multi-level Bézier extraction reconstructs an L2-projected field."""

    def _check(
        self,
        root: BsplineSpace,
        grid_factory: Callable[[], HierarchicalGrid],
        refines: list[tuple[int, list[int], list[int]]],
    ) -> None:
        grid = grid_factory()
        for level, lo_box, hi_box in refines:
            grid.refine(level, lo_box, hi_box)
        thb = THBSplineSpace(root, grid)

        def func(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.asarray(np.cos(2.0 * p[:, 0]) + (p[:, -1] if thb.dim > 1 else 0.0))

        coeffs = l2_project_thb(thb, func).coeffs
        spline = THBSpline(thb, coeffs)
        ext = MultiLevelExtraction(thb)
        degrees = list(thb.degrees)

        u = np.linspace(0.1, 0.9, 4)
        xi = np.stack([m.ravel() for m in np.meshgrid(*[u] * thb.dim, indexing="ij")], axis=-1)
        bern = tabulate_bernstein(degrees, xi)  # (n_pts, n_bernstein)
        for cid in range(thb.grid.num_cells):
            dofs = thb.active_basis(cid)
            # Field on the cell as a Bézier: control values g = Cᵉᵀ c[active].
            control = ext.operator(cid).T @ coeffs[dofs]
            from_bezier = bern @ control
            lo, hi = thb.grid.cell_bounds(cid)
            x = lo + (hi - lo) * xi
            np.testing.assert_allclose(from_bezier, spline.evaluate(x), atol=1e-10)

    def test_reconstruction_1d(self) -> None:
        self._check(
            _root_1d(),
            lambda: hierarchical_grid(uniform_grid([[0.0, 1.0]], 4), 2),
            [(0, [0], [2]), (1, [0], [2])],
        )

    def test_reconstruction_2d(self) -> None:
        self._check(
            _root_2d(),
            lambda: hierarchical_grid(uniform_grid([[0.0, 1.0], [0.0, 1.0]], 4), 2),
            [(0, [0, 0], [2, 2])],
        )
