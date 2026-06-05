"""Validation suite — group 4: cross-operator algebraic identities.

End-to-end identities that tie several THB operators together (the per-operator
identities themselves are covered in the feature-PR tests: ``test_thb_spline_space.py``
for refinement/prolongation/coarsening, ``test_multilevel_extraction.py`` for Bézier
extraction, ``test_quasi_interpolation.py`` for quasi-interpolation):

- refinement **nesting** ``V_H ⊂ V_h``: a coarse THB field prolonged to a finer THB
  space evaluates identically (via the public :class:`~pantr.bspline.THBSpline`), and
  prolongation is **transitive** across a multi-level chain;
- **coarsening**: ``restriction ∘ prolongation == I`` plus the non-automatic projection
  behaviour (restriction recovers a ``P``-independent coarse field; ``P ∘ R`` is a lossy
  idempotent projector);
- **Bézier reconstruction integrated with L2**: the per-cell multi-level extraction
  reconstructs an L2-projected field from its Bernstein control values (degrees 2 and 3).

(#164, PR8.)
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pantr.basis import tabulate_bernstein
from pantr.bspline import (
    BsplineSpace,
    BsplineSpace1D,
    MultiLevelExtraction,
    THBSpline,
    THBSplineSpace,
    create_uniform_space,
)
from pantr.grid import hierarchical_grid, uniform_grid
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

    def test_prolongation_transitivity(self) -> None:
        # P(coarse -> fine) == P(mid -> fine) @ P(coarse -> mid) over a 3-level chain.
        def space(refines: list[tuple[int, list[int], list[int]]]) -> THBSplineSpace:
            grid = hierarchical_grid(uniform_grid([[0.0, 1.0]], 4), 2)
            for level, lo, hi in refines:
                grid.refine(level, lo, hi)
            return THBSplineSpace(_root_1d(), grid)

        coarse = space([(0, [1], [3])])
        mid = space([(0, [1], [3]), (1, [2], [6])])
        fine = space([(0, [1], [3]), (1, [2], [6]), (2, [6], [10])])

        direct = coarse.prolongation_to(fine)
        chained = mid.prolongation_to(fine) @ coarse.prolongation_to(mid)
        np.testing.assert_allclose(direct, chained, atol=1e-9)


class TestCoarseningExactInverse:
    """restriction ∘ prolongation == I over a refinement chain.

    Necessary but weak: ``R := pinv(P)`` left-inverts *any* full-rank ``P``, and the
    same ``P`` cancels on both sides, so a wrong-but-full-rank prolongation would still
    pass.  :class:`TestCoarseningProjection` covers the non-automatic behaviour.
    """

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


class TestCoarseningProjection:
    """Restriction's coarsening behaviour, beyond the (vacuous) ``R∘P == I`` identity.

    ``R := pinv(P)`` left-inverts any full-rank ``P``, so ``R∘P == I`` only exercises
    numpy's pseudo-inverse — and since the same ``P`` appears on both sides, a faulty
    prolongation cancels out and stays hidden.  These tests pin the genuinely
    non-automatic behaviour:

    - restriction recovers a coarse field whose fine representation is built
      *independently* of ``P`` (L2-projected onto the fine space), so a wrong ``P`` is
      no longer masked;
    - ``P∘R`` is a non-trivial idempotent projector onto ``V_coarse ⊂ V_fine`` — i.e.
      restriction truly discards fine-level detail rather than acting as an embedding.
    """

    @staticmethod
    def _nested_1d() -> tuple[THBSplineSpace, THBSplineSpace]:
        coarse_grid = hierarchical_grid(uniform_grid([[0.0, 1.0]], 4), 2)
        coarse_grid.refine(0, [1], [3])
        coarse = THBSplineSpace(_root_1d(), coarse_grid)

        fine_grid = hierarchical_grid(uniform_grid([[0.0, 1.0]], 4), 2)
        fine_grid.refine(0, [1], [3])
        fine_grid.refine(1, [2], [6])
        fine = THBSplineSpace(_root_1d(), fine_grid)
        return coarse, fine

    @staticmethod
    def _nested_2d() -> tuple[THBSplineSpace, THBSplineSpace]:
        coarse_grid = hierarchical_grid(uniform_grid([[0.0, 1.0], [0.0, 1.0]], 4), 2)
        coarse_grid.refine(0, [0, 0], [2, 2])
        coarse = THBSplineSpace(_root_2d(), coarse_grid)

        fine_grid = hierarchical_grid(uniform_grid([[0.0, 1.0], [0.0, 1.0]], 4), 2)
        fine_grid.refine(0, [0, 0], [2, 2])
        fine_grid.refine(1, [0, 0], [4, 4])
        fine = THBSplineSpace(_root_2d(), fine_grid)
        return coarse, fine

    def test_recovers_independently_built_coarse_field_1d(self) -> None:
        # A quadratic lives in the (degree-2) coarse span. Build its FINE coefficients
        # by L2 projection — never touching the prolongation — then coarsen: restriction
        # must return coarse coefficients that reproduce the quadratic.  A wrong P (which
        # R∘P == I cannot detect) breaks this, because the fine field is P-independent.
        coarse, fine = self._nested_1d()

        def quad(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.asarray(p[:, 0] ** 2 - 0.3 * p[:, 0] + 0.1)

        c_fine = l2_project_thb(fine, quad).coeffs
        c_back = fine.restriction_to(coarse) @ c_fine
        xs = np.linspace(0.02, 0.98, 80).reshape(-1, 1)
        np.testing.assert_allclose(
            THBSpline(coarse, c_back).evaluate(xs).ravel(), quad(xs), atol=1e-8
        )

    def test_recovers_independently_built_coarse_field_2d(self) -> None:
        coarse, fine = self._nested_2d()

        def quad(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.asarray(p[:, 0] ** 2 + p[:, 0] * p[:, 1] - 0.5)

        c_fine = l2_project_thb(fine, quad).coeffs
        c_back = fine.restriction_to(coarse) @ c_fine
        u = np.linspace(0.05, 0.95, 7)
        pts = np.stack([m.ravel() for m in np.meshgrid(u, u, indexing="ij")], axis=-1)
        np.testing.assert_allclose(
            THBSpline(coarse, c_back).evaluate(pts).ravel(), quad(pts), atol=1e-8
        )

    def test_coarsening_is_a_lossy_idempotent_projector(self) -> None:
        coarse, fine = self._nested_1d()
        prolong = coarse.prolongation_to(fine)
        restrict = fine.restriction_to(coarse)
        proj = prolong @ restrict  # coarsen, then prolong back

        rng = np.random.default_rng(0)
        u = rng.standard_normal(fine.num_active_functions)
        # Genuine coarsening: a generic fine field carries detail the coarse space
        # cannot hold, so the round trip must change it (P∘R is not the identity).
        assert np.linalg.norm(proj @ u - u) > 1e-2 * np.linalg.norm(u)
        # ...yet it is an idempotent projection (re-applying it changes nothing)...
        np.testing.assert_allclose(proj @ (proj @ u), proj @ u, atol=1e-9)
        # ...onto exactly the coarse subspace.
        assert np.linalg.matrix_rank(proj) == coarse.num_active_functions


class TestBezierReconstruction:
    """Per-cell multi-level Bézier extraction reconstructs an L2-projected field.

    Cross-checks the extraction operator against the *independent* Bernstein basis
    (:func:`~pantr.basis.tabulate_bernstein`): ``bern @ (Cᵉᵀ c)`` must equal the spline's
    own evaluation on every active cell.  Covered for degrees 2 and 3, 1D and 2D.
    """

    @staticmethod
    def _check(degree: int, dim: int, refines: list[tuple[int, list[int], list[int]]]) -> None:
        root = create_uniform_space([degree] * dim, [4] * dim)
        grid = hierarchical_grid(uniform_grid([[0.0, 1.0]] * dim, 4), 2)
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

    @pytest.mark.parametrize("degree", [2, 3])
    def test_reconstruction_1d(self, degree: int) -> None:
        self._check(degree, 1, [(0, [0], [2]), (1, [0], [2])])

    @pytest.mark.parametrize("degree", [2, 3])
    def test_reconstruction_2d(self, degree: int) -> None:
        self._check(degree, 2, [(0, [0, 0], [2, 2])])
