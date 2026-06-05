"""Validation suite — group 2: L2 approximation in THB spaces.

Reproduces the L2-approximation behaviour of THB-splines:

- exact reproduction of a function already in the space (a consequence of the
  Giannelli-Jüttler-Speleers 2012 truncated basis spanning the hierarchical spline
  space, Thm 9);
- optimal order-``p+1`` convergence under refinement, and the adaptivity payoff —
  refining toward a localized feature reaches a smaller error with fewer degrees of
  freedom than uniform refinement.  The order-``p+1`` rate and the adaptive advantage
  are THB *approximation* results (Speleers-Manni 2016 / Speleers 2017); GJS 2012 itself
  reports a least-squares *fitting* study (L-infinity error, sparsity, conditioning), not
  an L2 convergence-order analysis.

The L2 projection is the test-only cell-assembly helper :func:`_thb_assembly.l2_project_thb`
(no global assembler is added to the library; #152 Q0).  Convergence orders and the
adaptive advantage are reproduced, not exact published magnitudes (#164, PR8).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline import (
    BsplineSpace,
    BsplineSpace1D,
    THBSpline,
    THBSplineSpace,
    create_uniform_space,
)
from pantr.grid import hierarchical_grid, uniform_grid
from tests._thb_assembly import l2_error, l2_project_thb

_KNOTS_DEG2_4 = np.array([0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0])


def _uniform_refined(degree: int, n: int, depth: int, dim: int) -> THBSplineSpace:
    """A THB space whose every cell is refined ``depth`` times (a uniform fine mesh).

    Note: refining *every* cell deactivates all coarse functions, so this is a
    single-level tensor-product space — it checks the tensor-product convergence base
    case.  Use :func:`_graded_refined` to exercise the genuinely truncated path.
    """
    root = create_uniform_space([degree] * dim, [n] * dim)
    grid = hierarchical_grid(uniform_grid([[0.0, 1.0]] * dim, n), 2)
    for level in range(depth):
        n_at = n * (2**level)
        grid.refine(level, [0] * dim, [n_at] * dim)
    return THBSplineSpace(root, grid)


def _graded_refined(degree: int, depth: int, dim: int) -> THBSplineSpace:
    """A genuinely truncated graded space (refine the left half only, doubling resolution).

    Coarse functions stay active over the right (unrefined-further) half, so truncated
    functions are present at every depth — unlike :func:`_uniform_refined`.  The global
    mesh size halves with ``depth``, so a smooth function still converges at order
    ``p+1`` (the error is dominated by the coarser half).
    """
    n = 4 * 2**depth
    root = create_uniform_space([degree] * dim, [n] * dim)
    grid = hierarchical_grid(uniform_grid([[0.0, 1.0]] * dim, n), 2)
    grid.refine(0, [0] * dim, [n // 2] * dim)
    return THBSplineSpace(root, grid)


def _observed_orders(errors: list[float]) -> list[float]:
    return [float(np.log2(errors[i] / errors[i + 1])) for i in range(len(errors) - 1)]


class TestL2Reproduction:
    """L2 projection reproduces any spline already in the space."""

    @pytest.mark.parametrize("dim", [1, 2])
    def test_reproduces_thb_spline(self, dim: int) -> None:
        knots = _KNOTS_DEG2_4
        root = BsplineSpace([BsplineSpace1D(knots, 2) for _ in range(dim)])
        grid = hierarchical_grid(uniform_grid([[0.0, 1.0]] * dim, 4), 2)
        grid.refine(0, [0] * dim, [2] * dim)
        thb = THBSplineSpace(root, grid)
        coeffs = np.random.default_rng(dim).standard_normal(thb.num_total_basis)
        target = THBSpline(thb, coeffs)
        proj = l2_project_thb(thb, lambda p: np.asarray(target.evaluate(p)))
        np.testing.assert_allclose(proj.coeffs, coeffs, atol=1e-9)


class TestL2Convergence:
    """Order-``p+1`` L2 convergence on a smooth function under refinement."""

    @staticmethod
    def _f(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray(np.sin(2.0 * np.pi * p[:, 0]))

    @pytest.mark.parametrize("degree", [2, 3])
    def test_order_1d(self, degree: int) -> None:
        # Tensor-product base case (uniform refinement deactivates coarse functions).
        errors = [
            l2_error(l2_project_thb(_uniform_refined(degree, 4, d, 1), self._f), self._f)
            for d in (1, 2, 3)
        ]
        assert min(_observed_orders(errors)) > degree + 0.5

    @pytest.mark.parametrize("degree", [2, 3])
    def test_order_1d_graded(self, degree: int) -> None:
        # Genuinely truncated hierarchy: coarse functions remain active, so this
        # exercises the THB path rather than a single-level tensor-product space.
        errors = [
            l2_error(l2_project_thb(_graded_refined(degree, d, 1), self._f), self._f)
            for d in (1, 2, 3)
        ]
        assert min(_observed_orders(errors)) > degree + 0.5


class TestAdaptiveEfficiency:
    """Adaptive refinement toward a localized feature beats uniform refinement."""

    @staticmethod
    def _bump(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray(np.exp(-(((p[:, 0] - 0.5) / 0.04) ** 2)))

    @staticmethod
    def _space(refines: list[tuple[int, list[int], list[int]]]) -> THBSplineSpace:
        root = create_uniform_space([2], [8])
        grid = hierarchical_grid(uniform_grid([[0.0, 1.0]], 8), 2)
        for level, lo, hi in refines:
            grid.refine(level, lo, hi)
        return THBSplineSpace(root, grid)

    def test_adaptive_beats_uniform(self) -> None:
        # Uniform: refine every cell once (resolution 16).
        uniform = self._space([(0, [0], [8])])
        # Adaptive: refine only the central band toward the bump, twice.
        adaptive = self._space([(0, [3], [5]), (1, [7], [9])])

        err_uniform = l2_error(l2_project_thb(uniform, self._bump), self._bump)
        err_adaptive = l2_error(l2_project_thb(adaptive, self._bump), self._bump)

        # Adaptive achieves a smaller error with fewer degrees of freedom.
        assert adaptive.num_total_basis < uniform.num_total_basis
        assert err_adaptive < err_uniform
