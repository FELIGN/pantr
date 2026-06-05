"""Validation suite — group 3: quasi-interpolation convergence (Speleers-Manni 2016).

Reproduces the approximation behaviour of the hierarchical quasi-interpolant: optimal
order-``p+1`` convergence under uniform refinement (1D and 2D, degrees 2 and 3),
consistency with the tensor-product QI on an unrefined space, and the adaptivity payoff
on a localized feature.  This goes beyond PR7's single convergence smoke test.

Orders and the adaptive advantage are reproduced, not exact published magnitudes
(#164, PR8).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline import (
    THBSplineSpace,
    create_uniform_space,
    quasi_interpolate_bspline,
    quasi_interpolate_thb_spline,
)
from pantr.grid import hierarchical_grid, uniform_grid
from tests._thb_assembly import l2_error


def _uniform_refined(degree: int, n: int, depth: int, dim: int) -> THBSplineSpace:
    """A THB space whose every cell is refined ``depth`` times (a uniform fine mesh)."""
    root = create_uniform_space([degree] * dim, [n] * dim)
    grid = hierarchical_grid(uniform_grid([[0.0, 1.0]] * dim, n), 2)
    for level in range(depth):
        n_at = n * (2**level)
        grid.refine(level, [0] * dim, [n_at] * dim)
    return THBSplineSpace(root, grid)


def _observed_orders(errors: list[float]) -> list[float]:
    return [float(np.log2(errors[i] / errors[i + 1])) for i in range(len(errors) - 1)]


def _f1(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray(np.sin(2.0 * np.pi * p[:, 0]))


def _f2(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray(np.sin(2.0 * np.pi * p[:, 0]) * np.cos(2.0 * np.pi * p[:, 1]))


class TestQIConvergence:
    """Order-``p+1`` QI convergence under uniform refinement."""

    @pytest.mark.parametrize("degree", [2, 3])
    def test_order_1d(self, degree: int) -> None:
        errors = [
            l2_error(quasi_interpolate_thb_spline(_f1, _uniform_refined(degree, 4, d, 1)), _f1)
            for d in (1, 2, 3)
        ]
        assert min(_observed_orders(errors)) > degree + 0.5

    @pytest.mark.parametrize("degree", [2, 3])
    def test_order_2d(self, degree: int) -> None:
        errors = [
            l2_error(quasi_interpolate_thb_spline(_f2, _uniform_refined(degree, 4, d, 2)), _f2)
            for d in (1, 2)
        ]
        assert _observed_orders(errors)[0] > degree + 0.5


class TestQIConsistency:
    """THB-QI on an unrefined space matches the tensor-product QI."""

    def test_unrefined_matches_tp(self) -> None:
        root = create_uniform_space([2], [8])
        grid = hierarchical_grid(uniform_grid([[0.0, 1.0]], 8), 2)
        thb = THBSplineSpace(root, grid)
        thb_qi = quasi_interpolate_thb_spline(_f1, thb)
        tp_qi = quasi_interpolate_bspline(_f1, root)
        # Identical coefficients ⇒ identical error.
        np.testing.assert_allclose(thb_qi.coeffs, tp_qi.control_points.ravel(), atol=1e-12)


class TestQIAdaptiveEfficiency:
    """Adaptive QI toward a localized feature beats uniform QI."""

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
        uniform = self._space([(0, [0], [8])])
        adaptive = self._space([(0, [3], [5]), (1, [7], [9])])
        err_uniform = l2_error(quasi_interpolate_thb_spline(self._bump, uniform), self._bump)
        err_adaptive = l2_error(quasi_interpolate_thb_spline(self._bump, adaptive), self._bump)
        assert adaptive.num_active_functions < uniform.num_active_functions
        assert err_adaptive < err_uniform
