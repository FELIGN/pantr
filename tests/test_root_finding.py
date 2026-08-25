"""Tests for Bernstein polynomial root-finding algorithms.

Covers:

- :func:`find_roots` -- auto-dispatch root finder (single or batch).
- :func:`find_monotone_root` -- Newton/bisection on monotone Beziers (single or batch).
- Internal helpers: de Casteljau scalar, split, subdivide, sign changes,
  convex hull clipping, Newton polish.
"""

import math
import unittest

import numpy as np
from numpy import typing as npt
from numpy.testing import assert_allclose

from pantr.bezier import (
    Bezier,
    find_monotone_root,
    find_roots,
)
from pantr.bezier._clipping_core import (
    _clip_roots_core,
    _dedup_roots_core,
)
from pantr.bezier._find_roots import _validate_coeff_array
from pantr.bezier._root_finding_core import (
    _clip_hull_to_zero,
    _count_sign_changes,
    _de_casteljau_eval_and_deriv_scalar,
    _de_casteljau_eval_scalar,
    _have_opposite_signs,
    _have_same_sign,
    _newton_polish_scalar,
    _spans_zero,
    _subdivide_scalar,
)
from pantr.bezier._yuksel_core import (
    _solve_monotone_root_kernel,
    _yuksel_roots,
)
from pantr.tolerance import get_default


class TestDeCasteljauEvalScalar(unittest.TestCase):
    """Tests for :func:`_de_casteljau_eval_scalar`."""

    def test_constant(self) -> None:
        """Degree-0 polynomial returns constant."""
        c = np.array([3.14], dtype=np.float64)
        self.assertAlmostEqual(_de_casteljau_eval_scalar(c, 0.5), 3.14)

    def test_linear(self) -> None:
        """Degree-1 linear interpolation."""
        c = np.array([1.0, 3.0], dtype=np.float64)
        self.assertAlmostEqual(_de_casteljau_eval_scalar(c, 0.0), 1.0)
        self.assertAlmostEqual(_de_casteljau_eval_scalar(c, 1.0), 3.0)
        self.assertAlmostEqual(_de_casteljau_eval_scalar(c, 0.5), 2.0)

    def test_quadratic(self) -> None:
        """Degree-2: f(0.5) for [-1, 0, 1] = 0."""
        c = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        self.assertAlmostEqual(_de_casteljau_eval_scalar(c, 0.5), 0.0)
        self.assertAlmostEqual(_de_casteljau_eval_scalar(c, 0.0), -1.0)
        self.assertAlmostEqual(_de_casteljau_eval_scalar(c, 1.0), 1.0)


class TestDeCasteljauEvalAndDerivScalar(unittest.TestCase):
    """Tests for :func:`_de_casteljau_eval_and_deriv_scalar`."""

    def test_constant_derivative_is_zero(self) -> None:
        """Degree-0: derivative is 0."""
        c = np.array([5.0], dtype=np.float64)
        f, df = _de_casteljau_eval_and_deriv_scalar(c, 0.5)
        self.assertAlmostEqual(f, 5.0)
        self.assertAlmostEqual(df, 0.0)

    def test_linear_derivative(self) -> None:
        """Degree-1: f'(t) = n * (c1 - c0) = 1 * (3 - 1) = 2."""
        c = np.array([1.0, 3.0], dtype=np.float64)
        f, df = _de_casteljau_eval_and_deriv_scalar(c, 0.25)
        self.assertAlmostEqual(f, 1.5)
        self.assertAlmostEqual(df, 2.0)

    def test_quadratic_derivative(self) -> None:
        """Degree-2: check value and derivative at midpoint."""
        c = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        f, df = _de_casteljau_eval_and_deriv_scalar(c, 0.5)
        self.assertAlmostEqual(f, 0.0, places=14)
        self.assertAlmostEqual(df, 2.0, places=14)


class TestSubdivideScalar(unittest.TestCase):
    """Tests for :func:`_subdivide_scalar`."""

    def test_full_interval(self) -> None:
        """Sub-interval [0, 1] returns a copy."""
        c = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        sub = _subdivide_scalar(c, 0.0, 1.0)
        assert_allclose(sub, c)

    def test_subdivision_evaluation(self) -> None:
        """Sub-interval coefficients evaluate correctly."""
        c = np.array([1.0, -2.0, 3.0, -1.0], dtype=np.float64)
        t_min, t_max = 0.2, 0.8
        sub = _subdivide_scalar(c, t_min, t_max)
        # Evaluate sub at u=0.5 should match original at 0.2 + 0.5*(0.8-0.2).
        t_global = t_min + 0.5 * (t_max - t_min)
        val_orig = _de_casteljau_eval_scalar(c, t_global)
        val_sub = _de_casteljau_eval_scalar(sub, 0.5)
        self.assertAlmostEqual(val_sub, val_orig, places=12)


class TestCountSignChanges(unittest.TestCase):
    """Tests for :func:`_count_sign_changes`."""

    def test_no_changes(self) -> None:
        """All positive: 0 sign changes."""
        self.assertEqual(_count_sign_changes(np.array([1.0, 2.0, 3.0])), 0)

    def test_one_change(self) -> None:
        """One sign change."""
        self.assertEqual(_count_sign_changes(np.array([1.0, -1.0, -2.0])), 1)

    def test_two_changes(self) -> None:
        """Two sign changes."""
        self.assertEqual(_count_sign_changes(np.array([1.0, -1.0, 1.0])), 2)

    def test_zeros_ignored(self) -> None:
        """Zeros are skipped in sign-change counting."""
        self.assertEqual(_count_sign_changes(np.array([1.0, 0.0, -1.0])), 1)


class TestClipHullToZero(unittest.TestCase):
    """Tests for :func:`_clip_hull_to_zero`."""

    def test_linear_crossing(self) -> None:
        """Linear polynomial [-1, 1] crosses zero at t=0.5."""
        t_lo, t_hi, found = _clip_hull_to_zero(np.array([-1.0, 1.0], dtype=np.float64))
        self.assertTrue(found)
        self.assertAlmostEqual(t_lo, 0.5, places=14)
        self.assertAlmostEqual(t_hi, 0.5, places=14)

    def test_no_crossing(self) -> None:
        """All-positive: no hull crossing."""
        _, _, found = _clip_hull_to_zero(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        self.assertFalse(found)

    def test_degree_zero(self) -> None:
        """Degree 0: not enough points for a hull."""
        _, _, found = _clip_hull_to_zero(np.array([1.0], dtype=np.float64))
        self.assertFalse(found)


class TestNewtonPolishScalar(unittest.TestCase):
    """Tests for :func:`_newton_polish_scalar`."""

    def test_simple_root_refined(self) -> None:
        """Newton polishes a simple root from bracket midpoint to near-exact."""
        c = np.array([-1.0, 1.0], dtype=np.float64)
        mid = 0.48
        polished, f_val, df_val = _newton_polish_scalar(c, mid, 0.0, 1.0, 1e-12)
        self.assertAlmostEqual(polished, 0.5, places=12)
        self.assertAlmostEqual(f_val, 0.0, places=12)
        self.assertNotEqual(df_val, 0.0)

    def test_no_improvement_keeps_original(self) -> None:
        """When Newton overshoots, the original midpoint is returned."""
        c = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        mid = 0.5
        polished, f_val, _ = _newton_polish_scalar(c, mid, 0.0, 1.0, 1e-12)
        self.assertAlmostEqual(f_val, _de_casteljau_eval_scalar(c, polished), places=14)

    def test_out_of_bounds_newton_rejected(self) -> None:
        """Newton candidate outside the neighborhood is rejected."""
        c = np.array([-1.0, 1.0], dtype=np.float64)
        mid = 0.9
        polished, _, _ = _newton_polish_scalar(c, mid, 0.8, 1.0, 0.0)
        self.assertAlmostEqual(polished, 0.9, places=14)


class TestSolveMonotoneRootKernel(unittest.TestCase):
    """Tests for :func:`_solve_monotone_root_kernel`."""

    def test_linear_root(self) -> None:
        """Linear polynomial: root at t = 0.5."""
        c = np.array([-1.0, 1.0], dtype=np.float64)
        root = _solve_monotone_root_kernel(c, 1e-14)
        self.assertAlmostEqual(root, 0.5, places=13)

    def test_quadratic_root(self) -> None:
        """Monotone quadratic [-1, 0, 1]: root at 0.5."""
        c = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        root = _solve_monotone_root_kernel(c, 1e-14)
        self.assertAlmostEqual(root, 0.5, places=12)

    def test_no_root_returns_nan(self) -> None:
        """All-positive: no sign change, returns NaN."""
        c = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        root = _solve_monotone_root_kernel(c, 1e-14)
        self.assertTrue(np.isnan(root))

    def test_root_at_boundary_zero(self) -> None:
        """Root at t = 0."""
        c = np.array([0.0, 1.0], dtype=np.float64)
        root = _solve_monotone_root_kernel(c, 1e-14)
        self.assertAlmostEqual(root, 0.0, places=10)

    def test_root_at_boundary_one(self) -> None:
        """Root at t = 1."""
        c = np.array([-1.0, 0.0], dtype=np.float64)
        root = _solve_monotone_root_kernel(c, 1e-14)
        self.assertAlmostEqual(root, 1.0, places=10)


class TestYukselRoots(unittest.TestCase):
    """Tests for :func:`_yuksel_roots`."""

    def test_linear_single_root(self) -> None:
        """Degree-1: root at t = 0.2."""
        c = np.array([-0.2, 0.8], dtype=np.float64)
        roots, count = _yuksel_roots(c, 1e-12)
        self.assertEqual(count, 1)
        self.assertAlmostEqual(roots[0], 0.2, places=10)

    def test_no_root_all_positive(self) -> None:
        """All-positive: no roots."""
        c = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        _, count = _yuksel_roots(c, 1e-12)
        self.assertEqual(count, 0)

    def test_quadratic_two_roots(self) -> None:
        """Quadratic with two roots in (0, 1)."""
        c = np.array([0.1, -0.3, 0.1], dtype=np.float64)
        roots, count = _yuksel_roots(c, 1e-12)
        self.assertEqual(count, 2)
        for i in range(count):
            val = _de_casteljau_eval_scalar(c, roots[i])
            self.assertAlmostEqual(val, 0.0, places=8)

    def test_constant_zero(self) -> None:
        """Degree-0 returns 0 roots."""
        c = np.array([0.0], dtype=np.float64)
        _, count = _yuksel_roots(c, 1e-12)
        self.assertEqual(count, 0)


class TestClipRootsCore(unittest.TestCase):
    """Tests for :func:`_clip_roots_core` and :func:`_dedup_roots_core`."""

    def _find_roots_clip(
        self,
        c: npt.NDArray[np.float64],
        param_tol: float = 1e-12,
        geom_tol: float = 1e-12,
    ) -> npt.NDArray[np.float64]:
        """Helper: run clipping + dedup."""
        if len(c) < 2:
            return np.empty(0, dtype=np.float64)
        if np.all(np.abs(c) <= geom_tol):
            return np.empty(0, dtype=np.float64)
        raw, n = _clip_roots_core(c, param_tol, geom_tol)
        if n == 0:
            return np.empty(0, dtype=np.float64)
        merged, count = _dedup_roots_core(raw, n, c, param_tol, geom_tol)
        unique: npt.NDArray[np.float64] = merged[:count].copy()
        return unique

    def test_linear_single_root(self) -> None:
        """Degree-1: root at t = 0.2."""
        c = np.array([-0.2, 0.8], dtype=np.float64)
        roots = self._find_roots_clip(c)
        self.assertEqual(len(roots), 1)
        self.assertAlmostEqual(roots[0], 0.2, places=10)

    def test_quadratic_two_roots(self) -> None:
        """Quadratic with two roots."""
        c = np.array([0.1, -0.3, 0.1], dtype=np.float64)
        roots = self._find_roots_clip(c)
        self.assertEqual(len(roots), 2)
        for r in roots:
            val = _de_casteljau_eval_scalar(c, r)
            self.assertAlmostEqual(val, 0.0, places=8)

    def test_no_root_all_positive(self) -> None:
        """All-positive: no roots."""
        c = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        roots = self._find_roots_clip(c)
        self.assertEqual(len(roots), 0)

    def test_no_root_all_negative(self) -> None:
        """All-negative: no roots."""
        c = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
        roots = self._find_roots_clip(c)
        self.assertEqual(len(roots), 0)

    def test_constant_zero_returns_empty(self) -> None:
        """All-zero: returns empty."""
        c = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        roots = self._find_roots_clip(c)
        self.assertEqual(len(roots), 0)

    def test_root_at_left_boundary(self) -> None:
        """Root at t = 0."""
        c = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        roots = self._find_roots_clip(c)
        self.assertGreaterEqual(len(roots), 1)
        self.assertAlmostEqual(roots[0], 0.0, places=10)

    def test_root_at_right_boundary(self) -> None:
        """Root at t = 1."""
        c = np.array([1.0, 2.0, 0.0], dtype=np.float64)
        roots = self._find_roots_clip(c)
        self.assertGreaterEqual(len(roots), 1)
        self.assertAlmostEqual(roots[-1], 1.0, delta=1e-10)

    def test_even_multiplicity_root(self) -> None:
        """Double root at t = 0.5."""
        c = np.array([0.25, -0.25, 0.25], dtype=np.float64)
        val_mid = _de_casteljau_eval_scalar(c, 0.5)
        self.assertAlmostEqual(val_mid, 0.0, places=14)

        roots = self._find_roots_clip(c)
        self.assertGreaterEqual(len(roots), 1)
        has_root_at_half = any(abs(r - 0.5) < 1e-6 for r in roots)
        self.assertTrue(has_root_at_half, f"Expected root near 0.5, got {roots}")

    def test_agrees_with_yuksel_quadratic(self) -> None:
        """Clipping and Yuksel agree on a quadratic."""
        c = np.array([0.1, -0.3, 0.1], dtype=np.float64)
        roots_clip = self._find_roots_clip(c)
        roots_yuk, n = _yuksel_roots(c, 1e-12)
        roots_yuk_sorted = np.sort(roots_yuk[:n])
        self.assertEqual(len(roots_clip), n)
        assert_allclose(roots_clip, roots_yuk_sorted, atol=1e-8)

    def test_agrees_with_yuksel_high_degree(self) -> None:
        """Clipping and Yuksel agree on a random degree-8 polynomial."""
        rng = np.random.default_rng(12345)
        c = rng.uniform(-2.0, 2.0, 9).astype(np.float64)
        roots_clip = self._find_roots_clip(c)
        roots_yuk, n = _yuksel_roots(c, 1e-12)
        roots_yuk_sorted = np.sort(roots_yuk[:n])
        self.assertEqual(len(roots_clip), n, f"clip={roots_clip}, yuksel={roots_yuk_sorted}")
        assert_allclose(roots_clip, roots_yuk_sorted, atol=1e-6)


class TestFindRoots(unittest.TestCase):
    """Tests for :func:`find_roots` (public API)."""

    def test_linear_root(self) -> None:
        """Linear: root at t = 0.2."""
        roots = find_roots(Bezier([-0.2, 0.8]))
        self.assertEqual(len(roots), 1)
        self.assertAlmostEqual(roots[0], 0.2, places=10)

    def test_quadratic_two_roots(self) -> None:
        """Quadratic with two roots."""
        c = np.array([0.1, -0.3, 0.1], dtype=np.float64)
        roots = find_roots(Bezier(c))
        self.assertEqual(len(roots), 2)
        for r in roots:
            val = _de_casteljau_eval_scalar(c, r)
            self.assertAlmostEqual(val, 0.0, places=8)

    def test_no_root_all_positive(self) -> None:
        """All-positive: no roots."""
        roots = find_roots(Bezier([1.0, 2.0, 3.0]))
        self.assertEqual(len(roots), 0)

    def test_constant_zero_returns_empty(self) -> None:
        """All-zero: returns empty."""
        roots = find_roots(Bezier([0.0, 0.0, 0.0]))
        self.assertEqual(len(roots), 0)

    def test_degree_zero(self) -> None:
        """Single element (degree 0): returns empty."""
        roots = find_roots(Bezier([5.0]))
        self.assertEqual(len(roots), 0)

    def test_low_degree_uses_yuksel(self) -> None:
        """Degree <= 5 routes to Yuksel (same results)."""
        c = np.array([0.1, -0.3, 0.1], dtype=np.float64)
        roots_auto = find_roots(Bezier(c), tol=1e-12)
        roots_yuk, n = _yuksel_roots(c, 1e-12)
        assert_allclose(roots_auto, np.sort(roots_yuk[:n]), atol=1e-14)

    def test_high_degree_well_conditioned(self) -> None:
        """Degree >= 6, well-conditioned: routes to clipping."""
        rng = np.random.default_rng(88)
        c = rng.uniform(-2.0, 2.0, 9).astype(np.float64)
        roots = find_roots(Bezier(c), tol=1e-12)
        # Verify all roots are actually roots.
        for r in roots:
            val = _de_casteljau_eval_scalar(c, r)
            self.assertAlmostEqual(val, 0.0, delta=1e-6)

    def test_extreme_range_falls_back_to_yuksel(self) -> None:
        """High dynamic range: falls back to Yuksel with valid residuals."""
        rng = np.random.default_rng(55)
        c = rng.uniform(-1.0, 1.0, 9).astype(np.float64)
        c[0] = 1e-8
        c[4] = -1e7
        roots = find_roots(Bezier(c), tol=1e-12)
        _roots_yuk, n = _yuksel_roots(c, 1e-12)
        self.assertEqual(len(roots), n)
        for r in roots:
            val = _de_casteljau_eval_scalar(c, float(np.clip(r, 0.0, 1.0)))
            self.assertAlmostEqual(val, 0.0, delta=1e-6)

    def test_custom_tolerance(self) -> None:
        """Custom tolerance is respected."""
        roots = find_roots(Bezier([-1.0, 1.0]), tol=1e-6)
        self.assertEqual(len(roots), 1)
        self.assertAlmostEqual(roots[0], 0.5, places=5)

    def test_float32_accepted(self) -> None:
        """float32 input is accepted and produces correct results."""
        c = np.array([[-1.0], [1.0]], dtype=np.float32)
        roots = find_roots(Bezier(c))
        self.assertEqual(roots.dtype, np.float64)
        self.assertEqual(len(roots), 1)
        self.assertAlmostEqual(roots[0], 0.5, places=5)

    def test_non_bezier_raises(self) -> None:
        """Non-Bezier input raises TypeError."""
        with self.assertRaises(TypeError):
            find_roots(np.array([1.0, -1.0]))  # type: ignore

    def test_dim_not_one_raises(self) -> None:
        """Bezier surface (dim=2) raises ValueError."""
        cp = np.ones((3, 3, 1), dtype=np.float64)
        with self.assertRaises(ValueError, msg="dim == 1"):
            find_roots(Bezier(cp))

    def test_rank_not_one_raises(self) -> None:
        """Multi-valued Bezier (rank=2) raises ValueError."""
        cp = np.array([[1.0, 0.0], [-1.0, 0.0], [0.5, 0.0]], dtype=np.float64)
        with self.assertRaises(ValueError, msg="rank == 1"):
            find_roots(Bezier(cp))

    def test_invalid_tol_raises(self) -> None:
        """Negative tolerance raises ValueError."""
        with self.assertRaises(ValueError, msg="tol must be positive"):
            find_roots(Bezier([1.0, -1.0]), tol=-1.0)

    # ---- Rational Bezier input ----

    def test_rational_bezier_numerator_extraction(self) -> None:
        """Rational scalar Bezier: roots found on numerator (x*w)."""
        # Construct a rational Bezier with rank=1.
        # control_points[:, 0] = x*w (numerator), control_points[:, 1] = w.
        # x*w = [-1, 0, 1], w = [1, 2, 1] => rational values = [-1, 0, 1].
        cp = np.array([[-1.0, 1.0], [0.0, 2.0], [1.0, 1.0]], dtype=np.float64)
        bez = Bezier(cp, is_rational=True)
        self.assertEqual(bez.rank, 1)
        roots = find_roots(bez)
        # Numerator [-1, 0, 1] has root at 0.5.
        self.assertEqual(len(roots), 1)
        self.assertAlmostEqual(roots[0], 0.5, places=10)

    def test_rational_quarter_circle(self) -> None:
        """Rational quarter-circle intersection with y = 0.5."""
        ctrl = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        w = np.array([1.0, math.sqrt(0.5), 1.0], dtype=np.float64)
        w_ctrl = ctrl * w[:, None]
        # Build scalar rational Bezier for (y*w - 0.5*w).
        numerator = w_ctrl[:, 1] - 0.5 * w
        cp_rational = np.column_stack([numerator, w])
        roots = find_roots(Bezier(cp_rational, is_rational=True))
        self.assertEqual(len(roots), 1)

    def test_rational_no_intersection(self) -> None:
        """Quarter-circle with y=1.5 produces no roots."""
        w = np.array([1.0, math.sqrt(0.5), 1.0], dtype=np.float64)
        w_ctrl = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64) * w[:, None]
        numerator = w_ctrl[:, 1] - 1.5 * w
        cp_rational = np.column_stack([numerator, w])
        roots = find_roots(Bezier(cp_rational, is_rational=True))
        self.assertEqual(len(roots), 0)


class TestSolveMonotoneRoot(unittest.TestCase):
    """Tests for :func:`find_monotone_root` (public API)."""

    def test_linear_root(self) -> None:
        """Linear: root at t = 0.5."""
        root = find_monotone_root(Bezier([-1.0, 1.0]))
        self.assertAlmostEqual(root, 0.5, places=13)

    def test_quadratic_root(self) -> None:
        """Quadratic [-1, 0, 1]: root at 0.5."""
        root = find_monotone_root(Bezier([-1.0, 0.0, 1.0]))
        self.assertAlmostEqual(root, 0.5, places=12)

    def test_no_root_returns_nan(self) -> None:
        """All-positive: returns NaN."""
        root = find_monotone_root(Bezier([1.0, 2.0, 3.0]))
        self.assertTrue(np.isnan(root))

    def test_custom_tolerance(self) -> None:
        """Custom tolerance works."""
        root = find_monotone_root(Bezier([-1.0, 1.0]), tol=1e-6)
        self.assertAlmostEqual(root, 0.5, places=5)

    def test_non_bezier_raises(self) -> None:
        """Non-Bezier input raises TypeError."""
        with self.assertRaises(TypeError):
            find_monotone_root(np.array([-1.0, 1.0]))  # type: ignore


class TestFindRootsBatch(unittest.TestCase):
    """Tests for :func:`find_roots` batch mode (public API)."""

    def test_single_polynomial(self) -> None:
        """Batch of one polynomial matches single-poly result."""
        bez = Bezier([0.1, -0.3, 0.1])
        roots_single = find_roots(bez, tol=1e-12)
        roots_batch, counts = find_roots([bez], tol=1e-12)
        self.assertEqual(counts[0], len(roots_single))
        assert_allclose(
            np.sort(roots_batch[0, : counts[0]]),
            roots_single,
            atol=1e-8,
        )

    def test_multiple_polynomials(self) -> None:
        """Batch of multiple Beziers."""
        beziers = [
            Bezier([-0.2, 0.8, 0.0]),  # has root(s)
            Bezier([1.0, 2.0, 3.0]),  # no roots
            Bezier([0.1, -0.3, 0.1]),  # two roots
        ]
        roots, counts = find_roots(beziers, tol=1e-12)
        self.assertEqual(roots.shape[0], 3)
        self.assertGreaterEqual(counts[0], 1)
        self.assertEqual(counts[1], 0)
        self.assertEqual(counts[2], 2)

    def test_degree_zero_batch(self) -> None:
        """Batch of degree-0 Beziers: all return 0 roots."""
        beziers = [Bezier([5.0]), Bezier([3.0])]
        _, counts = find_roots(beziers)
        self.assertEqual(counts[0], 0)
        self.assertEqual(counts[1], 0)

    def test_empty_batch(self) -> None:
        """Empty batch returns empty arrays without error."""
        roots, counts = find_roots([])
        self.assertEqual(roots.shape, (0, 1))
        self.assertEqual(counts.shape, (0,))

    def test_mismatched_degree_raises(self) -> None:
        """Beziers with different degrees raise ValueError."""
        with self.assertRaises(ValueError, msg="same degree"):
            find_roots([Bezier([1.0, -1.0]), Bezier([1.0, 0.0, -1.0])])

    def test_non_bezier_in_batch_raises(self) -> None:
        """Non-Bezier element in batch raises TypeError."""
        with self.assertRaises(TypeError):
            find_roots([Bezier([1.0, -1.0]), np.array([1.0, -1.0])])  # type: ignore[list-item]


class TestSolveMonotoneRootBatch(unittest.TestCase):
    """Tests for :func:`find_monotone_root` batch mode (public API)."""

    def test_mixed_roots(self) -> None:
        """Batch with some roots and some NaN."""
        beziers = [
            Bezier([-1.0, 1.0]),  # root at 0.5
            Bezier([1.0, 2.0]),  # no root
            Bezier([0.0, 1.0]),  # root at 0.0
        ]
        roots = find_monotone_root(beziers)
        self.assertAlmostEqual(roots[0], 0.5, places=12)
        self.assertTrue(np.isnan(roots[1]))
        self.assertAlmostEqual(roots[2], 0.0, places=10)

    def test_single_polynomial(self) -> None:
        """Batch of one matches single-poly result."""
        bez = Bezier([-1.0, 0.0, 1.0])
        root_single = find_monotone_root(bez)
        roots_batch = find_monotone_root([bez])
        self.assertAlmostEqual(roots_batch[0], root_single, places=12)

    def test_all_no_roots(self) -> None:
        """Batch where no Bezier has a root."""
        beziers = [Bezier([1.0, 2.0, 3.0]), Bezier([4.0, 5.0, 6.0])]
        roots = find_monotone_root(beziers)
        self.assertTrue(np.isnan(roots[0]))
        self.assertTrue(np.isnan(roots[1]))

    def test_empty_batch(self) -> None:
        """Empty batch returns empty array without error."""
        roots = find_monotone_root([])
        self.assertEqual(roots.shape, (0,))


class TestValidateCoeffArray(unittest.TestCase):
    """Tests for :func:`_validate_coeff_array` internal helper."""

    def test_empty_1d_raises(self) -> None:
        """Empty 1-D array (size 0) raises ValueError."""
        with self.assertRaises(ValueError):
            _validate_coeff_array(np.empty(0, dtype=np.float64), ndim=1, name="coeff")

    def test_empty_last_axis_2d_raises(self) -> None:
        """2-D array with zero columns raises ValueError."""
        with self.assertRaises(ValueError):
            _validate_coeff_array(np.empty((3, 0), dtype=np.float64), ndim=2, name="coeffs")

    def test_wrong_ndim_raises(self) -> None:
        """Array with wrong number of dimensions raises ValueError."""
        with self.assertRaises(ValueError):
            _validate_coeff_array(np.ones((2, 3), dtype=np.float64), ndim=1, name="coeff")

    def test_wrong_dtype_raises(self) -> None:
        """Integer array raises TypeError."""
        with self.assertRaises(TypeError):
            _validate_coeff_array(np.array([1, 2], dtype=np.int32), ndim=1, name="coeff")

    def test_name_in_error_message(self) -> None:
        """The ``name`` argument appears in error messages."""
        with self.assertRaises(ValueError) as ctx:
            _validate_coeff_array(np.empty(0, dtype=np.float64), ndim=1, name="my_arg")
        self.assertIn("my_arg", str(ctx.exception))

    def test_valid_1d_returns_contiguous(self) -> None:
        """Valid 1-D input is returned C-contiguous."""
        arr = np.array([1.0, -1.0], dtype=np.float64)
        result = _validate_coeff_array(arr, ndim=1, name="coeff")
        self.assertTrue(result.flags.c_contiguous)
        np.testing.assert_array_equal(result, arr)

    def test_valid_2d_returns_contiguous(self) -> None:
        """Valid 2-D input is returned C-contiguous."""
        arr = np.array([[1.0, -1.0], [0.5, -0.5]], dtype=np.float64)
        result = _validate_coeff_array(arr, ndim=2, name="coeffs")
        self.assertTrue(result.flags.c_contiguous)
        np.testing.assert_array_equal(result, arr)

    def test_float32_accepted(self) -> None:
        """float32 arrays are accepted."""
        arr = np.array([1.0, -1.0], dtype=np.float32)
        result = _validate_coeff_array(arr, ndim=1, name="coeff")
        self.assertEqual(result.dtype, np.float32)


def _root_delta(dtype: npt.DTypeLike) -> float:
    """Bound how far the root of ``a(1 - 2t)`` may sit from ``0.5`` in a given width.

    In this Bernstein form ``B(t) = a(1 - 2t)``, so ``|B'| = 2a`` while the evaluated
    value carries a handful of roundings of magnitude ``a``. The root's displacement is
    therefore bounded by ``k * u * a / (2 * a)``: a small multiple of the format's unit
    roundoff, with the scale ``a`` cancelling. That cancellation is why one bound serves
    every decade of the sweep below, which is the property those tests are about.

    :func:`~pantr.tolerance.get_default` is the right tier rather than a minted
    constant: it is a short algorithm plus build slack, and build slack is exactly what
    is spanned here, since the same assertion has to hold for the C++ backend and for
    the numba one compiled or interpreted. At ``float32`` an earlier ``places=10``
    demanded 5e-11 of a format whose own resolution near ``0.5`` is 6e-8, and it held
    only because the compiled path happened to land on ``0.5`` exactly.

    Args:
        dtype (npt.DTypeLike): The storage format the control points are given in.

    Returns:
        float: Absolute tolerance on the root parameter.
    """
    return 0.5 * get_default(dtype)


class TestSignTestUnderflow(unittest.TestCase):
    """A sign test written as a product must not underflow (FELIGN/pantr#351).

    Six sites in the root-finding kernels ask "do these two values share a sign?"
    by multiplying and comparing the product against zero. At ``float32`` the
    product of two operands of magnitude ``a`` is ``a**2``, which falls under
    that format's minimum subnormal (about ``1e-45``) once ``a`` drops below
    roughly ``1e-23`` -- while both operands remain perfectly representable.
    The test then answers as though one operand were zero.

    The cases below sit at that frontier rather than comfortably past it:
    ``1e-22`` squares to ``1e-44`` and survives, ``1e-23`` squares to ``1e-46``
    and does not. Each has a ``float64`` control, where the same product is
    nowhere near underflow.
    """

    #: Largest coefficient magnitude whose float32 square still underflows.
    FRONTIER = 1e-23

    def test_monotone_root_not_invented_at_frontier(self) -> None:
        """No root is reported for a strictly positive Bezier at the frontier."""
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=np.dtype(dtype).name):
                a = self.FRONTIER
                # B(t) > 0 on [0, 1]: every control value is positive, so the
                # convex-hull property forbids a root.
                bez = Bezier(np.array([[a], [0.5 * a], [2.0 * a]], dtype=dtype))
                self.assertTrue(math.isnan(find_monotone_root(bez)))

    def test_root_not_lost_at_frontier(self) -> None:
        """The single root of ``a(1 - 2t)`` survives at the frontier."""
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=np.dtype(dtype).name):
                a = self.FRONTIER
                bez = Bezier(np.array([[a], [0.0], [-a]], dtype=dtype))
                roots = find_roots(bez, tol=1e-40)
                self.assertEqual(len(roots), 1)
                self.assertAlmostEqual(roots[0], 0.5, delta=_root_delta(dtype))

    def test_sign_tests_are_scale_invariant_over_decades(self) -> None:
        """Both faces stay correct across the decades spanning the frontier.

        Scaling a root-finding problem by a positive constant moves no root, so
        every decade here must give the same answer. The sweep runs from two
        decades above the frontier to six below, at ``float32``, which is the
        width that distinguishes a fix of the *test* from one that merely moves
        the underflow threshold.
        """
        for exponent in range(-21, -30, -1):
            a = 10.0**exponent
            with self.subTest(a=f"1e{exponent}"):
                positive = Bezier(np.array([[a], [0.5 * a], [2.0 * a]], dtype=np.float32))
                self.assertTrue(math.isnan(find_monotone_root(positive)))

                crossing = Bezier(np.array([[a], [0.0], [-a]], dtype=np.float32))
                roots = find_roots(crossing, tol=1e-40)
                self.assertEqual(len(roots), 1)
                self.assertAlmostEqual(roots[0], 0.5, delta=_root_delta(np.float32))


class TestBoundaryEpsScaleInvariance(unittest.TestCase):
    """An absolute floor inside a scale-relative tolerance (FELIGN/pantr#352).

    ``_find_roots_at_level`` and ``_yuksel_roots`` grade "is this coefficient
    zero?" against ``max(scale * eps * 8, 1e-30)``. The first term is relative
    to the problem's own coefficient range and correct; the second is an
    absolute floor, and below it every coefficient reads as zero however well
    separated the coefficients are from each other.

    Scaling a root-finding problem by a positive constant does not move its
    roots, so this is an invariance being broken rather than a precision limit
    being reached. The sweep straddles the literal: ``1e-29`` is correct today
    and ``1e-30`` is not.
    """

    def test_single_root_is_scale_invariant(self) -> None:
        """``a(1 - 2t)`` has its only root at ``t = 0.5`` for every ``a != 0``."""
        for exponent in range(-25, -36, -1):
            a = 10.0**exponent
            with self.subTest(a=f"1e{exponent}"):
                bez = Bezier(np.array([[a], [0.0], [-a]], dtype=np.float64))
                roots = find_roots(bez, tol=1e-40)
                self.assertEqual(len(roots), 1)
                self.assertAlmostEqual(roots[0], 0.5, places=12)

    def test_no_root_is_scale_invariant(self) -> None:
        """A strictly positive Bezier reports no root at every scale."""
        for exponent in range(-25, -36, -1):
            a = 10.0**exponent
            with self.subTest(a=f"1e{exponent}"):
                bez = Bezier(np.array([[a], [0.5 * a], [2.0 * a]], dtype=np.float64))
                self.assertEqual(len(find_roots(bez, tol=1e-40)), 0)


class TestSignPredicateEquivalence(unittest.TestCase):
    """The sign predicates are exactly the product forms they replace.

    The three predicates in :mod:`pantr.bezier._root_finding_core` exist to
    remove a product from a sign test, and substituting them is only sound if
    they agree with the product everywhere the product is right. That is a claim
    about IEEE 754 semantics, so it is checked exhaustively over a matrix of
    special values rather than argued in a comment.

    Every pair drawn from ``VALUES`` has a well-defined product that neither
    underflows nor overflows, ``inf * 0.0 = nan`` included, so agreement here
    must be total. Where the product *is* wrong -- the underflow regime -- the
    predicates are supposed to differ from it, and that is asserted separately.
    """

    #: Signs, both zeros, both infinities and a NaN. All 81 pairs have an exact product.
    VALUES = (-np.inf, -3.0, -1.0, -0.0, 0.0, 1.0, 3.0, np.inf, np.nan)

    def test_same_sign_matches_product_gt_zero(self) -> None:
        """``_have_same_sign`` equals ``a * b > 0.0`` on every special-value pair."""
        for a in self.VALUES:
            for b in self.VALUES:
                with self.subTest(a=a, b=b):
                    self.assertEqual(_have_same_sign(a, b), a * b > 0.0)

    def test_opposite_signs_matches_product_lt_zero(self) -> None:
        """``_have_opposite_signs`` equals ``a * b < 0.0`` on every special-value pair."""
        for a in self.VALUES:
            for b in self.VALUES:
                with self.subTest(a=a, b=b):
                    self.assertEqual(_have_opposite_signs(a, b), a * b < 0.0)

    def test_spans_zero_matches_product_le_zero(self) -> None:
        """``_spans_zero`` equals ``a * b <= 0.0`` on every special-value pair.

        This is the predicate whose naive form is *not* equivalent: without its
        finiteness guards it would report a spanned bracket for ``(0.0, inf)``,
        where the product is NaN and the product form reports none.
        """
        for a in self.VALUES:
            for b in self.VALUES:
                with self.subTest(a=a, b=b):
                    self.assertEqual(_spans_zero(a, b), a * b <= 0.0)

    def test_zero_paired_with_infinity_is_not_spanned(self) -> None:
        """``(0, inf)`` is not a spanned bracket, because ``0 * inf`` is NaN.

        Called out on its own because it is the single pair that separates the
        exact predicate from the naive one, and a regression here would be
        invisible in the matrix above.
        """
        for zero in (0.0, -0.0):
            for infinity in (np.inf, -np.inf):
                with self.subTest(zero=zero, infinity=infinity):
                    self.assertFalse(_spans_zero(zero, infinity))
                    self.assertFalse(_spans_zero(infinity, zero))

    def test_predicates_survive_the_underflow_the_product_does_not(self) -> None:
        """The predicates are right in the regime where the product flushes to zero.

        Two operands of magnitude ``1e-200`` are ordinary ``float64`` values,
        but their product is ``1e-400`` and underflows to zero. The product form
        then answers as though an operand vanished; the predicates do not. This
        is the same mechanism as the ``float32`` defect, reached at ``float64``
        so that no dtype conversion is involved.
        """
        tiny = 1e-200
        # Same sign, and the product cannot see it.
        self.assertEqual(tiny * tiny, 0.0)
        self.assertTrue(_have_same_sign(tiny, tiny))
        self.assertFalse(tiny * tiny > 0.0)
        # Opposite signs, likewise.
        self.assertEqual(tiny * -tiny, 0.0)
        self.assertTrue(_have_opposite_signs(tiny, -tiny))
        self.assertFalse(tiny * -tiny < 0.0)
        # And the bracket test reports a span that is not there.
        self.assertFalse(_spans_zero(tiny, tiny))
        self.assertTrue(tiny * tiny <= 0.0)


if __name__ == "__main__":
    unittest.main()
