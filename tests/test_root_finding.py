"""Tests for Bernstein polynomial root-finding algorithms.

Covers:

- :func:`find_roots` -- auto-dispatch root finder (single polynomial).
- :func:`find_roots_batch` -- batch-parallel root finder.
- :func:`solve_monotone_root` -- Newton/bisection on monotone polynomials.
- :func:`solve_monotone_root_batch` -- batch-parallel monotone solver.
- Internal helpers: de Casteljau scalar, split, subdivide, sign changes,
  convex hull clipping, Newton polish.
"""

import math
import unittest

import numpy as np
from numpy import typing as npt
from numpy.testing import assert_allclose

from pantr.bezier import (
    find_roots,
    find_roots_batch,
    solve_monotone_root,
    solve_monotone_root_batch,
)
from pantr.bezier._clipping_core import (
    _clip_roots_core,
    _dedup_roots,
)
from pantr.bezier._root_finding_core import (
    _clip_hull_to_zero,
    _count_sign_changes,
    _de_casteljau_eval_and_deriv_scalar,
    _de_casteljau_eval_scalar,
    _newton_polish_scalar,
    _subdivide_scalar,
)
from pantr.bezier._yuksel_core import (
    _solve_monotone_root_kernel,
    _yuksel_roots,
)


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
    """Tests for :func:`_clip_roots_core` and :func:`_dedup_roots`."""

    def _find_roots_clip(
        self,
        c: npt.NDArray[np.float64],
        param_tol: float = 1e-12,
        geom_tol: float = 1e-12,
    ) -> npt.NDArray[np.float64]:
        """Helper: run clipping + dedup."""
        if len(c) < 2:  # noqa: PLR2004
            return np.empty(0, dtype=np.float64)
        if np.all(np.abs(c) <= geom_tol):
            return np.empty(0, dtype=np.float64)
        raw, n = _clip_roots_core(c, param_tol, geom_tol)
        return _dedup_roots(raw, n, c, param_tol, geom_tol)

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
        has_root_at_half = any(abs(r - 0.5) < 1e-6 for r in roots)  # noqa: PLR2004
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
        roots = find_roots(np.array([-0.2, 0.8]))
        self.assertEqual(len(roots), 1)
        self.assertAlmostEqual(roots[0], 0.2, places=10)

    def test_quadratic_two_roots(self) -> None:
        """Quadratic with two roots."""
        c = np.array([0.1, -0.3, 0.1], dtype=np.float64)
        roots = find_roots(c)
        self.assertEqual(len(roots), 2)
        for r in roots:
            val = _de_casteljau_eval_scalar(c, r)
            self.assertAlmostEqual(val, 0.0, places=8)

    def test_no_root_all_positive(self) -> None:
        """All-positive: no roots."""
        roots = find_roots(np.array([1.0, 2.0, 3.0]))
        self.assertEqual(len(roots), 0)

    def test_constant_zero_returns_empty(self) -> None:
        """All-zero: returns empty."""
        roots = find_roots(np.array([0.0, 0.0, 0.0]))
        self.assertEqual(len(roots), 0)

    def test_degree_zero(self) -> None:
        """Single element (degree 0): returns empty."""
        roots = find_roots(np.array([5.0]))
        self.assertEqual(len(roots), 0)

    def test_low_degree_uses_yuksel(self) -> None:
        """Degree <= 5 routes to Yuksel (same results)."""
        c = np.array([0.1, -0.3, 0.1], dtype=np.float64)
        roots_auto = find_roots(c, tol=1e-12)
        roots_yuk, n = _yuksel_roots(c, 1e-12)
        assert_allclose(roots_auto, np.sort(roots_yuk[:n]), atol=1e-14)

    def test_high_degree_well_conditioned(self) -> None:
        """Degree >= 6, well-conditioned: routes to clipping."""
        rng = np.random.default_rng(88)
        c = rng.uniform(-2.0, 2.0, 9).astype(np.float64)
        roots = find_roots(c, tol=1e-12)
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
        roots = find_roots(c, tol=1e-12)
        roots_yuk, n = _yuksel_roots(c, 1e-12)
        self.assertEqual(len(roots), n)
        for r in roots:
            val = _de_casteljau_eval_scalar(c, float(np.clip(r, 0.0, 1.0)))
            self.assertAlmostEqual(val, 0.0, delta=1e-6)

    def test_custom_tolerance(self) -> None:
        """Custom tolerance is respected."""
        c = np.array([-1.0, 1.0], dtype=np.float64)
        roots = find_roots(c, tol=1e-6)
        self.assertEqual(len(roots), 1)
        self.assertAlmostEqual(roots[0], 0.5, places=5)

    def test_float32_accepted(self) -> None:
        """float32 input is accepted and produces correct results."""
        c = np.array([-1.0, 1.0], dtype=np.float32)
        roots = find_roots(c)
        self.assertEqual(roots.dtype, np.float64)
        self.assertEqual(len(roots), 1)
        self.assertAlmostEqual(roots[0], 0.5, places=5)

    def test_unsupported_dtype_raises(self) -> None:
        """Non-float input raises TypeError."""
        with self.assertRaises(TypeError):
            find_roots(np.array([1, -1], dtype=np.int64))

    def test_list_raises(self) -> None:
        """Plain list raises TypeError (must be ndarray)."""
        with self.assertRaises(TypeError):
            find_roots([-1.0, 1.0])  # type: ignore[arg-type]

    def test_invalid_tol_raises(self) -> None:
        """Negative tolerance raises ValueError."""
        with self.assertRaises(ValueError, msg="tol must be positive"):
            find_roots(np.array([1.0, -1.0]), tol=-1.0)

    def test_invalid_shape_raises(self) -> None:
        """2-D input raises ValueError."""
        with self.assertRaises(ValueError):
            find_roots(np.array([[1.0, -1.0]]))

    def test_empty_raises(self) -> None:
        """Empty input raises ValueError."""
        with self.assertRaises(ValueError):
            find_roots(np.array([], dtype=np.float64))

    # ---- Ray-curve intersection equivalents ----

    def test_rational_quarter_circle(self) -> None:
        """Rational quarter-circle intersection with y = 0.5."""
        ctrl = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        w = np.array([1.0, math.sqrt(0.5), 1.0], dtype=np.float64)
        w_ctrl = ctrl * w[:, None]
        coeff = w_ctrl[:, 1] - 0.5 * w
        roots = find_roots(coeff, tol=1e-14)
        self.assertEqual(len(roots), 1)

    def test_rational_no_intersection(self) -> None:
        """Quarter-circle with y=1.5 produces no roots."""
        w = np.array([1.0, math.sqrt(0.5), 1.0], dtype=np.float64)
        w_ctrl = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64) * w[:, None]
        coeff = w_ctrl[:, 1] - 1.5 * w
        roots = find_roots(coeff, tol=1e-14)
        self.assertEqual(len(roots), 0)


class TestSolveMonotoneRoot(unittest.TestCase):
    """Tests for :func:`solve_monotone_root` (public API)."""

    def test_linear_root(self) -> None:
        """Linear: root at t = 0.5."""
        root = solve_monotone_root(np.array([-1.0, 1.0]))
        self.assertAlmostEqual(root, 0.5, places=13)

    def test_quadratic_root(self) -> None:
        """Quadratic [-1, 0, 1]: root at 0.5."""
        root = solve_monotone_root(np.array([-1.0, 0.0, 1.0]))
        self.assertAlmostEqual(root, 0.5, places=12)

    def test_no_root_returns_nan(self) -> None:
        """All-positive: returns NaN."""
        root = solve_monotone_root(np.array([1.0, 2.0, 3.0]))
        self.assertTrue(np.isnan(root))

    def test_custom_tolerance(self) -> None:
        """Custom tolerance works."""
        root = solve_monotone_root(np.array([-1.0, 1.0]), tol=1e-6)
        self.assertAlmostEqual(root, 0.5, places=5)


class TestFindRootsBatch(unittest.TestCase):
    """Tests for :func:`find_roots_batch` (public API)."""

    def test_single_polynomial(self) -> None:
        """Batch of one polynomial matches single-poly result."""
        c = np.array([0.1, -0.3, 0.1], dtype=np.float64)
        roots_single = find_roots(c, tol=1e-12)
        roots_batch, counts = find_roots_batch(c.reshape(1, -1), tol=1e-12)
        self.assertEqual(counts[0], len(roots_single))
        assert_allclose(
            np.sort(roots_batch[0, : counts[0]]),
            roots_single,
            atol=1e-8,
        )

    def test_multiple_polynomials(self) -> None:
        """Batch of multiple polynomials."""
        coeffs = np.array(
            [
                [-0.2, 0.8, 0.0],  # has root(s)
                [1.0, 2.0, 3.0],  # no roots
                [0.1, -0.3, 0.1],  # two roots
            ],
            dtype=np.float64,
        )
        roots, counts = find_roots_batch(coeffs, tol=1e-12)
        self.assertEqual(roots.shape[0], 3)
        self.assertGreaterEqual(counts[0], 1)
        self.assertEqual(counts[1], 0)
        self.assertEqual(counts[2], 2)

    def test_invalid_shape_raises(self) -> None:
        """1-D input raises ValueError."""
        with self.assertRaises(ValueError, msg="coeffs must be 2-D"):
            find_roots_batch(np.array([1.0, -1.0]))

    def test_degree_zero_batch(self) -> None:
        """Batch of degree-0 polynomials: all return 0 roots."""
        coeffs = np.array([[5.0], [3.0]], dtype=np.float64)
        _, counts = find_roots_batch(coeffs)
        self.assertEqual(counts[0], 0)
        self.assertEqual(counts[1], 0)

    def test_empty_batch(self) -> None:
        """Empty batch (0 polynomials) returns empty arrays without error."""
        coeffs = np.empty((0, 3), dtype=np.float64)
        roots, counts = find_roots_batch(coeffs)
        self.assertEqual(roots.shape, (0, 2))
        self.assertEqual(counts.shape, (0,))


class TestSolveMonotoneRootBatch(unittest.TestCase):
    """Tests for :func:`solve_monotone_root_batch` (public API)."""

    def test_mixed_roots(self) -> None:
        """Batch with some roots and some NaN."""
        coeffs = np.array(
            [
                [-1.0, 1.0],  # root at 0.5
                [1.0, 2.0],  # no root
                [0.0, 1.0],  # root at 0.0
            ],
            dtype=np.float64,
        )
        roots, found = solve_monotone_root_batch(coeffs)
        self.assertTrue(found[0])
        self.assertFalse(found[1])
        self.assertTrue(found[2])
        self.assertAlmostEqual(roots[0], 0.5, places=12)
        self.assertTrue(np.isnan(roots[1]))
        self.assertAlmostEqual(roots[2], 0.0, places=10)

    def test_single_polynomial(self) -> None:
        """Batch of one matches single-poly result."""
        c = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        root_single = solve_monotone_root(c)
        roots_batch, found = solve_monotone_root_batch(c.reshape(1, -1))
        self.assertTrue(found[0])
        self.assertAlmostEqual(roots_batch[0], root_single, places=12)

    def test_all_no_roots(self) -> None:
        """Batch where no polynomial has a root."""
        coeffs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        roots, found = solve_monotone_root_batch(coeffs)
        self.assertFalse(found[0])
        self.assertFalse(found[1])
        self.assertTrue(np.isnan(roots[0]))
        self.assertTrue(np.isnan(roots[1]))

    def test_empty_batch(self) -> None:
        """Empty batch (0 polynomials) returns empty arrays without error."""
        coeffs = np.empty((0, 3), dtype=np.float64)
        roots, found = solve_monotone_root_batch(coeffs)
        self.assertEqual(roots.shape, (0,))
        self.assertEqual(found.shape, (0,))


if __name__ == "__main__":
    unittest.main()
