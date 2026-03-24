"""Tests for Bézier auto degree reduction."""

from __future__ import annotations

import numpy as np
import pytest

from pantr.bezier import Bezier

# ---------------------------------------------------------------------------
# 1D tests
# ---------------------------------------------------------------------------


class TestAutoReduceDegree1D:
    """Auto-reduce for univariate Bézier curves."""

    def test_constant_elevated_to_high_degree(self) -> None:
        """A constant elevated to degree 5 should reduce back to degree 0."""
        b = Bezier(np.array([[3.0]] * 6))  # degree 5, all coeffs identical
        reduced = b.auto_reduce_degree()
        assert reduced.degree == (0,)
        np.testing.assert_allclose(reduced.control_points, [[3.0]], atol=1e-12)

    def test_linear_elevated_to_quartic(self) -> None:
        """A linear polynomial elevated to degree 4 should reduce to degree 1."""
        original = Bezier(np.array([[0.0, 0.0], [1.0, 2.0]]))
        elevated = original.elevate_degree(3)
        assert elevated.degree == (4,)
        reduced = elevated.auto_reduce_degree()
        assert reduced.degree == (1,)
        np.testing.assert_allclose(reduced.control_points, original.control_points, atol=1e-12)

    def test_quadratic_elevated(self) -> None:
        """A quadratic elevated to degree 6 should reduce back to degree 2."""
        original = Bezier(np.array([[0.0], [1.0], [0.0]]))
        elevated = original.elevate_degree(4)
        assert elevated.degree == (6,)
        reduced = elevated.auto_reduce_degree()
        assert reduced.degree == (2,)
        np.testing.assert_allclose(reduced.control_points, original.control_points, atol=1e-11)

    def test_genuine_cubic_not_reduced(self) -> None:
        """A genuine cubic should not reduce (it's not an elevated lower-degree)."""
        b = Bezier(np.array([[0.0, 0.0], [0.3, 1.0], [0.7, -1.0], [1.0, 0.0]]))
        reduced = b.auto_reduce_degree()
        assert reduced.degree == b.degree
        np.testing.assert_array_equal(reduced.control_points, b.control_points)
        assert reduced is not b  # always returns a copy

    def test_returns_copy_when_no_reduction(self) -> None:
        """When no reduction is possible, a copy is returned."""
        b = Bezier(np.array([[0.0], [1.0]]))  # already linear
        reduced = b.auto_reduce_degree()
        assert reduced is not b
        np.testing.assert_array_equal(reduced.control_points, b.control_points)

    def test_degree_0_unchanged(self) -> None:
        """A degree-0 Bézier (constant) cannot be reduced further."""
        b = Bezier(np.array([[42.0]]))
        reduced = b.auto_reduce_degree()
        assert reduced is not b
        assert reduced.degree == (0,)
        np.testing.assert_array_equal(reduced.control_points, b.control_points)


# ---------------------------------------------------------------------------
# Multi-dimensional tests
# ---------------------------------------------------------------------------


class TestAutoReduceDegreeMultiDim:
    """Auto-reduce for multivariate tensor-product Bézier."""

    def test_bilinear_elevated_both_dirs(self) -> None:
        """A bilinear surface elevated to (3, 4) should reduce to (1, 1)."""
        ctrl = np.array(
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 1.0], [1.0, 1.0]],
            ]
        )
        original = Bezier(ctrl)
        elevated = original.elevate_degree((2, 3))
        assert elevated.degree == (3, 4)
        reduced = elevated.auto_reduce_degree()
        assert reduced.degree == (1, 1)
        np.testing.assert_allclose(reduced.control_points, original.control_points, atol=1e-11)

    def test_mixed_reducible_directions(self) -> None:
        """Only the direction with elevated degree should be reduced."""
        # Degree (1, 2) surface: linear in dir 0, quadratic in dir 1.
        rng = np.random.default_rng(123)
        ctrl = rng.random((2, 3, 2))  # degree (1, 2), rank 2
        original = Bezier(ctrl)
        # Elevate only direction 0 by 3.
        elevated = original.elevate_degree((3, 0))
        assert elevated.degree == (4, 2)
        reduced = elevated.auto_reduce_degree()
        assert reduced.degree == (1, 2)
        np.testing.assert_allclose(reduced.control_points, original.control_points, atol=1e-11)

    def test_3d_volume_elevated(self) -> None:
        """A trilinear volume elevated to (3, 3, 3) reduces to (1, 1, 1)."""
        ctrl = np.zeros((2, 2, 2, 3))
        ctrl[0, 0, 0] = [0, 0, 0]
        ctrl[1, 0, 0] = [1, 0, 0]
        ctrl[0, 1, 0] = [0, 1, 0]
        ctrl[1, 1, 0] = [1, 1, 0]
        ctrl[0, 0, 1] = [0, 0, 1]
        ctrl[1, 0, 1] = [1, 0, 1]
        ctrl[0, 1, 1] = [0, 1, 1]
        ctrl[1, 1, 1] = [1, 1, 1]
        original = Bezier(ctrl)
        elevated = original.elevate_degree((2, 2, 2))
        assert elevated.degree == (3, 3, 3)
        reduced = elevated.auto_reduce_degree()
        assert reduced.degree == (1, 1, 1)
        np.testing.assert_allclose(reduced.control_points, original.control_points, atol=1e-10)

    def test_no_reduction_possible_2d(self) -> None:
        """A genuine (2, 2) surface should not be reduced."""
        rng = np.random.default_rng(99)
        ctrl = rng.random((3, 3, 2))
        b = Bezier(ctrl)
        reduced = b.auto_reduce_degree()
        assert reduced.degree == (2, 2)
        assert reduced is not b
        np.testing.assert_array_equal(reduced.control_points, b.control_points)


# ---------------------------------------------------------------------------
# Tolerance tests
# ---------------------------------------------------------------------------


class TestAutoReduceDegreeTolerance:
    """Tests for tolerance handling."""

    def test_tight_tolerance_prevents_reduction(self) -> None:
        """With a very tight tolerance, even elevated polynomials may not reduce."""
        # A cubic elevated by 1 — the round-trip is exact, should still reduce.
        original = Bezier(np.array([[0.0], [1.0], [0.0], [1.0]]))
        elevated = original.elevate_degree(1)
        # Even with tight tolerance, exact elevations should be detectable.
        reduced = elevated.auto_reduce_degree(tol=1e-14)
        assert reduced.degree == (3,)

    def test_loose_tolerance_allows_more_reduction(self) -> None:
        """With a loose tolerance, approximate reductions are accepted."""
        # A genuine cubic — normally not reduced. With loose tol it might be.
        b = Bezier(np.array([[0.0], [0.3], [0.7], [1.0]]))
        reduced = b.auto_reduce_degree(tol=0.5)
        assert reduced.degree[0] < b.degree[0]

    def test_custom_tolerance(self) -> None:
        """Auto-reduce with explicit tolerance value."""
        original = Bezier(np.array([[1.0, 2.0], [3.0, 4.0]]))
        elevated = original.elevate_degree(5)
        reduced = elevated.auto_reduce_degree(tol=1e-10)
        assert reduced.degree == (1,)
        np.testing.assert_allclose(reduced.control_points, original.control_points, atol=1e-10)


# ---------------------------------------------------------------------------
# dtype tests
# ---------------------------------------------------------------------------


class TestAutoReduceDegreeDtype:
    """Dtype handling."""

    def test_float32(self) -> None:
        """Auto-reduce works with float32 control points."""
        ctrl = np.array([[0.0], [1.0]], dtype=np.float32)
        original = Bezier(ctrl)
        elevated = original.elevate_degree(3)
        reduced = elevated.auto_reduce_degree()
        assert reduced.degree == (1,)
        assert reduced.control_points.dtype == np.float32
        np.testing.assert_allclose(reduced.control_points, original.control_points, atol=1e-4)

    def test_float64(self) -> None:
        """Auto-reduce with float64 control points (default)."""
        ctrl = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 0.0]], dtype=np.float64)
        original = Bezier(ctrl)
        elevated = original.elevate_degree(4)
        reduced = elevated.auto_reduce_degree()
        assert reduced.degree == (2,)
        assert reduced.control_points.dtype == np.float64
        np.testing.assert_allclose(reduced.control_points, original.control_points, atol=1e-12)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestAutoReduceDegreeErrors:
    """Error cases."""

    def test_rational_raises(self) -> None:
        """Rational Bézier should raise TypeError."""
        ctrl = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        b = Bezier(ctrl, is_rational=True)
        with pytest.raises(TypeError, match="non-rational"):
            b.auto_reduce_degree()

    def test_negative_tolerance_raises(self) -> None:
        """Negative tolerance should raise ValueError."""
        b = Bezier(np.array([[0.0], [1.0]]))
        with pytest.raises(ValueError, match="positive"):
            b.auto_reduce_degree(tol=-1.0)

    def test_zero_tolerance_raises(self) -> None:
        """Zero tolerance should raise ValueError."""
        b = Bezier(np.array([[0.0], [1.0]]))
        with pytest.raises(ValueError, match="positive"):
            b.auto_reduce_degree(tol=0.0)
