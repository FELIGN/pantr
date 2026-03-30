"""Tests for Bézier and B-spline degree reduction."""

from __future__ import annotations

import numpy as np
import pytest

from pantr.bezier import Bezier
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D, create_uniform_periodic


def _make_bezier_1d(ctrl: list[list[float]], rational: bool = False) -> Bezier:
    """Create a 1D Bézier from a list of control points."""
    return Bezier(np.array(ctrl), is_rational=rational)


# ---------------------------------------------------------------------------
# Round-trip tests: elevate then reduce should recover the original
# ---------------------------------------------------------------------------


class TestBezierReduceDegreeRoundTrip:
    """Elevate by t then reduce by t should recover the original exactly."""

    def test_linear_elevate_1_reduce_1(self) -> None:
        """Linear Bézier → elevate by 1 → reduce by 1."""
        b = _make_bezier_1d([[0.0, 0.0], [1.0, 2.0]])
        reduced = b.elevate_degree(1).reduce_degree(1)
        np.testing.assert_allclose(reduced.control_points, b.control_points, atol=1e-14)

    def test_linear_elevate_3_reduce_3(self) -> None:
        """Linear Bézier → elevate by 3 → reduce by 3."""
        b = _make_bezier_1d([[0.0], [5.0]])
        reduced = b.elevate_degree(3).reduce_degree(3)
        np.testing.assert_allclose(reduced.control_points, b.control_points, atol=1e-13)

    def test_quadratic_elevate_2_reduce_2(self) -> None:
        """Quadratic Bézier → elevate by 2 → reduce by 2."""
        b = _make_bezier_1d([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]])
        reduced = b.elevate_degree(2).reduce_degree(2)
        np.testing.assert_allclose(reduced.control_points, b.control_points, atol=1e-13)

    def test_cubic_elevate_4_reduce_4(self) -> None:
        """Cubic Bézier → elevate by 4 → reduce by 4."""
        b = _make_bezier_1d([[0.0], [1.0], [2.0], [3.0]])
        reduced = b.elevate_degree(4).reduce_degree(4)
        np.testing.assert_allclose(reduced.control_points, b.control_points, atol=1e-12)

    def test_2d_surface_round_trip(self) -> None:
        """2D tensor-product Bézier (bilinear) → elevate → reduce."""
        ctrl = np.array(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[0.0, 1.0], [1.0, 2.0], [2.0, 1.0]],
            ]
        )
        b = Bezier(ctrl)
        reduced = b.elevate_degree((1, 2)).reduce_degree((1, 2))
        np.testing.assert_allclose(reduced.control_points, b.control_points, atol=1e-12)

    def test_2d_different_decrements(self) -> None:
        """2D Bézier with different elevations per direction."""
        rng = np.random.default_rng(42)
        ctrl = rng.random((3, 4, 2))  # degree (2, 3), rank 2
        b = Bezier(ctrl)
        reduced = b.elevate_degree((2, 1)).reduce_degree((2, 1))
        np.testing.assert_allclose(reduced.control_points, b.control_points, atol=1e-12)

    def test_rational_round_trip(self) -> None:
        """Rational Bézier: elevate then reduce preserves geometry."""
        ctrl_h = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0 / np.sqrt(2)], [0.0, 1.0, 1.0]])
        b = Bezier(ctrl_h, is_rational=True)
        reduced = b.elevate_degree(2).reduce_degree(2)
        np.testing.assert_allclose(reduced.control_points, b.control_points, atol=1e-13)
        assert reduced.is_rational


# ---------------------------------------------------------------------------
# Approximate reduction: reduce a genuine polynomial
# ---------------------------------------------------------------------------


class TestBezierReduceDegreeApproximate:
    """Reducing a polynomial that is NOT an elevated lower-degree is approximate."""

    def test_cubic_to_quadratic_geometry(self) -> None:
        """Reducing a true cubic to quadratic preserves geometry approximately."""
        b = _make_bezier_1d([[0.0, 0.0], [0.3, 1.0], [0.7, 1.0], [1.0, 0.0]])
        reduced = b.reduce_degree(1)
        assert reduced.degree == (2,)

        # Evaluate both at sample points and compare
        pts = np.linspace(0, 1, 50)
        vals_orig = b.evaluate(pts)
        vals_red = reduced.evaluate(pts)
        # The error should be reasonably small (not exact)
        max_err_tol = 0.1
        assert np.max(np.abs(vals_orig - vals_red)) < max_err_tol

    def test_endpoints_preserved(self) -> None:
        """After reduction, endpoints should be close to the original."""
        b = _make_bezier_1d([[0.0, 0.0], [0.3, 1.5], [0.7, -0.5], [1.0, 1.0]])
        reduced = b.reduce_degree(1)

        pts = np.array([0.0, 1.0])
        vals_orig = b.evaluate(pts)
        vals_red = reduced.evaluate(pts)
        np.testing.assert_allclose(vals_red, vals_orig, atol=0.5)

    def test_reduce_degree_result_type(self) -> None:
        """Reduced Bézier has correct degree and dtype."""
        b = _make_bezier_1d([[0.0], [1.0], [2.0], [3.0], [4.0]])
        assert b.degree == (4,)

        reduced = b.reduce_degree(2)
        assert reduced.degree == (2,)
        assert reduced.control_points.dtype == b.control_points.dtype

    def test_reduce_by_1_from_degree_1(self) -> None:
        """Reducing a linear Bézier by 1 gives degree 0 (constant)."""
        b = _make_bezier_1d([[0.0, 0.0], [2.0, 4.0]])
        reduced = b.reduce_degree(1)
        assert reduced.degree == (0,)
        # The constant is the least-squares fit: average of endpoints
        expected = np.array([[1.0, 2.0]])
        np.testing.assert_allclose(reduced.control_points, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestBezierReduceDegreeErrors:
    """Test that invalid inputs raise appropriate errors."""

    def test_decrement_exceeds_degree(self) -> None:
        """Decrement > degree should raise ValueError."""
        b = _make_bezier_1d([[0.0], [1.0], [2.0]])  # degree 2
        with pytest.raises(ValueError, match=r"exceeds current degree"):
            b.reduce_degree(3)

    def test_negative_decrement(self) -> None:
        """Negative decrement should raise ValueError."""
        b = _make_bezier_1d([[0.0], [1.0], [2.0]])
        with pytest.raises(ValueError, match=r"non-negative"):
            b.reduce_degree(-1)

    def test_all_zero_decrements(self) -> None:
        """All-zero decrements should raise ValueError."""
        b = _make_bezier_1d([[0.0], [1.0], [2.0]])
        with pytest.raises(ValueError, match=r"(?i)at least one"):
            b.reduce_degree(0)

    def test_wrong_length(self) -> None:
        """Wrong number of decrements should raise ValueError."""
        b = _make_bezier_1d([[0.0], [1.0], [2.0]])
        with pytest.raises(ValueError, match=r"must match dimension"):
            b.reduce_degree((1, 1))

    def test_decrement_exceeds_per_direction(self) -> None:
        """Per-direction decrement exceeding degree should raise."""
        ctrl = np.zeros((2, 3, 1))  # degree (1, 2)
        b = Bezier(ctrl)
        with pytest.raises(ValueError, match=r"exceeds current degree"):
            b.reduce_degree((2, 0))

    def test_reduce_degree_0(self) -> None:
        """Reducing a degree-0 Bézier should raise."""
        b = Bezier(np.array([[42.0]]))
        with pytest.raises(ValueError, match=r"exceeds current degree"):
            b.reduce_degree(1)


# ---------------------------------------------------------------------------
# Minimize degree
# ---------------------------------------------------------------------------


class TestBezierMinimizeDegree:
    """Test Bezier.minimize_degree."""

    def test_constant_reduces_to_degree_0(self) -> None:
        """A constant elevated to degree 2 should reduce back to degree 0."""
        b = _make_bezier_1d([[3.0], [3.0], [3.0]])
        b_min = b.minimize_degree()
        assert b_min.degree[0] < b.degree[0]
        pts = np.linspace(0.0, 1.0, 10, dtype=np.float64)
        np.testing.assert_allclose(b_min.evaluate(pts), b.evaluate(pts), atol=1e-12)

    def test_true_quadratic_not_reduced(self) -> None:
        """A genuine quadratic should not be reduced."""
        b = _make_bezier_1d([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]])
        b_min = b.minimize_degree()
        assert b_min.degree == b.degree

    def test_linear_elevated_reduces(self) -> None:
        """A linear elevated to degree 3 should reduce back."""
        b = _make_bezier_1d([[1.0], [3.0]])
        b_elev = b.elevate_degree(2)
        assert b_elev.degree == (3,)
        b_min = b_elev.minimize_degree()
        assert b_min.degree[0] < b_elev.degree[0]
        pts = np.linspace(0.0, 1.0, 10, dtype=np.float64)
        np.testing.assert_allclose(b_min.evaluate(pts), b.evaluate(pts), atol=1e-12)

    def test_2d_constant_in_one_direction(self) -> None:
        """A 2D polynomial constant in one direction reduces along it."""
        ctrl = np.array([[[0.0], [1.0], [0.0]], [[0.0], [1.0], [0.0]]])  # (2, 3, 1)
        b = Bezier(ctrl)
        assert b.degree == (1, 2)
        b_min = b.minimize_degree()
        assert b_min.degree[0] < b.degree[0]
        assert b_min.degree[1] == b.degree[1]

    def test_vector_valued(self) -> None:
        """Vector-valued Bezier: all components checked together."""
        # Linear in both components, elevated to degree 2
        b = Bezier(np.array([[0.0, 0.0], [1.0, 2.0]]))
        b_elev = b.elevate_degree(1)
        assert b_elev.degree == (2,)
        b_min = b_elev.minimize_degree()
        assert b_min.degree[0] < b_elev.degree[0]
        pts = np.linspace(0.0, 1.0, 10, dtype=np.float64)
        np.testing.assert_allclose(b_min.evaluate(pts), b.evaluate(pts), atol=1e-12)


# ---------------------------------------------------------------------------
# Float32 support
# ---------------------------------------------------------------------------


class TestBezierReduceDegreeFloat32:
    """Test that float32 inputs produce float32 outputs."""

    def test_float32_round_trip(self) -> None:
        """Float32 elevation + reduction round-trip."""
        ctrl = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float32)
        b = Bezier(ctrl)
        reduced = b.elevate_degree(2).reduce_degree(2)
        assert reduced.control_points.dtype == np.float32
        np.testing.assert_allclose(reduced.control_points, b.control_points, atol=1e-5)


# ===========================================================================
# B-spline degree reduction
# ===========================================================================


def _make_bspline_1d(knots: list[float], degree: int, ctrl: list[list[float]]) -> Bspline:
    """Create a simple 1D open B-spline."""
    space = BsplineSpace([BsplineSpace1D(np.array(knots), degree)])
    return Bspline(space, np.array(ctrl))


class TestBsplineReduceDegreeRoundTrip:
    """Elevate by t then reduce by t should recover the original geometry."""

    def test_single_segment_linear(self) -> None:
        """Single-segment linear B-spline → elevate 2 → reduce 2."""
        bsp = _make_bspline_1d([0, 0, 1, 1], 1, [[0.0], [1.0]])
        reduced = bsp.elevate_degree(2).reduce_degree(2)
        pts = np.linspace(0, 1, 20)
        np.testing.assert_allclose(bsp.evaluate(pts), reduced.evaluate(pts), atol=1e-13)

    def test_multi_segment_quadratic(self) -> None:
        """Quadratic B-spline with interior knot → elevate 2 → reduce 2."""
        bsp = _make_bspline_1d([0, 0, 0, 0.5, 1, 1, 1], 2, [[0.0], [1.0], [0.0], [1.0]])
        reduced = bsp.elevate_degree(2).reduce_degree(2)
        pts = np.linspace(0, 1, 30)
        np.testing.assert_allclose(bsp.evaluate(pts), reduced.evaluate(pts), atol=1e-12)

    def test_multi_segment_cubic(self) -> None:
        """Cubic B-spline with multiple interior knots."""
        knots = [0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1]
        rng = np.random.default_rng(42)
        ctrl = rng.random((7, 2))
        bsp = _make_bspline_1d(knots, 3, ctrl.tolist())
        reduced = bsp.elevate_degree(1).reduce_degree(1)
        pts = np.linspace(0, 1, 50)
        np.testing.assert_allclose(bsp.evaluate(pts), reduced.evaluate(pts), atol=1e-12)

    def test_2d_surface(self) -> None:
        """2D B-spline surface → elevate → reduce."""
        knots1 = np.array([0.0, 0.0, 1.0, 1.0])
        knots2 = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        space = BsplineSpace([BsplineSpace1D(knots1, 1), BsplineSpace1D(knots2, 2)])
        rng = np.random.default_rng(42)
        ctrl = rng.random((2, 3, 2))
        bsp = Bspline(space, ctrl)

        reduced = bsp.elevate_degree([1, 1]).reduce_degree([1, 1])

        pts = rng.random((20, 2))
        np.testing.assert_allclose(bsp.evaluate(pts), reduced.evaluate(pts), atol=1e-12)

    def test_rational(self) -> None:
        """Rational B-spline (NURBS): elevate then reduce."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        ctrl_h = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0 / np.sqrt(2)], [0.0, 1.0, 1.0]])
        space = BsplineSpace([BsplineSpace1D(knots, 2)])
        bsp = Bspline(space, ctrl_h, is_rational=True)

        reduced = bsp.elevate_degree(1).reduce_degree(1)
        assert reduced.is_rational

        pts = np.linspace(0, 1, 20)
        np.testing.assert_allclose(bsp.evaluate(pts), reduced.evaluate(pts), atol=1e-12)


class TestBsplineReduceDegreePeriodic:
    """Test degree reduction for periodic B-splines."""

    @pytest.mark.parametrize(
        "degree,continuity,dec",
        [
            (2, None, 1),
            (3, None, 1),
            (3, None, 2),
            (3, 1, 1),
        ],
    )
    def test_periodic_preserves_geometry(
        self, degree: int, continuity: int | None, dec: int
    ) -> None:
        """Elevate then reduce a periodic B-spline preserves geometry."""
        knots = create_uniform_periodic(num_intervals=4, degree=degree, continuity=continuity)
        space = BsplineSpace([BsplineSpace1D(knots, degree, periodic=True)])
        rng = np.random.default_rng(42)
        ctrl = rng.random((space.num_total_basis, 2))
        bsp = Bspline(space, ctrl)

        reduced = bsp.elevate_degree(dec).reduce_degree(dec)

        assert reduced.space.spaces[0].periodic
        assert reduced.degree == (degree,)

        pts = np.linspace(0.01, 0.99, 50)
        orig = bsp.to_open_bspline().evaluate(pts)
        red = reduced.to_open_bspline().evaluate(pts)
        np.testing.assert_allclose(orig, red, atol=1e-11)

    def test_mixed_periodic_open_2d(self) -> None:
        """2D mixed periodic/open B-spline: elevate then reduce."""
        knots_per = create_uniform_periodic(num_intervals=4, degree=2)
        knots_open = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
        space = BsplineSpace(
            [
                BsplineSpace1D(knots_per, 2, periodic=True),
                BsplineSpace1D(knots_open, 2),
            ]
        )
        rng = np.random.default_rng(42)
        ctrl = rng.random((*space.num_basis, 2))
        bsp = Bspline(space, ctrl)

        reduced = bsp.elevate_degree([1, 1]).reduce_degree([1, 1])

        assert reduced.space.spaces[0].periodic
        assert not reduced.space.spaces[1].periodic
        assert reduced.degree == (2, 2)

        pts = rng.random((30, 2))
        pts[:, 0] = pts[:, 0] * 0.98 + 0.01
        orig = bsp.to_open_bspline().evaluate(pts)
        red = reduced.to_open_bspline().evaluate(pts)
        np.testing.assert_allclose(orig, red, atol=1e-11)


class TestBsplineReduceDegreeErrors:
    """Test that invalid inputs raise appropriate errors."""

    def test_decrement_exceeds_degree(self) -> None:
        """Decrement > degree should raise ValueError."""
        bsp = _make_bspline_1d([0, 0, 1, 1], 1, [[0.0], [1.0]])
        with pytest.raises(ValueError, match=r"exceeds current degree"):
            bsp.reduce_degree(2)

    def test_negative_decrement(self) -> None:
        """Negative decrement should raise ValueError."""
        bsp = _make_bspline_1d([0, 0, 1, 1], 1, [[0.0], [1.0]])
        with pytest.raises(ValueError, match=r"non-negative"):
            bsp.reduce_degree(-1)

    def test_all_zero_decrements(self) -> None:
        """All-zero decrements should raise ValueError."""
        bsp = _make_bspline_1d([0, 0, 1, 1], 1, [[0.0], [1.0]])
        with pytest.raises(ValueError, match=r"(?i)at least one"):
            bsp.reduce_degree(0)

    def test_wrong_length(self) -> None:
        """Wrong number of decrements should raise ValueError."""
        bsp = _make_bspline_1d([0, 0, 1, 1], 1, [[0.0], [1.0]])
        with pytest.raises(ValueError, match=r"must match dimension"):
            bsp.reduce_degree((1, 1))
