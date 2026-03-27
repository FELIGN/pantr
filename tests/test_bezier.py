"""Tests for the Bezier class."""

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bezier import Bezier
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
from pantr.quad import PointsLattice

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bezier_1d(
    ctrl: list[float] | list[list[float]],
    is_rational: bool = False,
    dtype: type = np.float64,
) -> Bezier:
    """Create a 1D Bezier from control points."""
    cp: npt.NDArray[np.float32 | np.float64] = np.array(ctrl, dtype=dtype)
    return Bezier(cp, is_rational=is_rational)


# ---------------------------------------------------------------------------
# Init + properties
# ---------------------------------------------------------------------------


class TestBezierInit:
    """Test Bezier initialization."""

    def test_valid_initialization_1d_scalar(self) -> None:
        """Test 1D scalar Bezier."""
        b = _make_bezier_1d([1.0, 2.0, 3.0])
        assert b.dim == 1
        assert b.degree == (2,)
        assert b.rank == 1
        assert b.is_rational is False
        assert b.control_points.shape == (3, 1)
        assert b.dtype == np.float64

    def test_valid_initialization_1d_vector(self) -> None:
        """Test 1D vector Bezier."""
        b = _make_bezier_1d([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert b.dim == 1
        assert b.degree == (2,)
        assert b.rank == 2  # noqa: PLR2004
        assert b.control_points.shape == (3, 2)

    def test_valid_initialization_2d_scalar(self) -> None:
        """Test 2D scalar Bezier."""
        ctrl = np.array([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]])
        b = Bezier(ctrl)
        assert b.dim == 2  # noqa: PLR2004
        assert b.degree == (2, 1)
        assert b.rank == 1
        assert b.control_points.shape == (3, 2, 1)

    def test_valid_initialization_2d_vector(self) -> None:
        """Test 2D vector Bezier."""
        ctrl = np.arange(36, dtype=np.float64).reshape(3, 2, 6)
        b = Bezier(ctrl)
        assert b.dim == 2  # noqa: PLR2004
        assert b.degree == (2, 1)
        assert b.rank == 6  # noqa: PLR2004

    def test_valid_initialization_3d(self) -> None:
        """Test 3D Bezier."""
        ctrl = np.ones((3, 2, 4, 1), dtype=np.float64)
        b = Bezier(ctrl)
        assert b.dim == 3  # noqa: PLR2004
        assert b.degree == (2, 1, 3)

    def test_rational(self) -> None:
        """Test rational Bezier."""
        ctrl = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
        b = Bezier(ctrl, is_rational=True)
        assert b.is_rational is True
        assert b.rank == 2  # noqa: PLR2004
        assert b.control_points.shape == (3, 3)

    def test_1d_array_reshaped_to_scalar(self) -> None:
        """Test that 1D input is reshaped to scalar field."""
        b = Bezier(np.array([1.0, 2.0, 3.0]))
        assert b.dim == 1
        assert b.control_points.shape == (3, 1)

    def test_integer_control_points_cast_to_float64(self) -> None:
        """Test that integer control points are cast to float64."""
        b = Bezier(np.array([[1, 2], [3, 4], [5, 6]]))
        assert b.dtype == np.float64

    def test_invalid_rank_zero(self) -> None:
        """Test that rational Bezier with rank 0 raises."""
        ctrl = np.array([[1.0], [2.0], [3.0]])
        with pytest.raises(ValueError, match="rank"):
            Bezier(ctrl, is_rational=True)

    def test_float32(self) -> None:
        """Test float32 Bezier."""
        b = Bezier(np.array([[1.0, 2.0]], dtype=np.float32))
        assert b.dtype == np.float32


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


class TestBezierConversion:
    """Test Bezier to/from Bspline conversion."""

    def test_to_bspline_1d(self) -> None:
        """Test conversion to B-spline preserves Bezier knot structure."""
        b = _make_bezier_1d([1.0, 2.0, 3.0])
        bs = b.to_bspline()
        assert bs.space.has_Bezier_like_knots()
        assert bs.degree == (2,)
        np.testing.assert_array_equal(bs.space.spaces[0].knots, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    def test_to_bspline_2d(self) -> None:
        """Test 2D conversion."""
        ctrl = np.ones((3, 2, 1), dtype=np.float64)
        b = Bezier(ctrl)
        bs = b.to_bspline()
        assert bs.space.has_Bezier_like_knots()
        assert bs.degree == (2, 1)

    def test_from_bspline_valid(self) -> None:
        """Test conversion from Bezier-like B-spline."""
        b_orig = _make_bezier_1d([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        bs = b_orig.to_bspline()
        b_back = Bezier.from_bspline(bs)
        np.testing.assert_array_equal(b_back.control_points, b_orig.control_points)

    def test_from_bspline_invalid(self) -> None:
        """Test that non-Bezier B-spline raises."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        space = BsplineSpace([BsplineSpace1D(knots, 2)])
        cp = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        bs = Bspline(space, cp)
        with pytest.raises(ValueError, match="Bézier-like"):
            Bezier.from_bspline(bs)

    def test_roundtrip(self) -> None:
        """Test to_bspline -> from_bspline roundtrip."""
        b = _make_bezier_1d([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]], is_rational=True)
        b2 = Bezier.from_bspline(b.to_bspline())
        np.testing.assert_array_equal(b2.control_points, b.control_points)
        assert b2.is_rational == b.is_rational

    def test_to_bspline_copy_true(self) -> None:
        """Test that to_bspline with copy=True creates independent arrays."""
        b = _make_bezier_1d([1.0, 2.0, 3.0])
        bs = b.to_bspline(copy=True)
        assert not np.shares_memory(b.control_points, bs.control_points)

    def test_to_bspline_copy_false(self) -> None:
        """Test that to_bspline with copy=False shares the control point array."""
        b = _make_bezier_1d([1.0, 2.0, 3.0])
        bs = b.to_bspline(copy=False)
        assert np.shares_memory(b.control_points, bs.control_points)

    def test_from_bspline_copy_true(self) -> None:
        """Test that from_bspline with copy=True creates independent arrays."""
        b_orig = _make_bezier_1d([1.0, 2.0, 3.0])
        bs = b_orig.to_bspline(copy=False)
        b_back = Bezier.from_bspline(bs, copy=True)
        assert not np.shares_memory(bs.control_points, b_back.control_points)

    def test_from_bspline_copy_false(self) -> None:
        """Test that from_bspline with copy=False shares the control point array."""
        b_orig = _make_bezier_1d([1.0, 2.0, 3.0])
        bs = b_orig.to_bspline(copy=False)
        b_back = Bezier.from_bspline(bs, copy=False)
        assert np.shares_memory(bs.control_points, b_back.control_points)

    def test_to_bspline_default_copies(self) -> None:
        """Test that to_bspline copies by default."""
        b = _make_bezier_1d([1.0, 2.0, 3.0])
        bs = b.to_bspline()
        assert not np.shares_memory(b.control_points, bs.control_points)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------


class TestBezierEvaluate:
    """Test Bezier evaluate."""

    def test_linear_1d(self) -> None:
        """Test linear Bezier is exact line."""
        b = _make_bezier_1d([[0.0, 0.0], [1.0, 2.0]])
        pts = np.array([0.0, 0.5, 1.0])
        result = b.evaluate(pts)
        expected = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 2.0]])
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_quadratic_1d(self) -> None:
        """Test quadratic Bezier at midpoint."""
        # Quadratic Bezier: P0=(0,0), P1=(0.5,1), P2=(1,0)
        # At t=0.5: B(0.5) = 0.25*P0 + 0.5*P1 + 0.25*P2 = (0.5, 0.5)
        b = _make_bezier_1d([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]])
        pts = np.array([0.5])
        result = b.evaluate(pts)
        np.testing.assert_allclose(result, [[0.5, 0.5]], atol=1e-14)

    def test_scalar_1d(self) -> None:
        """Test scalar Bezier returns 1D array."""
        b = _make_bezier_1d([1.0, 3.0])
        pts = np.array([0.0, 0.5, 1.0])
        result = b.evaluate(pts)
        assert result.ndim == 1
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0], atol=1e-14)

    def test_2d_surface(self) -> None:
        """Test 2D Bezier surface evaluation."""
        # Bilinear: ctrl[i, j] = (i, j)
        ctrl = np.array(
            [
                [[0.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [1.0, 1.0]],
            ]
        )
        b = Bezier(ctrl)
        pts = np.array([[0.5, 0.5]], dtype=np.float64)
        result = b.evaluate(pts)
        np.testing.assert_allclose(result, [[0.5, 0.5]], atol=1e-14)

    def test_rational_quarter_circle(self) -> None:
        """Test rational quadratic Bezier for quarter circle."""
        w = 1.0 / np.sqrt(2.0)
        ctrl = np.array(
            [
                [1.0, 0.0, 1.0],
                [w, w, w],
                [0.0, 1.0, 1.0],
            ]
        )
        b = Bezier(ctrl, is_rational=True)
        pts = np.linspace(0.0, 1.0, 50, dtype=np.float64)
        result = b.evaluate(pts)
        # Points should lie on unit circle
        radii = np.sqrt(result[:, 0] ** 2 + result[:, 1] ** 2)
        np.testing.assert_allclose(radii, 1.0, atol=1e-12)

    def test_evaluate_with_out(self) -> None:
        """Test evaluate with pre-allocated output."""
        b = _make_bezier_1d([1.0, 3.0])
        pts = np.array([0.0, 0.5, 1.0])
        out = np.empty(3, dtype=np.float64)
        result = b.evaluate(pts, out=out)
        assert result is out
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0], atol=1e-14)

    def test_matches_bspline_evaluate(self) -> None:
        """Test that Bezier evaluate matches Bspline evaluate."""
        ctrl = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]], dtype=np.float64)
        b = Bezier(ctrl)
        bs = b.to_bspline()
        pts = np.linspace(0.0, 1.0, 20, dtype=np.float64)
        np.testing.assert_allclose(b.evaluate(pts), bs.evaluate(pts), atol=1e-14)

    def test_2d_lattice(self) -> None:
        """Test 2D Bezier evaluation on a lattice."""
        ctrl = np.array(
            [
                [[0.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [1.0, 1.0]],
            ]
        )
        b = Bezier(ctrl)
        pts_u = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        pts_v = np.array([0.0, 1.0], dtype=np.float64)
        lattice = PointsLattice([pts_u, pts_v])
        result = b.evaluate(lattice)
        assert result.shape == (3, 2, 2)
        np.testing.assert_allclose(result[1, 1], [0.5, 1.0], atol=1e-14)


# ---------------------------------------------------------------------------
# Evaluate derivatives
# ---------------------------------------------------------------------------


class TestBezierEvaluateDerivatives:
    """Test Bezier evaluate_derivatives."""

    def test_first_derivative_1d(self) -> None:
        """Test first derivative against finite differences."""
        b = _make_bezier_1d([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]])
        pts = np.array([0.25, 0.5, 0.75])
        h = 1e-7
        deriv = b.evaluate_derivatives(pts, 1)

        # Finite differences
        pts_p = np.clip(pts + h, 0, 1)
        pts_m = np.clip(pts - h, 0, 1)
        fd = (b.evaluate(pts_p) - b.evaluate(pts_m)) / (pts_p - pts_m)[:, None]
        np.testing.assert_allclose(deriv, fd, atol=1e-5)

    def test_second_derivative_1d(self) -> None:
        """Test second derivative of quadratic is constant."""
        # Quadratic Bezier has constant second derivative
        b = _make_bezier_1d([[0.0], [0.5], [1.0]])
        pts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        deriv2 = b.evaluate_derivatives(pts, 2)
        # Second derivative of B(t) = (1-t)^2*0 + 2t(1-t)*0.5 + t^2*1 = t
        # B'(t) = 1, B''(t) = 0
        np.testing.assert_allclose(deriv2, 0.0, atol=1e-12)

    def test_derivative_matches_bspline(self) -> None:
        """Test derivative evaluation matches B-spline."""
        ctrl = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]], dtype=np.float64)
        b = Bezier(ctrl)
        bs = b.to_bspline()
        pts = np.linspace(0.0, 1.0, 15, dtype=np.float64)
        np.testing.assert_allclose(
            b.evaluate_derivatives(pts, 1), bs.evaluate_derivatives(pts, 1), atol=1e-12
        )

    def test_2d_partial_derivatives(self) -> None:
        """Test 2D partial derivatives."""
        # Bilinear surface: f(u, v) = (u, v)
        ctrl = np.array(
            [
                [[0.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [1.0, 1.0]],
            ]
        )
        b = Bezier(ctrl)
        pts = np.array([[0.5, 0.5]], dtype=np.float64)
        # df/du = (1, 0), df/dv = (0, 1)
        du = b.evaluate_derivatives(pts, [1, 0])
        dv = b.evaluate_derivatives(pts, [0, 1])
        np.testing.assert_allclose(du, [[1.0, 0.0]], atol=1e-14)
        np.testing.assert_allclose(dv, [[0.0, 1.0]], atol=1e-14)

    def test_rational_derivative(self) -> None:
        """Test rational Bezier derivative against finite differences."""
        w = 1.0 / np.sqrt(2.0)
        ctrl = np.array([[1.0, 0.0, 1.0], [w, w, w], [0.0, 1.0, 1.0]])
        b = Bezier(ctrl, is_rational=True)
        pts = np.array([0.25, 0.5, 0.75])
        h = 1e-7
        deriv = b.evaluate_derivatives(pts, 1)
        pts_p = np.clip(pts + h, 0, 1)
        pts_m = np.clip(pts - h, 0, 1)
        fd = (b.evaluate(pts_p) - b.evaluate(pts_m)) / (pts_p - pts_m)[:, None]
        np.testing.assert_allclose(deriv, fd, atol=1e-5)


# ---------------------------------------------------------------------------
# Derivative (returns new Bezier)
# ---------------------------------------------------------------------------


class TestBezierDerivative:
    """Test Bezier derivative method."""

    def test_linear_to_constant(self) -> None:
        """Test derivative of linear Bezier is constant."""
        b = _make_bezier_1d([[0.0, 0.0], [1.0, 2.0]])
        d = b.derivative()
        assert d.degree == (0,)
        # Derivative: p * (P1 - P0) = 1 * (1, 2) = (1, 2)
        np.testing.assert_allclose(d.control_points, [[1.0, 2.0]], atol=1e-14)

    def test_quadratic_to_linear(self) -> None:
        """Test derivative of quadratic Bezier is linear."""
        b = _make_bezier_1d([[0.0], [1.0], [0.0]])
        d = b.derivative()
        assert d.degree == (1,)
        # Q0 = 2*(1-0) = 2, Q1 = 2*(0-1) = -2
        np.testing.assert_allclose(d.control_points, [[2.0], [-2.0]], atol=1e-14)

    def test_derivative_matches_evaluate_derivatives(self) -> None:
        """Test that derivative().evaluate matches evaluate_derivatives."""
        ctrl = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]], dtype=np.float64)
        b = Bezier(ctrl)
        pts = np.linspace(0.0, 1.0, 20, dtype=np.float64)
        from_deriv = b.derivative().evaluate(pts)
        from_eval_deriv = b.evaluate_derivatives(pts, 1)
        np.testing.assert_allclose(from_deriv, from_eval_deriv, atol=1e-12)

    def test_2d_direction0(self) -> None:
        """Test 2D partial derivative in direction 0."""
        ctrl = np.array(
            [
                [[0.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [1.0, 1.0]],
            ]
        )
        b = Bezier(ctrl)
        d = b.derivative(direction=0)
        assert d.degree == (0, 1)

    def test_2d_direction1(self) -> None:
        """Test 2D partial derivative in direction 1."""
        ctrl = np.array(
            [
                [[0.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [1.0, 1.0]],
            ]
        )
        b = Bezier(ctrl)
        d = b.derivative(direction=1)
        assert d.degree == (1, 0)

    def test_invalid_direction(self) -> None:
        """Test that invalid direction raises."""
        b = _make_bezier_1d([1.0, 2.0])
        with pytest.raises(ValueError, match="direction"):
            b.derivative(direction=1)

    def test_degree_0_raises(self) -> None:
        """Test that degree 0 raises."""
        b = _make_bezier_1d([1.0])
        with pytest.raises(ValueError, match="degree-0"):
            b.derivative()

    def test_rational_derivative(self) -> None:
        """Test rational Bezier derivative matches evaluate_derivatives."""
        w = 1.0 / np.sqrt(2.0)
        ctrl = np.array([[1.0, 0.0, 1.0], [w, w, w], [0.0, 1.0, 1.0]])
        b = Bezier(ctrl, is_rational=True)
        d = b.derivative()
        pts = np.linspace(0.01, 0.99, 20, dtype=np.float64)
        from_deriv = d.evaluate(pts)
        from_eval = b.evaluate_derivatives(pts, 1)
        np.testing.assert_allclose(from_deriv, from_eval, atol=1e-10)


# ---------------------------------------------------------------------------
# Derivative with keep_degree
# ---------------------------------------------------------------------------


class TestBezierDerivativeKeepDegree:
    """Test Bezier derivative with keep_degree=True."""

    def test_nonrational_1d_degree_preserved(self) -> None:
        """Degree is preserved when keep_degree=True."""
        b = _make_bezier_1d([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]])
        d = b.derivative(keep_degree=True)
        assert d.degree == b.degree

    def test_nonrational_1d_matches_derivative_then_elevate(self) -> None:
        """Fused result matches derivative + elevate_degree."""
        b = _make_bezier_1d([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]])
        d_fused = b.derivative(keep_degree=True)
        d_ref = b.derivative().elevate_degree(1)
        pts = np.linspace(0.0, 1.0, 30, dtype=np.float64)
        np.testing.assert_allclose(d_fused.evaluate(pts), d_ref.evaluate(pts), atol=1e-13)

    def test_nonrational_1d_matches_evaluate_derivatives(self) -> None:
        """Derivative values match evaluate_derivatives."""
        ctrl = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [4.0, 0.0]], dtype=np.float64)
        b = Bezier(ctrl)
        d = b.derivative(keep_degree=True)
        pts = np.linspace(0.0, 1.0, 30, dtype=np.float64)
        np.testing.assert_allclose(d.evaluate(pts), b.evaluate_derivatives(pts, 1), atol=1e-12)

    def test_nonrational_2d_direction0(self) -> None:
        """2D partial derivative in direction 0 preserves degree."""
        ctrl = np.random.default_rng(42).standard_normal((4, 3, 2))
        b = Bezier(ctrl)
        d = b.derivative(direction=0, keep_degree=True)
        assert d.degree == b.degree

    def test_nonrational_2d_direction1(self) -> None:
        """2D partial derivative in direction 1 preserves degree."""
        ctrl = np.random.default_rng(42).standard_normal((4, 3, 2))
        b = Bezier(ctrl)
        d = b.derivative(direction=1, keep_degree=True)
        assert d.degree == b.degree

    def test_nonrational_2d_matches_evaluate_derivatives(self) -> None:
        """2D keep_degree derivative matches evaluate_derivatives."""
        ctrl = np.random.default_rng(42).standard_normal((4, 3, 2))
        b = Bezier(ctrl)
        pts = np.random.default_rng(99).uniform(0.0, 1.0, (20, 2))
        for direction in range(2):
            d = b.derivative(direction=direction, keep_degree=True)
            orders = [0, 0]
            orders[direction] = 1
            np.testing.assert_allclose(
                d.evaluate(pts), b.evaluate_derivatives(pts, orders), atol=1e-11
            )

    def test_rational_1d_same_as_without_keep_degree(self) -> None:
        """Rational derivative with keep_degree matches without it.

        For rational Bézier of degree p, the derivative has degree 2p > p,
        so keep_degree does not further elevate the result.
        """
        w = 1.0 / np.sqrt(2.0)
        ctrl = np.array([[1.0, 0.0, 1.0], [w, w, w], [0.0, 1.0, 1.0]])
        b = Bezier(ctrl, is_rational=True)
        d_normal = b.derivative()
        d_keep = b.derivative(keep_degree=True)
        pts = np.linspace(0.01, 0.99, 20, dtype=np.float64)
        np.testing.assert_allclose(d_keep.evaluate(pts), d_normal.evaluate(pts), atol=1e-14)


# ---------------------------------------------------------------------------
# Degree elevation
# ---------------------------------------------------------------------------


class TestBezierElevateDegree:
    """Test Bezier degree elevation."""

    def test_elevate_1d(self) -> None:
        """Test degree elevation preserves curve values."""
        b = _make_bezier_1d([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]])
        b_elev = b.elevate_degree(1)
        assert b_elev.degree == (3,)
        pts = np.linspace(0.0, 1.0, 30, dtype=np.float64)
        np.testing.assert_allclose(b.evaluate(pts), b_elev.evaluate(pts), atol=1e-13)

    def test_elevate_by_2(self) -> None:
        """Test degree elevation by 2."""
        b = _make_bezier_1d([1.0, 3.0])
        b_elev = b.elevate_degree(2)
        assert b_elev.degree == (3,)
        pts = np.linspace(0.0, 1.0, 20, dtype=np.float64)
        np.testing.assert_allclose(b.evaluate(pts), b_elev.evaluate(pts), atol=1e-13)

    def test_elevate_2d_per_direction(self) -> None:
        """Test 2D degree elevation per direction."""
        ctrl = np.ones((3, 2, 1), dtype=np.float64)
        b = Bezier(ctrl)
        b_elev = b.elevate_degree([1, 2])
        assert b_elev.degree == (3, 3)

    def test_zero_increment_raises(self) -> None:
        """Test that zero increment raises."""
        b = _make_bezier_1d([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="(?i)at least one.*positive"):
            b.elevate_degree(0)

    def test_negative_increment_raises(self) -> None:
        """Test that negative increment raises."""
        b = _make_bezier_1d([1.0, 2.0])
        with pytest.raises(ValueError, match="non-negative"):
            b.elevate_degree(-1)

    def test_wrong_length_raises(self) -> None:
        """Test that wrong increment length raises."""
        b = _make_bezier_1d([1.0, 2.0])
        with pytest.raises(ValueError, match="must match dimension"):
            b.elevate_degree([1, 2])


# ---------------------------------------------------------------------------
# Multiply
# ---------------------------------------------------------------------------


class TestBezierMultiply:
    """Test Bezier multiply."""

    def test_multiply_1d_scalars(self) -> None:
        """Test product of two scalar Beziers."""
        # f(t) = t (linear), g(t) = 1-t (linear)
        # product = t(1-t), quadratic, max at 0.25
        f = _make_bezier_1d([0.0, 1.0])
        g = _make_bezier_1d([1.0, 0.0])
        h = f.multiply(g)
        assert h.degree == (2,)
        pts = np.linspace(0.0, 1.0, 30, dtype=np.float64)
        np.testing.assert_allclose(h.evaluate(pts), f.evaluate(pts) * g.evaluate(pts), atol=1e-14)

    def test_multiply_1d_vectors(self) -> None:
        """Test product of vector Beziers."""
        f = _make_bezier_1d([[1.0, 2.0], [3.0, 4.0]])
        g = _make_bezier_1d([[1.0, 1.0], [0.0, 2.0]])
        h = f.multiply(g)
        assert h.degree == (2,)
        pts = np.linspace(0.0, 1.0, 20, dtype=np.float64)
        np.testing.assert_allclose(h.evaluate(pts), f.evaluate(pts) * g.evaluate(pts), atol=1e-13)

    def test_multiply_different_degrees(self) -> None:
        """Test product with different degrees."""
        f = _make_bezier_1d([1.0, 2.0])  # degree 1
        g = _make_bezier_1d([1.0, 0.0, 1.0])  # degree 2
        h = f.multiply(g)
        assert h.degree == (3,)
        pts = np.linspace(0.0, 1.0, 20, dtype=np.float64)
        np.testing.assert_allclose(h.evaluate(pts), f.evaluate(pts) * g.evaluate(pts), atol=1e-13)

    def test_multiply_2d(self) -> None:
        """Test 2D Bezier product."""
        ctrl_f = np.ones((2, 2, 1), dtype=np.float64)
        ctrl_f[1, 1, 0] = 2.0
        ctrl_g = np.ones((2, 2, 1), dtype=np.float64) * 3.0
        f = Bezier(ctrl_f)
        g = Bezier(ctrl_g)
        h = f.multiply(g)
        assert h.degree == (2, 2)
        pts = np.array([[0.5, 0.5]], dtype=np.float64)
        np.testing.assert_allclose(h.evaluate(pts), f.evaluate(pts) * g.evaluate(pts), atol=1e-13)

    def test_dunder_mul(self) -> None:
        """Test __mul__ operator."""
        f = _make_bezier_1d([1.0, 2.0])
        g = _make_bezier_1d([3.0, 4.0])
        h = f * g
        pts = np.linspace(0.0, 1.0, 10, dtype=np.float64)
        np.testing.assert_allclose(h.evaluate(pts), f.evaluate(pts) * g.evaluate(pts), atol=1e-14)

    def test_different_dim_raises(self) -> None:
        """Test that different dimensions raise."""
        f = _make_bezier_1d([1.0, 2.0])
        g = Bezier(np.ones((2, 2, 1), dtype=np.float64))
        with pytest.raises(ValueError, match="dimension"):
            f.multiply(g)

    def test_different_dtype_raises(self) -> None:
        """Test that different dtypes raise."""
        f = Bezier(np.array([[1.0]], dtype=np.float64))
        g = Bezier(np.array([[1.0]], dtype=np.float32))
        with pytest.raises(ValueError, match="dtype"):
            f.multiply(g)

    def test_different_rank_raises(self) -> None:
        """Test that different ranks raise."""
        f = _make_bezier_1d([[1.0, 2.0], [3.0, 4.0]])
        g = _make_bezier_1d([1.0, 2.0])
        with pytest.raises(ValueError, match="rank"):
            f.multiply(g)

    def test_rational_product(self) -> None:
        """Test rational Bezier product."""
        ctrl_f = np.array([[1.0, 1.0], [2.0, 1.0]], dtype=np.float64)
        ctrl_g = np.array([[1.0, 1.0], [3.0, 2.0]], dtype=np.float64)
        f = Bezier(ctrl_f, is_rational=True)
        g = Bezier(ctrl_g, is_rational=True)
        h = f.multiply(g)
        assert h.is_rational
        pts = np.linspace(0.01, 0.99, 20, dtype=np.float64)
        np.testing.assert_allclose(h.evaluate(pts), f.evaluate(pts) * g.evaluate(pts), atol=1e-12)


# ---------------------------------------------------------------------------
# Resultant matrices
# ---------------------------------------------------------------------------


class TestSylvesterMatrix:
    """Test _sylvester_matrix for Bernstein polynomials."""

    def test_two_linears(self) -> None:
        """Resultant of two linear Bernstein polynomials.

        a(t) = 1 - t  (coeffs [1, 0])
        b(t) = t       (coeffs [0, 1])
        They share a root at t in (0,1)? No — a(1)=0, b(0)=0, no common root.
        Resultant should be nonzero.  For p=q=1 the Sylvester matrix is 2×2.
        """
        from pantr.bezier._resultant_matrices import _sylvester_matrix

        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        S = _sylvester_matrix(a, b)
        assert S.shape == (2, 2)
        # det != 0 means no common root.
        assert np.abs(np.linalg.det(S)) > 1e-14

    def test_common_root_gives_zero_det(self) -> None:
        """Two Bernstein polynomials sharing a root → singular Sylvester matrix.

        a(t) = t(1-t) in Bernstein form of degree 2: coeffs [0, 0.5, 0]
        Actually, t(1-t) = 2*B_{1,2}(t)*0.5... let's use a simpler approach.

        a(t) = t,  b(t) = t  both vanish at t=0 → common root → det = 0.
        Bernstein coeffs: a = [0, 1], b = [0, 1].
        """
        from pantr.bezier._resultant_matrices import _sylvester_matrix

        a = np.array([0.0, 1.0])
        b = np.array([0.0, 1.0])
        S = _sylvester_matrix(a, b)
        assert np.abs(np.linalg.det(S)) < 1e-14

    def test_different_degrees(self) -> None:
        """Sylvester matrix for polynomials of different degrees."""
        from pantr.bezier._resultant_matrices import _sylvester_matrix

        a = np.array([1.0, -1.0, 0.5])  # degree 2
        b = np.array([2.0, 3.0])  # degree 1
        S = _sylvester_matrix(a, b)
        assert S.shape == (3, 3)
        assert S.dtype == np.float64

    def test_out_parameter(self) -> None:
        """Test that the out parameter works."""
        from pantr.bezier._resultant_matrices import _sylvester_matrix

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0])
        out = np.zeros((3, 3), dtype=np.float64)
        result = _sylvester_matrix(a, b, out=out)
        assert result is out
        np.testing.assert_array_equal(result, _sylvester_matrix(a, b))

    def test_out_wrong_shape_raises(self) -> None:
        """Test that wrong out shape raises."""
        from pantr.bezier._resultant_matrices import _sylvester_matrix

        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        out = np.zeros((3, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="shape"):
            _sylvester_matrix(a, b, out=out)

    def test_non_1d_raises(self) -> None:
        """Test that 2-D input raises."""
        from pantr.bezier._resultant_matrices import _sylvester_matrix

        a = np.array([[1.0, 2.0]])
        b = np.array([3.0, 4.0])
        with pytest.raises(ValueError, match="1-D"):
            _sylvester_matrix(a, b)

    def test_degree_zero_raises(self) -> None:
        """Test that a constant polynomial (degree 0) raises."""
        from pantr.bezier._resultant_matrices import _sylvester_matrix

        a = np.array([1.0])  # degree 0
        b = np.array([2.0, 3.0])
        with pytest.raises(ValueError, match="at least 2"):
            _sylvester_matrix(a, b)

    def test_integer_input_raises(self) -> None:
        """Test that integer dtype raises."""
        from pantr.bezier._resultant_matrices import _sylvester_matrix

        a = np.array([1, 2])
        b = np.array([3, 4])
        with pytest.raises(ValueError, match="floating"):
            _sylvester_matrix(a, b)

    def test_float32(self) -> None:
        """Test that float32 inputs produce float32 output."""
        from pantr.bezier._resultant_matrices import _sylvester_matrix

        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        S = _sylvester_matrix(a, b)
        assert S.dtype == np.float32


class TestBezoutMatrix:
    """Test _bezout_matrix for Bernstein polynomials."""

    def test_symmetry(self) -> None:
        """Bezout matrix must be symmetric."""
        from pantr.bezier._resultant_matrices import _bezout_matrix

        rng = np.random.default_rng(42)
        a = rng.standard_normal(6)
        b = rng.standard_normal(6)
        B = _bezout_matrix(a, b)
        np.testing.assert_allclose(B, B.T, atol=1e-15)

    def test_degree_1(self) -> None:
        """Bezout matrix for degree-1 polynomials is 1×1."""
        from pantr.bezier._resultant_matrices import _bezout_matrix

        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        B = _bezout_matrix(a, b)
        assert B.shape == (1, 1)
        # B[0,0] = (a[1]*b[0] - a[0]*b[1]) * n/1 = (2*3 - 1*4) * 1 = 2
        np.testing.assert_allclose(B[0, 0], 2.0, atol=1e-15)

    def test_common_root_gives_zero_det(self) -> None:
        """Two polynomials sharing a root → singular Bezout matrix."""
        from pantr.bezier._resultant_matrices import _bezout_matrix

        # Both are t: coeffs [0, 1]
        a = np.array([0.0, 1.0])
        b = np.array([0.0, 1.0])
        B = _bezout_matrix(a, b)
        assert np.abs(np.linalg.det(B)) < 1e-14

    def test_sylvester_bezout_zero_det_agreement(self) -> None:
        """Both matrices agree on whether the resultant is zero.

        Construct polynomial pairs with and without a common root and verify
        that both matrices give zero / nonzero determinant consistently.
        """
        from pantr.bezier._resultant_matrices import _bezout_matrix, _sylvester_matrix

        # Pair with common root at t=0.5.
        # f(t) = t - 0.5 → Bernstein coeffs [-0.5, 0.5]
        # g(t) = 2t - 1  → Bernstein coeffs [-1.0, 1.0]  (same root, different scale)
        a = np.array([-0.5, 0.5])
        b = np.array([-1.0, 1.0])
        assert np.abs(np.linalg.det(_sylvester_matrix(a, b))) < 1e-14
        assert np.abs(np.linalg.det(_bezout_matrix(a, b))) < 1e-14

        # Pair without common root.
        a = np.array([-0.3, 0.7])
        b = np.array([-0.8, 0.2])
        assert np.abs(np.linalg.det(_sylvester_matrix(a, b))) > 1e-2
        assert np.abs(np.linalg.det(_bezout_matrix(a, b))) > 1e-2

    def test_bezout_det_equals_resultant(self) -> None:
        """Bezout determinant matches the classical resultant for small degrees.

        For two degree-1 Bernstein polynomials a(t)=a0(1-t)+a1·t and
        b(t)=b0(1-t)+b1·t, the resultant is a0·b1 - a1·b0.  The 1×1
        Bezout matrix entry is (a1·b0 - a0·b1)·n/1 = -(a0·b1-a1·b0).
        """
        from pantr.bezier._resultant_matrices import _bezout_matrix

        rng = np.random.default_rng(77)
        for _ in range(20):
            a = rng.standard_normal(2)
            b = rng.standard_normal(2)
            B = _bezout_matrix(a, b)
            expected = a[1] * b[0] - a[0] * b[1]
            np.testing.assert_allclose(B[0, 0], expected, atol=1e-14)

    def test_unequal_lengths_raises(self) -> None:
        """Test that arrays of different lengths raise."""
        from pantr.bezier._resultant_matrices import _bezout_matrix

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0])
        with pytest.raises(ValueError, match="equal length"):
            _bezout_matrix(a, b)

    def test_out_parameter(self) -> None:
        """Test that the out parameter works."""
        from pantr.bezier._resultant_matrices import _bezout_matrix

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        out = np.zeros((2, 2), dtype=np.float64)
        result = _bezout_matrix(a, b, out=out)
        assert result is out
        np.testing.assert_array_equal(result, _bezout_matrix(a, b))

    def test_degree_zero_raises(self) -> None:
        """Test that a constant polynomial (degree 0) raises."""
        from pantr.bezier._resultant_matrices import _bezout_matrix

        a = np.array([1.0])
        b = np.array([2.0])
        with pytest.raises(ValueError, match="at least 2"):
            _bezout_matrix(a, b)

    def test_resultant_via_power_basis(self) -> None:
        """Cross-check: resultant of (t - 0.3) and (t - 0.7) via Bezout.

        In Bernstein form on [0,1]:
            f(t) = t - 0.3 = -0.3·B₀(t) + 0.7·B₁(t)  → coeffs [-0.3, 0.7]
            g(t) = t - 0.7 = -0.7·B₀(t) + 0.3·B₁(t)  → coeffs [-0.7, 0.3]

        Resultant of f and g (as degree-1 polys) = f(0.7) = 0.4 (or g(0.3) = -0.4).
        The Bezout/Sylvester det captures this up to a binomial scaling factor.
        For degree p=q=1, the Bezout matrix is 1×1 and the Sylvester is 2×2.
        """
        from pantr.bezier._resultant_matrices import _bezout_matrix, _sylvester_matrix

        a = np.array([-0.3, 0.7])
        b = np.array([-0.7, 0.3])
        B = _bezout_matrix(a, b)
        S = _sylvester_matrix(a, b)

        # Both should give a nonzero determinant (no common root).
        assert np.abs(np.linalg.det(B)) > 0.1
        assert np.abs(np.linalg.det(S)) > 0.1

    def test_quadratic_with_known_common_root(self) -> None:
        """Two quadratics sharing root at t=0.5 → zero resultant.

        f(t) = (t - 0.5)(t - 0.2) in Bernstein degree-2 form.
        g(t) = (t - 0.5)(t - 0.8) in Bernstein degree-2 form.

        Bernstein basis for degree 2: B₀ = (1-t)², B₁ = 2t(1-t), B₂ = t².
        f(t) = t² - 0.7t + 0.1 = 0.1·B₀ + (0.5·(-0.7) + 0.1 + 0.5·1)·... easier to compute directly.
        """
        from pantr.bezier._resultant_matrices import _bezout_matrix

        # Convert power basis coefficients to Bernstein via the known relation.
        # f(t) = t^2 - 0.7t + 0.1
        # Bernstein coeffs: c0 = f(0) = 0.1, c2 = f(1) = 0.4, c1 = f(0) + f'(0)/(2) = 0.1 + (-0.7)/2 = -0.25
        # Wait: for degree 2, c0=f(0), c1 = f(0) + f'(0)/2, c2=f(1)
        a = np.array([0.1, -0.25, 0.4])

        # g(t) = t^2 - 1.3t + 0.4
        # c0 = 0.4, c1 = 0.4 + (-1.3)/2 = -0.25, c2 = g(1) = 0.1
        b = np.array([0.4, -0.25, 0.1])

        B = _bezout_matrix(a, b)
        assert np.abs(np.linalg.det(B)) < 1e-12
