"""Tests for Bezier.interpolate and the Bernstein interpolation pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.testing as nptest
import numpy.typing as npt
import pytest

from pantr.bezier import Bezier

# ---------------------------------------------------------------------------
# 1D scalar interpolation
# ---------------------------------------------------------------------------


class TestInterpolate1DScalar:
    """Tests for 1D scalar-valued interpolation."""

    def test_linear(self) -> None:
        """Interpolating a linear function with 2 points recovers it exactly."""
        b = Bezier.interpolate(lambda x: 2.0 * x + 1.0, 2)
        assert b.degree == (1,)
        assert b.rank == 1
        pts = np.array([0.0, 0.5, 1.0])
        vals = b.evaluate(pts)
        nptest.assert_allclose(vals, [1.0, 2.0, 3.0], atol=1e-14)

    def test_quadratic(self) -> None:
        """Interpolating x^2 with 3 points recovers it exactly."""
        b = Bezier.interpolate(lambda x: x**2, 3)
        assert b.degree == (2,)
        pts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        expected = pts**2
        nptest.assert_allclose(b.evaluate(pts), expected, atol=1e-13)

    def test_cubic(self) -> None:
        """Interpolating x^3 with 4 points recovers it exactly."""
        b = Bezier.interpolate(lambda x: x**3, 4)
        assert b.degree == (3,)
        pts = np.linspace(0, 1, 10)
        nptest.assert_allclose(b.evaluate(pts), pts**3, atol=1e-12)

    def test_single_point(self) -> None:
        """A single interpolation point gives a degree-0 (constant) Bezier."""
        b = Bezier.interpolate(lambda x: np.full_like(x, 7.0), 1)
        assert b.degree == (0,)
        nptest.assert_allclose(b.evaluate(np.array([0.5])), [7.0], atol=1e-14)


# ---------------------------------------------------------------------------
# 1D vector-valued interpolation
# ---------------------------------------------------------------------------


class TestInterpolate1DVector:
    """Tests for 1D vector-valued interpolation (parametric curves)."""

    def test_circle_arc(self) -> None:
        """Interpolate a quarter-circle parametric curve."""

        def quarter_circle(t: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
            theta = t * (np.pi / 2.0)
            return np.stack([np.cos(theta), np.sin(theta)], axis=-1)

        b = Bezier.interpolate(quarter_circle, 8)
        assert b.degree == (7,)
        nptest.assert_equal(b.rank, 2)

        # Evaluate at midpoint
        pts = np.array([0.5])
        result = b.evaluate(pts)
        theta = 0.5 * np.pi / 2.0
        nptest.assert_allclose(result, [[np.cos(theta), np.sin(theta)]], atol=1e-6)

    def test_linear_curve(self) -> None:
        """A linear vector-valued function is recovered exactly."""

        def line(t: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
            return np.stack([t, 2.0 * t + 1.0, -t + 3.0], axis=-1)

        b = Bezier.interpolate(line, 2)
        assert b.degree == (1,)
        nptest.assert_equal(b.rank, 3)
        pts = np.array([0.0, 0.5, 1.0])
        expected = np.array([[0.0, 1.0, 3.0], [0.5, 2.0, 2.5], [1.0, 3.0, 2.0]])
        nptest.assert_allclose(b.evaluate(pts), expected, atol=1e-14)


# ---------------------------------------------------------------------------
# 2D scalar interpolation
# ---------------------------------------------------------------------------


class TestInterpolate2DScalar:
    """Tests for 2D scalar-valued interpolation."""

    def test_bilinear(self) -> None:
        """Interpolating a bilinear function with (2,2) points recovers it."""
        b = Bezier.interpolate(lambda x, y: x + y, [2, 2])
        assert b.degree == (1, 1)
        assert b.rank == 1
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
        expected = pts[:, 0] + pts[:, 1]
        nptest.assert_allclose(b.evaluate(pts), expected, atol=1e-13)

    def test_biquadratic(self) -> None:
        """Interpolating x^2 + y^2 with (3,3) points."""
        b = Bezier.interpolate(lambda x, y: x**2 + y**2, [3, 3])
        assert b.degree == (2, 2)
        pts = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
        expected = pts[:, 0] ** 2 + pts[:, 1] ** 2
        nptest.assert_allclose(b.evaluate(pts), expected, atol=1e-12)


# ---------------------------------------------------------------------------
# 2D vector-valued interpolation
# ---------------------------------------------------------------------------


class TestInterpolate2DVector:
    """Tests for 2D vector-valued interpolation (parametric surfaces)."""

    def test_planar_surface(self) -> None:
        """A linear vector-valued 2D function is recovered exactly."""

        def plane(
            x: npt.NDArray[np.floating[Any]], y: npt.NDArray[np.floating[Any]]
        ) -> npt.NDArray[np.floating[Any]]:
            return np.stack([x, y, x + y], axis=-1)

        b = Bezier.interpolate(plane, [2, 2])
        assert b.degree == (1, 1)
        nptest.assert_equal(b.rank, 3)
        pts = np.array([[0.5, 0.5], [0.0, 1.0], [1.0, 0.0]])
        expected = np.array([[0.5, 0.5, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
        nptest.assert_allclose(b.evaluate(pts), expected, atol=1e-13)


# ---------------------------------------------------------------------------
# Node selection
# ---------------------------------------------------------------------------


class TestNodeSelection:
    """Tests for different node selection strategies."""

    def test_chebyshev_default(self) -> None:
        """Default (Chebyshev) nodes recover a quadratic exactly."""
        b = Bezier.interpolate(lambda x: x**2, 3, nodes=None)
        pts = np.array([0.0, 0.5, 1.0])
        nptest.assert_allclose(b.evaluate(pts), pts**2, atol=1e-13)

    def test_chebyshev_explicit(self) -> None:
        """Explicitly requesting 'chebyshev' is the same as default."""
        b = Bezier.interpolate(lambda x: x**2, 3, nodes="chebyshev")
        pts = np.array([0.0, 0.5, 1.0])
        nptest.assert_allclose(b.evaluate(pts), pts**2, atol=1e-13)

    def test_uniform_nodes(self) -> None:
        """Uniform nodes can also recover polynomials (less stable for high degree)."""
        b = Bezier.interpolate(lambda x: x**2, 3, nodes="uniform")
        pts = np.array([0.0, 0.5, 1.0])
        nptest.assert_allclose(b.evaluate(pts), pts**2, atol=1e-12)

    def test_custom_nodes_array(self) -> None:
        """User-provided custom nodes as a single array."""
        custom = np.array([0.0, 0.5, 1.0])
        b = Bezier.interpolate(lambda x: x**2, 3, nodes=custom)
        pts = np.array([0.0, 0.5, 1.0])
        nptest.assert_allclose(b.evaluate(pts), pts**2, atol=1e-12)

    def test_custom_nodes_sequence(self) -> None:
        """User-provided custom nodes as a sequence of arrays (2D)."""
        nodes_x = np.array([0.0, 0.5, 1.0])
        nodes_y = np.array([0.0, 1.0])
        b = Bezier.interpolate(lambda x, y: x + y, [3, 2], nodes=[nodes_x, nodes_y])
        assert b.degree == (2, 1)
        pts = np.array([[0.5, 0.5]])
        nptest.assert_allclose(b.evaluate(pts), [1.0], atol=1e-12)


# ---------------------------------------------------------------------------
# dtype handling
# ---------------------------------------------------------------------------


class TestDtype:
    """Tests for dtype propagation."""

    def test_float64_default(self) -> None:
        """Default dtype is float64."""
        b = Bezier.interpolate(lambda x: x, 2)
        assert b.dtype == np.float64

    def test_float32(self) -> None:
        """Requesting float32 propagates through."""
        b = Bezier.interpolate(lambda x: x, 2, dtype=np.float32)
        assert b.dtype == np.float32


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestInterpolateValidation:
    """Input validation tests."""

    def test_n_pts_too_small(self) -> None:
        """n_pts < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_pts.*>= 1"):
            Bezier.interpolate(lambda x: x, 0)

    def test_non_floating_dtype(self) -> None:
        """Non-floating dtype raises ValueError."""
        with pytest.raises(ValueError, match="floating"):
            Bezier.interpolate(lambda x: x, 3, dtype=np.int32)

    def test_mismatched_nodes_n_pts(self) -> None:
        """Custom nodes with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="does not match"):
            Bezier.interpolate(lambda x: x, 3, nodes=np.array([0.0, 1.0]))

    def test_wrong_number_of_node_arrays(self) -> None:
        """Wrong number of node arrays for 2D raises ValueError."""
        with pytest.raises(ValueError, match="Expected 2"):
            Bezier.interpolate(
                lambda x, y: x + y,
                [3, 3],
                nodes=[np.array([0.0, 0.5, 1.0])],
            )

    def test_bad_function_output_shape(self) -> None:
        """Function returning wrong shape raises ValueError."""
        with pytest.raises(ValueError, match="Function returned shape"):
            Bezier.interpolate(lambda x: np.ones((2, 3)), 3)
