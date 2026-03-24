"""Tests for Bézier L2 norm computation."""

from __future__ import annotations

import numpy as np
import pytest

from pantr.bezier import Bezier


class TestSquaredL2Norm:
    """Tests for ``Bezier.squared_l2_norm``."""

    def test_constant_1d(self) -> None:
        """Constant p(x) = 3: ||p||^2 = 9."""
        b = Bezier([3.0, 3.0, 3.0])
        np.testing.assert_allclose(b.squared_l2_norm(), 9.0, atol=1e-14)

    def test_constant_2d(self) -> None:
        """Constant p(x,y) = 2: ||p||^2 = 4."""
        ctrl = 2.0 * np.ones((3, 4, 1))
        b = Bezier(ctrl)
        np.testing.assert_allclose(b.squared_l2_norm(), 4.0, atol=1e-14)

    def test_constant_3d(self) -> None:
        """Constant p(x,y,z) = 5: ||p||^2 = 25."""
        ctrl = 5.0 * np.ones((2, 2, 2, 1))
        b = Bezier(ctrl)
        np.testing.assert_allclose(b.squared_l2_norm(), 25.0, atol=1e-14)

    def test_linear_1d_identity(self) -> None:
        """p(x) = x: ||p||^2 = integral(x^2, 0, 1) = 1/3."""
        b = Bezier([0.0, 1.0])
        np.testing.assert_allclose(b.squared_l2_norm(), 1.0 / 3.0, atol=1e-14)

    def test_linear_1d_complement(self) -> None:
        """p(x) = 1-x: ||p||^2 = integral((1-x)^2, 0, 1) = 1/3."""
        b = Bezier([1.0, 0.0])
        np.testing.assert_allclose(b.squared_l2_norm(), 1.0 / 3.0, atol=1e-14)

    def test_quadratic_1d(self) -> None:
        """p(x) = (1-x)^2 via coefficients [1, 0, 0]: ||p||^2 = 1/5."""
        b = Bezier([1.0, 0.0, 0.0])
        np.testing.assert_allclose(b.squared_l2_norm(), 1.0 / 5.0, atol=1e-14)

    def test_degree_zero(self) -> None:
        """Degree-0 Bézier p(x) = 7: ||p||^2 = 49."""
        b = Bezier([7.0])
        np.testing.assert_allclose(b.squared_l2_norm(), 49.0, atol=1e-14)

    def test_2d_separable(self) -> None:
        """Separable 2D: p(x,y) = f(x)*g(y), ||p||^2 = ||f||^2 * ||g||^2."""
        f_ctrl = np.array([0.0, 1.0])  # f(x) = x
        g_ctrl = np.array([1.0, 0.0])  # g(y) = 1-y
        ctrl_2d = np.outer(f_ctrl, g_ctrl)[:, :, np.newaxis]
        b2d = Bezier(ctrl_2d)

        bf = Bezier(f_ctrl)
        bg = Bezier(g_ctrl)
        expected = float(bf.squared_l2_norm()) * float(bg.squared_l2_norm())
        np.testing.assert_allclose(b2d.squared_l2_norm(), expected, atol=1e-14)

    def test_vector_valued(self) -> None:
        """Vector-valued: ||p||^2 = sum of component-wise squared norms."""
        ctrl = np.array([[0.0, 1.0], [1.0, 0.0]])  # rank 2
        b = Bezier(ctrl)

        # Component 0: coefficients [0, 1] -> p(x) = x, ||p||^2 = 1/3
        # Component 1: coefficients [1, 0] -> p(x) = 1-x, ||p||^2 = 1/3
        np.testing.assert_allclose(b.squared_l2_norm(), 2.0 / 3.0, atol=1e-14)

    def test_rational_raises(self) -> None:
        """Rational Bézier raises ValueError."""
        ctrl = np.array([[1.0, 1.0], [2.0, 1.0]])
        b = Bezier(ctrl, is_rational=True)
        with pytest.raises(ValueError, match="rational"):
            b.squared_l2_norm()

    def test_float32(self) -> None:
        """Float32 dtype is supported."""
        ctrl = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        b = Bezier(ctrl)
        result = b.squared_l2_norm()
        assert result.dtype == np.float32
        np.testing.assert_allclose(float(result), 1.0, atol=1e-6)

    def test_cross_validate_via_product_2d(self) -> None:
        """Cross-validate against product + coefficient-mean integration."""
        rng = np.random.default_rng(42)
        ctrl = rng.standard_normal((4, 5, 1))
        b = Bezier(ctrl)

        # ||p||^2 = integral(p^2) = integral of (b*b).
        # For a Bernstein polynomial q of degree r_d per direction,
        # integral over [0,1]^D = mean of all coefficients.
        product = b * b
        product_ctrl = product.control_points
        # Sum over spatial axes, then sum over rank.
        ref = float(np.mean(product_ctrl[..., 0]))
        np.testing.assert_allclose(b.squared_l2_norm(), ref, rtol=1e-14)


class TestL2Norm:
    """Tests for ``Bezier.l2_norm``."""

    def test_matches_sqrt_of_squared(self) -> None:
        """l2_norm == sqrt(abs(squared_l2_norm))."""
        b = Bezier([0.0, 1.0])
        expected = np.sqrt(np.abs(b.squared_l2_norm()))
        np.testing.assert_allclose(b.l2_norm(), expected, atol=1e-14)

    def test_constant(self) -> None:
        """Constant p(x) = 3: ||p|| = 3."""
        b = Bezier([3.0, 3.0])
        np.testing.assert_allclose(b.l2_norm(), 3.0, atol=1e-14)

    def test_rational_raises(self) -> None:
        """Rational Bézier raises ValueError."""
        ctrl = np.array([[1.0, 1.0], [2.0, 1.0]])
        b = Bezier(ctrl, is_rational=True)
        with pytest.raises(ValueError, match="rational"):
            b.l2_norm()
