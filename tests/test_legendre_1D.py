"""Tests for normalized Shifted Legendre basis 1D."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.testing as nptest
import pytest
from numpy import typing as npt
from scipy.special import eval_sh_legendre

from pantr.basis import tabulate_legendre_1d
from pantr.quad import get_gauss_legendre_1d
from pantr.tolerance import get_default


class TestEvalLegendreBasis1D:
    """Test suite for tabulate_legendre_1d function."""

    def test_degree_zero(self) -> None:
        """Test Legendre basis of degree 0."""
        pts = np.array([0.0, 0.5, 1.0])
        result = tabulate_legendre_1d(0, pts)
        # p_0(x) = 1
        expected = np.ones((3, 1))
        nptest.assert_allclose(result, expected)

    def test_degree_one(self) -> None:
        """Test Legendre basis of degree 1."""
        pts = np.array([0.0, 0.5, 1.0])
        result = tabulate_legendre_1d(1, pts)
        # p_0(x) = 1
        # p_1(x) = sqrt(3)(2x-1)
        # at 0: sqrt(3)(-1) = -sqrt(3)
        # at 0.5: sqrt(3)(0) = 0
        # at 1: sqrt(3)(1) = sqrt(3)
        sqrt3 = np.sqrt(3.0)
        expected = np.array([[1.0, -sqrt3], [1.0, 0.0], [1.0, sqrt3]])
        nptest.assert_allclose(result, expected)

    def test_degree_two(self) -> None:
        """Test Legendre basis of degree 2."""
        pts = np.array([0.0, 0.5, 1.0])
        result = tabulate_legendre_1d(2, pts)
        # p_2(x) = sqrt(5) * (6x^2 - 6x + 1)
        # at 0: sqrt(5) * 1 = sqrt(5)
        # at 0.5: sqrt(5) * (1.5 - 3 + 1) = sqrt(5) * (-0.5) = -sqrt(5)/2
        # at 1: sqrt(5) * (6 - 6 + 1) = sqrt(5)
        sqrt5 = np.sqrt(5.0)
        expected_p2 = np.array([sqrt5, -0.5 * sqrt5, sqrt5])
        nptest.assert_allclose(result[:, 2], expected_p2)

    @pytest.mark.parametrize("degree", [0, 1, 2, 3, 5, 10])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_matches_scipy(self, degree: int, dtype: type) -> None:
        """Compare against scipy.special.eval_sh_legendre scaled by sqrt(2n+1)."""
        pts: npt.NDArray[np.floating[Any]] = np.linspace(0.0, 1.0, 11, dtype=dtype)
        pantr_vals = tabulate_legendre_1d(degree, pts)

        scipy_vals = np.zeros_like(pantr_vals)
        for n in range(degree + 1):
            # Scipy gives non-normalized shifted Legendre polynomials
            # We need to scale by sqrt(2n+1)
            scale = np.sqrt(2 * n + 1)
            scipy_vals[:, n] = eval_sh_legendre(n, pts) * scale

        rtol = get_default(dtype)
        nptest.assert_allclose(
            pantr_vals, scipy_vals, rtol=rtol, atol=1e-6 if dtype == np.float32 else 1e-14
        )

    def test_orthogonality(self) -> None:
        """Test that the basis functions are orthonormal on [0, 1]."""
        degree = 5
        # Use Gauss-Legendre quadrature which is exact for polynomials of degree 2n-1
        # We are integrating product of two polynomials of degree up to 5, so max degree is 10.
        # We need quadrature with enough points. n_quad points integrates
        # exactly degree 2*n_quad - 1.
        # So 6 points integrates degree 11.
        pts, weights = get_gauss_legendre_1d(10)

        basis_vals = tabulate_legendre_1d(degree, pts)

        # Compute mass matrix: M_ij = int p_i(x) p_j(x) dx
        # M = B.T * W * B
        weighted_basis = basis_vals * weights[:, None]
        mass_matrix = basis_vals.T @ weighted_basis

        # Should be identity matrix
        expected = np.eye(degree + 1)
        nptest.assert_allclose(mass_matrix, expected, atol=1e-14)

    def test_negative_degree_raises_error(self) -> None:
        """Test that negative degree raises ValueError."""
        pts = np.array([0.0, 0.5, 1.0])
        with pytest.raises(ValueError, match="degree must be non-negative"):
            tabulate_legendre_1d(-1, pts)

    def test_scalar_input(self) -> None:
        """Test with scalar input."""
        result = tabulate_legendre_1d(1, 0.5)
        # p_0(0.5) = 1, p_1(0.5) = 0
        expected = np.array([1.0, 0.0])
        nptest.assert_allclose(result, expected)

    def test_list_input(self) -> None:
        """Test with list input."""
        result = tabulate_legendre_1d(1, [0.0, 0.5, 1.0])
        sqrt3 = np.sqrt(3.0)
        expected = np.array([[1.0, -sqrt3], [1.0, 0.0], [1.0, sqrt3]])
        nptest.assert_allclose(result, expected)

    def test_outside_unit_interval(self) -> None:
        """Test evaluation outside [0, 1]."""
        pts = np.array([-0.5, 1.5])
        result = tabulate_legendre_1d(1, pts)
        # p_1(x) = sqrt(3)(2x-1)
        # at -0.5: sqrt(3)(-1 - 1) = -2sqrt(3)
        # at 1.5: sqrt(3)(3 - 1) = 2sqrt(3)
        sqrt3 = np.sqrt(3.0)
        expected = np.array([[1.0, -2.0 * sqrt3], [1.0, 2.0 * sqrt3]])
        nptest.assert_allclose(result, expected)
