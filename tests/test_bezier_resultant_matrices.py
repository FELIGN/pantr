"""Tests for Sylvester and Bezout matrix construction."""

import numpy as np
import pytest

from pantr.bezier._resultant_matrices import _bezout_matrix, _sylvester_matrix

_DET_ATOL = 1e-12
"""Absolute tolerance for determinant-is-zero checks."""


class TestSylvesterMatrix:
    """Test _sylvester_matrix for Bernstein polynomials."""

    def test_two_linears(self) -> None:
        """Resultant of two linear Bernstein polynomials is nonzero (no common root)."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        mat = _sylvester_matrix(a, b)
        assert mat.shape == (2, 2)
        assert np.abs(np.linalg.det(mat)) > _DET_ATOL

    def test_common_root_gives_zero_det(self) -> None:
        """Two identical polynomials share a root, giving a singular matrix."""
        a = np.array([0.0, 1.0])
        b = np.array([0.0, 1.0])
        mat = _sylvester_matrix(a, b)
        np.testing.assert_allclose(np.linalg.det(mat), 0.0, atol=_DET_ATOL)

    def test_different_degrees(self) -> None:
        """Sylvester matrix for polynomials of different degrees."""
        a = np.array([1.0, -1.0, 0.5])  # degree 2
        b = np.array([2.0, 3.0])  # degree 1
        mat = _sylvester_matrix(a, b)
        assert mat.shape == (3, 3)
        assert mat.dtype == np.float64

    def test_out_parameter(self) -> None:
        """Test that the out parameter is written to and returned."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0])
        out = np.zeros((3, 3), dtype=np.float64)
        result = _sylvester_matrix(a, b, out=out)
        assert result is out
        np.testing.assert_array_equal(result, _sylvester_matrix(a, b))

    def test_out_wrong_shape_raises(self) -> None:
        """Test that wrong out shape raises."""
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        out = np.zeros((3, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="shape"):
            _sylvester_matrix(a, b, out=out)

    def test_non_1d_raises(self) -> None:
        """Test that 2-D input raises."""
        a = np.array([[1.0, 2.0]])
        b = np.array([3.0, 4.0])
        with pytest.raises(ValueError, match="1-D"):
            _sylvester_matrix(a, b)

    def test_degree_zero_raises(self) -> None:
        """Test that a constant polynomial (degree 0) raises."""
        a = np.array([1.0])
        b = np.array([2.0, 3.0])
        with pytest.raises(ValueError, match="at least 2"):
            _sylvester_matrix(a, b)

    def test_integer_input_raises(self) -> None:
        """Test that integer dtype raises."""
        a = np.array([1, 2])
        b = np.array([3, 4])
        with pytest.raises(ValueError, match="floating"):
            _sylvester_matrix(a, b)

    def test_float32(self) -> None:
        """Test that float32 inputs produce float32 output."""
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        mat = _sylvester_matrix(a, b)
        assert mat.dtype == np.float32


class TestBezoutMatrix:
    """Test _bezout_matrix for Bernstein polynomials."""

    def test_symmetry(self) -> None:
        """Bezout matrix must be symmetric."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal(6)
        b = rng.standard_normal(6)
        mat = _bezout_matrix(a, b)
        np.testing.assert_allclose(mat, mat.T, atol=1e-15)

    def test_degree_1(self) -> None:
        """Bezout matrix for degree-1 polynomials is 1x1."""
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        mat = _bezout_matrix(a, b)
        assert mat.shape == (1, 1)
        # B[0,0] = (a[1]*b[0] - a[0]*b[1]) * n/1 = (2*3 - 1*4) * 1 = 2
        np.testing.assert_allclose(mat[0, 0], 2.0, atol=1e-15)

    def test_common_root_gives_zero_det(self) -> None:
        """Two identical polynomials share a root, giving a singular matrix."""
        a = np.array([0.0, 1.0])
        b = np.array([0.0, 1.0])
        mat = _bezout_matrix(a, b)
        np.testing.assert_allclose(np.linalg.det(mat), 0.0, atol=_DET_ATOL)

    def test_sylvester_bezout_zero_det_agreement(self) -> None:
        """Both matrices agree on whether the resultant is zero."""
        # Pair with common root at t=0.5: both det should be zero.
        a = np.array([-0.5, 0.5])
        b = np.array([-1.0, 1.0])
        np.testing.assert_allclose(np.linalg.det(_sylvester_matrix(a, b)), 0.0, atol=_DET_ATOL)
        np.testing.assert_allclose(np.linalg.det(_bezout_matrix(a, b)), 0.0, atol=_DET_ATOL)

        # Pair without common root: both det should be nonzero.
        a = np.array([-0.3, 0.7])
        b = np.array([-0.8, 0.2])
        assert np.abs(np.linalg.det(_sylvester_matrix(a, b))) > _DET_ATOL
        assert np.abs(np.linalg.det(_bezout_matrix(a, b))) > _DET_ATOL

    def test_bezout_det_equals_resultant(self) -> None:
        """Bezout entry matches the classical resultant for degree-1 polynomials."""
        rng = np.random.default_rng(77)
        for _ in range(20):
            a = rng.standard_normal(2)
            b = rng.standard_normal(2)
            mat = _bezout_matrix(a, b)
            expected = a[1] * b[0] - a[0] * b[1]
            np.testing.assert_allclose(mat[0, 0], expected, atol=1e-14)

    def test_unequal_lengths_raises(self) -> None:
        """Test that arrays of different lengths raise."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0])
        with pytest.raises(ValueError, match="equal length"):
            _bezout_matrix(a, b)

    def test_out_parameter(self) -> None:
        """Test that the out parameter is written to and returned."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        out = np.zeros((2, 2), dtype=np.float64)
        result = _bezout_matrix(a, b, out=out)
        assert result is out
        np.testing.assert_array_equal(result, _bezout_matrix(a, b))

    def test_degree_zero_raises(self) -> None:
        """Test that a constant polynomial (degree 0) raises."""
        a = np.array([1.0])
        b = np.array([2.0])
        with pytest.raises(ValueError, match="at least 2"):
            _bezout_matrix(a, b)

    def test_resultant_via_power_basis(self) -> None:
        """Both Bezout and Sylvester detect no common root for (t-0.3) vs (t-0.7)."""
        a = np.array([-0.3, 0.7])
        b = np.array([-0.7, 0.3])
        mat_b = _bezout_matrix(a, b)
        mat_s = _sylvester_matrix(a, b)
        assert np.abs(np.linalg.det(mat_b)) > _DET_ATOL
        assert np.abs(np.linalg.det(mat_s)) > _DET_ATOL

    def test_quadratic_with_known_common_root(self) -> None:
        """Two quadratics sharing root at t=0.5 give zero resultant.

        f(t) = (t-0.5)(t-0.2), g(t) = (t-0.5)(t-0.8) in Bernstein form.
        """
        # f(t) = t^2 - 0.7t + 0.1
        # Bernstein: c0=f(0)=0.1, c1=f(0)+f'(0)/2=-0.25, c2=f(1)=0.4
        a = np.array([0.1, -0.25, 0.4])
        # g(t) = t^2 - 1.3t + 0.4
        # Bernstein: c0=0.4, c1=0.4+(-1.3)/2=-0.25, c2=g(1)=0.1
        b = np.array([0.4, -0.25, 0.1])
        mat = _bezout_matrix(a, b)
        np.testing.assert_allclose(np.linalg.det(mat), 0.0, atol=_DET_ATOL)
