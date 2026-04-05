"""Tests for Sylvester and Bezout matrix construction, and QR determinant."""

import numpy as np
import pytest

from pantr.bezier._resultant_matrices import (
    _bezout_matrix,
    _det_qr,
    _givens_rotation,
    _sylvester_matrix,
)

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


class TestGivensRotation:
    """Test _givens_rotation helper."""

    def test_zeroes_second_component(self) -> None:
        """Applying the rotation to (a, b) gives (r, 0)."""
        a, b = 3.0, 4.0
        c, s = _givens_rotation(a, b)
        r = c * a + s * b
        zero = -s * a + c * b
        np.testing.assert_allclose(zero, 0.0, atol=1e-15)
        np.testing.assert_allclose(r, 5.0, atol=1e-15)

    def test_identity_when_b_is_zero(self) -> None:
        """When b == 0, the rotation is the identity."""
        c, s = _givens_rotation(7.0, 0.0)
        assert c == 1.0
        assert s == 0.0

    def test_abs_b_greater_than_abs_a(self) -> None:
        """Branch where |b| > |a|."""
        c, s = _givens_rotation(1.0, 10.0)
        zero = -s * 1.0 + c * 10.0
        np.testing.assert_allclose(zero, 0.0, atol=1e-15)
        assert c * c + s * s == pytest.approx(1.0)

    def test_negative_values(self) -> None:
        """Works correctly with negative inputs."""
        c, s = _givens_rotation(-3.0, 4.0)
        zero = -s * (-3.0) + c * 4.0
        np.testing.assert_allclose(zero, 0.0, atol=1e-15)


class TestDetQr:
    """Test _det_qr determinant and rank computation."""

    def test_identity_matrix(self) -> None:
        """Determinant of identity is 1, rank is n."""
        n = 5
        A = np.eye(n)
        det, rank = _det_qr(A)
        np.testing.assert_allclose(det, 1.0, atol=1e-12)
        assert rank == n

    def test_known_determinant(self) -> None:
        """Compare with np.linalg.det on a random matrix."""
        rng = np.random.default_rng(123)
        for n in [2, 3, 5, 8]:
            A = rng.standard_normal((n, n))
            expected = np.linalg.det(A)
            det, rank = _det_qr(A.copy())
            np.testing.assert_allclose(det, expected, rtol=1e-10)
            assert rank == n

    def test_singular_matrix_zero_det(self) -> None:
        """Singular matrix has det ~ 0 and rank < n."""
        A = np.array([[1.0, 2.0], [2.0, 4.0]])
        det, rank = _det_qr(A)
        np.testing.assert_allclose(det, 0.0, atol=1e-12)
        assert rank < 2  # noqa: PLR2004

    def test_rank_deficient_matrix(self) -> None:
        """Matrix with known rank deficiency."""
        # Rank-2 matrix of size 4x4.
        rng = np.random.default_rng(42)
        U = rng.standard_normal((4, 2))
        V = rng.standard_normal((2, 4))
        A = U @ V
        det, rank = _det_qr(A.copy())
        np.testing.assert_allclose(det, 0.0, atol=1e-10)
        assert rank == 2  # noqa: PLR2004

    def test_diagonal_matrix(self) -> None:
        """Determinant of a diagonal matrix is the product of diagonal entries."""
        diag = np.array([2.0, -3.0, 0.5, 4.0])
        A = np.diag(diag)
        det, rank = _det_qr(A)
        np.testing.assert_allclose(det, np.prod(diag), rtol=1e-12)
        assert rank == 4  # noqa: PLR2004

    def test_overwrites_input(self) -> None:
        """The input matrix is modified in place."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        original = A.copy()
        _det_qr(A)
        assert not np.array_equal(A, original)

    def test_1x1_matrix(self) -> None:
        """1x1 matrix determinant is the single entry."""
        A = np.array([[42.0]])
        det, rank = _det_qr(A)
        np.testing.assert_allclose(det, 42.0, atol=1e-15)
        assert rank == 1

    def test_resultant_matrix_det(self) -> None:
        """det_qr agrees with np.linalg.det on a Sylvester matrix."""
        a = np.array([1.0, -1.0, 0.5])
        b = np.array([0.0, 1.0])
        mat = _sylvester_matrix(a, b)
        expected = np.linalg.det(mat)
        det, _rank = _det_qr(mat.copy())
        np.testing.assert_allclose(det, expected, rtol=1e-10)

    def test_non_square_raises(self) -> None:
        """Non-square input raises ValueError."""
        A = np.ones((2, 3))
        with pytest.raises(ValueError, match="square"):
            _det_qr(A)

    def test_integer_dtype_raises(self) -> None:
        """Integer dtype raises ValueError."""
        A = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="floating"):
            _det_qr(A)

    def test_non_writeable_raises(self) -> None:
        """Non-writeable array raises ValueError."""
        A = np.eye(3)
        A.flags.writeable = False
        with pytest.raises(ValueError, match="writeable"):
            _det_qr(A)

    def test_non_array_raises(self) -> None:
        """Non-ndarray input raises ValueError."""
        with pytest.raises(ValueError, match="numpy array"):
            _det_qr([[1.0, 0.0], [0.0, 1.0]])  # type: ignore[arg-type]

    def test_non_2d_raises(self) -> None:
        """1-D array raises ValueError."""
        with pytest.raises(ValueError, match="square"):
            _det_qr(np.array([1.0, 2.0, 3.0]))

    def test_empty_matrix_raises(self) -> None:
        """0x0 matrix raises ValueError."""
        with pytest.raises(ValueError, match="size >= 1"):
            _det_qr(np.empty((0, 0)))

    def test_non_positive_tol_raises(self) -> None:
        """Non-positive tol raises ValueError."""
        A = np.eye(2)
        with pytest.raises(ValueError, match="tol"):
            _det_qr(A, tol=0.0)
        with pytest.raises(ValueError, match="tol"):
            _det_qr(A.copy(), tol=-1.0)
