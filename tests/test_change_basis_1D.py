"""Tests for change_basis_1D module."""

import numpy as np
import numpy.typing as npt
import pytest

from pantr.basis import (
    LagrangeVariant,
    tabulate_bernstein_1d,
    tabulate_cardinal_bspline_1d,
    tabulate_lagrange_1d,
)
from pantr.change_basis import (
    _cached_cardinal_to_bernstein_matrix,
    _cached_lagrange_to_bernstein_matrix,
    _compute_change_basis_1D,
    compute_bernstein_to_cardinal_1d,
    compute_bernstein_to_lagrange_1d,
    compute_cardinal_to_bernstein_1d,
    compute_lagrange_to_bernstein_1d,
    compute_monomial_to_bernstein_1d,
)


class TestLagrangeToBernsteinBasisOperator:
    """Test the compute_lagrange_to_bernstein_1d function."""

    def test_degree_zero_error(self) -> None:
        """Test that degree lower than 1 raises ValueError."""
        with pytest.raises(ValueError, match="Degree must at least 1"):
            compute_lagrange_to_bernstein_1d(0)

    def test_negative_degree_error(self) -> None:
        """Test that negative degree raises ValueError."""
        with pytest.raises(ValueError, match="Degree must at least 1"):
            compute_lagrange_to_bernstein_1d(-1)

    def test_invalid_dtype_error(self) -> None:
        """Test that invalid dtype raises ValueError."""
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            compute_lagrange_to_bernstein_1d(2, dtype=np.int32)
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            compute_lagrange_to_bernstein_1d(2, dtype=np.float16)

    def test_out_parameter(self) -> None:
        """Test that out parameter works correctly."""
        degree = 2
        variant = LagrangeVariant.EQUISPACES

        # Test with None (default)
        result1 = compute_lagrange_to_bernstein_1d(degree, variant)
        assert result1.shape == (degree + 1, degree + 1)
        assert result1.dtype == np.float64

        # Test with provided out array (correct shape and dtype)
        out = np.empty((degree + 1, degree + 1), dtype=np.float64)
        result2 = compute_lagrange_to_bernstein_1d(degree, variant, out=out)
        assert result2 is out
        np.testing.assert_array_almost_equal(result1, result2)

        # Test with float32
        out_f32 = np.empty((degree + 1, degree + 1), dtype=np.float32)
        result3 = compute_lagrange_to_bernstein_1d(degree, variant, dtype=np.float32, out=out_f32)
        assert result3 is out_f32
        assert result3.dtype == np.float32

    def test_out_parameter_validation(self) -> None:
        """Test that out parameter validation works correctly."""
        degree = 2
        variant = LagrangeVariant.EQUISPACES

        # Wrong shape
        out_wrong_shape = np.empty((degree + 2, degree + 1), dtype=np.float64)
        with pytest.raises(ValueError, match="Output array has shape"):
            compute_lagrange_to_bernstein_1d(degree, variant, out=out_wrong_shape)

        # Wrong dtype
        out_wrong_dtype = np.empty((degree + 1, degree + 1), dtype=np.float32)
        with pytest.raises(ValueError, match="Output array has dtype"):
            compute_lagrange_to_bernstein_1d(degree, variant, out=out_wrong_dtype)

        # Not writeable
        out_readonly = np.empty((degree + 1, degree + 1), dtype=np.float64)
        out_readonly.setflags(write=False)
        with pytest.raises(ValueError, match="Output array is not writeable"):
            compute_lagrange_to_bernstein_1d(degree, variant, out=out_readonly)


class TestBernsteinToLagrangeBasisOperator:
    """Test the create_Bernstein_to_Lagrange_basis function."""

    def test_degree_zero_error(self) -> None:
        """Test that degree lower than 1 raises ValueError."""
        with pytest.raises(ValueError, match="Degree must at least 1"):
            compute_bernstein_to_lagrange_1d(0)

    def test_negative_degree_error(self) -> None:
        """Test that negative degree raises ValueError."""
        with pytest.raises(ValueError, match="Degree must at least 1"):
            compute_bernstein_to_lagrange_1d(-1)

    def test_invalid_dtype_error(self) -> None:
        """Test that invalid dtype raises ValueError."""
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            compute_bernstein_to_lagrange_1d(2, dtype=np.int32)
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            compute_bernstein_to_lagrange_1d(2, dtype=np.float16)

    def test_out_parameter(self) -> None:
        """Test that out parameter works correctly."""
        degree = 2
        variant = LagrangeVariant.EQUISPACES

        # Test with None (default)
        result1 = compute_bernstein_to_lagrange_1d(degree, variant)
        assert result1.shape == (degree + 1, degree + 1)
        assert result1.dtype == np.float64

        # Test with provided out array (correct shape and dtype)
        out = np.empty((degree + 1, degree + 1), dtype=np.float64)
        result2 = compute_bernstein_to_lagrange_1d(degree, variant, out=out)
        assert result2 is out
        np.testing.assert_array_almost_equal(result1, result2)

        # Test with float32
        out_f32 = np.empty((degree + 1, degree + 1), dtype=np.float32)
        result3 = compute_bernstein_to_lagrange_1d(degree, variant, dtype=np.float32, out=out_f32)
        assert result3 is out_f32
        assert result3.dtype == np.float32

    def test_out_parameter_validation(self) -> None:
        """Test that out parameter validation works correctly."""
        degree = 2
        variant = LagrangeVariant.EQUISPACES

        # Wrong shape
        out_wrong_shape = np.empty((degree + 2, degree + 1), dtype=np.float64)
        with pytest.raises(ValueError, match="Output array has shape"):
            compute_bernstein_to_lagrange_1d(degree, variant, out=out_wrong_shape)

        # Wrong dtype
        out_wrong_dtype = np.empty((degree + 1, degree + 1), dtype=np.float32)
        with pytest.raises(ValueError, match="Output array has dtype"):
            compute_bernstein_to_lagrange_1d(degree, variant, out=out_wrong_dtype)

        # Not writeable
        out_readonly = np.empty((degree + 1, degree + 1), dtype=np.float64)
        out_readonly.setflags(write=False)
        with pytest.raises(ValueError, match="Output array is not writeable"):
            compute_bernstein_to_lagrange_1d(degree, variant, out=out_readonly)

    def test_inverse_relationship(self) -> None:
        """Test that Bernstein to Lagrange is inverse of Lagrange to Bernstein."""
        degree = 2
        variant = LagrangeVariant.EQUISPACES
        lagrange_to_bernstein = compute_lagrange_to_bernstein_1d(degree, variant)
        bernstein_to_lagrange = compute_bernstein_to_lagrange_1d(degree, variant)

        # Should be inverse matrices
        identity = lagrange_to_bernstein @ bernstein_to_lagrange
        np.testing.assert_array_almost_equal(identity, np.eye(degree + 1))

    @pytest.mark.parametrize(
        "variant",
        [
            LagrangeVariant.EQUISPACES,
            LagrangeVariant.GAUSS_LEGENDRE,
            LagrangeVariant.GAUSS_LOBATTO_LEGENDRE,
            LagrangeVariant.CHEBYSHEV_1ST,
            LagrangeVariant.CHEBYSHEV_2ND,
        ],
    )
    @pytest.mark.parametrize("degree", [1, 2, 3, 4])
    def test_values(self, degree: int, variant: LagrangeVariant) -> None:
        """Test Bernstein evaluations transformed with the operator return Lagrange evaluations."""
        n_pts = 10
        tt = np.linspace(0.0, 1.0, n_pts)

        bernsteins = tabulate_bernstein_1d(degree, tt)
        lagranges = tabulate_lagrange_1d(degree, variant, tt)

        C = compute_bernstein_to_lagrange_1d(degree, variant)
        np.testing.assert_array_almost_equal(bernsteins @ C.T, lagranges)

        C_inv = compute_lagrange_to_bernstein_1d(degree, variant)
        np.testing.assert_array_almost_equal(lagranges @ C_inv.T, bernsteins)


class TestCardinalToBernsteinBasisOperator:
    """Test the create_cardinal_to_Bernstein_basis function."""

    def test_inverse_relationship(self) -> None:
        """Test that cardinal to Bernstein is inverse of Bernstein to cardinal."""
        degree = 2
        bernstein_to_cardinal = compute_bernstein_to_cardinal_1d(degree)
        cardinal_to_bernstein = compute_cardinal_to_bernstein_1d(degree)

        # Should be inverse matrices
        identity = bernstein_to_cardinal @ cardinal_to_bernstein
        np.testing.assert_array_almost_equal(identity, np.eye(degree + 1))

    def test_negative_degree_error(self) -> None:
        """Test that negative degree raises ValueError."""
        with pytest.raises(ValueError, match="Degree must be non-negative"):
            compute_cardinal_to_bernstein_1d(-1)
        with pytest.raises(ValueError, match="Degree must be non-negative"):
            compute_bernstein_to_cardinal_1d(-1)

    def test_invalid_dtype_error(self) -> None:
        """Test that invalid dtype raises ValueError."""
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            compute_bernstein_to_cardinal_1d(2, dtype=np.int32)
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            compute_bernstein_to_cardinal_1d(2, dtype=np.float16)
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            compute_cardinal_to_bernstein_1d(2, dtype=np.int32)
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            compute_cardinal_to_bernstein_1d(2, dtype=np.float16)

    def test_values(self) -> None:
        """Test that cardinal evaluations transformed with operator return Bernstein evaluations."""
        for degree in [1, 2, 3, 4]:
            n_pts = 10
            tt = np.linspace(0.0, 1.0, n_pts)
            bernsteins = tabulate_bernstein_1d(degree, tt)
            cardinals = tabulate_cardinal_bspline_1d(degree, tt)

            C = compute_cardinal_to_bernstein_1d(degree)
            np.testing.assert_array_almost_equal(bernsteins, cardinals @ C.T)

            C_inv = compute_bernstein_to_cardinal_1d(degree)
            np.testing.assert_array_almost_equal(cardinals, bernsteins @ C_inv.T)

    def test_out_parameter(self) -> None:
        """Test that out parameter works correctly for cardinal-Bernstein functions."""
        degree = 2

        # Test compute_bernstein_to_cardinal_1d
        result1 = compute_bernstein_to_cardinal_1d(degree)
        assert result1.shape == (degree + 1, degree + 1)
        assert result1.dtype == np.float64

        out = np.empty((degree + 1, degree + 1), dtype=np.float64)
        result2 = compute_bernstein_to_cardinal_1d(degree, out=out)
        assert result2 is out
        np.testing.assert_array_almost_equal(result1, result2)

        # Test compute_cardinal_to_bernstein_1d
        result3 = compute_cardinal_to_bernstein_1d(degree)
        assert result3.shape == (degree + 1, degree + 1)
        assert result3.dtype == np.float64

        out2 = np.empty((degree + 1, degree + 1), dtype=np.float64)
        result4 = compute_cardinal_to_bernstein_1d(degree, out=out2)
        assert result4 is out2
        np.testing.assert_array_almost_equal(result3, result4)

        # Test with float32
        out_f32 = np.empty((degree + 1, degree + 1), dtype=np.float32)
        result5 = compute_bernstein_to_cardinal_1d(degree, dtype=np.float32, out=out_f32)
        assert result5 is out_f32
        assert result5.dtype == np.float32

    def test_out_parameter_validation(self) -> None:
        """Test that out parameter validation works correctly for cardinal-Bernstein functions."""
        degree = 2

        # Wrong shape
        out_wrong_shape = np.empty((degree + 2, degree + 1), dtype=np.float64)
        with pytest.raises(ValueError, match="Output array has shape"):
            compute_bernstein_to_cardinal_1d(degree, out=out_wrong_shape)
        with pytest.raises(ValueError, match="Output array has shape"):
            compute_cardinal_to_bernstein_1d(degree, out=out_wrong_shape)

        # Wrong dtype
        out_wrong_dtype = np.empty((degree + 1, degree + 1), dtype=np.float32)
        with pytest.raises(ValueError, match="Output array has dtype"):
            compute_bernstein_to_cardinal_1d(degree, out=out_wrong_dtype)
        with pytest.raises(ValueError, match="Output array has dtype"):
            compute_cardinal_to_bernstein_1d(degree, out=out_wrong_dtype)

        # Not writeable
        out_readonly = np.empty((degree + 1, degree + 1), dtype=np.float64)
        out_readonly.setflags(write=False)
        with pytest.raises(ValueError, match="Output array is not writeable"):
            compute_bernstein_to_cardinal_1d(degree, out=out_readonly)
        out_readonly2 = np.empty((degree + 1, degree + 1), dtype=np.float64)
        out_readonly2.setflags(write=False)
        with pytest.raises(ValueError, match="Output array is not writeable"):
            compute_cardinal_to_bernstein_1d(degree, out=out_readonly2)


class TestCreateChangeBasis:
    """Test the _create_change_basis private function."""

    def test_invalid_n_quad_pts_error(self) -> None:
        """Test that non-positive n_quad_pts raises ValueError."""
        degree = 2

        def bernstein(
            pts: npt.NDArray[np.float32 | np.float64],
        ) -> npt.NDArray[np.float32 | np.float64]:
            return tabulate_bernstein_1d(degree, pts)

        def cardinal(
            pts: npt.NDArray[np.float32 | np.float64],
        ) -> npt.NDArray[np.float32 | np.float64]:
            return tabulate_cardinal_bspline_1d(degree, pts)

        with pytest.raises(ValueError, match="Number of quadrature points must be positive"):
            _compute_change_basis_1D(bernstein, cardinal, n_quad_pts=0)

        with pytest.raises(ValueError, match="Number of quadrature points must be positive"):
            _compute_change_basis_1D(bernstein, cardinal, n_quad_pts=-1)

    def test_invalid_dtype_error(self) -> None:
        """Test that invalid dtype raises ValueError."""
        degree = 2

        def bernstein(
            pts: npt.NDArray[np.float32 | np.float64],
        ) -> npt.NDArray[np.float32 | np.float64]:
            return tabulate_bernstein_1d(degree, pts)

        def cardinal(
            pts: npt.NDArray[np.float32 | np.float64],
        ) -> npt.NDArray[np.float32 | np.float64]:
            return tabulate_cardinal_bspline_1d(degree, pts)

        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            _compute_change_basis_1D(bernstein, cardinal, n_quad_pts=3, dtype=np.int32)

        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            _compute_change_basis_1D(bernstein, cardinal, n_quad_pts=3, dtype=np.float16)

    def test_out_parameter(self) -> None:
        """Test that out parameter works correctly for _compute_change_basis_1D."""
        degree = 2

        def bernstein(
            pts: npt.NDArray[np.float32 | np.float64],
        ) -> npt.NDArray[np.float32 | np.float64]:
            return tabulate_bernstein_1d(degree, pts)

        def cardinal(
            pts: npt.NDArray[np.float32 | np.float64],
        ) -> npt.NDArray[np.float32 | np.float64]:
            return tabulate_cardinal_bspline_1d(degree, pts)

        # Test with None (default)
        result1 = _compute_change_basis_1D(bernstein, cardinal, n_quad_pts=degree + 1)
        assert result1.shape == (degree + 1, degree + 1)
        assert result1.dtype == np.float64

        # Test with provided out array (correct shape and dtype)
        out = np.empty((degree + 1, degree + 1), dtype=np.float64)
        result2 = _compute_change_basis_1D(bernstein, cardinal, n_quad_pts=degree + 1, out=out)
        assert result2 is out
        np.testing.assert_array_almost_equal(result1, result2)

        # Test with float32
        out_f32 = np.empty((degree + 1, degree + 1), dtype=np.float32)
        result3 = _compute_change_basis_1D(
            bernstein, cardinal, n_quad_pts=degree + 1, dtype=np.float32, out=out_f32
        )
        assert result3 is out_f32
        assert result3.dtype == np.float32

    def test_out_parameter_validation(self) -> None:
        """Test that out parameter validation works correctly for _compute_change_basis_1D."""
        degree = 2

        def bernstein(
            pts: npt.NDArray[np.float32 | np.float64],
        ) -> npt.NDArray[np.float32 | np.float64]:
            return tabulate_bernstein_1d(degree, pts)

        def cardinal(
            pts: npt.NDArray[np.float32 | np.float64],
        ) -> npt.NDArray[np.float32 | np.float64]:
            return tabulate_cardinal_bspline_1d(degree, pts)

        # Wrong shape
        out_wrong_shape = np.empty((degree + 2, degree + 1), dtype=np.float64)
        with pytest.raises(ValueError, match="Output array has shape"):
            _compute_change_basis_1D(
                bernstein, cardinal, n_quad_pts=degree + 1, out=out_wrong_shape
            )

        # Wrong dtype
        out_wrong_dtype = np.empty((degree + 1, degree + 1), dtype=np.float32)
        with pytest.raises(ValueError, match="Output array has dtype"):
            _compute_change_basis_1D(
                bernstein, cardinal, n_quad_pts=degree + 1, out=out_wrong_dtype
            )

        # Not writeable
        out_readonly = np.empty((degree + 1, degree + 1), dtype=np.float64)
        out_readonly.setflags(write=False)
        with pytest.raises(ValueError, match="Output array is not writeable"):
            _compute_change_basis_1D(bernstein, cardinal, n_quad_pts=degree + 1, out=out_readonly)


class TestCachedChangeBasisMatrices:
    """Tests for the LRU-cached change-of-basis helpers."""

    def test_lagrange_cached_values_match_uncached(self) -> None:
        """Cached matrix must be numerically identical to the uncached reference."""
        degree = 3
        variant = LagrangeVariant.GAUSS_LOBATTO_LEGENDRE
        dtype = np.dtype(np.float64)
        expected = compute_lagrange_to_bernstein_1d(degree, variant, dtype)
        cached = _cached_lagrange_to_bernstein_matrix(degree, variant, dtype)
        np.testing.assert_array_equal(cached, expected)

    def test_cardinal_cached_values_match_uncached(self) -> None:
        """Cached matrix must be numerically identical to the uncached reference."""
        degree = 3
        dtype = np.dtype(np.float64)
        expected = compute_cardinal_to_bernstein_1d(degree, dtype)
        cached = _cached_cardinal_to_bernstein_matrix(degree, dtype)
        np.testing.assert_array_equal(cached, expected)

    def test_lagrange_cached_array_is_readonly(self) -> None:
        """The cached array must be read-only to prevent mutation of the shared copy."""
        degree = 2
        dtype = np.dtype(np.float32)
        mat = _cached_lagrange_to_bernstein_matrix(degree, LagrangeVariant.EQUISPACES, dtype)
        assert not mat.flags.writeable
        with pytest.raises((ValueError, TypeError)):
            mat[0, 0] = 0.0

    def test_cardinal_cached_array_is_readonly(self) -> None:
        """The cached array must be read-only to prevent mutation of the shared copy."""
        degree = 2
        dtype = np.dtype(np.float64)
        mat = _cached_cardinal_to_bernstein_matrix(degree, dtype)
        assert not mat.flags.writeable
        with pytest.raises((ValueError, TypeError)):
            mat[0, 0] = 0.0

    def test_lagrange_cache_returns_same_object(self) -> None:
        """Repeated calls with identical arguments must return the exact same object."""
        degree = 2
        variant = LagrangeVariant.EQUISPACES
        dtype = np.dtype(np.float64)
        mat1 = _cached_lagrange_to_bernstein_matrix(degree, variant, dtype)
        mat2 = _cached_lagrange_to_bernstein_matrix(degree, variant, dtype)
        assert mat1 is mat2

    def test_cardinal_cache_returns_same_object(self) -> None:
        """Repeated calls with identical arguments must return the exact same object."""
        degree = 3
        dtype = np.dtype(np.float64)
        mat1 = _cached_cardinal_to_bernstein_matrix(degree, dtype)
        mat2 = _cached_cardinal_to_bernstein_matrix(degree, dtype)
        assert mat1 is mat2

    def test_lagrange_cache_is_bounded(self) -> None:
        """The Lagrange cache must have a finite maxsize."""
        info = _cached_lagrange_to_bernstein_matrix.cache_info()
        assert info.maxsize is not None
        assert info.maxsize > 0

    def test_cardinal_cache_is_bounded(self) -> None:
        """The cardinal cache must have a finite maxsize."""
        info = _cached_cardinal_to_bernstein_matrix.cache_info()
        assert info.maxsize is not None
        assert info.maxsize > 0


class TestMonomialToBernsteinBasisOperator:
    """Test the compute_monomial_to_bernstein_1d function."""

    def test_negative_degree_error(self) -> None:
        """Test that negative degree raises ValueError."""
        with pytest.raises(ValueError, match="Degree must be non-negative"):
            compute_monomial_to_bernstein_1d(-1)

    def test_invalid_dtype_error(self) -> None:
        """Test that invalid dtype raises ValueError."""
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            compute_monomial_to_bernstein_1d(2, dtype=np.int32)
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            compute_monomial_to_bernstein_1d(2, dtype=np.float16)

    def test_out_parameter(self) -> None:
        """Test that out parameter works correctly."""
        degree = 3

        result1 = compute_monomial_to_bernstein_1d(degree)
        assert result1.shape == (degree + 1, degree + 1)
        assert result1.dtype == np.float64

        out = np.empty((degree + 1, degree + 1), dtype=np.float64)
        result2 = compute_monomial_to_bernstein_1d(degree, out=out)
        assert result2 is out
        np.testing.assert_array_equal(result1, result2)

        out_f32 = np.empty((degree + 1, degree + 1), dtype=np.float32)
        result3 = compute_monomial_to_bernstein_1d(degree, dtype=np.float32, out=out_f32)
        assert result3 is out_f32
        assert result3.dtype == np.float32

    def test_out_parameter_validation(self) -> None:
        """Test that out parameter validation works correctly."""
        degree = 2

        out_wrong_shape = np.empty((degree + 2, degree + 1), dtype=np.float64)
        with pytest.raises(ValueError, match="Output array has shape"):
            compute_monomial_to_bernstein_1d(degree, out=out_wrong_shape)

        out_wrong_dtype = np.empty((degree + 1, degree + 1), dtype=np.float32)
        with pytest.raises(ValueError, match="Output array has dtype"):
            compute_monomial_to_bernstein_1d(degree, out=out_wrong_dtype)

        out_readonly = np.empty((degree + 1, degree + 1), dtype=np.float64)
        out_readonly.setflags(write=False)
        with pytest.raises(ValueError, match="Output array is not writeable"):
            compute_monomial_to_bernstein_1d(degree, out=out_readonly)

    def test_degree_zero(self) -> None:
        """Degree 0 returns the 1x1 identity."""
        mat = compute_monomial_to_bernstein_1d(0)
        np.testing.assert_array_equal(mat, np.array([[1.0]]))

    def test_known_values_degree_2(self) -> None:
        """Degree 2: 1 = B0+B1+B2, t = (1/2)B1 + B2, t^2 = B2."""
        expected = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        np.testing.assert_array_almost_equal(compute_monomial_to_bernstein_1d(2), expected)

    def test_known_values_degree_3(self) -> None:
        """Degree 3 reference values: M[i, j] = C(i, j) / C(3, j)."""
        expected = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0 / 3.0, 0.0, 0.0],
                [1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        )
        np.testing.assert_array_almost_equal(compute_monomial_to_bernstein_1d(3), expected)

    @pytest.mark.parametrize("degree", [1, 2, 3, 4, 5, 6])
    def test_polynomial_reconstruction(self, degree: int) -> None:
        """Bernstein coefficients from the matrix must reproduce the monomial polynomial."""
        rng = np.random.default_rng(degree)
        mono = rng.standard_normal(degree + 1)

        bern_coeffs = compute_monomial_to_bernstein_1d(degree) @ mono

        tt = np.linspace(0.0, 1.0, 25)
        p_mono = sum(mono[k] * tt**k for k in range(degree + 1))
        p_bern = tabulate_bernstein_1d(degree, tt) @ bern_coeffs

        np.testing.assert_array_almost_equal(p_bern, p_mono)
