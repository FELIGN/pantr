"""Tests for the resultant and discriminant computation pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.testing as nptest
import numpy.typing as npt
import pytest

from pantr.basis._basis_core import _tabulate_Bernstein_basis_1D_core
from pantr.bezier._resultant import (
    _auto_reduction,
    _bernstein_derivative_1d,
    _bernstein_derivative_nd,
    _bernstein_interpolate,
    _bernstein_interpolate_1d,
    _discriminant,
    _discriminant_order,
    _normalise,
    _resultant,
    _resultant_order,
)
from pantr.quad import get_modified_chebyshev_nodes_1d

# ---------------------------------------------------------------------------
# Bernstein derivative
# ---------------------------------------------------------------------------


class TestBernsteinDerivative1D:
    """Tests for _bernstein_derivative_1d."""

    def test_linear(self) -> None:
        """Derivative of a linear: d/dx(a0(1-x) + a1*x) = a1 - a0."""
        coeffs = np.array([1.0, 3.0])
        result = _bernstein_derivative_1d(coeffs)
        assert result.shape == (1,)
        nptest.assert_allclose(result, [2.0])

    def test_quadratic(self) -> None:
        """Derivative of degree-2 Bernstein: out[i] = 2*(a[i+1]-a[i])."""
        coeffs = np.array([1.0, 2.0, 5.0])
        result = _bernstein_derivative_1d(coeffs)
        assert result.shape == (2,)
        nptest.assert_allclose(result, [2.0, 6.0])

    def test_constant(self) -> None:
        """Derivative of constant (degree 1 Bernstein [c, c]) is 0."""
        coeffs = np.array([3.0, 3.0])
        result = _bernstein_derivative_1d(coeffs)
        nptest.assert_allclose(result, [0.0], atol=1e-15)


class TestBernsteinDerivativeND:
    """Tests for _bernstein_derivative_nd."""

    def test_2d_dim0(self) -> None:
        """Derivative along dim=0 reduces order in that direction."""
        coeffs = np.array([[1.0, 2.0], [3.0, 4.0], [7.0, 8.0]])
        result = _bernstein_derivative_nd(coeffs, 0)
        assert result.shape == (2, 2)
        # d/dx along dim=0: (P-1)*(a[i+1,j] - a[i,j]) with P=3
        expected = 2.0 * np.array([[2.0, 2.0], [4.0, 4.0]])
        nptest.assert_allclose(result, expected)

    def test_2d_dim1(self) -> None:
        """Derivative along dim=1 reduces order in that direction."""
        coeffs = np.array([[1.0, 2.0, 5.0], [3.0, 4.0, 9.0]])
        result = _bernstein_derivative_nd(coeffs, 1)
        assert result.shape == (2, 2)
        # d/dy along dim=1: (Q-1)*(a[i,j+1] - a[i,j]) with Q=3
        expected = 2.0 * np.array([[1.0, 3.0], [1.0, 5.0]])
        nptest.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# Bernstein interpolation
# ---------------------------------------------------------------------------


class TestBernsteinInterpolate1D:
    """Tests for _bernstein_interpolate_1d."""

    def test_roundtrip_linear(self) -> None:
        """Interpolating linear Bernstein at 2 Chebyshev nodes recovers coefficients."""
        coeffs = np.array([1.0, 3.0])
        vals = self._eval_bernstein_at_chebyshev(coeffs)
        recovered = _bernstein_interpolate_1d(vals)
        nptest.assert_allclose(recovered, coeffs, atol=1e-14)

    def test_roundtrip_quadratic(self) -> None:
        """Interpolating quadratic Bernstein at 3 Chebyshev nodes recovers coefficients."""
        coeffs = np.array([1.0, 2.0, 5.0])
        vals = self._eval_bernstein_at_chebyshev(coeffs)
        recovered = _bernstein_interpolate_1d(vals)
        nptest.assert_allclose(recovered, coeffs, atol=1e-13)

    def test_roundtrip_cubic(self) -> None:
        """Interpolating cubic Bernstein at 4 nodes recovers coefficients."""
        coeffs = np.array([1.0, 2.0, 0.5, 3.0])
        vals = self._eval_bernstein_at_chebyshev(coeffs)
        recovered = _bernstein_interpolate_1d(vals)
        nptest.assert_allclose(recovered, coeffs, atol=1e-13)

    def test_single_coefficient(self) -> None:
        """Single-element input returns a copy."""
        coeffs = np.array([42.0])
        result = _bernstein_interpolate_1d(coeffs)
        nptest.assert_array_equal(result, coeffs)

    @staticmethod
    def _eval_bernstein_at_chebyshev(
        coeffs: npt.NDArray[np.floating[Any]],
    ) -> npt.NDArray[np.floating[Any]]:
        """Evaluate Bernstein polynomial at modified Chebyshev nodes."""
        n = len(coeffs)
        nodes = get_modified_chebyshev_nodes_1d(n)
        basis = np.empty((n, n), dtype=np.float64)
        _tabulate_Bernstein_basis_1D_core(np.int32(n - 1), nodes, basis)
        return basis @ coeffs


class TestBernsteinInterpolateND:
    """Tests for _bernstein_interpolate (N-D)."""

    def test_2d_roundtrip(self) -> None:
        """Interpolating 2D Bernstein coefficients at Chebyshev nodes recovers them."""
        coeffs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        # Evaluate at tensor-product Chebyshev nodes
        n0, n1 = coeffs.shape
        nodes0 = get_modified_chebyshev_nodes_1d(n0)
        nodes1 = get_modified_chebyshev_nodes_1d(n1)
        B0 = np.empty((n0, n0), dtype=np.float64)
        B1 = np.empty((n1, n1), dtype=np.float64)
        _tabulate_Bernstein_basis_1D_core(np.int32(n0 - 1), nodes0, B0)
        _tabulate_Bernstein_basis_1D_core(np.int32(n1 - 1), nodes1, B1)
        vals = B0 @ coeffs @ B1.T

        recovered = _bernstein_interpolate(vals)
        nptest.assert_allclose(recovered, coeffs, atol=1e-12)


# ---------------------------------------------------------------------------
# Normalise
# ---------------------------------------------------------------------------


class TestNormalise:
    """Tests for _normalise."""

    def test_scales_by_max(self) -> None:
        """Max absolute value becomes 1 after normalisation."""
        arr = np.array([2.0, -4.0, 1.0])
        _normalise(arr)
        nptest.assert_allclose(np.max(np.abs(arr)), 1.0, atol=1e-15)

    def test_zero_array_unchanged(self) -> None:
        """Zero array stays zero."""
        arr = np.zeros(5)
        _normalise(arr)
        nptest.assert_array_equal(arr, np.zeros(5))

    def test_modifies_in_place(self) -> None:
        """Normalise modifies the input array."""
        arr = np.array([2.0, -4.0])
        result = _normalise(arr)
        assert result is arr


# ---------------------------------------------------------------------------
# Auto-reduction
# ---------------------------------------------------------------------------


class TestAutoReduction:
    """Tests for _auto_reduction."""

    def test_constant_polynomial_reducible(self) -> None:
        """A constant represented as degree 2 should reduce to degree 0."""
        coeffs = np.array([3.0, 3.0, 3.0])
        reduced, changed = _auto_reduction(coeffs)
        assert changed
        assert reduced.shape[0] < coeffs.shape[0]

    def test_non_constant_not_reduced(self) -> None:
        """A true quadratic should not be reduced."""
        coeffs = np.array([0.0, 1.0, 0.0])  # x(1-x) in Bernstein
        _, changed = _auto_reduction(coeffs)
        assert not changed

    def test_2d_reducible_in_one_direction(self) -> None:
        """A 2D polynomial constant in one direction reduces along it."""
        # Constant in dim=0: all rows identical, strongly non-linear in dim=1
        coeffs = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        reduced, changed = _auto_reduction(coeffs)
        assert changed
        assert reduced.shape[0] < coeffs.shape[0]


# ---------------------------------------------------------------------------
# Extent computation
# ---------------------------------------------------------------------------


class TestResultantOrder:
    """Tests for _resultant_order."""

    def test_two_linears_2d(self) -> None:
        """Two (2,2) polynomials along dim=0."""
        ord_ = _resultant_order((2, 2), (2, 2), 0)
        # (2-1)*(2-1) + (2-1)*(2-1) + 1 = 3
        assert ord_ == (3,)

    def test_two_linears_2d_dim1(self) -> None:
        """Two (2,2) polynomials along dim=1."""
        ord_ = _resultant_order((2, 2), (2, 2), 1)
        assert ord_ == (3,)

    def test_mixed_orders(self) -> None:
        """Different orders in each direction."""
        ord_ = _resultant_order((3, 2), (2, 3), 0)
        # dim=0 eliminated. Remaining: dim=1.
        # (3-1)*(3-1) + (2-1)*(2-1) + 1 = 4 + 1 + 1 = 6
        assert ord_ == (6,)


class TestDiscriminantOrder:
    """Tests for _discriminant_order."""

    def test_quadratic_2d(self) -> None:
        """Discriminant of a (3, 2) polynomial along dim=0."""
        ord_ = _discriminant_order((3, 2), 0)
        # (2*3 - 3)*(2-1) + 1 = 3*1 + 1 = 4
        assert ord_ == (4,)

    def test_linear_2d(self) -> None:
        """Discriminant of a (2, 2) polynomial along dim=0."""
        ord_ = _discriminant_order((2, 2), 0)
        # (2*2-3)*(2-1) + 1 = 1*1 + 1 = 2
        assert ord_ == (2,)


# ---------------------------------------------------------------------------
# Resultant
# ---------------------------------------------------------------------------


class TestResultant1D:
    """Tests for 1D resultant computation."""

    def test_common_root(self) -> None:
        """Resultant is zero when polynomials share a root."""
        # p(x) = 1-2x, q(x) = x-0.5 — both zero at x=0.5
        p = np.array([1.0, -1.0])
        q = np.array([-0.5, 0.5])
        res = _resultant(p, q, 0)
        nptest.assert_allclose(float(res), 0.0, atol=1e-14)

    def test_coprime(self) -> None:
        """Resultant is nonzero for coprime polynomials."""
        p = np.array([0.0, 1.0])  # x
        q = np.array([1.0, 1.0])  # 1
        res = _resultant(p, q, 0)
        assert float(res) != 0.0

    def test_quadratic_vs_linear(self) -> None:
        """Resultant of quadratic and linear with different degrees."""
        # p(x) = x^2 in Bernstein degree 2: [0, 0, 1]
        # q(x) = x in Bernstein degree 1: [0, 1]
        # Common root at x=0
        p = np.array([0.0, 0.0, 1.0])
        q = np.array([0.0, 1.0])
        res = _resultant(p, q, 0)
        nptest.assert_allclose(float(res), 0.0, atol=1e-14)


class TestResultant2D:
    """Tests for 2D resultant computation."""

    def test_eliminates_dim0(self) -> None:
        """Resultant along dim=0 produces a 1D polynomial."""
        p = np.array([[0.0, 0.0], [1.0, 1.0]])
        q = np.array([[1.0, -1.0], [1.0, -1.0]])
        res = _resultant(p, q, 0)
        assert res.ndim == 1

    def test_result_proportional_to_expected(self) -> None:
        """Res_x(x, 1-2y) ~ (1-2y) in Bernstein basis."""
        p = np.array([[0.0, 0.0], [1.0, 1.0]])  # x
        q = np.array([[1.0, -1.0], [1.0, -1.0]])  # 1-2y
        res = _resultant(p, q, 0)
        # Result should be proportional to [1, -1] (i.e. 1-2y in Bernstein)
        _normalise(res)
        nptest.assert_allclose(np.abs(res), [1.0, 1.0], atol=0.1)

    def test_coprime_gives_nonzero(self) -> None:
        """Coprime polynomials in x give nonzero resultant."""
        p = np.array([[0.0, 0.0], [1.0, 1.0]])  # x
        q = np.array([[-0.5, -0.5], [0.5, 0.5]])  # x - 0.5
        res = _resultant(p, q, 0)
        assert np.any(res != 0.0)


class TestResultantValidation:
    """Input validation tests for _resultant."""

    def test_mismatched_ndim(self) -> None:
        with pytest.raises(ValueError, match="same number of dimensions"):
            _resultant(np.array([1.0, 2.0]), np.array([[1.0, 2.0]]), 0)

    def test_non_floating(self) -> None:
        with pytest.raises(ValueError, match="floating dtype"):
            _resultant(np.array([1, 2]), np.array([1, 2]), 0)

    def test_dim_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="dim must be"):
            _resultant(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 1)

    def test_order_too_small(self) -> None:
        with pytest.raises(ValueError, match="order >= 2"):
            _resultant(np.array([1.0]), np.array([1.0, 2.0]), 0)


# ---------------------------------------------------------------------------
# Discriminant
# ---------------------------------------------------------------------------


class TestDiscriminant1D:
    """Tests for 1D discriminant computation."""

    def test_double_root(self) -> None:
        """Discriminant vanishes for polynomial with double root."""
        # (x-0.5)^2 in Bernstein degree 2: [0.25, -0.25, 0.25]
        p = np.array([0.25, -0.25, 0.25])
        disc = _discriminant(p, 0)
        nptest.assert_allclose(float(disc), 0.0, atol=1e-14)

    def test_distinct_roots(self) -> None:
        """Discriminant is nonzero for polynomial with distinct roots."""
        # x(1-x) in Bernstein: [0, 0.5, 0]
        p = np.array([0.0, 0.5, 0.0])
        disc = _discriminant(p, 0)
        assert float(disc) != 0.0

    def test_linear_raises(self) -> None:
        """Discriminant of a linear raises because degree must be >= 2."""
        p = np.array([1.0, 3.0])
        with pytest.raises(ValueError, match="order >= 3"):
            _discriminant(p, 0)


class TestDiscriminant2D:
    """Tests for 2D discriminant computation."""

    def test_reduces_dimension(self) -> None:
        """Discriminant along one dim produces result with one fewer dim."""
        p = np.array([[1.0, 0.0, 1.0], [2.0, 1.0, 2.0]])
        disc = _discriminant(p, 1)
        assert disc.ndim == 1

    def test_discriminant_validation(self) -> None:
        """Non-floating input raises ValueError."""
        with pytest.raises(ValueError, match="floating dtype"):
            _discriminant(np.array([1, 2, 3]), 0)
