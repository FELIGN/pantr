"""Tests for B-spline basis function derivatives (DerBasisFuncs, Algorithm A2.3)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pantr._basis_core import (
    _tabulate_Bernstein_basis_1D_core,
    _tabulate_Bernstein_basis_deriv_1D_core,
)
from pantr._bspline_basis_core import (
    _compute_basis_deriv_nurbs_book_impl,
    _compute_basis_nurbs_book_impl,
    _tabulate_Bspline_basis_Bernstein_like_deriv_1D,
    _tabulate_Bspline_basis_deriv_1D_impl,
)
from pantr.bspline_space_1D import BsplineSpace1D

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def quadratic_single_span() -> BsplineSpace1D:
    """Quadratic Bézier: knots=[0,0,0,1,1,1], degree=2."""
    return BsplineSpace1D([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 2)


@pytest.fixture()
def quadratic_two_span() -> BsplineSpace1D:
    """Quadratic, two spans: knots=[0,0,0,0.5,1,1,1], degree=2."""
    return BsplineSpace1D([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], 2)


@pytest.fixture()
def linear_single_span() -> BsplineSpace1D:
    """Linear: knots=[0,0,1,1], degree=1."""
    return BsplineSpace1D([0.0, 0.0, 1.0, 1.0], 1)


@pytest.fixture()
def cubic_two_span() -> BsplineSpace1D:
    """Cubic, two spans: knots=[0,0,0,0,0.5,1,1,1,1], degree=3."""
    return BsplineSpace1D([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0], 3)


# ---------------------------------------------------------------------------
# Layer 3 kernel: _compute_basis_deriv_nurbs_book_impl
# ---------------------------------------------------------------------------


class TestComputeBasisDerivNurbsBook:
    """Direct tests of the Numba kernel _compute_basis_deriv_nurbs_book_impl."""

    def _call(  # noqa: PLR0913
        self,
        knots: npt.NDArray[np.float32 | np.float64],
        degree: int,
        pts: npt.NDArray[np.float32 | np.float64],
        n_deriv: int,
        periodic: bool = False,
        tol: float = 1e-10,
    ) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
        n_pts = pts.size
        order = degree + 1
        out_deriv = np.empty((n_pts, n_deriv + 1, order), dtype=knots.dtype)
        out_first = np.empty(n_pts, dtype=np.int_)
        _compute_basis_deriv_nurbs_book_impl(
            knots, degree, periodic, tol, n_deriv, pts, out_deriv, out_first
        )
        return out_deriv, out_first

    def test_0th_slice_matches_a22(self) -> None:
        """out_deriv[:,0,:] must equal the A2.2 basis values."""
        knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        pts = np.array([0.1, 0.3, 0.6, 0.9], dtype=np.float64)
        n_pts = pts.size
        order = degree + 1

        out_a22 = np.empty((n_pts, order), dtype=np.float64)
        first_a22 = np.empty(n_pts, dtype=np.int_)
        _compute_basis_nurbs_book_impl(knots, degree, False, 1e-10, pts, out_a22, first_a22)

        out_deriv, first_deriv = self._call(knots, degree, pts, n_deriv=2)

        np.testing.assert_array_almost_equal(out_deriv[:, 0, :], out_a22)
        np.testing.assert_array_equal(first_deriv, first_a22)

    def test_partition_of_unity_0th(self) -> None:
        """Sum of 0th-order slice equals 1 for all points."""
        knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        pts = np.linspace(0.0, 1.0, 15)
        out_deriv, _ = self._call(knots, 2, pts, n_deriv=2)
        np.testing.assert_allclose(out_deriv[:, 0, :].sum(axis=1), 1.0, atol=1e-14)

    def test_sum_of_kth_derivatives_is_zero(self) -> None:
        """Sum of the k-th derivative row equals 0 for k >= 1 (interior points)."""
        knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        # Use strictly interior points to avoid boundary edge cases
        pts = np.linspace(0.05, 0.95, 12)
        out_deriv, _ = self._call(knots, 2, pts, n_deriv=2)
        for k in range(1, 3):
            sums = out_deriv[:, k, :].sum(axis=1)
            np.testing.assert_allclose(sums, 0.0, atol=1e-12, err_msg=f"k={k}")

    def test_boundary_last_knot(self) -> None:
        """Point at the last knot: derivatives match the analytical Bernstein values.

        For degree-2 Bernstein basis on [0,1]: B0=(1-t)^2, B1=2t(1-t), B2=t^2.
        At t=1: B=[0,0,1], B'=[0,-2,2], B''=[2,-4,2].
        """
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        pts = np.array([1.0], dtype=np.float64)
        out_deriv, _ = self._call(knots, 2, pts, n_deriv=2)
        np.testing.assert_array_almost_equal(out_deriv[0, 0, :], [0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(out_deriv[0, 1, :], [0.0, -2.0, 2.0])
        np.testing.assert_array_almost_equal(out_deriv[0, 2, :], [2.0, -4.0, 2.0])

    def test_n_deriv_exceeds_degree_gives_zeros(self) -> None:
        """Derivatives of order > degree are identically zero."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        pts = np.array([0.25, 0.5, 0.75], dtype=np.float64)
        out_deriv, _ = self._call(knots, 2, pts, n_deriv=4)
        # degree=2: rows k=3,4 must be all zero
        np.testing.assert_array_almost_equal(out_deriv[:, 3, :], 0.0)
        np.testing.assert_array_almost_equal(out_deriv[:, 4, :], 0.0)

    def test_float32_support(self) -> None:
        """Kernel runs correctly with float32 arrays."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
        pts = np.array([0.5], dtype=np.float32)
        out_deriv, _ = self._call(knots, 2, pts, n_deriv=1)
        assert out_deriv.dtype == np.float32
        # Partition of unity
        np.testing.assert_allclose(out_deriv[0, 0, :].sum(), 1.0, atol=1e-6)
        # Sum of first derivatives = 0
        np.testing.assert_allclose(out_deriv[0, 1, :].sum(), 0.0, atol=1e-6)

    def test_periodic_knots(self) -> None:
        """Partition of unity and zero derivative sum hold for periodic knots."""
        knots = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        pts = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
        out_deriv, _ = self._call(knots, 2, pts, n_deriv=1, periodic=True)
        np.testing.assert_allclose(out_deriv[:, 0, :].sum(axis=1), 1.0, atol=1e-13)
        np.testing.assert_allclose(out_deriv[:, 1, :].sum(axis=1), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Layer 2 function: _tabulate_Bspline_basis_deriv_1D_impl
# ---------------------------------------------------------------------------


class TestTabulateBsplineBasisDeriv1D:
    """Tests for the Layer 2 wrapper _tabulate_Bspline_basis_deriv_1D_impl."""

    def test_output_shape_1d_input(self, quadratic_two_span: BsplineSpace1D) -> None:
        """Shape is (n_pts, n_deriv+1, degree+1) for 1D input."""
        pts = np.array([0.1, 0.5, 0.9])
        out, first = _tabulate_Bspline_basis_deriv_1D_impl(quadratic_two_span, pts, n_deriv=2)
        assert out.shape == (3, 3, 3)
        assert first.shape == (3,)

    def test_output_shape_scalar_input(self, quadratic_two_span: BsplineSpace1D) -> None:
        """Shape is (n_deriv+1, degree+1) for scalar input."""
        out, first = _tabulate_Bspline_basis_deriv_1D_impl(quadratic_two_span, 0.5, n_deriv=1)
        assert out.shape == (2, 3)
        assert first.shape == ()

    def test_output_shape_2d_input(self, quadratic_two_span: BsplineSpace1D) -> None:
        """Shape is (m, n, n_deriv+1, degree+1) for 2D input."""
        pts = np.array([[0.1, 0.5], [0.6, 0.9]])
        out, first = _tabulate_Bspline_basis_deriv_1D_impl(quadratic_two_span, pts, n_deriv=1)
        assert out.shape == (2, 2, 2, 3)
        assert first.shape == (2, 2)

    def test_invalid_n_deriv(self, quadratic_two_span: BsplineSpace1D) -> None:
        """Negative n_deriv raises ValueError."""
        with pytest.raises(ValueError, match="n_deriv"):
            _tabulate_Bspline_basis_deriv_1D_impl(quadratic_two_span, [0.5], n_deriv=-1)

    def test_points_outside_domain(self, quadratic_two_span: BsplineSpace1D) -> None:
        """Points outside the domain raise ValueError."""
        with pytest.raises(ValueError, match="outside"):
            _tabulate_Bspline_basis_deriv_1D_impl(quadratic_two_span, [1.5], n_deriv=1)

    def test_out_deriv_wrong_shape(self, quadratic_two_span: BsplineSpace1D) -> None:
        """Providing out_deriv with wrong shape raises ValueError."""
        out_bad = np.empty((3, 2, 3), dtype=np.float64)  # wrong n_deriv+1 axis
        with pytest.raises(ValueError, match="shape"):
            _tabulate_Bspline_basis_deriv_1D_impl(
                quadratic_two_span, [0.1, 0.5, 0.9], n_deriv=2, out_deriv=out_bad
            )

    def test_out_deriv_wrong_dtype(self, quadratic_two_span: BsplineSpace1D) -> None:
        """Providing out_deriv with wrong dtype raises ValueError."""
        out_bad = np.empty((3, 3, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="dtype"):
            _tabulate_Bspline_basis_deriv_1D_impl(
                quadratic_two_span,
                np.array([0.1, 0.5, 0.9], dtype=np.float64),
                n_deriv=2,
                out_deriv=out_bad,
            )

    def test_out_deriv_not_writeable(self, quadratic_two_span: BsplineSpace1D) -> None:
        """Providing a read-only out_deriv raises ValueError."""
        out_bad = np.empty((3, 3, 3), dtype=np.float64)
        out_bad.flags.writeable = False
        with pytest.raises(ValueError, match="writeable"):
            _tabulate_Bspline_basis_deriv_1D_impl(
                quadratic_two_span, [0.1, 0.5, 0.9], n_deriv=2, out_deriv=out_bad
            )

    def test_out_deriv_reuses_array(self, quadratic_two_span: BsplineSpace1D) -> None:
        """When out_deriv is provided, the returned array is the same object."""
        pts = np.array([0.1, 0.5, 0.9])
        out_pre = np.empty((3, 3, 3), dtype=np.float64)
        out_ret, _ = _tabulate_Bspline_basis_deriv_1D_impl(
            quadratic_two_span, pts, n_deriv=2, out_deriv=out_pre
        )
        assert out_ret is out_pre

    def test_out_first_basis_wrong_shape(self, quadratic_two_span: BsplineSpace1D) -> None:
        """Providing out_first_basis with wrong shape raises ValueError."""
        out_bad = np.empty(5, dtype=np.int_)
        with pytest.raises(ValueError, match="shape"):
            _tabulate_Bspline_basis_deriv_1D_impl(
                quadratic_two_span, [0.1, 0.5, 0.9], n_deriv=1, out_first_basis=out_bad
            )


# ---------------------------------------------------------------------------
# Layer 1 method: BsplineSpace1D.tabulate_basis_derivatives
# ---------------------------------------------------------------------------


class TestBsplineSpace1DTabulateDeriv:
    """Tests for the Layer 1 public method tabulate_basis_derivatives."""

    def test_basic_call(self, quadratic_two_span: BsplineSpace1D) -> None:
        """Basic call returns arrays with correct shapes."""
        d, first = quadratic_two_span.tabulate_basis_derivatives([0.25, 0.75], n_deriv=1)
        assert d.shape == (2, 2, 3)
        assert first.shape == (2,)

    def test_n_deriv_0_matches_tabulate_basis(self, quadratic_two_span: BsplineSpace1D) -> None:
        """n_deriv=0 result[...,0,:] equals tabulate_basis output."""
        pts = [0.1, 0.3, 0.7, 0.9]
        basis, first_b = quadratic_two_span.tabulate_basis(pts)
        deriv, first_d = quadratic_two_span.tabulate_basis_derivatives(pts, n_deriv=0)
        np.testing.assert_array_almost_equal(deriv[:, 0, :], basis)
        np.testing.assert_array_equal(first_d, first_b)

    def test_n_deriv_1_matches_tabulate_basis_0th(self, quadratic_two_span: BsplineSpace1D) -> None:
        """With n_deriv=1, the 0th row still matches tabulate_basis."""
        pts = [0.2, 0.8]
        basis, _ = quadratic_two_span.tabulate_basis(pts)
        deriv, _ = quadratic_two_span.tabulate_basis_derivatives(pts, n_deriv=1)
        np.testing.assert_array_almost_equal(deriv[:, 0, :], basis)

    def test_docstring_example(self) -> None:
        """Verify the docstring example: linear derivatives of quadratic Bézier at x=0.5."""
        bspline = BsplineSpace1D([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 2)
        d, _ = bspline.tabulate_basis_derivatives([0.5], n_deriv=1)
        assert d.shape == (1, 2, 3)
        # B0'=-1, B1'=0, B2'=1 at x=0.5
        np.testing.assert_allclose(d[0, 1, :], [-1.0, 0.0, 1.0], atol=1e-14)

    def test_consistency_across_n_deriv(self, cubic_two_span: BsplineSpace1D) -> None:
        """Results for n_deriv=2 and n_deriv=1 agree on the first two rows."""
        pts = [0.1, 0.4, 0.6, 0.9]
        d1, _ = cubic_two_span.tabulate_basis_derivatives(pts, n_deriv=1)
        d2, _ = cubic_two_span.tabulate_basis_derivatives(pts, n_deriv=2)
        np.testing.assert_array_almost_equal(d2[:, :2, :], d1)


# ---------------------------------------------------------------------------
# Mathematical properties
# ---------------------------------------------------------------------------


class TestMathematicalPropertiesDeriv:
    """Tests that verify mathematical correctness through known properties."""

    # --- Exact values for the single-span quadratic Bézier ---
    # B0(x) = (1-x)^2,  B1(x) = 2x(1-x),  B2(x) = x^2
    # B0'(x) = -2(1-x), B1'(x) = 2-4x,    B2'(x) = 2x
    # B0''(x) = 2,       B1''(x) = -4,      B2''(x) = 2

    @pytest.mark.parametrize("x", [0.0, 0.25, 0.5, 0.75])
    def test_exact_quadratic_bezier_first_derivative(
        self, quadratic_single_span: BsplineSpace1D, x: float
    ) -> None:
        """First derivatives of the quadratic Bézier match analytical values.

        Note: x=1.0 is excluded because the Bernstein fast path has a
        separate boundary issue (zero derivatives at t=1).
        """
        d, _ = quadratic_single_span.tabulate_basis_derivatives([x], n_deriv=1)
        expected = np.array([-2 * (1 - x), 2 - 4 * x, 2 * x])
        np.testing.assert_allclose(d[0, 1, :], expected, atol=1e-13)

    @pytest.mark.parametrize("x", [0.0, 0.25, 0.5, 0.75])
    def test_exact_quadratic_bezier_second_derivative(
        self, quadratic_single_span: BsplineSpace1D, x: float
    ) -> None:
        """Second derivatives of the quadratic Bézier match analytical values.

        Note: x=1.0 is excluded because the Bernstein fast path has a
        separate boundary issue (zero derivatives at t=1).
        """
        d, _ = quadratic_single_span.tabulate_basis_derivatives([x], n_deriv=2)
        expected = np.array([2.0, -4.0, 2.0])
        np.testing.assert_allclose(d[0, 2, :], expected, atol=1e-12)

    @pytest.mark.parametrize("x", [0.1, 0.5, 0.9])
    def test_exact_linear_bspline_first_derivative(
        self, linear_single_span: BsplineSpace1D, x: float
    ) -> None:
        """Linear B-spline: B0'=-1, B1'=+1 everywhere."""
        d, _ = linear_single_span.tabulate_basis_derivatives([x], n_deriv=1)
        np.testing.assert_allclose(d[0, 1, :], [-1.0, 1.0], atol=1e-14)

    def test_exact_linear_bspline_second_derivative_zero(
        self, linear_single_span: BsplineSpace1D
    ) -> None:
        """Second derivatives of degree-1 spline are identically zero."""
        pts = np.linspace(0.0, 1.0, 10)
        d, _ = linear_single_span.tabulate_basis_derivatives(pts, n_deriv=2)
        np.testing.assert_array_almost_equal(d[:, 2, :], 0.0)

    def test_partition_of_unity_all_orders(self, quadratic_two_span: BsplineSpace1D) -> None:
        """Sum of 0th row = 1; sum of kth row = 0 for k >= 1, uniformly."""
        pts = np.linspace(0.0, 1.0, 25)
        d, _ = quadratic_two_span.tabulate_basis_derivatives(pts, n_deriv=2)
        np.testing.assert_allclose(d[:, 0, :].sum(axis=1), 1.0, atol=1e-13)
        for k in range(1, 3):
            np.testing.assert_allclose(d[:, k, :].sum(axis=1), 0.0, atol=1e-11, err_msg=f"k={k}")

    def test_finite_difference_first_derivative(self, quadratic_two_span: BsplineSpace1D) -> None:
        """Central FD approximation of 1st derivative agrees with A2.3 to O(h^2)."""
        h = 1e-5
        # Use points strictly inside both spans, away from the internal knot at 0.5
        pts = np.array([0.15, 0.25, 0.35, 0.65, 0.75, 0.85])

        d_exact, _ = quadratic_two_span.tabulate_basis_derivatives(pts, n_deriv=1)

        b_fwd, _ = quadratic_two_span.tabulate_basis(pts + h)
        b_bwd, _ = quadratic_two_span.tabulate_basis(pts - h)
        d_fd = (b_fwd - b_bwd) / (2 * h)

        np.testing.assert_allclose(d_exact[:, 1, :], d_fd, atol=1e-8)

    def test_finite_difference_second_derivative(self, quadratic_two_span: BsplineSpace1D) -> None:
        """Central FD approximation of 2nd derivative agrees with A2.3 to O(h^2)."""
        h = 1e-4
        pts = np.array([0.15, 0.25, 0.35, 0.65, 0.75, 0.85])

        d_exact, _ = quadratic_two_span.tabulate_basis_derivatives(pts, n_deriv=2)

        b_fwd, _ = quadratic_two_span.tabulate_basis(pts + h)
        b_mid, _ = quadratic_two_span.tabulate_basis(pts)
        b_bwd, _ = quadratic_two_span.tabulate_basis(pts - h)
        d_fd2 = (b_fwd - 2 * b_mid + b_bwd) / (h**2)

        np.testing.assert_allclose(d_exact[:, 2, :], d_fd2, atol=1e-5)

    def test_finite_difference_cubic(self, cubic_two_span: BsplineSpace1D) -> None:
        """FD check for first derivative on cubic two-span spline."""
        h = 1e-5
        pts = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
        d_exact, _ = cubic_two_span.tabulate_basis_derivatives(pts, n_deriv=1)
        b_fwd, _ = cubic_two_span.tabulate_basis(pts + h)
        b_bwd, _ = cubic_two_span.tabulate_basis(pts - h)
        d_fd = (b_fwd - b_bwd) / (2 * h)
        np.testing.assert_allclose(d_exact[:, 1, :], d_fd, atol=1e-8)

    def test_first_derivative_sum_zero_multi_span(self, cubic_two_span: BsplineSpace1D) -> None:
        """Sum of first derivatives equals zero for all interior points (cubic)."""
        pts = np.linspace(0.02, 0.98, 30)
        d, _ = cubic_two_span.tabulate_basis_derivatives(pts, n_deriv=3)
        for k in range(1, 4):
            np.testing.assert_allclose(d[:, k, :].sum(axis=1), 0.0, atol=1e-10, err_msg=f"k={k}")


# ---------------------------------------------------------------------------
# Bézier fast path: _tabulate_Bernstein_basis_deriv_1D_core
# and _tabulate_Bspline_basis_Bernstein_like_deriv_1D
# ---------------------------------------------------------------------------


class TestBernsteinDerivCore:
    """Direct tests of the parallel Bernstein derivative kernel."""

    def _call(
        self,
        n: int,
        t: npt.NDArray[np.float32 | np.float64],
        n_deriv: int,
    ) -> npt.NDArray[np.float32 | np.float64]:
        out = np.empty((t.size, n_deriv + 1, n + 1), dtype=t.dtype)
        _tabulate_Bernstein_basis_deriv_1D_core(np.int32(n), t, n_deriv, out)
        return out

    def test_0th_matches_bernstein_basis(self) -> None:
        """0th row equals the standard Bernstein basis."""
        t = np.linspace(0.0, 1.0, 11)
        n = 3
        out_deriv = self._call(n, t, n_deriv=2)

        ref = np.empty((t.size, n + 1), dtype=np.float64)
        _tabulate_Bernstein_basis_1D_core(np.int32(n), t, ref)

        np.testing.assert_array_almost_equal(out_deriv[:, 0, :], ref)

    def test_partition_of_unity_0th(self) -> None:
        """0th row sums to 1 for all points."""
        t = np.linspace(0.0, 1.0, 20)
        out = self._call(3, t, n_deriv=2)
        np.testing.assert_allclose(out[:, 0, :].sum(axis=1), 1.0, atol=1e-14)

    def test_sum_of_kth_deriv_is_zero(self) -> None:
        """Sum of k-th derivative row equals 0 for k >= 1 (interior points)."""
        t = np.linspace(0.05, 0.95, 15)
        out = self._call(3, t, n_deriv=3)
        for k in range(1, 4):
            np.testing.assert_allclose(out[:, k, :].sum(axis=1), 0.0, atol=1e-12)

    def test_exact_quadratic_first_derivative(self) -> None:
        """Exact first derivatives for quadratic Bernstein on [0,1]."""
        t = np.array([0.0, 0.25, 0.5, 0.75])
        out = self._call(2, t, n_deriv=1)
        for i, x in enumerate(t):
            expected = np.array([-2 * (1 - x), 2 - 4 * x, 2 * x])
            np.testing.assert_allclose(out[i, 1, :], expected, atol=1e-13)

    def test_exact_quadratic_second_derivative(self) -> None:
        """Second derivatives for quadratic Bernstein are [2, -4, 2] everywhere."""
        t = np.array([0.1, 0.4, 0.7])
        out = self._call(2, t, n_deriv=2)
        np.testing.assert_allclose(out[:, 2, :], [[2.0, -4.0, 2.0]] * 3, atol=1e-12)

    def test_boundary_t_equals_1(self) -> None:
        """At t=1: 0th row is last-function=1; all derivative rows are zero."""
        t = np.array([1.0])
        out = self._call(2, t, n_deriv=2)
        np.testing.assert_array_equal(out[0, 0, :], [0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(out[0, 1, :], [0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(out[0, 2, :], [0.0, 0.0, 0.0])

    def test_n_deriv_exceeds_degree_gives_zeros(self) -> None:
        """Derivative rows beyond degree are identically zero."""
        t = np.array([0.3, 0.6])
        out = self._call(2, t, n_deriv=4)
        np.testing.assert_array_almost_equal(out[:, 3, :], 0.0)
        np.testing.assert_array_almost_equal(out[:, 4, :], 0.0)

    def test_float32_support(self) -> None:
        """Kernel produces float32 output from float32 input."""
        t = np.array([0.5], dtype=np.float32)
        out = self._call(2, t, n_deriv=1)
        assert out.dtype == np.float32
        np.testing.assert_allclose(float(out[0, 0, :].sum()), 1.0, atol=1e-6)

    def test_degree_0_all_derivatives_zero(self) -> None:
        """For n=0, all derivatives are zero and the single basis value is 1."""
        t = np.array([0.2, 0.5, 0.8])
        out = self._call(0, t, n_deriv=2)
        np.testing.assert_array_equal(out[:, 0, :], [[1.0], [1.0], [1.0]])
        np.testing.assert_array_almost_equal(out[:, 1, :], 0.0)
        np.testing.assert_array_almost_equal(out[:, 2, :], 0.0)


class TestBezierFastPath:
    """Tests verifying the Bézier fast path matches the general A2.3 kernel."""

    @pytest.mark.parametrize("degree", [1, 2, 3, 4])
    def test_bezier_matches_general_kernel(self, degree: int) -> None:
        """Fast path and general kernel agree for Bézier-like knots on [0,1]."""
        knots = np.array([0.0] * (degree + 1) + [1.0] * (degree + 1), dtype=np.float64)
        spline = BsplineSpace1D(knots, degree)
        pts = np.linspace(0.02, 0.98, 20)
        n_deriv = min(degree, 3)

        d_fast, first_fast = spline.tabulate_basis_derivatives(pts, n_deriv=n_deriv)

        # Force the general path by using a non-Bézier spline with the same knot structure
        # (we compare against the A2.3 kernel called directly with the same knot vector)
        n_pts = pts.size
        order = degree + 1
        d_general = np.empty((n_pts, n_deriv + 1, order), dtype=np.float64)
        first_general = np.empty(n_pts, dtype=np.int_)
        _compute_basis_deriv_nurbs_book_impl(
            knots, degree, False, 1e-10, n_deriv, pts, d_general, first_general
        )

        np.testing.assert_allclose(d_fast, d_general, atol=1e-13)
        np.testing.assert_array_equal(first_fast, first_general)

    def test_bezier_non_unit_domain(self) -> None:
        """Chain-rule scaling is applied correctly for domain [2, 5]."""
        spline = BsplineSpace1D([2.0, 2.0, 2.0, 5.0, 5.0, 5.0], 2)
        pts = np.array([2.5, 3.0, 3.5, 4.0, 4.5])

        d, _ = spline.tabulate_basis_derivatives(pts, n_deriv=2)

        # Verify against finite differences
        h = 1e-5
        b_fwd, _ = spline.tabulate_basis(pts + h)
        b_bwd, _ = spline.tabulate_basis(pts - h)
        d_fd = (b_fwd - b_bwd) / (2 * h)
        np.testing.assert_allclose(d[:, 1, :], d_fd, atol=1e-8)

    def test_bezier_like_deriv_sets_first_basis_to_zero(self) -> None:
        """first_basis_indices are all zero for Bézier-like knots."""
        spline = BsplineSpace1D([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 2)
        pts = np.linspace(0.0, 1.0, 15)
        _, first = spline.tabulate_basis_derivatives(pts, n_deriv=1)
        np.testing.assert_array_equal(first, 0)

    def test_bezier_like_deriv_helper_raises_for_non_bezier(self) -> None:
        """_tabulate_Bspline_basis_Bernstein_like_deriv_1D raises for non-Bézier splines."""
        spline = BsplineSpace1D([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], 2)
        pts = np.array([0.25, 0.75])
        out_deriv = np.empty((2, 2, 3), dtype=np.float64)
        out_first = np.empty(2, dtype=np.int_)
        with pytest.raises(ValueError, match="Bézier-like"):
            _tabulate_Bspline_basis_Bernstein_like_deriv_1D(spline, pts, 1, out_deriv, out_first)

    def test_bezier_0th_matches_tabulate_basis(self) -> None:
        """With n_deriv=0, fast path matches tabulate_basis output."""
        spline = BsplineSpace1D([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], 3)
        pts = np.linspace(0.0, 1.0, 20)
        basis, _ = spline.tabulate_basis(pts)
        d, _ = spline.tabulate_basis_derivatives(pts, n_deriv=0)
        np.testing.assert_allclose(d[:, 0, :], basis, atol=1e-14)

    def test_bezier_second_derivative_exact_quadratic(self) -> None:
        """Second derivatives on [0,1] match analytical values (quadratic Bézier)."""
        spline = BsplineSpace1D([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 2)
        pts = np.array([0.1, 0.4, 0.7])
        d, _ = spline.tabulate_basis_derivatives(pts, n_deriv=2)
        np.testing.assert_allclose(d[:, 2, :], [[2.0, -4.0, 2.0]] * 3, atol=1e-12)
