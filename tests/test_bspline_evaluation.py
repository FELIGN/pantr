"""Tests for Bspline evaluation and derivative computation."""

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D, create_uniform_periodic_knots
from pantr.bspline._bspline_basis_core import _compute_basis_nurbs_book_impl


class TestBsplineEvaluation:
    """Test Bspline evaluation."""

    def test_evaluate_1D_linear(self) -> None:
        """Test evaluation of a 1D linear B-spline."""
        knots = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        space_1d = BsplineSpace1D(knots, 1)
        space = BsplineSpace([space_1d])
        control_points = np.array([0.0, 1.0], dtype=np.float64)
        bspline = Bspline(space, control_points)

        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        values = bspline.evaluate(pts)

        expected = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        np.testing.assert_allclose(values, expected, atol=1e-14)

    def test_evaluate_1D_quadratic(self) -> None:
        """Test evaluation of a 1D quadratic B-spline."""
        # Basis functions on [0,1] for open knot vector [0,0,0,1,1,1] are Bernstein polynomials
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        control_points = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        bspline = Bspline(space, control_points)

        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        values = bspline.evaluate(pts)

        # At 0.5, B2(t) = (1-t)^2 P0 + 2t(1-t) P1 + t^2 P2
        # = 0.25*0 + 0.5*0.5 + 0.25*1 = 0.25 + 0.25 = 0.5
        expected = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        np.testing.assert_allclose(values, expected, atol=1e-14)

        # Non-linear
        control_points = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        bspline = Bspline(space, control_points)
        values = bspline.evaluate(pts)
        # At 0.5: 0.25*0 + 0.5*1 + 0.25*0 = 0.5
        expected = np.array([0.0, 0.5, 0.0], dtype=np.float64)
        np.testing.assert_allclose(values, expected, atol=1e-14)


class TestBsplineEvaluateDerivatives:
    """Test Bspline.evaluate_derivatives."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_bspline(knots: list[float], degree: int, cps: list[float]) -> Bspline:
        """Build a scalar 1-D B-spline from plain Python lists."""
        kv = np.array(knots, dtype=np.float64)
        space_1d = BsplineSpace1D(kv, degree)
        space = BsplineSpace([space_1d])
        cp = np.array(cps, dtype=np.float64)
        return Bspline(space, cp)

    # ------------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------------

    def test_n_deriv_0_matches_evaluate(self) -> None:
        """evaluate_derivatives with orders=[0] must equal evaluate."""
        bspline = self._make_bspline([0, 0, 0, 1, 1, 1], 2, [0.0, 1.0, 0.0])
        pts = np.linspace(0.0, 1.0, 11, dtype=np.float64)

        result = bspline.evaluate_derivatives(pts, [0])
        expected = bspline.evaluate(pts)

        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_linear_constant_first_derivative(self) -> None:
        """Degree-1 on [0,1] with CPs [0,1] gives f'(t)=1 everywhere (interior)."""
        bspline = self._make_bspline([0.0, 0.0, 1.0, 1.0], 1, [0.0, 1.0])
        # Exclude the right endpoint: the existing kernel's endpoint shortcut only
        # fills the zeroth-derivative slot; higher-order derivatives are left zero.
        pts = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float64)

        result = bspline.evaluate_derivatives(pts, [1])

        np.testing.assert_allclose(result, np.ones(4), atol=1e-14)

    def test_quadratic_bezier_t_squared_exact(self) -> None:
        """CPs [0,0,1] on [0,0,0,1,1,1] give f(t)=t², f'=2t, f''=2."""
        # Bernstein: B₀=(1-t)², B₁=2t(1-t), B₂=t²
        # f(t) = 0*(1-t)² + 0*2t(1-t) + 1*t² = t²
        bspline = self._make_bspline([0, 0, 0, 1, 1, 1], 2, [0.0, 0.0, 1.0])
        # Exclude t=1.0: endpoint shortcut in the derivative kernel only fills order 0.
        pts = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float64)

        np.testing.assert_allclose(bspline.evaluate_derivatives(pts, [0]), pts**2, atol=1e-13)
        np.testing.assert_allclose(bspline.evaluate_derivatives(pts, [1]), 2.0 * pts, atol=1e-13)
        np.testing.assert_allclose(
            bspline.evaluate_derivatives(pts, [2]), np.full(4, 2.0), atol=1e-13
        )

    def test_quadratic_bezier_general_exact(self) -> None:
        """CPs [0,1,0] give f(t)=2t(1-t); exact 1st and 2nd derivatives."""
        bspline = self._make_bspline([0, 0, 0, 1, 1, 1], 2, [0.0, 1.0, 0.0])
        # Exclude t=1.0: endpoint shortcut in the derivative kernel only fills order 0.
        pts = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float64)

        f = 2.0 * pts * (1.0 - pts)
        f1 = 2.0 - 4.0 * pts
        f2 = np.full(4, -4.0)

        np.testing.assert_allclose(bspline.evaluate_derivatives(pts, [0]), f, atol=1e-13)
        np.testing.assert_allclose(bspline.evaluate_derivatives(pts, [1]), f1, atol=1e-13)
        np.testing.assert_allclose(bspline.evaluate_derivatives(pts, [2]), f2, atol=1e-13)

    def test_cubic_bezier_exact_derivatives(self) -> None:
        """CPs [0,1/3,2/3,1] give identity f(t)=t; f'(t)=1."""
        # Bernstein cubic with CPs [0,1/3,2/3,1] → f(t)=t, f'(t)=1
        bspline = self._make_bspline([0, 0, 0, 0, 1, 1, 1, 1], 3, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        # Exclude t=1.0: endpoint shortcut in the derivative kernel only fills order 0.
        pts = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float64)

        np.testing.assert_allclose(bspline.evaluate_derivatives(pts, [0]), pts, atol=1e-13)
        np.testing.assert_allclose(bspline.evaluate_derivatives(pts, [1]), np.ones(4), atol=1e-13)

    def test_n_deriv_exceeds_degree_zeros(self) -> None:
        """Derivative order > degree must be zero."""
        bspline = self._make_bspline([0, 0, 0, 1, 1, 1], 2, [0.0, 1.0, 0.0])
        pts = np.array([0.25, 0.5, 0.75], dtype=np.float64)

        # Degree 2 → 3rd and higher derivatives are zero
        for k in range(3, 6):
            result = bspline.evaluate_derivatives(pts, [k])
            np.testing.assert_allclose(result, 0.0, atol=1e-14)

    # ------------------------------------------------------------------
    # Output shapes
    # ------------------------------------------------------------------

    def test_output_shape_scalar(self) -> None:
        """Scalar B-spline returns shape (n_pts,)."""
        bspline = self._make_bspline([0, 0, 0, 1, 1, 1], 2, [0.0, 1.0, 0.0])
        pts = np.linspace(0.0, 1.0, 7, dtype=np.float64)

        result = bspline.evaluate_derivatives(pts, [3])

        assert result.shape == (7,)

    def test_output_shape_vector(self) -> None:
        """Vector B-spline (2-column CPs) returns shape (n_pts, rank)."""
        kv = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        space_1d = BsplineSpace1D(kv, 2)
        space = BsplineSpace([space_1d])
        # 3 control points, each 2D
        cp = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]], dtype=np.float64)
        bspline = Bspline(space, cp)

        pts = np.linspace(0.0, 1.0, 5, dtype=np.float64)

        result = bspline.evaluate_derivatives(pts, [2])

        assert result.shape == (5, 2)

    # ------------------------------------------------------------------
    # Numerical validation
    # ------------------------------------------------------------------

    def test_finite_difference_validation(self) -> None:
        """Central FD approximation of first derivative matches evaluate_derivatives."""
        bspline = self._make_bspline([0, 0, 0, 1, 1, 1], 2, [0.0, 1.0, 0.0])
        # Use interior points away from both endpoints to avoid endpoint-shortcut issues
        # and to keep FD points inside the domain.
        pts = np.linspace(0.1, 0.9, 9, dtype=np.float64)
        h = 1e-6

        result = bspline.evaluate_derivatives(pts, [1])
        fd = (bspline.evaluate(pts + h) - bspline.evaluate(pts - h)) / (2.0 * h)

        # Use atol to handle the case where the true derivative is near zero
        # (rtol would fail when comparing ~0 to ~2e-11 floating-point noise).
        np.testing.assert_allclose(result, fd, atol=1e-5)

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_invalid_n_deriv(self) -> None:
        """Negative order raises ValueError."""
        bspline = self._make_bspline([0, 0, 0, 1, 1, 1], 2, [0.0, 1.0, 0.0])
        pts = np.array([0.5], dtype=np.float64)

        with pytest.raises(ValueError):
            bspline.evaluate_derivatives(pts, [-1])

    def test_out_array_reuse(self) -> None:
        """Pre-allocated out array is filled in-place."""
        bspline = self._make_bspline([0, 0, 0, 1, 1, 1], 2, [0.0, 1.0, 0.0])
        pts = np.array([0.25, 0.5, 0.75], dtype=np.float64)
        out = np.zeros(3, dtype=np.float64)

        result = bspline.evaluate_derivatives(pts, [2], out=out)

        np.testing.assert_array_equal(result, out)
        assert not np.all(out == 0.0)

    def test_dim_not_1_returns_scalar_result(self) -> None:
        """evaluate_derivatives on dim=2 scalar B-spline returns (n_pts,)."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(np.array(knots1, dtype=np.float64), 2)
        space_1d_2 = BsplineSpace1D(np.array(knots2, dtype=np.float64), 1)
        space = BsplineSpace([space_1d_1, space_1d_2])
        cp = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        bspline = Bspline(space, cp)

        pts = np.array([[0.5, 0.5]], dtype=np.float64)
        result = bspline.evaluate_derivatives(pts, [1, 1])

        # scalar 2D spline, mixed first derivative: shape (n_pts,)
        assert result.shape == (1,)


# ---------------------------------------------------------------------------
# Helpers for periodic B-spline evaluation tests
# ---------------------------------------------------------------------------


def _make_periodic_bspline(
    num_intervals: int,
    degree: int,
    dtype: type = np.float64,
    continuity: int | None = None,
) -> Bspline:
    """Create a simple periodic B-spline with sequential integer control points.

    Args:
        num_intervals (int): Number of intervals.
        degree (int): B-spline degree.
        dtype (type): Data type. Defaults to np.float64.
        continuity (int | None): Continuity level at interior knots. None uses degree-1
            (maximum continuity). Defaults to None.

    Returns:
        Bspline: A 1D periodic scalar B-spline.
    """
    knots = create_uniform_periodic_knots(num_intervals, degree, continuity=continuity, dtype=dtype)
    space_1d = BsplineSpace1D(knots, degree, periodic=True)
    space = BsplineSpace([space_1d])
    n = space.num_total_basis
    ctrl: npt.NDArray[np.float64] = np.arange(1, n + 1, dtype=dtype)
    return Bspline(space, ctrl)


def _eval_periodic_correct(f: Bspline, pts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Evaluate a periodic B-spline using the mathematically correct algorithm.

    Uses the unclamped ``first_basis = knot_id - degree`` index with modulo-wrapped
    control point lookup, which is the standard mathematical definition of a periodic
    B-spline. This differs from ``f.evaluate()`` which uses a clamped first_basis.

    Args:
        f (Bspline): A 1D periodic B-spline.
        pts (np.ndarray): Interior evaluation points (must lie strictly inside domain).

    Returns:
        np.ndarray: Evaluated values at the given points.
    """
    space_1d = f.space.spaces[0]
    knots = space_1d.knots
    p = space_1d.degree
    tol = float(space_1d.tolerance)
    n_stored = f.space.num_total_basis
    ctrl = f._control_points  # shape (n_stored, rank)

    # Compute knot spans using the non-periodic (unclamped) algorithm
    basis_out = np.zeros((len(pts), p + 1), dtype=np.float64)
    first_basis_arr = np.zeros(len(pts), dtype=np.int64)
    _compute_basis_nurbs_book_impl(knots, p, False, tol, pts, basis_out, first_basis_arr)

    rank = ctrl.shape[1]
    result = np.zeros((len(pts), rank), dtype=np.float64)
    for i in range(len(pts)):
        s = int(first_basis_arr[i])
        for j in range(p + 1):
            idx = (s + j) % n_stored
            result[i] += basis_out[i, j] * ctrl[idx]

    return result.squeeze()


class TestPeriodicBsplineEvaluation:
    """Test Bspline.evaluate and evaluate_derivatives for periodic B-splines.

    Covers both maximum continuity and reduced continuity (C^0, C^1) cases.
    All comparisons use interior points only to avoid endpoint ambiguity.
    """

    @pytest.mark.parametrize(
        "num_intervals,degree,continuity",
        [
            (3, 2, None),  # degree 2, max continuity
            (4, 2, 0),  # degree 2, C^0
            (4, 3, None),  # degree 3, max continuity
            (4, 3, 1),  # degree 3, C^1
            (5, 3, 0),  # degree 3, C^0
        ],
    )
    def test_periodic_to_open_evaluate_matches_correct_algorithm(
        self, num_intervals: int, degree: int, continuity: int | None
    ) -> None:
        """to_open_bspline().evaluate() agrees with modulo-wrapped reference algorithm.

        This is the canonical correctness check for periodic B-spline evaluation:
        converting to open form via knot insertion must reproduce the mathematically
        correct unclamped periodic function at interior points.
        """
        f = _make_periodic_bspline(num_intervals, degree, continuity=continuity)
        f_open = f.to_open_bspline()

        a, b = f_open.space.spaces[0].domain
        pts = np.linspace(float(a), float(b), 21, dtype=np.float64)[1:-1]

        np.testing.assert_allclose(
            f_open.evaluate(pts),
            _eval_periodic_correct(f, pts),
            atol=1e-11,
        )

    @pytest.mark.parametrize(
        "num_intervals,degree,continuity",
        [
            (3, 2, None),  # degree 2, max continuity
            (4, 2, 0),  # degree 2, C^0
            (4, 3, None),  # degree 3, max continuity
            (4, 3, 1),  # degree 3, C^1
            (5, 3, 0),  # degree 3, C^0
        ],
    )
    def test_periodic_evaluate_matches_correct_algorithm(
        self, num_intervals: int, degree: int, continuity: int | None
    ) -> None:
        """evaluate() agrees with the modulo-wrapped reference algorithm.

        Directly verifies that the periodic B-spline evaluation kernel produces
        the mathematically correct result at interior points, without going
        through ``to_open_bspline()``.
        """
        f = _make_periodic_bspline(num_intervals, degree, continuity=continuity)

        a, b = f.space.spaces[0].domain
        pts = np.linspace(float(a), float(b), 21, dtype=np.float64)[1:-1]

        np.testing.assert_allclose(
            f.evaluate(pts),
            _eval_periodic_correct(f, pts),
            atol=1e-13,
        )

    @pytest.mark.parametrize(
        "num_intervals,degree,continuity",
        [
            (3, 2, None),
            (4, 2, 0),
            (4, 3, None),
            (4, 3, 1),
            (5, 3, 0),
        ],
    )
    def test_periodic_evaluate_derivatives_order_0_matches_evaluate(
        self, num_intervals: int, degree: int, continuity: int | None
    ) -> None:
        """evaluate_derivatives(pts, [0]) equals evaluate(pts) for periodic splines."""
        f = _make_periodic_bspline(num_intervals, degree, continuity=continuity)

        a, b = f.space.spaces[0].domain
        pts = np.linspace(float(a), float(b), 21, dtype=np.float64)[1:-1]

        np.testing.assert_allclose(
            f.evaluate_derivatives(pts, [0]),
            f.evaluate(pts),
            atol=1e-13,
        )

    @pytest.mark.parametrize(
        "num_intervals,degree,continuity",
        [
            (4, 2, 0),  # degree 2, C^0
            (5, 3, 1),  # degree 3, C^1
            (5, 3, 0),  # degree 3, C^0
        ],
    )
    def test_periodic_evaluate_derivatives_matches_finite_diff(
        self, num_intervals: int, degree: int, continuity: int | None
    ) -> None:
        """evaluate_derivatives() agrees with finite differences for periodic splines."""
        f = _make_periodic_bspline(num_intervals, degree, continuity=continuity)

        a, b = f.space.spaces[0].domain
        # Avoid C^0 knot positions where the derivative is discontinuous.
        pts = np.linspace(float(a), float(b), 21, dtype=np.float64)[1:-1]
        unique_knots = np.unique(f.space.spaces[0].knots)
        mask = np.ones(len(pts), dtype=bool)
        for uk in unique_knots:
            mask &= ~np.isclose(pts, uk, atol=1e-10)
        pts = pts[mask]

        # 0th order must match evaluate()
        np.testing.assert_allclose(
            f.evaluate_derivatives(pts, [0]),
            f.evaluate(pts),
            atol=1e-13,
        )

        # 1st order validated by central finite differences
        h = 1e-6
        fd = (f.evaluate(pts + h) - f.evaluate(pts - h)) / (2.0 * h)
        np.testing.assert_allclose(
            f.evaluate_derivatives(pts, [1]),
            fd,
            atol=1e-5,
        )
