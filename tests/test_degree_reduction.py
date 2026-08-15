"""Tests for Bézier and B-spline degree reduction."""

from __future__ import annotations

import itertools
import math

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bezier import Bezier
from pantr.bezier._bezier_degree import (
    _elevation_matrix_exact,
    _interpolating_reduction_operator,
    _l2_reduction_operator,
    _squared_l2_norm,
)
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D, create_uniform_periodic_knots
from pantr.bspline._bspline_degree_core import _degree_reduce_1d_core

_EPS = float(np.finfo(np.float64).eps)

_ROUND_TRIP_FACTOR = 8.0
"""Safety factor on the round-trip bound ``(p + 1) * eps * max|ctrl|``.

Reduction applies a dense operator, so each output coefficient is a dot product
of ``p + 1`` terms and carries the classic forward error ``(p + 1) * u *
sum_j |R_ij| |c_j|``.  The operator entries stay near unity over the whole
tested range (``max |R| = 1.52`` at degree 30), so a factor of 8 covers the row
sums with room to spare.  Measured worst over degrees 1 to 22 and decrements 1
to 3: ``1.3e-15``, that is 55 times inside the bound.
"""

_MAX_INTERPOLATION_PRICE = 5.0
"""Largest admissible ratio of interpolating to unconstrained :math:`L^2` error.

Endpoint interpolation removes two degrees of freedom, so it can only make the
:math:`L^2` error larger; the ratio is 1 exactly when the unconstrained optimum
already interpolates.  Measured over degrees 1 to 20 and every decrement, the
ratio stays in ``[1.002, 4.547]``, the worst cases being reductions to a
straight line.  A regression that dropped the constraints, or one that solved
the wrong system, would leave this window immediately.
"""


def _make_bezier_1d(ctrl: list[list[float]], rational: bool = False) -> Bezier:
    """Create a 1D Bézier from a list of control points."""
    return Bezier(np.array(ctrl), is_rational=rational)


def _l2_norm_squared(bezier: Bezier) -> float:
    """Sum the squared L2 norms of every rank component of a Bézier."""
    ctrl = bezier.control_points
    return float(sum(_squared_l2_norm(ctrl[..., r]) for r in range(ctrl.shape[-1])))


def _difference(left: Bezier, right: Bezier) -> Bezier:
    """Coefficient-wise difference of two Béziers of equal degree."""
    return Bezier(left.control_points - right.control_points)


# ---------------------------------------------------------------------------
# Round-trip tests: elevate then reduce should recover the original
# ---------------------------------------------------------------------------


class TestBezierReduceDegreeRoundTrip:
    """Elevate by t then reduce by t should recover the original exactly."""

    def test_linear_elevate_1_reduce_1(self) -> None:
        """Linear Bézier → elevate by 1 → reduce by 1."""
        b = _make_bezier_1d([[0.0, 0.0], [1.0, 2.0]])
        reduced = b.elevate_degree(1).reduce_degree(1)
        np.testing.assert_allclose(reduced.control_points, b.control_points, atol=1e-14)

    def test_linear_elevate_3_reduce_3(self) -> None:
        """Linear Bézier → elevate by 3 → reduce by 3."""
        b = _make_bezier_1d([[0.0], [5.0]])
        reduced = b.elevate_degree(3).reduce_degree(3)
        np.testing.assert_allclose(reduced.control_points, b.control_points, atol=1e-13)

    def test_quadratic_elevate_2_reduce_2(self) -> None:
        """Quadratic Bézier → elevate by 2 → reduce by 2."""
        b = _make_bezier_1d([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]])
        reduced = b.elevate_degree(2).reduce_degree(2)
        np.testing.assert_allclose(reduced.control_points, b.control_points, atol=1e-13)

    def test_cubic_elevate_4_reduce_4(self) -> None:
        """Cubic Bézier → elevate by 4 → reduce by 4."""
        b = _make_bezier_1d([[0.0], [1.0], [2.0], [3.0]])
        reduced = b.elevate_degree(4).reduce_degree(4)
        np.testing.assert_allclose(reduced.control_points, b.control_points, atol=1e-12)

    def test_2d_surface_round_trip(self) -> None:
        """2D tensor-product Bézier (bilinear) → elevate → reduce."""
        ctrl = np.array(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[0.0, 1.0], [1.0, 2.0], [2.0, 1.0]],
            ]
        )
        b = Bezier(ctrl)
        reduced = b.elevate_degree((1, 2)).reduce_degree((1, 2))
        np.testing.assert_allclose(reduced.control_points, b.control_points, atol=1e-12)

    def test_2d_different_decrements(self) -> None:
        """2D Bézier with different elevations per direction."""
        rng = np.random.default_rng(42)
        ctrl = rng.random((3, 4, 2))  # degree (2, 3), rank 2
        b = Bezier(ctrl)
        reduced = b.elevate_degree((2, 1)).reduce_degree((2, 1))
        np.testing.assert_allclose(reduced.control_points, b.control_points, atol=1e-12)

    def test_rational_round_trip(self) -> None:
        """Rational Bézier: elevate then reduce preserves geometry."""
        ctrl_h = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0 / np.sqrt(2)], [0.0, 1.0, 1.0]])
        b = Bezier(ctrl_h, is_rational=True)
        reduced = b.elevate_degree(2).reduce_degree(2)
        np.testing.assert_allclose(reduced.control_points, b.control_points, atol=1e-13)
        assert reduced.is_rational


# ---------------------------------------------------------------------------
# Approximate reduction: reduce a genuine polynomial
# ---------------------------------------------------------------------------


class TestBezierReduceDegreeApproximate:
    """Reducing a polynomial that is NOT an elevated lower-degree is approximate."""

    def test_cubic_to_quadratic_geometry(self) -> None:
        """Reducing a true cubic to quadratic preserves geometry approximately."""
        b = _make_bezier_1d([[0.0, 0.0], [0.3, 1.0], [0.7, 1.0], [1.0, 0.0]])
        reduced = b.reduce_degree(1)
        assert reduced.degree == (2,)

        # Evaluate both at sample points and compare
        pts = np.linspace(0, 1, 50)
        vals_orig = b.evaluate(pts)
        vals_red = reduced.evaluate(pts)
        # The error should be reasonably small (not exact)
        max_err_tol = 0.1
        assert np.max(np.abs(vals_orig - vals_red)) < max_err_tol

    def test_endpoints_preserved(self) -> None:
        """After reduction, the endpoints are reproduced exactly."""
        b = _make_bezier_1d([[0.0, 0.0], [0.3, 1.5], [0.7, -0.5], [1.0, 1.0]])
        reduced = b.reduce_degree(1)

        pts = np.array([0.0, 1.0])
        assert np.array_equal(reduced.evaluate(pts), b.evaluate(pts))

    def test_reduce_degree_result_type(self) -> None:
        """Reduced Bézier has correct degree and dtype."""
        b = _make_bezier_1d([[0.0], [1.0], [2.0], [3.0], [4.0]])
        assert b.degree == (4,)

        reduced = b.reduce_degree(2)
        assert reduced.degree == (2,)
        assert reduced.control_points.dtype == b.control_points.dtype

    def test_reduce_by_1_from_degree_1(self) -> None:
        """Reducing a linear Bézier by 1 gives degree 0 (constant)."""
        b = _make_bezier_1d([[0.0, 0.0], [2.0, 4.0]])
        reduced = b.reduce_degree(1)
        assert reduced.degree == (0,)
        # The constant is the least-squares fit: average of endpoints
        expected = np.array([[1.0, 2.0]])
        np.testing.assert_allclose(reduced.control_points, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# The reduction operator and what makes it optimal
# ---------------------------------------------------------------------------


class TestReductionOperator:
    """The cached operators that drive every reduction."""

    @pytest.mark.parametrize("degree", [1, 2, 3, 5, 8, 12, 16, 20])
    @pytest.mark.parametrize("increment", [1, 2, 5])
    def test_exact_elevation_matrix_matches_the_elevation_kernel(
        self, degree: int, increment: int
    ) -> None:
        """The rational elevation matrix reproduces the compiled elevation kernel."""
        rng = np.random.default_rng(degree * 100 + increment)
        ctrl = rng.standard_normal((degree + 1, 2))

        from_kernel = Bezier(ctrl).elevate_degree(increment).control_points
        matrix = np.array(
            [[float(v) for v in row] for row in _elevation_matrix_exact(degree, increment)]
        )

        # Both routes sum O(degree) products, so compare against the magnitude
        # of the result rather than entry by entry: a coefficient near zero is
        # the difference of larger terms and has no relative accuracy.
        scale = float(np.max(np.abs(from_kernel)))
        assert float(np.max(np.abs(matrix @ ctrl - from_kernel))) <= 8.0 * _EPS * scale

    @pytest.mark.parametrize("degree,decrement", [(1, 1), (4, 1), (7, 3), (12, 2)])
    def test_operator_is_shared_and_read_only(self, degree: int, decrement: int) -> None:
        """Repeated calls hand back the same immutable array."""
        first = _interpolating_reduction_operator(degree, decrement)
        second = _interpolating_reduction_operator(degree, decrement)

        assert first is second
        assert not first.flags.writeable
        assert first.shape == (degree - decrement + 1, degree + 1)

    def test_reduction_to_a_constant_is_the_mean(self) -> None:
        """Two endpoint conditions cannot hold with one coefficient; the mean does."""
        operator = _interpolating_reduction_operator(3, 3)

        np.testing.assert_allclose(operator, np.full((1, 4), 0.25), rtol=0.0, atol=_EPS)

    def test_the_two_operators_differ(self) -> None:
        """The interpolating operator is not the plain projection."""
        interpolating = _interpolating_reduction_operator(5, 1)
        unconstrained = _l2_reduction_operator(5, 1)

        assert not np.allclose(interpolating, unconstrained)


def _bernstein_values(degree: int, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Tabulate the degree-``n`` Bernstein basis by evaluating the unit Béziers.

    Goes through the public evaluator, so it shares nothing with the assembly of
    the reduction operator.

    Args:
        degree (int): Polynomial degree.
        points (npt.NDArray[np.float64]): Evaluation points in ``[0, 1]``.

    Returns:
        npt.NDArray[np.float64]: ``(len(points), degree + 1)`` basis values.
    """
    return np.column_stack(
        [np.ravel(Bezier(np.eye(degree + 1)[:, [i]]).evaluate(points)) for i in range(degree + 1)]
    )


def _reference_reduction(ctrl: npt.NDArray[np.float64], decrement: int) -> npt.NDArray[np.float64]:
    """Solve the endpoint-constrained L2 fit by Gauss-Legendre quadrature.

    An ``n``-node Gauss-Legendre rule integrates polynomials of degree
    ``2n - 1`` exactly, so with ``n = degree + 2`` nodes the discrete weighted
    least-squares problem *is* the continuous one, not an approximation of it.
    The endpoint conditions are imposed by substitution and the remaining
    overdetermined system goes through :func:`numpy.linalg.lstsq`, so the
    reference shares no algebra with the implementation.

    Args:
        ctrl (npt.NDArray[np.float64]): Control points of shape ``(p + 1, rank)``.
        decrement (int): Degrees to remove.

    Returns:
        npt.NDArray[np.float64]: Reference reduced control points.
    """
    degree = ctrl.shape[0] - 1
    target = degree - decrement

    nodes, weights = np.polynomial.legendre.leggauss(degree + 2)
    nodes = 0.5 * (nodes + 1.0)
    root_weights = np.sqrt(0.5 * weights)[:, None]

    high = _bernstein_values(degree, nodes) * root_weights
    low = _bernstein_values(target, nodes) * root_weights

    residual = high @ ctrl - np.outer(low[:, 0], ctrl[0]) - np.outer(low[:, target], ctrl[-1])
    reduced = np.empty((target + 1, ctrl.shape[1]))
    reduced[0] = ctrl[0]
    reduced[target] = ctrl[-1]
    if target >= 2:
        reduced[1:target] = np.linalg.lstsq(low[:, 1:target], residual, rcond=None)[0]
    return reduced


class TestBezierReductionOptimality:
    """Optimality against an independent oracle, and variationally."""

    @pytest.mark.parametrize("degree", [3, 4, 5, 8, 10])
    @pytest.mark.parametrize("decrement", [1, 2, 3])
    def test_matches_a_quadrature_reference(self, degree: int, decrement: int) -> None:
        """The reduction agrees with a Gauss-Legendre constrained least-squares fit."""
        if decrement >= degree:
            pytest.skip("a reduction to a constant keeps no endpoint condition")

        rng = np.random.default_rng(degree * 10 + decrement)
        ctrl = rng.standard_normal((degree + 1, 2))

        reduced = Bezier(ctrl).reduce_degree(decrement).control_points
        reference = _reference_reduction(ctrl, decrement)

        # The reference solves the same problem through a design matrix whose
        # condition number is the square root of the Gram's; over these degrees
        # that is at most ~2e2, so half a dozen digits of head-room is plenty.
        scale = float(np.max(np.abs(reference)))
        assert float(np.max(np.abs(reduced - reference))) <= 1e-9 * scale

    @pytest.mark.parametrize("degree", [3, 5, 8, 12])
    def test_moving_off_the_optimum_costs_error_in_every_direction(self, degree: int) -> None:
        """Any admissible move increases the L2 error, at every interior coefficient."""
        rng = np.random.default_rng(degree * 3)
        bezier = Bezier(rng.standard_normal((degree + 1, 1)))
        reduced = bezier.reduce_degree(1)
        best = _l2_norm_squared(_difference(reduced.elevate_degree(1), bezier))

        for index in range(1, degree - 1):
            for step in (-1e-3, 1e-3):
                moved = reduced.control_points.copy()
                moved[index] += step
                perturbed = _l2_norm_squared(_difference(Bezier(moved).elevate_degree(1), bezier))
                assert perturbed > best

    @pytest.mark.parametrize("degree", [2, 3, 5, 8, 12, 16])
    def test_endpoint_conditions_cost_a_bounded_factor(self, degree: int) -> None:
        """Interpolating is never better than projecting, and never much worse."""
        rng = np.random.default_rng(degree + 500)
        for decrement in range(1, degree):
            target = degree - decrement
            for _ in range(3):
                ctrl = rng.standard_normal((degree + 1, 1))
                bezier = Bezier(ctrl)

                interpolating = Bezier(_interpolating_reduction_operator(degree, decrement) @ ctrl)
                unconstrained = Bezier(_l2_reduction_operator(degree, decrement) @ ctrl)
                assert interpolating.degree == (target,)

                error_i = math.sqrt(
                    _l2_norm_squared(_difference(interpolating.elevate_degree(decrement), bezier))
                )
                error_u = math.sqrt(
                    _l2_norm_squared(_difference(unconstrained.elevate_degree(decrement), bezier))
                )

                assert error_u <= error_i * (1.0 + 1e-12)
                assert error_i <= _MAX_INTERPOLATION_PRICE * error_u


class TestBezierEndpointInterpolation:
    """The endpoint values survive reduction untouched."""

    @pytest.mark.parametrize("degree", [1, 2, 3, 5, 8, 12, 16, 20])
    @pytest.mark.parametrize("rank", [1, 3])
    def test_endpoints_are_reproduced_bit_for_bit(self, degree: int, rank: int) -> None:
        """Every decrement short of a constant leaves both ends exactly where they were."""
        rng = np.random.default_rng(degree * 7 + rank)
        ctrl = rng.standard_normal((degree + 1, rank))

        for decrement in range(1, degree):
            reduced = Bezier(ctrl).reduce_degree(decrement).control_points
            assert np.array_equal(reduced[0], ctrl[0])
            assert np.array_equal(reduced[-1], ctrl[-1])

    @pytest.mark.parametrize(
        "target,increment", [(1, 1), (2, 3), (5, 2), (12, 1), (16, 2), (20, 2)]
    )
    def test_elevated_data_round_trips(self, target: int, increment: int) -> None:
        """Reduction inverts elevation to the accuracy of applying one dense operator."""
        rng = np.random.default_rng(target * 13 + increment)
        ctrl = rng.standard_normal((target + 1, 3))
        degree = target + increment

        back = Bezier(ctrl).elevate_degree(increment).reduce_degree(increment).control_points

        bound = _ROUND_TRIP_FACTOR * (degree + 1) * _EPS * float(np.max(np.abs(ctrl)))
        assert float(np.max(np.abs(back - ctrl))) <= bound

    def test_float32_stays_float32(self) -> None:
        """A float32 Bézier round trips within single-precision rounding."""
        rng = np.random.default_rng(4)
        ctrl = rng.standard_normal((6, 2)).astype(np.float32)

        back = Bezier(ctrl).elevate_degree(2).reduce_degree(2).control_points

        assert back.dtype == np.float32
        eps32 = float(np.finfo(np.float32).eps)
        assert float(np.max(np.abs(back - ctrl))) <= 8.0 * eps32 * float(np.max(np.abs(ctrl)))


class TestBezierDegreeReductionError:
    """The exact L2 error reported alongside a reduction."""

    @pytest.mark.parametrize("degree,decrement", [(3, 1), (5, 1), (5, 2), (8, 3)])
    def test_matches_numerical_quadrature(self, degree: int, decrement: int) -> None:
        """The reported norm matches a fine trapezoidal integration of the error."""
        rng = np.random.default_rng(degree * 3 + decrement)
        bezier = Bezier(rng.standard_normal((degree + 1, 1)))

        reported = bezier.degree_reduction_error(decrement)

        samples = np.linspace(0.0, 1.0, 200001)
        diff = np.ravel(bezier.evaluate(samples)) - np.ravel(
            bezier.reduce_degree(decrement).evaluate(samples)
        )
        quadrature = float(np.sqrt(np.trapezoid(diff**2, samples)))

        # The composite trapezoidal rule converges as h^2; at 2e5 intervals that
        # leaves well under six digits for a smooth integrand.
        assert reported == pytest.approx(quadrature, rel=1e-6)

    def test_vanishes_for_exactly_reducible_input(self) -> None:
        """An elevated curve can be reduced back for free."""
        rng = np.random.default_rng(11)
        bezier = Bezier(rng.standard_normal((4, 2))).elevate_degree(2)

        reported = bezier.degree_reduction_error(2)

        scale = float(np.max(np.abs(bezier.control_points)))
        assert reported <= _ROUND_TRIP_FACTOR * (bezier.degree[0] + 1) * _EPS * scale

    def test_grows_with_the_decrement(self) -> None:
        """Removing more degrees cannot reduce the error, while a target degree remains.

        The interpolating targets are nested — a degree-``q`` polynomial through
        both endpoints is also a degree-``q+1`` one — so the optimum over the
        smaller set cannot be closer.  The chain stops short of a constant,
        which is not in that family (see the companion test).
        """
        rng = np.random.default_rng(12)
        bezier = Bezier(rng.standard_normal((7, 1)))

        errors = [bezier.degree_reduction_error(dec) for dec in range(1, 6)]

        assert errors == sorted(errors)

    def test_reduction_to_a_constant_breaks_the_monotone_chain(self) -> None:
        """Dropping to degree 0 can beat degree 1, because it stops interpolating.

        Degree 1 is pinned to the chord between the endpoints; degree 0 keeps no
        endpoint condition at all and is free to sit at the mean, which for this
        curve is closer.  Worth pinning: it is the visible edge of the
        interpolating family, not an accident of the arithmetic.
        """
        rng = np.random.default_rng(12)
        bezier = Bezier(rng.standard_normal((7, 1)))

        to_line = bezier.degree_reduction_error(5)
        to_constant = bezier.degree_reduction_error(6)

        assert to_constant < to_line

    def test_rejects_the_same_arguments_reduce_degree_does(self) -> None:
        """Validation is shared with :meth:`Bezier.reduce_degree`."""
        bezier = _make_bezier_1d([[0.0], [1.0], [2.0]])

        with pytest.raises(ValueError, match=r"exceeds current degree"):
            bezier.degree_reduction_error(3)
        with pytest.raises(ValueError, match=r"non-negative"):
            bezier.degree_reduction_error(-1)
        with pytest.raises(ValueError, match=r"(?i)at least one"):
            bezier.degree_reduction_error(0)
        with pytest.raises(ValueError, match=r"must match dimension"):
            bezier.degree_reduction_error((1, 1))


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestBezierReduceDegreeErrors:
    """Test that invalid inputs raise appropriate errors."""

    def test_decrement_exceeds_degree(self) -> None:
        """Decrement > degree should raise ValueError."""
        b = _make_bezier_1d([[0.0], [1.0], [2.0]])  # degree 2
        with pytest.raises(ValueError, match=r"exceeds current degree"):
            b.reduce_degree(3)

    def test_negative_decrement(self) -> None:
        """Negative decrement should raise ValueError."""
        b = _make_bezier_1d([[0.0], [1.0], [2.0]])
        with pytest.raises(ValueError, match=r"non-negative"):
            b.reduce_degree(-1)

    def test_all_zero_decrements(self) -> None:
        """All-zero decrements should raise ValueError."""
        b = _make_bezier_1d([[0.0], [1.0], [2.0]])
        with pytest.raises(ValueError, match=r"(?i)at least one"):
            b.reduce_degree(0)

    def test_wrong_length(self) -> None:
        """Wrong number of decrements should raise ValueError."""
        b = _make_bezier_1d([[0.0], [1.0], [2.0]])
        with pytest.raises(ValueError, match=r"must match dimension"):
            b.reduce_degree((1, 1))

    def test_decrement_exceeds_per_direction(self) -> None:
        """Per-direction decrement exceeding degree should raise."""
        ctrl = np.zeros((2, 3, 1))  # degree (1, 2)
        b = Bezier(ctrl)
        with pytest.raises(ValueError, match=r"exceeds current degree"):
            b.reduce_degree((2, 0))

    def test_reduce_degree_0(self) -> None:
        """Reducing a degree-0 Bézier should raise."""
        b = Bezier(np.array([[42.0]]))
        with pytest.raises(ValueError, match=r"exceeds current degree"):
            b.reduce_degree(1)


# ---------------------------------------------------------------------------
# Minimize degree
# ---------------------------------------------------------------------------


class TestBezierMinimizeDegree:
    """Test Bezier.minimize_degree."""

    def test_constant_reduces_to_degree_0(self) -> None:
        """A constant elevated to degree 2 should reduce back to degree 0."""
        b = _make_bezier_1d([[3.0], [3.0], [3.0]])
        b_min = b.minimize_degree()
        assert b_min.degree[0] < b.degree[0]
        pts = np.linspace(0.0, 1.0, 10, dtype=np.float64)
        np.testing.assert_allclose(b_min.evaluate(pts), b.evaluate(pts), atol=1e-12)

    def test_true_quadratic_not_reduced(self) -> None:
        """A genuine quadratic should not be reduced."""
        b = _make_bezier_1d([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]])
        b_min = b.minimize_degree()
        assert b_min.degree == b.degree

    def test_linear_elevated_reduces(self) -> None:
        """A linear elevated to degree 3 should reduce back."""
        b = _make_bezier_1d([[1.0], [3.0]])
        b_elev = b.elevate_degree(2)
        assert b_elev.degree == (3,)
        b_min = b_elev.minimize_degree()
        assert b_min.degree[0] < b_elev.degree[0]
        pts = np.linspace(0.0, 1.0, 10, dtype=np.float64)
        np.testing.assert_allclose(b_min.evaluate(pts), b.evaluate(pts), atol=1e-12)

    def test_2d_constant_in_one_direction(self) -> None:
        """A 2D polynomial constant in one direction reduces along it."""
        ctrl = np.array([[[0.0], [1.0], [0.0]], [[0.0], [1.0], [0.0]]])  # (2, 3, 1)
        b = Bezier(ctrl)
        assert b.degree == (1, 2)
        b_min = b.minimize_degree()
        assert b_min.degree[0] < b.degree[0]
        assert b_min.degree[1] == b.degree[1]

    def test_vector_valued(self) -> None:
        """Vector-valued Bezier: all components checked together."""
        # Linear in both components, elevated to degree 2
        b = Bezier(np.array([[0.0, 0.0], [1.0, 2.0]]))
        b_elev = b.elevate_degree(1)
        assert b_elev.degree == (2,)
        b_min = b_elev.minimize_degree()
        assert b_min.degree[0] < b_elev.degree[0]
        pts = np.linspace(0.0, 1.0, 10, dtype=np.float64)
        np.testing.assert_allclose(b_min.evaluate(pts), b.evaluate(pts), atol=1e-12)


# ---------------------------------------------------------------------------
# Float32 support
# ---------------------------------------------------------------------------


class TestBezierReduceDegreeFloat32:
    """Test that float32 inputs produce float32 outputs."""

    def test_float32_round_trip(self) -> None:
        """Float32 elevation + reduction round-trip."""
        ctrl = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float32)
        b = Bezier(ctrl)
        reduced = b.elevate_degree(2).reduce_degree(2)
        assert reduced.control_points.dtype == np.float32
        np.testing.assert_allclose(reduced.control_points, b.control_points, atol=1e-5)


# ===========================================================================
# B-spline degree reduction
# ===========================================================================


def _make_bspline_1d(knots: list[float], degree: int, ctrl: list[list[float]]) -> Bspline:
    """Create a simple 1D open B-spline."""
    space = BsplineSpace([BsplineSpace1D(np.array(knots), degree)])
    return Bspline(space, np.array(ctrl))


def _bspline_multiplicity(spline: Bspline, xi: float) -> int:
    """Multiplicity of the breakpoint ``xi`` in a 1D spline's knot vector."""
    knots = spline.space.spaces[0].knots
    return int(np.count_nonzero(np.abs(knots - xi) <= 1.0e-12))


class TestBsplineReduceDegreeRoundTrip:
    """Elevate by t then reduce by t should recover the original geometry."""

    def test_single_segment_linear(self) -> None:
        """Single-segment linear B-spline → elevate 2 → reduce 2."""
        bsp = _make_bspline_1d([0, 0, 1, 1], 1, [[0.0], [1.0]])
        reduced = bsp.elevate_degree(2).reduce_degree(2)
        pts = np.linspace(0, 1, 20)
        np.testing.assert_allclose(bsp.evaluate(pts), reduced.evaluate(pts), atol=1e-13)

    def test_multi_segment_quadratic(self) -> None:
        """Quadratic B-spline with interior knot → elevate 2 → reduce 2."""
        bsp = _make_bspline_1d([0, 0, 0, 0.5, 1, 1, 1], 2, [[0.0], [1.0], [0.0], [1.0]])
        reduced = bsp.elevate_degree(2).reduce_degree(2)
        pts = np.linspace(0, 1, 30)
        np.testing.assert_allclose(bsp.evaluate(pts), reduced.evaluate(pts), atol=1e-12)

    def test_multi_segment_cubic(self) -> None:
        """Cubic B-spline with multiple interior knots."""
        knots = [0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1]
        rng = np.random.default_rng(42)
        ctrl = rng.random((7, 2))
        bsp = _make_bspline_1d(knots, 3, ctrl.tolist())
        reduced = bsp.elevate_degree(1).reduce_degree(1)
        pts = np.linspace(0, 1, 50)
        np.testing.assert_allclose(bsp.evaluate(pts), reduced.evaluate(pts), atol=1e-12)

    def test_2d_surface(self) -> None:
        """2D B-spline surface → elevate → reduce."""
        knots1 = np.array([0.0, 0.0, 1.0, 1.0])
        knots2 = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        space = BsplineSpace([BsplineSpace1D(knots1, 1), BsplineSpace1D(knots2, 2)])
        rng = np.random.default_rng(42)
        ctrl = rng.random((2, 3, 2))
        bsp = Bspline(space, ctrl)

        reduced = bsp.elevate_degree([1, 1]).reduce_degree([1, 1])

        pts = rng.random((20, 2))
        np.testing.assert_allclose(bsp.evaluate(pts), reduced.evaluate(pts), atol=1e-12)

    def test_rational(self) -> None:
        """Rational B-spline (NURBS): elevate then reduce."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        ctrl_h = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0 / np.sqrt(2)], [0.0, 1.0, 1.0]])
        space = BsplineSpace([BsplineSpace1D(knots, 2)])
        bsp = Bspline(space, ctrl_h, is_rational=True)

        reduced = bsp.elevate_degree(1).reduce_degree(1)
        assert reduced.is_rational

        pts = np.linspace(0, 1, 20)
        np.testing.assert_allclose(bsp.evaluate(pts), reduced.evaluate(pts), atol=1e-12)


def _open_uniform_bspline(degree: int, n_el: int, rank: int = 2, seed: int = 0) -> Bspline:
    """Create an open B-spline with ``n_el`` uniform elements and random control points."""
    knots = np.concatenate([np.zeros(degree), np.linspace(0.0, 1.0, n_el + 1), np.ones(degree)])
    space = BsplineSpace([BsplineSpace1D(knots, degree)])
    rng = np.random.default_rng(seed)
    return Bspline(space, rng.random((space.num_total_basis, rank)))


class TestBsplineReduceDegreeOutputSizing:
    """Regression: the kernel sized its output buffers from ``len(knots)``.

    The reduced spline is in Bézier form, so it holds ``n_seg * new_degree + 1``
    control points, a count that grows with the number of elements while
    ``len(knots)`` grows more slowly.  Past the crossover the kernel wrote beyond
    the end of both output arrays: unchecked under ``nopython``, and surfacing
    downstream as a control-point/basis-count mismatch.
    """

    @pytest.mark.parametrize(
        "degree,n_el,dec",
        [
            (5, 8, 1),  # the smallest failing case at degree 5
            (4, 13, 1),
            (6, 7, 1),
            (7, 6, 1),
            (8, 5, 1),
            (5, 16, 2),
            (7, 11, 3),
            (8, 13, 4),
        ],
    )
    def test_reduction_fills_a_consistent_spline(self, degree: int, n_el: int, dec: int) -> None:
        """Reducing an open uniform B-spline yields a well-formed spline."""
        bsp = _open_uniform_bspline(degree, n_el)
        reduced = bsp.reduce_degree(dec)

        assert reduced.degree == (degree - dec,)
        space_1d = reduced.space.spaces[0]
        assert len(space_1d.knots) == space_1d.num_basis + space_1d.degree + 1
        assert reduced.control_points.shape[0] == space_1d.num_basis

    def test_kernel_output_matches_the_bezier_form_count(self) -> None:
        """The kernel returns exactly ``n_seg * new_degree + 1`` control points."""
        degree, n_el, dec = 5, 8, 1
        bsp = _open_uniform_bspline(degree, n_el, rank=1)
        space_1d = bsp.space.spaces[0]

        new_ctrl, new_knots = _degree_reduce_1d_core(
            degree,
            bsp.control_points,
            space_1d.knots,
            dec,
            _interpolating_reduction_operator(degree, dec),
        )

        new_degree = degree - dec
        assert new_ctrl.shape[0] == n_el * new_degree + 1
        assert new_knots.shape[0] == new_ctrl.shape[0] + new_degree + 1

    def test_round_trip_on_the_triggering_configuration(self) -> None:
        """Degree 5 with 8 elements: elevate then reduce recovers the geometry."""
        bsp = _open_uniform_bspline(4, 8, rank=2, seed=3)
        reduced = bsp.elevate_degree(1).reduce_degree(1)

        pts = np.linspace(0.0, 1.0, 100)
        np.testing.assert_allclose(reduced.evaluate(pts), bsp.evaluate(pts), atol=1e-12)

    def test_periodic_spline_of_the_same_size(self) -> None:
        """The periodic path reaches the same kernel."""
        knots = create_uniform_periodic_knots(num_intervals=8, degree=4)
        space = BsplineSpace([BsplineSpace1D(knots, 4, periodic=True)])
        rng = np.random.default_rng(1)
        bsp = Bspline(space, rng.random((space.num_total_basis, 2)))

        reduced = bsp.elevate_degree(1).reduce_degree(1)

        assert reduced.degree == (4,)
        assert reduced.space.spaces[0].periodic

    def test_surface_direction_of_the_same_size(self) -> None:
        """A 2D surface reduced along one direction only."""
        knots_x = np.concatenate([np.zeros(5), np.linspace(0.0, 1.0, 9), np.ones(5)])
        knots_y = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
        space = BsplineSpace([BsplineSpace1D(knots_x, 5), BsplineSpace1D(knots_y, 2)])
        rng = np.random.default_rng(2)
        bsp = Bspline(space, rng.random((*space.num_basis, 3)))

        reduced = bsp.reduce_degree((1, 0))

        assert reduced.degree == (4, 2)
        assert reduced.control_points.shape[:2] == reduced.space.num_basis


class TestBsplineSegmentStitching:
    """Reduced segments meet exactly, so nothing has to be averaged."""

    @pytest.mark.parametrize("degree,n_el", [(3, 5), (4, 7), (5, 8), (6, 4)])
    def test_neighbouring_segments_agree_bit_for_bit(self, degree: int, n_el: int) -> None:
        """Independently reduced segments share their junction control point exactly.

        This is what removes the averaging step: the old kernel replaced both
        sides with their mean, moving each segment off its own optimum to buy a
        C0 join that endpoint interpolation now provides for free.
        """
        bsp = _open_uniform_bspline(degree, n_el, rank=2, seed=degree * 3 + n_el)
        operator = _interpolating_reduction_operator(degree, 1)

        segments = [bezier.control_points for bezier in bsp.to_beziers().ravel()]
        reduced = [operator @ segment for segment in segments]

        assert len(reduced) == n_el
        for left, right in itertools.pairwise(reduced):
            assert np.array_equal(left[-1], right[0])

    @pytest.mark.parametrize("degree,n_el", [(3, 4), (4, 6)])
    def test_the_stitched_spline_interpolates_the_breakpoints(self, degree: int, n_el: int) -> None:
        """Before knot coarsening, the reduced spline meets the original at every breakpoint."""
        bsp = _open_uniform_bspline(degree, n_el, rank=2, seed=degree + n_el)
        space_1d = bsp.space.spaces[0]

        new_ctrl, new_knots = _degree_reduce_1d_core(
            degree,
            bsp.control_points,
            space_1d.knots,
            1,
            _interpolating_reduction_operator(degree, 1),
        )
        bezier_form = Bspline(
            BsplineSpace([BsplineSpace1D(new_knots, degree - 1)]),
            new_ctrl,
        )

        breakpoints = np.linspace(0.0, 1.0, n_el + 1)
        # Bézier form: the value at a breakpoint is a control point, and the
        # reduction pinned it to the original segment's endpoint.
        np.testing.assert_allclose(
            bezier_form.evaluate(breakpoints),
            bsp.evaluate(breakpoints),
            rtol=0.0,
            atol=8.0 * _EPS * float(np.max(np.abs(bsp.control_points))),
        )


class TestBsplineReduceDegreePeriodic:
    """Test degree reduction for periodic B-splines."""

    @pytest.mark.parametrize(
        "degree,continuity,dec",
        [
            (2, None, 1),
            (3, None, 1),
            (3, None, 2),
            (3, 1, 1),
        ],
    )
    def test_periodic_preserves_geometry(
        self, degree: int, continuity: int | None, dec: int
    ) -> None:
        """Elevate then reduce a periodic B-spline preserves geometry."""
        knots = create_uniform_periodic_knots(num_intervals=4, degree=degree, continuity=continuity)
        space = BsplineSpace([BsplineSpace1D(knots, degree, periodic=True)])
        rng = np.random.default_rng(42)
        ctrl = rng.random((space.num_total_basis, 2))
        bsp = Bspline(space, ctrl)

        reduced = bsp.elevate_degree(dec).reduce_degree(dec)

        assert reduced.space.spaces[0].periodic
        assert reduced.degree == (degree,)

        pts = np.linspace(0.01, 0.99, 50)
        orig = bsp.to_open_bspline().evaluate(pts)
        red = reduced.to_open_bspline().evaluate(pts)
        np.testing.assert_allclose(orig, red, atol=1e-11)

    def test_mixed_periodic_open_2d(self) -> None:
        """2D mixed periodic/open B-spline: elevate then reduce."""
        knots_per = create_uniform_periodic_knots(num_intervals=4, degree=2)
        knots_open = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
        space = BsplineSpace(
            [
                BsplineSpace1D(knots_per, 2, periodic=True),
                BsplineSpace1D(knots_open, 2),
            ]
        )
        rng = np.random.default_rng(42)
        ctrl = rng.random((*space.num_basis, 2))
        bsp = Bspline(space, ctrl)

        reduced = bsp.elevate_degree([1, 1]).reduce_degree([1, 1])

        assert reduced.space.spaces[0].periodic
        assert not reduced.space.spaces[1].periodic
        assert reduced.degree == (2, 2)

        pts = rng.random((30, 2))
        pts[:, 0] = pts[:, 0] * 0.98 + 0.01
        orig = bsp.to_open_bspline().evaluate(pts)
        red = reduced.to_open_bspline().evaluate(pts)
        np.testing.assert_allclose(orig, red, atol=1e-11)


class TestBsplineReduceDegreeDiscontinuous:
    """Reduction preserves smoothness, so a C^-1 knot stays C^-1.

    A breakpoint of multiplicity ``m`` is ``C^{p-m}``, and ``C^{p-m} =
    C^{(p-t)-(m-t)}``, so the reduced multiplicity is ``m - t``.  At ``m = p + 1``
    that is ``new_degree + 1``: clamping it at ``new_degree`` asks for a C0 space
    where the function jumps, and the reduction stops being exact on curves that
    reduce exactly.
    """

    @pytest.mark.parametrize("degree", [2, 3, 4, 5])
    def test_exactly_reducible_curve_with_a_jump(self, degree: int) -> None:
        """A degree-elevated C^-1 spline reduces back to itself, not to a C0 fit."""
        knots = np.concatenate([np.zeros(degree), np.full(degree, 0.5), np.ones(degree)]).tolist()
        rng = np.random.default_rng(degree)
        base = _make_bspline_1d(knots, degree - 1, rng.standard_normal((2 * degree, 1)).tolist())
        assert _bspline_multiplicity(base, 0.5) == degree  # (degree - 1) + 1: C^-1

        elevated = base.elevate_degree(1)
        reduced = elevated.reduce_degree(1)

        assert reduced.degree == (degree - 1,)
        assert _bspline_multiplicity(reduced, 0.5) == degree

        pts = np.linspace(0.0, 1.0, 97)[1:-1] + 1.0e-7
        scale = float(np.max(np.abs(base.control_points)))
        np.testing.assert_allclose(
            reduced.evaluate(pts),
            base.evaluate(pts),
            rtol=0.0,
            atol=_ROUND_TRIP_FACTOR * (degree + 1) * _EPS * scale,
        )

    def test_mixed_multiplicities_keep_their_continuity(self) -> None:
        """A C^-1 knot alongside smooth and C^0 ones: each loses exactly the decrement."""
        degree = 4
        knots = np.concatenate(
            [
                np.zeros(degree + 1),
                [0.25],
                np.full(degree + 1, 0.5),
                np.full(degree, 0.75),
                np.ones(degree + 1),
            ]
        ).tolist()
        rng = np.random.default_rng(4321)
        n_basis = len(knots) - degree - 1
        bsp = _make_bspline_1d(knots, degree, rng.standard_normal((n_basis, 2)).tolist())

        reduced = bsp.reduce_degree(1)

        assert reduced.degree == (3,)
        assert _bspline_multiplicity(reduced, 0.25) == 1
        assert _bspline_multiplicity(reduced, 0.5) == 4  # new_degree + 1: still a jump
        assert _bspline_multiplicity(reduced, 0.75) == 3
        space_1d = reduced.space.spaces[0]
        assert reduced.control_points.shape[0] == space_1d.num_basis

    def test_kernel_output_sizing_accounts_for_the_jumps(self) -> None:
        """The Bézier form holds one extra control point and knot per C^-1 breakpoint."""
        degree, dec = 3, 1
        knots = np.concatenate(
            [np.zeros(degree + 1), np.full(4, 0.3), [0.6], np.full(4, 0.8), np.ones(degree + 1)]
        )
        rng = np.random.default_rng(99)
        n_basis = len(knots) - degree - 1
        bsp = _make_bspline_1d(knots.tolist(), degree, rng.standard_normal((n_basis, 1)).tolist())

        new_ctrl, new_knots = _degree_reduce_1d_core(
            degree,
            bsp.control_points,
            bsp.space.spaces[0].knots,
            dec,
            _interpolating_reduction_operator(degree, dec),
        )

        new_degree = degree - dec
        n_seg, n_jump = 4, 2
        assert new_ctrl.shape[0] == n_seg * new_degree + 1 + n_jump
        assert new_knots.shape[0] == new_ctrl.shape[0] + new_degree + 1


class TestBsplineReduceDegreeErrors:
    """Test that invalid inputs raise appropriate errors."""

    def test_decrement_exceeds_degree(self) -> None:
        """Decrement > degree should raise ValueError."""
        bsp = _make_bspline_1d([0, 0, 1, 1], 1, [[0.0], [1.0]])
        with pytest.raises(ValueError, match=r"exceeds current degree"):
            bsp.reduce_degree(2)

    def test_negative_decrement(self) -> None:
        """Negative decrement should raise ValueError."""
        bsp = _make_bspline_1d([0, 0, 1, 1], 1, [[0.0], [1.0]])
        with pytest.raises(ValueError, match=r"non-negative"):
            bsp.reduce_degree(-1)

    def test_all_zero_decrements(self) -> None:
        """All-zero decrements should raise ValueError."""
        bsp = _make_bspline_1d([0, 0, 1, 1], 1, [[0.0], [1.0]])
        with pytest.raises(ValueError, match=r"(?i)at least one"):
            bsp.reduce_degree(0)

    def test_wrong_length(self) -> None:
        """Wrong number of decrements should raise ValueError."""
        bsp = _make_bspline_1d([0, 0, 1, 1], 1, [[0.0], [1.0]])
        with pytest.raises(ValueError, match=r"must match dimension"):
            bsp.reduce_degree((1, 1))
