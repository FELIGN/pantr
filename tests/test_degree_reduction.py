"""Tests for Bézier and B-spline degree reduction."""

from __future__ import annotations

import itertools
import math

import numpy as np
import numpy.typing as npt
import pytest

from pantr._numba_compat import wait_for_jit_warmup
from pantr.bezier import Bezier
from pantr.bezier._bezier_degree import (
    _degree_elevate_bezier,
    _degree_reduce_bezier,
    _elevation_matrix_exact,
    _interpolating_reduction_operator,
    _l2_reduction_operator,
    _projected_quadrature_size,
    _projected_relative_deviation,
    _sample_projected,
    _squared_l2_norm,
    _tensor_gauss_weights,
)
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D, create_uniform_periodic_knots
from pantr.bspline._bspline_degree_core import _degree_reduce_1d_core

# Several tests here reach `Bezier.evaluate` within milliseconds of the process
# starting, and it dispatches to a `parallel=True` kernel. Numba's default workqueue
# layer is not safe against that racing `pantr/__init__.py`'s background warmup thread:
# the interpreter *aborts* rather than raising. Measured on this file with a warm Numba
# cache, before this barrier was added: 3 of 4 runs aborted. Same mitigation, and the
# same reason, as `tests/test_bernstein_underflow.py`.
wait_for_jit_warmup()

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

_QUADRATURE_AGREEMENT = 1.0e-2
"""Admissible relative gap between the projected measure and a dense reference.

The rational integrand of the projected deviation is not integrated exactly by any
Gauss rule, so the accept/reject measure is an estimate; ``_projected_quadrature_size``
records that over degrees 3 to 20, weight ratios to ``1e2`` and coordinate offsets to
``1e3`` it agreed with a 2e5-point reference to five decimal digits at the median and
to within 3% at worst.  1% bounds the case pinned here (measured 3.0e-6) with a wide
margin while still catching a rule that has stopped resolving the integrand: dropping
the node count from ``2p + 2`` to ``p + 1`` moves this case to 1.6e-2, outside the bound.
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
# Minimize degree, rational
# ---------------------------------------------------------------------------

_NUDGED_WEIGHT_NETS: dict[tuple[float, float], list[list[float]]] = {
    (1.0, 1.0e-6): [
        [0.0, 0.0, 1.0],
        [0.09999999999999999, 0.19999999999999998, 0.8666666666666666],
        [0.22666666666666668, 0.32, 0.7866666666666666],
        [0.38, 0.36000000000000004, 0.7600007599999999],
        [0.56, 0.32, 0.7866666666666666],
        [0.7666666666666666, 0.19999999999999998, 0.8666666666666666],
        [1.0, 0.0, 1.0],
    ],
    (1.0e3, 1.0e-6): [
        [0.0, 0.0, 1.0],
        [100.0, 200.0, 0.8666666666666666],
        [226.66666666666669, 320.0, 0.7866666666666666],
        [380.0, 360.00000000000006, 0.7600007599999999],
        [560.0, 320.0, 0.7866666666666666],
        [766.6666666666666, 200.0, 0.8666666666666666],
        [1000.0, 0.0, 1.0],
    ],
    (1.0e6, 1.0e-6): [
        [0.0, 0.0, 1.0],
        [100000.0, 200000.0, 0.8666666666666666],
        [226666.6666666667, 320000.0, 0.7866666666666666],
        [380000.0, 360000.00000000006, 0.7600007599999999],
        [560000.0, 320000.0, 0.7866666666666666],
        [766666.6666666666, 200000.0, 0.8666666666666666],
        [1000000.0, 0.0, 1.0],
    ],
    (1.0, 1.0e-3): [
        [0.0, 0.0, 1.0],
        [0.09999999999999999, 0.19999999999999998, 0.8666666666666666],
        [0.22666666666666668, 0.32, 0.7866666666666666],
        [0.38, 0.36000000000000004, 0.7607599999999999],
        [0.56, 0.32, 0.7866666666666666],
        [0.7666666666666666, 0.19999999999999998, 0.8666666666666666],
        [1.0, 0.0, 1.0],
    ],
    (1.0e3, 1.0e-3): [
        [0.0, 0.0, 1.0],
        [100.0, 200.0, 0.8666666666666666],
        [226.66666666666669, 320.0, 0.7866666666666666],
        [380.0, 360.00000000000006, 0.7607599999999999],
        [560.0, 320.0, 0.7866666666666666],
        [766.6666666666666, 200.0, 0.8666666666666666],
        [1000.0, 0.0, 1.0],
    ],
    (1.0e6, 1.0e-3): [
        [0.0, 0.0, 1.0],
        [100000.0, 200000.0, 0.8666666666666666],
        [226666.6666666667, 320000.0, 0.7866666666666666],
        [380000.0, 360000.00000000006, 0.7607599999999999],
        [560000.0, 320000.0, 0.7866666666666666],
        [766666.6666666666, 200000.0, 0.8666666666666666],
        [1000000.0, 0.0, 1.0],
    ],
}
"""Homogeneous control nets of the case in issue #297, keyed by (scale, weight nudge).

Each is the degree-2 rational Bézier through ``(0, 0)``, ``(0.5, 1)``, ``(1, 0)`` with
weights ``(1, 0.6, 1)``, scaled by ``scale``, elevated to degree 6, and with the middle
control weight multiplied by ``1 + nudge``.  Elevation is applied *after* scaling, so the
three scales are not exact multiples of each other in ``float64``; each is written out
rather than derived, so the triggering data is exactly the data the issue reports.

The perturbation lives entirely in the weight column, which is what makes the case
diagnostic: a homogeneous norm reads it as ``O(nudge / scale)`` relative while the
projected deviation it causes is ``O(nudge)`` relative.
"""

_ELEVATED_ARC_NETS: dict[float, list[list[float]]] = {
    1.0: [
        [1.0, 0.0, 1.0],
        [0.8828427124746191, 0.282842712474619, 0.8828427124746191],
        [0.7242640687119286, 0.5242640687119285, 0.8242640687119286],
        [0.5242640687119285, 0.7242640687119286, 0.8242640687119286],
        [0.282842712474619, 0.8828427124746191, 0.8828427124746191],
        [0.0, 1.0, 1.0],
    ],
    1.0e6: [
        [1000000.0, 0.0, 1.0],
        [882842.712474619, 282842.712474619, 0.8828427124746191],
        [724264.0687119286, 524264.06871192856, 0.8242640687119286],
        [524264.06871192856, 724264.0687119286, 0.8242640687119286],
        [282842.712474619, 882842.712474619, 0.8828427124746191],
        [0.0, 1000000.0, 1.0],
    ],
}
"""Exact quarter circles of radius ``scale``, degree-elevated from 2 to 5.

Reducing these back to degree 2 is exact up to round-off, so they are the curves the
default tolerance must keep accepting.
"""


def _projected_deviation(source: Bezier, reduced: Bezier) -> tuple[float, float]:
    """Measure a reduction's deviation and the curve's extent, both in projected space.

    Args:
        source (Bezier): The Bézier that was minimized.
        reduced (Bezier): What ``minimize_degree`` returned.

    Returns:
        tuple[float, float]: The largest projected deviation over a dense parameter
        sample, and the largest projected magnitude of ``source`` over the same sample.
    """
    pts = np.linspace(0.0, 1.0, 801, dtype=np.float64)
    values = source.evaluate(pts)
    return (
        float(np.linalg.norm(reduced.evaluate(pts) - values, axis=1).max()),
        float(np.linalg.norm(values, axis=1).max()),
    )


class TestBezierMinimizeDegreeRational:
    """A rational round-trip is graded in projected space, not in homogeneous one (#297)."""

    @pytest.mark.parametrize(("scale", "tol"), [(1.0, 1.0e-6), (1.0e3, 1.0e-9), (1.0e6, 1.0e-12)])
    def test_accepted_reduction_stays_within_the_requested_budget(
        self, scale: float, tol: float
    ) -> None:
        """No accepted reduction exceeds ``tol`` times the extent of the geometry.

        This is the case reported in issue #297.  Before the fix the reduction from
        degree 6 to degree 2 was accepted at all three scales, overshooting the budget
        by a factor of 61 at ``scale = 1e3`` and 6.1e4 at ``scale = 1e6``.
        """
        source = Bezier(np.array(_NUDGED_WEIGHT_NETS[scale, 1.0e-6]), is_rational=True)
        deviation, span = _projected_deviation(source, source.minimize_degree(tol=tol))
        assert deviation <= tol * span

    def test_reduction_within_budget_is_still_accepted(self) -> None:
        """A reduction the caller's budget does allow is not refused by the new measure."""
        source = Bezier(np.array(_NUDGED_WEIGHT_NETS[1.0, 1.0e-6]), is_rational=True)
        assert source.minimize_degree(tol=1.0e-6).degree == (2,)

    @pytest.mark.parametrize("tol", [1.0e-6, 1.0e-4])
    def test_verdict_is_scale_invariant(self, tol: float) -> None:
        """The same geometry written at three coordinate scales gets the same verdict.

        Scaling every coordinate is a similarity, so the *relative* deviation of a
        reduction is unchanged by it and the accept/reject verdict must be too.  The
        homogeneous measure is not scale invariant: at ``tol = 1e-6`` it refused the
        reduction at scale 1 and accepted it at scale 1e3 and 1e6.
        """
        degrees = {
            Bezier(np.array(_NUDGED_WEIGHT_NETS[scale, 1.0e-3]), is_rational=True)
            .minimize_degree(tol=tol)
            .degree
            for scale in (1.0, 1.0e3, 1.0e6)
        }
        assert len(degrees) == 1

    def test_weight_carried_deviation_is_refused_at_large_coordinate_scale(self) -> None:
        """A weight perturbation invisible to the homogeneous measure is now caught.

        At ``scale = 1e6`` the first trial's homogeneous relative error is 1.1e-11
        against a true projected 7.1e-6, so every trial cleared a ``1e-6`` budget and
        the accepted degree 6 to 2 reduction moved the curve by 4.9e-5 relative.
        """
        source = Bezier(np.array(_NUDGED_WEIGHT_NETS[1.0e6, 1.0e-3]), is_rational=True)
        assert source.minimize_degree(tol=1.0e-6).degree == (6,)

    @pytest.mark.parametrize("scale", [1.0, 1.0e6])
    def test_exactly_reducible_curve_still_reduces_at_the_default_tolerance(
        self, scale: float
    ) -> None:
        """An elevated rational curve is recovered by the default tolerance.

        The default is ``1e3 * eps`` on a *relative* error, and the projected round-trip
        error of an exactly reducible net sits at the round-off floor just as the
        homogeneous one does, so the default keeps its meaning.
        """
        source = Bezier(np.array(_ELEVATED_ARC_NETS[scale]), is_rational=True)
        assert source.degree == (5,)
        assert source.minimize_degree().degree == (2,)

    def test_non_rational_measure_is_unchanged(self) -> None:
        """The same net read as non-rational keeps the exact Bernstein-Gram verdict.

        The weight column is then an ordinary coordinate, so the round-trip error is
        1.1e-14 relative and a ``1e-12`` budget accepts the reduction.  The fix must not
        tighten this path.
        """
        net = np.array(_NUDGED_WEIGHT_NETS[1.0e6, 1.0e-6])
        assert Bezier(net, is_rational=False).minimize_degree(tol=1.0e-12).degree == (2,)

    def test_quadrature_measure_agrees_with_a_dense_reference(self) -> None:
        """The Gauss estimate of the projected deviation matches a dense sample.

        The integrand is rational, so the rule is not exact and the measure is an
        estimate; this pins how good an estimate.  See
        :func:`~pantr.bezier._bezier_degree._projected_quadrature_size`.
        """
        source = Bezier(np.array(_NUDGED_WEIGHT_NETS[1.0e6, 1.0e-3]), is_rational=True)
        num_nodes = tuple(_projected_quadrature_size(p) for p in source.degree)
        trial = _degree_elevate_bezier(_degree_reduce_bezier(source, (1,)), (1,)).control_points
        estimate = _projected_relative_deviation(
            _sample_projected(source.control_points, num_nodes),
            _sample_projected(trial, num_nodes),
            _tensor_gauss_weights(num_nodes),
        )

        pts = np.linspace(0.0, 1.0, 200001, dtype=np.float64)
        values = source.evaluate(pts)
        difference = Bezier(trial, is_rational=True).evaluate(pts) - values
        reference = math.sqrt(float(np.sum(difference**2)) / float(np.sum(values**2)))

        assert estimate == pytest.approx(reference, rel=_QUADRATURE_AGREEMENT)

    def test_weight_sign_change_refuses_every_reduction(self) -> None:
        """A weight function that changes sign has a pole, so nothing reduces.

        Asserting the sampler's verdict as well as the degree is what keeps this from
        passing for the wrong reason: merely zeroing a control weight leaves ``w(t)``
        positive everywhere (Bernstein positivity), and the reduction is then refused
        because the geometry genuinely changed, not because the measure is undefined.
        A weight of ``-3`` drives ``min w(t)`` to ``-0.46``.
        """
        net = np.array(_ELEVATED_ARC_NETS[1.0])
        net[2, -1] = -3.0
        num_nodes = tuple(_projected_quadrature_size(p) for p in (5,))
        assert _sample_projected(net, num_nodes) is None
        assert Bezier(net, is_rational=True).minimize_degree().degree == (5,)

    def test_two_dimensional_patch_reduces_only_the_redundant_direction(self) -> None:
        """A rational patch elevated in one direction is reduced in that one alone.

        Sampling a patch contracts the collocation matrix along every parametric axis in
        turn, which a curve exercises only once; a mistake in that loop would be
        invisible in the 1D tests.
        """
        root_half = 1.0 / math.sqrt(2.0)
        arc = np.array([[1.0, 0.0, 1.0], [root_half, root_half, root_half], [0.0, 1.0, 1.0]])
        # Extrude the quarter circle linearly along a third coordinate: the surface is
        # genuinely quadratic in direction 1 and genuinely linear in direction 0.
        net = np.stack(
            [
                np.column_stack([arc[:, :2], np.zeros(3), arc[:, 2]]),
                np.column_stack([arc[:, :2], arc[:, 2], arc[:, 2]]),
            ]
        )
        patch = Bezier(net, is_rational=True).elevate_degree((3, 0))
        assert patch.degree == (4, 2)
        assert patch.minimize_degree().degree == (1, 2)

    def test_float32_rational_keeps_its_dtype(self) -> None:
        """A float32 rational Bézier reduces and stays float32.

        The measure is computed in float64 regardless, so what is checked here is that
        the sampling path accepts a float32 net and does not promote the result.
        """
        source = Bezier(np.array(_ELEVATED_ARC_NETS[1.0], dtype=np.float32), is_rational=True)
        reduced = source.minimize_degree()
        assert reduced.degree == (2,)
        assert reduced.control_points.dtype == np.float32


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


class TestBsplineReduceDegreePeriodicSeam:
    """The seam multiplicity is floored at 1, exactly as interior knots are."""

    def test_maximally_smooth_periodic_spline_reduces(self) -> None:
        """``m_bdy = 1`` used to ask for multiplicity 0 and fail on the knot vector.

        The subtraction ``m_bdy - dec`` reaches 0 for a maximally smooth periodic
        spline; multiplicity 0 means "no breakpoint at the seam", which the
        periodic knot-vector builder cannot express.  Flooring at 1 asks for less
        smoothness than the seam already has, which is always representable.
        """
        knots = create_uniform_periodic_knots(num_intervals=6, degree=2)
        space = BsplineSpace([BsplineSpace1D(knots, 2, periodic=True)])
        rng = np.random.default_rng(14)
        bsp = Bspline(space, rng.standard_normal((space.num_total_basis, 1)))
        assert _bspline_multiplicity(bsp, 0.0) == 1  # precondition: maximally smooth seam

        reduced = bsp.reduce_degree(1)

        assert reduced.degree == (1,)
        assert reduced.space.spaces[0].periodic
        assert reduced.control_points.shape[0] == reduced.space.spaces[0].num_basis
        assert _bspline_multiplicity(reduced, 0.0) == 1  # floored, not 1 - 1 = 0
        assert reduced.space.spaces[0].num_basis == 6

        # It is an approximation, but a periodic one that stays near the original.
        pts = np.linspace(0.01, 0.99, 61)
        original = bsp.to_open_bspline().evaluate(pts)
        got = reduced.to_open_bspline().evaluate(pts)
        assert float(np.max(np.abs(got - original))) < 0.5 * float(np.max(np.abs(original)))

    def test_degree_one_still_rejects_the_degree_zero_periodic_result(self) -> None:
        """Reducing to degree 0 stays rejected: no ghost knots, no periodic form."""
        knots = create_uniform_periodic_knots(num_intervals=6, degree=1)
        space = BsplineSpace([BsplineSpace1D(knots, 1, periodic=True)])
        rng = np.random.default_rng(15)
        bsp = Bspline(space, rng.standard_normal((space.num_total_basis, 1)))

        with pytest.raises(ValueError, match=r"boundary multiplicity in \[1, degree\]"):
            bsp.reduce_degree(1)


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
