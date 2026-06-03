"""Tests for B-spline derivative (hodograph) computation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline import (
    Bspline,
    BsplineSpace,
    BsplineSpace1D,
    create_uniform_open_knots,
    create_uniform_periodic_knots,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_bspline_1d(
    knots: list[float],
    degree: int,
    ctrl: list[float] | list[list[float]],
    is_rational: bool = False,
    dtype: type = np.float64,
) -> Bspline:
    """Create a 1D B-spline from plain Python lists."""
    space_1d = BsplineSpace1D(np.array(knots, dtype=dtype), degree)
    space = BsplineSpace([space_1d])
    cp: npt.NDArray[np.float32 | np.float64] = np.array(ctrl, dtype=dtype)
    return Bspline(space, cp, is_rational=is_rational)


def eval_pts(a: float = 0.0, b: float = 1.0, n: int = 201) -> npt.NDArray[np.float64]:
    """Return n evenly-spaced evaluation points in [a, b]."""
    return np.linspace(a, b, n, dtype=np.float64)


def _make_open(
    num_intervals: int, degree: int, domain: tuple[float, float] = (0.0, 1.0)
) -> Bspline:
    """Create an open 1D B-spline with random-ish CPs."""
    knots = create_uniform_open_knots(num_intervals, degree, domain=domain)
    space_1d = BsplineSpace1D(knots, degree)
    space = BsplineSpace([space_1d])
    n = space.num_total_basis
    rng = np.random.default_rng(42 + degree + num_intervals)
    ctrl = rng.standard_normal(n)
    return Bspline(space, ctrl)


def _make_periodic(
    num_intervals: int, degree: int, domain: tuple[float, float] = (0.0, 1.0)
) -> Bspline:
    """Create a periodic 1D B-spline."""
    knots = create_uniform_periodic_knots(num_intervals, degree, domain=domain)
    space_1d = BsplineSpace1D(knots, degree, periodic=True)
    space = BsplineSpace([space_1d])
    n = space.num_total_basis
    rng = np.random.default_rng(17 + degree + num_intervals)
    ctrl = rng.standard_normal(n)
    return Bspline(space, ctrl)


def _make_nonopen(
    num_intervals: int, degree: int, domain: tuple[float, float] = (0.0, 1.0)
) -> Bspline:
    """Create a non-open, non-periodic B-spline (unclamped boundary knots)."""
    knots = create_uniform_periodic_knots(num_intervals, degree, domain=domain)
    space_1d = BsplineSpace1D(knots, degree, periodic=False)
    space = BsplineSpace([space_1d])
    n = space.num_total_basis
    rng = np.random.default_rng(31 + degree + num_intervals)
    ctrl = rng.standard_normal(n)
    return Bspline(space, ctrl)


def _assert_derivative_matches_evaluate(
    f: Bspline,
    direction: int = 0,
    atol: float = 1e-9,
) -> Bspline:
    """Assert that f.derivative() matches f.evaluate_derivatives() pointwise.

    Returns the derivative B-spline for further checks.
    """
    space_d = f.space.spaces[direction]
    a, b = float(space_d.domain[0]), float(space_d.domain[1])

    f_prime = f.derivative(direction=direction)

    if f.dim == 1:
        pts = eval_pts(a, b, 201)
        expected = f.evaluate_derivatives(pts, 1)
        actual = f_prime.evaluate(pts)
    else:
        # Build pts array for nD: use linspace in each direction.
        n_pts = 51
        dim = f.dim
        coords = np.empty((n_pts, dim), dtype=np.float64)
        rng = np.random.default_rng(99)
        for d in range(dim):
            ad, bd = float(f.space.spaces[d].domain[0]), float(f.space.spaces[d].domain[1])
            coords[:, d] = rng.uniform(ad, bd, n_pts)
        orders = [0] * dim
        orders[direction] = 1
        expected = f.evaluate_derivatives(coords, orders)
        actual = f_prime.evaluate(coords)

    np.testing.assert_allclose(actual, expected, atol=atol)
    return f_prime


# ---------------------------------------------------------------------------
# Non-rational 1D open tests
# ---------------------------------------------------------------------------


class TestNonRationalOpen1D:
    """Derivative of non-rational, open (clamped) 1D B-splines."""

    def test_linear(self) -> None:
        """Derivative of a linear B-spline (line) is a constant."""
        f = make_bspline_1d([0.0, 0.0, 1.0, 1.0], 1, [2.0, 5.0])
        f_prime = f.derivative()

        assert f_prime.space.spaces[0].degree == 0
        pts = eval_pts()
        vals = f_prime.evaluate(pts)
        np.testing.assert_allclose(vals, 3.0, atol=1e-14)

    def test_quadratic(self) -> None:
        """Derivative of a quadratic B-spline matches evaluate_derivatives."""
        f = _make_open(3, 2)
        _assert_derivative_matches_evaluate(f)

    def test_cubic(self) -> None:
        """Derivative of a cubic B-spline matches evaluate_derivatives."""
        f = _make_open(4, 3)
        _assert_derivative_matches_evaluate(f)

    def test_high_degree(self) -> None:
        """Derivative of degree-5 B-spline."""
        f = _make_open(3, 5)
        _assert_derivative_matches_evaluate(f)

    def test_single_element_bezier(self) -> None:
        """Derivative of a single-element Bezier."""
        f = make_bspline_1d([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 2, [1.0, 3.0, 2.0])
        f_prime = _assert_derivative_matches_evaluate(f)
        assert f_prime.space.spaces[0].has_Bezier_like_knots()

    def test_vector_valued(self) -> None:
        """Derivative of a vector-valued B-spline (2D curve)."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        ctrl = [[0.0, 0.0], [1.0, 2.0], [2.0, 1.0], [3.0, 3.0]]
        f = make_bspline_1d(knots, 2, ctrl)
        f_prime = f.derivative()

        pts = eval_pts()
        expected = f.evaluate_derivatives(pts, 1)
        actual = f_prime.evaluate(pts)
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_second_derivative_composable(self) -> None:
        """f.derivative().derivative() matches second derivative."""
        f = _make_open(4, 3)
        f_pp = f.derivative().derivative()
        a, b = float(f.space.spaces[0].domain[0]), float(f.space.spaces[0].domain[1])
        pts = eval_pts(a, b, 201)
        expected = f.evaluate_derivatives(pts, 2)
        actual = f_pp.evaluate(pts)
        np.testing.assert_allclose(actual, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# Non-rational 1D non-open tests
# ---------------------------------------------------------------------------


class TestNonRationalNonOpen1D:
    """Derivative of non-rational, non-open (unclamped) 1D B-splines."""

    def test_nonopen_stays_nonopen(self) -> None:
        """Derivative of a non-open B-spline is still non-open."""
        f = _make_nonopen(4, 2)
        f_prime = f.derivative()
        space_d = f_prime.space.spaces[0]
        assert not space_d.has_open_knots()
        assert not space_d.periodic

    def test_nonopen_matches_evaluate_derivatives(self) -> None:
        """Derivative evaluation matches evaluate_derivatives for non-open."""
        f = _make_nonopen(4, 3)
        f_open = f.to_open_bspline()
        f_prime = f.derivative()
        f_prime_open = f_prime.to_open_bspline()

        a, b = float(f_open.space.spaces[0].domain[0]), float(f_open.space.spaces[0].domain[1])
        pts = eval_pts(a, b, 201)
        expected = f_open.evaluate_derivatives(pts, 1)
        actual = f_prime_open.evaluate(pts)
        np.testing.assert_allclose(actual, expected, atol=1e-9)

    def test_nonopen_degree_reduced(self) -> None:
        """Degree decreases by 1."""
        f = _make_nonopen(5, 3)
        f_prime = f.derivative()
        assert f_prime.space.spaces[0].degree == 2


# ---------------------------------------------------------------------------
# Non-rational 1D periodic tests
# ---------------------------------------------------------------------------


class TestNonRationalPeriodic1D:
    """Derivative of non-rational, periodic 1D B-splines."""

    def test_periodic_stays_periodic(self) -> None:
        """Derivative of a periodic B-spline is still periodic."""
        f = _make_periodic(4, 2)
        f_prime = f.derivative()
        assert f_prime.space.spaces[0].periodic

    def test_periodic_matches_evaluate_derivatives(self) -> None:
        """Derivative evaluation matches evaluate_derivatives for periodic."""
        f = _make_periodic(4, 3)
        f_prime = f.derivative()

        a, b = float(f.space.spaces[0].domain[0]), float(f.space.spaces[0].domain[1])
        pts = eval_pts(a, b, 201)
        expected = f.evaluate_derivatives(pts, 1)
        actual = f_prime.evaluate(pts)
        np.testing.assert_allclose(actual, expected, atol=1e-9)

    def test_periodic_degree_reduced(self) -> None:
        """Degree decreases by 1."""
        f = _make_periodic(5, 3)
        f_prime = f.derivative()
        assert f_prime.space.spaces[0].degree == 2

    def test_periodic_second_derivative(self) -> None:
        """Composable second derivative of periodic B-spline."""
        f = _make_periodic(5, 3)
        f_pp = f.derivative().derivative()
        assert f_pp.space.spaces[0].periodic
        assert f_pp.space.spaces[0].degree == 1  # p=3 -> p=2 -> p=1


# ---------------------------------------------------------------------------
# Non-rational nD tests
# ---------------------------------------------------------------------------


class TestNonRationalND:
    """Derivative of non-rational tensor-product B-splines (nD)."""

    def _make_2d_surface(self) -> Bspline:
        """Create a 2D B-spline surface."""
        knots0 = create_uniform_open_knots(3, 2, domain=(0.0, 1.0))
        knots1 = create_uniform_open_knots(2, 3, domain=(0.0, 1.0))
        s0 = BsplineSpace1D(knots0, 2)
        s1 = BsplineSpace1D(knots1, 3)
        space = BsplineSpace([s0, s1])
        rng = np.random.default_rng(55)
        ctrl = rng.standard_normal((*space.num_basis, 2))
        return Bspline(space, ctrl)

    def test_partial_direction_0(self) -> None:
        """Partial derivative in direction 0 matches evaluate_derivatives."""
        f = self._make_2d_surface()
        _assert_derivative_matches_evaluate(f, direction=0, atol=1e-8)

    def test_partial_direction_1(self) -> None:
        """Partial derivative in direction 1 matches evaluate_derivatives."""
        f = self._make_2d_surface()
        _assert_derivative_matches_evaluate(f, direction=1, atol=1e-8)

    def test_degree_reduced_only_in_direction(self) -> None:
        """Only the differentiated direction's degree is reduced."""
        f = self._make_2d_surface()
        f_prime = f.derivative(direction=0)
        assert f_prime.space.spaces[0].degree == 1  # was 2
        assert f_prime.space.spaces[1].degree == 3  # unchanged

    def test_periodic_direction(self) -> None:
        """Periodic in one direction, open in another."""
        knots0 = create_uniform_open_knots(3, 2, domain=(0.0, 1.0))
        knots1 = create_uniform_periodic_knots(4, 2, domain=(0.0, 1.0))
        s0 = BsplineSpace1D(knots0, 2)
        s1 = BsplineSpace1D(knots1, 2, periodic=True)
        space = BsplineSpace([s0, s1])
        rng = np.random.default_rng(77)
        ctrl = rng.standard_normal((*space.num_basis, 1))
        f = Bspline(space, ctrl)

        # Derivative in periodic direction.
        f_prime = f.derivative(direction=1)
        assert f_prime.space.spaces[1].periodic
        assert f_prime.space.spaces[1].degree == 1  # p=2 -> p=1
        assert f_prime.space.spaces[0].degree == 2  # unchanged


# ---------------------------------------------------------------------------
# Rational 1D tests
# ---------------------------------------------------------------------------


class TestRational1D:
    """Derivative of rational (NURBS) 1D B-splines."""

    def test_rational_matches_evaluate_derivatives(self) -> None:
        """Rational derivative matches evaluate_derivatives pointwise."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        # CPs store [w*x, w*y, w]: a weighted 2D curve.
        ctrl = [[1.0, 0.0, 1.0], [2.0, 4.0, 2.0], [1.5, 3.0, 1.0], [3.0, 1.5, 1.5]]
        f = make_bspline_1d(knots, 2, ctrl, is_rational=True)
        f_prime = _assert_derivative_matches_evaluate(f, atol=1e-8)
        assert f_prime.is_rational

    def test_rational_unit_weights_matches_nonrational(self) -> None:
        """Rational with unit weights should match non-rational derivative."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        ctrl_nr = [1.0, 2.0, 0.5, 3.0]
        f_nr = make_bspline_1d(knots, 2, ctrl_nr)
        # Same curve, rational with w=1: CPs store [w*P, w] = [P, 1].
        ctrl_r = [[1.0, 1.0], [2.0, 1.0], [0.5, 1.0], [3.0, 1.0]]
        f_r = make_bspline_1d(knots, 2, ctrl_r, is_rational=True)

        pts = eval_pts()
        deriv_nr = f_nr.derivative().evaluate(pts)
        deriv_r = f_r.derivative().evaluate(pts)
        np.testing.assert_allclose(deriv_r, deriv_nr, atol=1e-9)

    def test_rational_circle_arc(self) -> None:
        """Derivative of a quarter circle NURBS arc."""
        # Quarter circle: P0=(1,0), P1=(1,1) w=1/sqrt(2), P2=(0,1)
        w = 1.0 / np.sqrt(2.0)
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        # Homogeneous CPs: [w*x, w*y, w]
        ctrl = [[1.0, 0.0, 1.0], [w * 1.0, w * 1.0, w], [0.0, 1.0, 1.0]]
        f = make_bspline_1d(knots, 2, ctrl, is_rational=True)

        f_prime = f.derivative()
        pts = eval_pts()
        expected = f.evaluate_derivatives(pts, 1)
        actual = f_prime.evaluate(pts)
        np.testing.assert_allclose(actual, expected, atol=1e-9)

    def test_rational_degree(self) -> None:
        """Rational derivative has degree 2p in the given direction."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        ctrl = [[1.0, 0.0, 1.0], [2.0, 4.0, 2.0], [1.5, 3.0, 1.0], [3.0, 1.5, 1.5]]
        f = make_bspline_1d(knots, 2, ctrl, is_rational=True)
        f_prime = f.derivative()
        assert f_prime.space.spaces[0].degree == 4  # 2 * 2 = 4


# ---------------------------------------------------------------------------
# Rational nD tests
# ---------------------------------------------------------------------------


class TestRationalND:
    """Derivative of rational (NURBS) nD B-splines."""

    def test_rational_2d_surface(self) -> None:
        """Rational 2D surface derivative matches evaluate_derivatives."""
        knots0 = create_uniform_open_knots(2, 2, domain=(0.0, 1.0))
        knots1 = create_uniform_open_knots(2, 2, domain=(0.0, 1.0))
        s0 = BsplineSpace1D(knots0, 2)
        s1 = BsplineSpace1D(knots1, 2)
        space = BsplineSpace([s0, s1])
        rng = np.random.default_rng(88)
        n0, n1 = space.num_basis
        # [w*x, w*y, w] — rank-2 rational surface
        ctrl = np.empty((n0, n1, 3), dtype=np.float64)
        ctrl[..., :2] = rng.standard_normal((n0, n1, 2))
        ctrl[..., 2] = rng.uniform(0.5, 2.0, (n0, n1))
        # Weight coords: multiply x,y by w.
        ctrl[..., 0] *= ctrl[..., 2]
        ctrl[..., 1] *= ctrl[..., 2]
        f = Bspline(space, ctrl, is_rational=True)

        _assert_derivative_matches_evaluate(f, direction=0, atol=1e-7)
        _assert_derivative_matches_evaluate(f, direction=1, atol=1e-7)


# ---------------------------------------------------------------------------
# keep_degree tests
# ---------------------------------------------------------------------------


def _assert_keep_degree_matches_evaluate(
    f: Bspline,
    direction: int = 0,
    atol: float = 1e-9,
) -> Bspline:
    """Assert that derivative(keep_degree=True) matches evaluate_derivatives.

    Returns the derivative B-spline for further checks.
    """
    space_d = f.space.spaces[direction]
    a, b = float(space_d.domain[0]), float(space_d.domain[1])

    f_prime = f.derivative(direction=direction, keep_degree=True)

    # Degree must be preserved.
    assert f_prime.space.spaces[direction].degree == space_d.degree

    if f.dim == 1:
        pts = eval_pts(a, b, 201)
        expected = f.evaluate_derivatives(pts, 1)
        actual = f_prime.evaluate(pts)
    else:
        n_pts = 51
        dim = f.dim
        coords = np.empty((n_pts, dim), dtype=np.float64)
        rng = np.random.default_rng(99)
        for d in range(dim):
            ad, bd = float(f.space.spaces[d].domain[0]), float(f.space.spaces[d].domain[1])
            coords[:, d] = rng.uniform(ad, bd, n_pts)
        orders = [0] * dim
        orders[direction] = 1
        expected = f.evaluate_derivatives(coords, orders)
        actual = f_prime.evaluate(coords)

    np.testing.assert_allclose(actual, expected, atol=atol)
    return f_prime


class TestKeepDegreeNonRational:
    """Derivative with keep_degree=True for non-rational B-splines."""

    def test_1d_degree_preserved(self) -> None:
        """Degree is preserved in 1D case."""
        f = _make_open(3, 2)
        f_prime = f.derivative(keep_degree=True)
        assert f_prime.space.spaces[0].degree == 2

    def test_1d_matches_evaluate_derivatives(self) -> None:
        """Values match evaluate_derivatives for 1D open B-spline."""
        f = _make_open(4, 3)
        _assert_keep_degree_matches_evaluate(f)

    def test_1d_cubic(self) -> None:
        """Cubic keep_degree derivative."""
        f = _make_open(3, 3)
        _assert_keep_degree_matches_evaluate(f)

    def test_1d_high_degree(self) -> None:
        """Degree-5 keep_degree derivative."""
        f = _make_open(3, 5)
        _assert_keep_degree_matches_evaluate(f)

    def test_1d_matches_derivative_then_elevate(self) -> None:
        """Result matches derivative() followed by elevate_degree()."""
        f = _make_open(4, 3)
        d_keep = f.derivative(keep_degree=True)
        d_ref = f.derivative().elevate_degree(1)
        pts = eval_pts()
        np.testing.assert_allclose(d_keep.evaluate(pts), d_ref.evaluate(pts), atol=1e-12)

    def test_vector_valued(self) -> None:
        """Vector-valued B-spline with keep_degree."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        ctrl = [[0.0, 0.0], [1.0, 2.0], [2.0, 1.0], [3.0, 3.0]]
        f = make_bspline_1d(knots, 2, ctrl)
        f_prime = f.derivative(keep_degree=True)
        assert f_prime.space.spaces[0].degree == 2
        pts = eval_pts()
        expected = f.evaluate_derivatives(pts, 1)
        actual = f_prime.evaluate(pts)
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_periodic(self) -> None:
        """Periodic B-spline with keep_degree (converts to open)."""
        f = _make_periodic(4, 3)
        f_prime = f.derivative(keep_degree=True)
        # Degree elevation doesn't support periodic, so result is open.
        assert f_prime.space.spaces[0].degree == 3
        assert not f_prime.space.spaces[0].periodic

        # Values must still match.
        a, b = float(f.space.spaces[0].domain[0]), float(f.space.spaces[0].domain[1])
        pts = eval_pts(a, b, 201)
        expected = f.evaluate_derivatives(pts, 1)
        actual = f_prime.evaluate(pts)
        np.testing.assert_allclose(actual, expected, atol=1e-9)

    def test_2d_direction0(self) -> None:
        """2D surface keep_degree in direction 0."""
        knots0 = create_uniform_open_knots(3, 2, domain=(0.0, 1.0))
        knots1 = create_uniform_open_knots(2, 3, domain=(0.0, 1.0))
        s0 = BsplineSpace1D(knots0, 2)
        s1 = BsplineSpace1D(knots1, 3)
        space = BsplineSpace([s0, s1])
        rng = np.random.default_rng(55)
        ctrl = rng.standard_normal((*space.num_basis, 2))
        f = Bspline(space, ctrl)
        _assert_keep_degree_matches_evaluate(f, direction=0, atol=1e-8)

    def test_2d_direction1(self) -> None:
        """2D surface keep_degree in direction 1."""
        knots0 = create_uniform_open_knots(3, 2, domain=(0.0, 1.0))
        knots1 = create_uniform_open_knots(2, 3, domain=(0.0, 1.0))
        s0 = BsplineSpace1D(knots0, 2)
        s1 = BsplineSpace1D(knots1, 3)
        space = BsplineSpace([s0, s1])
        rng = np.random.default_rng(55)
        ctrl = rng.standard_normal((*space.num_basis, 2))
        f = Bspline(space, ctrl)
        _assert_keep_degree_matches_evaluate(f, direction=1, atol=1e-8)

    def test_2d_other_directions_unchanged(self) -> None:
        """Degrees in non-differentiated directions remain unchanged."""
        knots0 = create_uniform_open_knots(3, 2, domain=(0.0, 1.0))
        knots1 = create_uniform_open_knots(2, 3, domain=(0.0, 1.0))
        s0 = BsplineSpace1D(knots0, 2)
        s1 = BsplineSpace1D(knots1, 3)
        space = BsplineSpace([s0, s1])
        rng = np.random.default_rng(55)
        ctrl = rng.standard_normal((*space.num_basis, 2))
        f = Bspline(space, ctrl)

        f_prime = f.derivative(direction=0, keep_degree=True)
        assert f_prime.space.spaces[0].degree == 2
        assert f_prime.space.spaces[1].degree == 3  # unchanged


class TestKeepDegreeRational:
    """Derivative with keep_degree=True for rational (NURBS) B-splines.

    For rational B-splines of degree ``p``, the derivative has degree ``2p``,
    which is already higher than the original degree. In this case,
    ``keep_degree`` does not further elevate the result.
    """

    def test_rational_1d_same_as_without_keep_degree(self) -> None:
        """Rational derivative with keep_degree matches without it."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        ctrl = [[1.0, 0.0, 1.0], [2.0, 4.0, 2.0], [1.5, 3.0, 1.0], [3.0, 1.5, 1.5]]
        f = make_bspline_1d(knots, 2, ctrl, is_rational=True)
        d_normal = f.derivative()
        d_keep = f.derivative(keep_degree=True)
        pts = eval_pts()
        np.testing.assert_allclose(d_keep.evaluate(pts), d_normal.evaluate(pts), atol=1e-14)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_degree_0_raises(self) -> None:
        """Derivative of a degree-0 B-spline raises ValueError."""
        f = make_bspline_1d([0.0, 0.5, 1.0], 0, [1.0, 2.0])
        with pytest.raises(ValueError, match="degree-0"):
            f.derivative()

    def test_direction_out_of_range_raises(self) -> None:
        """Out-of-range direction raises ValueError."""
        f = _make_open(3, 2)
        with pytest.raises(ValueError, match="direction"):
            f.derivative(direction=1)
        with pytest.raises(ValueError, match="direction"):
            f.derivative(direction=-1)

    def test_float32_dtype(self) -> None:
        """Float32 B-spline produces float32 derivative."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        ctrl = [1.0, 2.0, 3.0]
        f = make_bspline_1d(knots, 2, ctrl, dtype=np.float32)
        f_prime = f.derivative()
        assert f_prime.dtype == np.float32

    def test_degree_1_produces_degree_0(self) -> None:
        """Derivative of degree 1 produces degree 0."""
        f = _make_open(3, 1)
        f_prime = f.derivative()
        assert f_prime.space.spaces[0].degree == 0

    def test_knot_structure_preserved(self) -> None:
        """Interior knot multiplicities are preserved in derivative."""
        # Degree 3 with an interior knot of multiplicity 2 (C^1).
        knots = [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]
        ctrl = [1.0, 2.0, 0.5, 3.0, 1.0, 2.5]
        f = make_bspline_1d(knots, 3, ctrl)
        f_prime = f.derivative()

        # Derivative has degree 2, same interior knot at 0.5 with mult 2.
        space_d = f_prime.space.spaces[0]
        assert space_d.degree == 2
        unique, mults = space_d.get_unique_knots_and_multiplicity(in_domain=True)
        # Interior knots (exclude boundaries).
        interior_mask = (unique > 0.0 + 1e-10) & (unique < 1.0 - 1e-10)
        assert np.sum(interior_mask) == 1
        assert mults[interior_mask][0] == 2
