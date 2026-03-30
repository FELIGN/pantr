from __future__ import annotations

import numpy as np
import pytest

from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D, create_uniform_periodic_knots


def test_degree_elevation_1d_linear_to_quadratic() -> None:
    """Test degree elevation of a 1D linear B-spline to quadratic."""
    knots = np.array([0.0, 0.0, 1.0, 1.0])
    ctrl = np.array([[0.0], [1.0]])
    space = BsplineSpace([BsplineSpace1D(knots, 1)])
    bspline = Bspline(space, ctrl)

    # Elevate by 1
    elevated = bspline.elevate_degree(1)

    assert elevated.degree == (2,)
    # New knot vector for p=2 should be [0,0,0,1,1,1] for a single segment
    expected_knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(elevated.space.spaces[0].knots, expected_knots)

    # Evaluate at some points
    pts = np.linspace(0, 1, 10)
    vals_orig = bspline.evaluate(pts)
    vals_elev = elevated.evaluate(pts)

    np.testing.assert_allclose(vals_orig, vals_elev, atol=1e-14)


def test_degree_elevation_1d_quadratic_to_quartic() -> None:
    """Test degree elevation of a 1D quadratic B-spline by 2 degrees."""
    knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
    ctrl = np.array([[0.0], [1.0], [0.0], [1.0]])
    space = BsplineSpace([BsplineSpace1D(knots, 2)])
    bspline = Bspline(space, ctrl)

    # Elevate by 2
    elevated = bspline.elevate_degree(2)

    assert elevated.degree == (4,)

    # Evaluate at some points
    pts = np.linspace(0, 1, 20)
    vals_orig = bspline.evaluate(pts)
    vals_elev = elevated.evaluate(pts)

    np.testing.assert_allclose(vals_orig, vals_elev, atol=1e-14)


def test_degree_elevation_2d() -> None:
    """Test degree elevation of a 2D B-spline surface."""
    knots1 = np.array([0.0, 0.0, 1.0, 1.0])
    knots2 = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    space = BsplineSpace([BsplineSpace1D(knots1, 1), BsplineSpace1D(knots2, 2)])

    rng = np.random.default_rng(42)
    ctrl = rng.random((2, 3, 2))
    bspline = Bspline(space, ctrl)

    # Elevate direction 0 by 2, direction 1 by 1
    elevated = bspline.elevate_degree([1, 1])

    assert elevated.degree == (2, 3)

    # Evaluate at random points
    n_pts = 10
    pts = rng.random((n_pts, 2))
    vals_orig = bspline.evaluate(pts)
    vals_elev = elevated.evaluate(pts)

    np.testing.assert_allclose(vals_orig, vals_elev, atol=1e-14)


def test_degree_elevation_rational() -> None:
    """Test degree elevation of a rational B-spline (NURBS)."""
    # 1D rational curve (e.g. circle arc)
    knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    # Control points in homogeneous coordinates (x, y, w)
    ctrl_h = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0 / np.sqrt(2)], [0.0, 1.0, 1.0]])
    space = BsplineSpace([BsplineSpace1D(knots, 2)])
    bspline = Bspline(space, ctrl_h, is_rational=True)

    # Elevate by 1
    elevated = bspline.elevate_degree(1)

    assert elevated.degree == (3,)
    assert elevated.is_rational

    # Evaluate
    pts = np.linspace(0, 1, 10)
    vals_orig = bspline.evaluate(pts)
    vals_elev = elevated.evaluate(pts)

    np.testing.assert_allclose(vals_orig, vals_elev, atol=1e-14)


def test_degree_elevation_zero_increment_raises() -> None:
    """Test that zero increment raises ValueError."""
    knots = np.array([0.0, 0.0, 1.0, 1.0])
    ctrl = np.array([[0.0], [1.0]])
    space = BsplineSpace([BsplineSpace1D(knots, 1)])
    bspline = Bspline(space, ctrl)

    with pytest.raises(ValueError, match=r"(?i)at least one.*positive"):
        bspline.elevate_degree(0)

    with pytest.raises(ValueError, match=r"(?i)at least one.*positive"):
        bspline.elevate_degree((0,))


def test_degree_elevation_invalid_inputs() -> None:
    """Test invalid inputs for degree elevation."""
    knots = np.array([0.0, 0.0, 1.0, 1.0])
    ctrl = np.array([[0.0], [1.0]])
    space = BsplineSpace([BsplineSpace1D(knots, 1)])
    bspline = Bspline(space, ctrl)

    with pytest.raises(ValueError, match="match dimension"):
        bspline.elevate_degree((1, 1))

    with pytest.raises(ValueError, match="non-negative"):
        bspline.elevate_degree(-1)


# ---------------------------------------------------------------------------
# Periodic Bspline: degree elevation preserves periodicity
# ---------------------------------------------------------------------------


def _make_periodic_bspline(
    num_intervals: int,
    degree: int,
    continuity: int | None = None,
    rank: int = 2,
) -> Bspline:
    """Create a 1D periodic B-spline with random control points."""
    knots = create_uniform_periodic_knots(num_intervals, degree, continuity=continuity)
    space_1d = BsplineSpace1D(knots, degree, periodic=True)
    space = BsplineSpace([space_1d])
    rng = np.random.default_rng(42)
    ctrl = rng.random((space.num_total_basis, rank))
    return Bspline(space, ctrl)


class TestPeriodicBsplineElevateDegree:
    """Test that Bspline.elevate_degree preserves periodicity and geometry."""

    @pytest.mark.parametrize(
        "degree,continuity,inc",
        [
            (2, None, 1),
            (3, None, 1),
            (3, None, 2),
            (3, 1, 1),
            (3, 0, 1),
            (2, 0, 1),
        ],
    )
    def test_elevate_degree_preserves_periodic(
        self, degree: int, continuity: int | None, inc: int
    ) -> None:
        """elevate_degree on a periodic Bspline returns a periodic Bspline."""
        bsp = _make_periodic_bspline(4, degree, continuity)
        elevated = bsp.elevate_degree((inc,))

        assert elevated.space.spaces[0].periodic
        assert elevated.space.spaces[0].degree == degree + inc

    @pytest.mark.parametrize(
        "degree,continuity,inc",
        [
            (2, None, 1),
            (3, None, 1),
            (3, None, 2),
            (3, 1, 1),
            (3, 0, 1),
            (2, 0, 1),
        ],
    )
    def test_elevate_degree_preserves_geometry(
        self, degree: int, continuity: int | None, inc: int
    ) -> None:
        """elevate_degree on a periodic Bspline preserves geometry."""
        bsp = _make_periodic_bspline(4, degree, continuity)
        elevated = bsp.elevate_degree((inc,))

        pts = np.linspace(0.01, 0.99, 50)
        orig = bsp.to_open_bspline().evaluate(pts)
        elev = elevated.to_open_bspline().evaluate(pts)
        np.testing.assert_allclose(orig, elev, atol=1e-12)

    def test_elevate_degree_multidim_mixed_periodic_open(self) -> None:
        """elevate_degree preserves periodicity for mixed periodic/open 2D splines."""
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

        elevated = bsp.elevate_degree([1, 1])

        assert elevated.space.spaces[0].periodic
        assert not elevated.space.spaces[1].periodic
        assert elevated.degree == (3, 3)

        pts = rng.random((30, 2))
        pts[:, 0] = pts[:, 0] * 0.98 + 0.01
        orig = bsp.to_open_bspline().evaluate(pts)
        elev = elevated.to_open_bspline().evaluate(pts)
        np.testing.assert_allclose(orig, elev, atol=1e-12)

    def test_elevate_degree_rational_periodic(self) -> None:
        """elevate_degree preserves periodic NURBS geometry."""
        knots = create_uniform_periodic_knots(num_intervals=4, degree=3)
        space_1d = BsplineSpace1D(knots, 3, periodic=True)
        space = BsplineSpace([space_1d])
        n = space.num_total_basis
        rng = np.random.default_rng(42)
        ctrl_h = rng.random((n, 3))  # (x, y, w) homogeneous
        ctrl_h[:, -1] = np.abs(ctrl_h[:, -1]) + 0.5  # positive weights
        bsp = Bspline(space, ctrl_h, is_rational=True)

        elevated = bsp.elevate_degree(1)

        assert elevated.space.spaces[0].periodic
        assert elevated.is_rational
        expected_degree = 3 + 1
        assert elevated.space.spaces[0].degree == expected_degree

        pts = np.linspace(0.01, 0.99, 50)
        orig = bsp.to_open_bspline().evaluate(pts)
        elev = elevated.to_open_bspline().evaluate(pts)
        np.testing.assert_allclose(orig, elev, atol=1e-12)
