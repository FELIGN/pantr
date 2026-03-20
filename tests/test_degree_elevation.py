from __future__ import annotations

import numpy as np
import pytest

from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D


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

    with pytest.raises(ValueError, match="(?i)at least one.*positive"):
        bspline.elevate_degree(0)

    with pytest.raises(ValueError, match="(?i)at least one.*positive"):
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
