"""Tests for Bspline.split() method."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_1d_curve(degree: int = 3, dtype: type = np.float64) -> Bspline:
    """Create a 1D cubic curve with 5 control points."""
    knots: npt.NDArray[np.float64] = np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1], dtype=dtype)
    ctrl: npt.NDArray[np.float64] = np.array([[0, 0], [1, 2], [3, 3], [5, 1], [6, 0]], dtype=dtype)
    space = BsplineSpace([BsplineSpace1D(knots, degree)])
    return Bspline(space, ctrl)


def _make_2d_surface(dtype: type = np.float64) -> Bspline:
    """Create a 2D quadratic surface with interior knot."""
    knots_u: npt.NDArray[np.float64] = np.array([0, 0, 0, 0.5, 1, 1, 1], dtype=dtype)
    knots_v: npt.NDArray[np.float64] = np.array([0, 0, 0, 1, 1, 1], dtype=dtype)
    space = BsplineSpace([BsplineSpace1D(knots_u, 2), BsplineSpace1D(knots_v, 2)])
    rng = np.random.default_rng(42)
    ctrl: npt.NDArray[np.float64] = rng.standard_normal((4, 3, 3)).astype(dtype)
    return Bspline(space, ctrl)


def _make_rational_curve(dtype: type = np.float64) -> Bspline:
    """Create a rational quadratic B-spline (quarter circle)."""
    w = np.sqrt(2.0) / 2.0
    knots: npt.NDArray[np.float64] = np.array([0, 0, 0, 1, 1, 1], dtype=dtype)
    ctrl: npt.NDArray[np.float64] = np.array([[1, 0, 1], [w, w, w], [0, 1, 1]], dtype=dtype)
    space = BsplineSpace([BsplineSpace1D(knots, 2)])
    return Bspline(space, ctrl, is_rational=True)


# ---------------------------------------------------------------------------
# Tests: Bspline.split() 1D
# ---------------------------------------------------------------------------


class TestSplit1D:
    """Test splitting a 1D B-spline curve."""

    def test_split_returns_two_bsplines(self) -> None:
        """Split returns a tuple of two Bsplines with same degree."""
        crv = _make_1d_curve()
        left, right = crv.split(0, 0.3)
        assert isinstance(left, Bspline)
        assert isinstance(right, Bspline)
        assert left.degree == crv.degree
        assert right.degree == crv.degree

    def test_split_domains(self) -> None:
        """Split produces correct sub-domains."""
        crv = _make_1d_curve()
        left, right = crv.split(0, 0.3)
        np.testing.assert_allclose(left.space.spaces[0].domain, (0.0, 0.3))
        np.testing.assert_allclose(right.space.spaces[0].domain, (0.3, 1.0))

    def test_split_continuity(self) -> None:
        """Left and right match at the split point."""
        crv = _make_1d_curve()
        for val in [0.1, 0.3, 0.5, 0.7, 0.9]:
            left, right = crv.split(0, val)
            pt_left = left.evaluate(np.array([val])).squeeze()
            pt_right = right.evaluate(np.array([val])).squeeze()
            np.testing.assert_allclose(pt_left, pt_right, atol=1e-14)

    def test_split_matches_original(self) -> None:
        """Evaluations on halves match the original curve."""
        crv = _make_1d_curve()
        val = 0.4
        left, right = crv.split(0, val)

        # Left half.
        pts_left = np.linspace(0, val, 20)
        np.testing.assert_allclose(left.evaluate(pts_left), crv.evaluate(pts_left), atol=1e-13)

        # Right half.
        pts_right = np.linspace(val, 1.0, 20)
        np.testing.assert_allclose(right.evaluate(pts_right), crv.evaluate(pts_right), atol=1e-13)

    def test_split_at_existing_knot(self) -> None:
        """Splitting at an existing interior knot works correctly."""
        crv = _make_1d_curve()
        left, right = crv.split(0, 0.5)
        np.testing.assert_allclose(left.space.spaces[0].domain, (0.0, 0.5))
        np.testing.assert_allclose(right.space.spaces[0].domain, (0.5, 1.0))

        pt_l = left.evaluate(np.array([0.5])).squeeze()
        pt_r = right.evaluate(np.array([0.5])).squeeze()
        pt_o = crv.evaluate(np.array([0.5])).squeeze()
        np.testing.assert_allclose(pt_l, pt_o, atol=1e-14)
        np.testing.assert_allclose(pt_r, pt_o, atol=1e-14)

    def test_split_clamped_knots(self) -> None:
        """Both halves have clamped knot vectors at the split point."""
        crv = _make_1d_curve()
        left, right = crv.split(0, 0.3)
        p = crv.degree[0]

        # Left should end with p+1 copies of 0.3.
        left_knots = left.space.spaces[0].knots
        assert left.space.spaces[0].has_right_end_open()
        np.testing.assert_allclose(left_knots[-p - 1 :], 0.3)

        # Right should start with p+1 copies of 0.3.
        right_knots = right.space.spaces[0].knots
        assert right.space.spaces[0].has_left_end_open()
        np.testing.assert_allclose(right_knots[: p + 1], 0.3)


# ---------------------------------------------------------------------------
# Tests: Bspline.split() multi-dim
# ---------------------------------------------------------------------------


class TestSplitMultiDim:
    """Test splitting multi-dimensional B-splines."""

    def test_split_surface_dir0(self) -> None:
        """Split a surface along direction 0."""
        srf = _make_2d_surface()
        val = 0.3
        left, right = srf.split(0, val)

        assert left.dim == srf.dim
        assert right.dim == srf.dim
        assert left.degree == srf.degree
        assert right.degree == srf.degree

        # Check domains.
        np.testing.assert_allclose(left.space.spaces[0].domain, (0.0, val))
        np.testing.assert_allclose(right.space.spaces[0].domain, (val, 1.0))
        # Direction 1 unchanged.
        np.testing.assert_array_equal(left.space.spaces[1].knots, srf.space.spaces[1].knots)

    def test_split_surface_dir1(self) -> None:
        """Split a surface along direction 1."""
        srf = _make_2d_surface()
        val = 0.6
        left, right = srf.split(1, val)

        np.testing.assert_allclose(left.space.spaces[1].domain, (0.0, val))
        np.testing.assert_allclose(right.space.spaces[1].domain, (val, 1.0))
        # Direction 0 unchanged.
        np.testing.assert_array_equal(left.space.spaces[0].knots, srf.space.spaces[0].knots)

    def test_split_surface_continuity(self) -> None:
        """Split surface evaluations match at the split boundary."""
        srf = _make_2d_surface()
        val = 0.4
        left, right = srf.split(0, val)

        n = 15
        v_pts = np.linspace(0, 1, n)
        pts = np.column_stack([np.full(n, val), v_pts])
        vals_left = left.evaluate(pts)
        vals_right = right.evaluate(pts)
        vals_orig = srf.evaluate(pts)

        np.testing.assert_allclose(vals_left, vals_orig, atol=1e-13)
        np.testing.assert_allclose(vals_right, vals_orig, atol=1e-13)


# ---------------------------------------------------------------------------
# Tests: Bspline.split() rational
# ---------------------------------------------------------------------------


class TestSplitRational:
    """Test splitting rational B-splines."""

    def test_split_rational_preserves_flag(self) -> None:
        """Split preserves the rational flag."""
        crv = _make_rational_curve()
        left, right = crv.split(0, 0.5)
        assert left.is_rational
        assert right.is_rational
        assert left.rank == crv.rank

    def test_split_rational_matches_original(self) -> None:
        """Split of a rational curve matches original evaluations."""
        crv = _make_rational_curve()
        val = 0.4
        left, right = crv.split(0, val)

        pts_left = np.linspace(0, val, 20)
        np.testing.assert_allclose(left.evaluate(pts_left), crv.evaluate(pts_left), atol=1e-13)

        pts_right = np.linspace(val, 1.0, 20)
        np.testing.assert_allclose(right.evaluate(pts_right), crv.evaluate(pts_right), atol=1e-13)


# ---------------------------------------------------------------------------
# Tests: Bspline.split() error cases
# ---------------------------------------------------------------------------


class TestSplitErrors:
    """Test that split raises errors for invalid inputs."""

    def test_direction_out_of_range(self) -> None:
        """Direction >= dim raises ValueError."""
        crv = _make_1d_curve()
        with pytest.raises(ValueError, match="direction"):
            crv.split(1, 0.5)

    def test_direction_negative(self) -> None:
        """Negative direction raises ValueError."""
        crv = _make_1d_curve()
        with pytest.raises(ValueError, match="direction"):
            crv.split(-1, 0.5)

    def test_value_at_domain_start(self) -> None:
        """Value at domain start raises ValueError."""
        crv = _make_1d_curve()
        with pytest.raises(ValueError, match="strictly inside"):
            crv.split(0, 0.0)

    def test_value_at_domain_end(self) -> None:
        """Value at domain end raises ValueError."""
        crv = _make_1d_curve()
        with pytest.raises(ValueError, match="strictly inside"):
            crv.split(0, 1.0)

    def test_value_outside_domain(self) -> None:
        """Value outside domain raises ValueError."""
        crv = _make_1d_curve()
        with pytest.raises(ValueError, match="strictly inside"):
            crv.split(0, 1.5)
