"""Tests for the compat (B-spline compatibility) function."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
from pantr.cad import create_circle, create_line, make_compat

_KNOT_TOL = 1e-14


def _make_curve(knots: list[float], degree: int, rank: int = 1) -> Bspline:
    """Build a simple 1D B-spline curve with sequential control points."""
    sp = BsplineSpace1D(np.array(knots, dtype=np.float64), degree)
    n = sp.num_basis
    cp = np.arange(n * rank, dtype=np.float64).reshape(n, rank)
    return Bspline(BsplineSpace([sp]), cp)


def _make_surface(
    knots_u: list[float],
    deg_u: int,
    knots_v: list[float],
    deg_v: int,
) -> Bspline:
    """Build a simple 2D B-spline surface with sequential control points."""
    sp_u = BsplineSpace1D(np.array(knots_u, dtype=np.float64), deg_u)
    sp_v = BsplineSpace1D(np.array(knots_v, dtype=np.float64), deg_v)
    space = BsplineSpace([sp_u, sp_v])
    n = space.num_total_basis
    cp = np.arange(n * 3, dtype=np.float64).reshape(*space.num_basis, 3)
    return Bspline(space, cp)


class TestCompatBasic:
    """Test basic compat behavior."""

    def test_single_input_returned_unchanged(self) -> None:
        """Test that a single B-spline is returned as-is."""
        crv = _make_curve([0, 0, 0, 1, 1, 1], degree=2)
        result = make_compat(crv)
        assert len(result) == 1
        assert_allclose(result[0].control_points, crv.control_points)

    def test_identical_curves_unchanged(self) -> None:
        """Test that two identical curves come back with same structure."""
        crv = _make_curve([0, 0, 0, 1, 1, 1], degree=2)
        r1, _r2 = make_compat(crv, crv)
        assert r1.degree == crv.degree
        assert_allclose(r1.space.spaces[0].knots, crv.space.spaces[0].knots)

    def test_different_dim_raises(self) -> None:
        """Test that mixing curve and surface raises ValueError."""
        crv = _make_curve([0, 0, 1, 1], degree=1)
        srf = _make_surface([0, 0, 1, 1], 1, [0, 0, 1, 1], 1)
        with pytest.raises(ValueError, match="dimension"):
            make_compat(crv, srf)

    def test_axis_out_of_range_raises(self) -> None:
        """Test that an out-of-range axis raises ValueError."""
        crv = _make_curve([0, 0, 1, 1], degree=1)
        with pytest.raises(ValueError, match="out of range"):
            make_compat(crv, crv, axes=5)


class TestCompatDegreeElevation:
    """Test that compat elevates degrees to match."""

    def test_different_degrees(self) -> None:
        """Test curves with different degrees get elevated to max."""
        c1 = _make_curve([0, 0, 1, 1], degree=1)
        c2 = _make_curve([0, 0, 0, 1, 1, 1], degree=2)
        r1, r2 = make_compat(c1, c2)
        assert r1.degree == (2,)
        assert r2.degree == (2,)

    def test_geometric_invariance_after_elevation(self) -> None:
        """Test that degree elevation preserves the geometry."""
        c1 = create_line([0, 0, 0], [1, 0, 0])
        c2 = create_circle(angle=np.pi / 2)
        t = np.linspace(0, 1, 20)
        pts_before_1 = c1.evaluate(t)
        pts_before_2 = c2.evaluate(t)
        r1, r2 = make_compat(c1, c2)
        pts_after_1 = r1.evaluate(t)
        pts_after_2 = r2.evaluate(t)
        assert_allclose(pts_after_1, pts_before_1, atol=1e-13)
        assert_allclose(pts_after_2, pts_before_2, atol=1e-13)


class TestCompatDomainRemap:
    """Test that compat remaps domains to a common envelope."""

    def test_different_domains(self) -> None:
        """Test curves with different domains get remapped."""
        c1 = _make_curve([0, 0, 1, 1], degree=1)
        c2 = _make_curve([2, 2, 3, 3], degree=1)
        r1, r2 = make_compat(c1, c2)
        assert_allclose(r1.space.spaces[0].domain, [0, 3])
        assert_allclose(r2.space.spaces[0].domain, [0, 3])


class TestCompatKnotMerge:
    """Test that compat merges knot vectors."""

    def test_different_interior_knots(self) -> None:
        """Test curves with different interior knots get merged."""
        # c1 has interior knot at 0.5; c2 has interior knot at 0.3
        c1 = _make_curve([0, 0, 0, 0.5, 1, 1, 1], degree=2)
        c2 = _make_curve([0, 0, 0, 0.3, 1, 1, 1], degree=2)
        r1, r2 = make_compat(c1, c2)
        # Both should have interior knots at 0.3 and 0.5
        knots1 = r1.space.spaces[0].knots
        knots2 = r2.space.spaces[0].knots
        assert_allclose(knots1, knots2)
        # Check that both 0.3 and 0.5 appear in the knot vector
        assert np.any(np.abs(knots1 - 0.3) < _KNOT_TOL)
        assert np.any(np.abs(knots1 - 0.5) < _KNOT_TOL)

    def test_geometric_invariance_after_knot_insertion(self) -> None:
        """Test that knot insertion preserves the geometry."""
        c1 = _make_curve([0, 0, 0, 0.5, 1, 1, 1], degree=2, rank=3)
        c2 = _make_curve([0, 0, 0, 0.3, 1, 1, 1], degree=2, rank=3)
        t = np.linspace(0, 1, 30)
        pts_before_1 = c1.evaluate(t)
        pts_before_2 = c2.evaluate(t)
        r1, r2 = make_compat(c1, c2)
        pts_after_1 = r1.evaluate(t)
        pts_after_2 = r2.evaluate(t)
        assert_allclose(pts_after_1, pts_before_1, atol=1e-12)
        assert_allclose(pts_after_2, pts_before_2, atol=1e-12)

    def test_multiplicity_max(self) -> None:
        """Test that merged knot has max multiplicity across inputs."""
        # c1: knot at 0.5 with mult 1; c2: knot at 0.5 with mult 2
        c1 = _make_curve([0, 0, 0, 0.5, 1, 1, 1], degree=2)
        c2 = _make_curve([0, 0, 0, 0.5, 0.5, 1, 1, 1], degree=2)
        r1, r2 = make_compat(c1, c2)
        knots1 = r1.space.spaces[0].knots
        knots2 = r2.space.spaces[0].knots
        assert_allclose(knots1, knots2)
        # 0.5 should appear with multiplicity 2
        count_05 = np.sum(np.abs(knots1 - 0.5) < _KNOT_TOL)
        assert count_05 == 2  # noqa: PLR2004


class TestCompatSurface:
    """Test compat on surfaces with selective axes."""

    def test_selective_axis(self) -> None:
        """Test compat on a single axis of a 2D B-spline."""
        s1 = _make_surface([0, 0, 1, 1], 1, [0, 0, 0, 1, 1, 1], 2)
        s2 = _make_surface([0, 0, 0, 1, 1, 1], 2, [0, 0, 0, 1, 1, 1], 2)
        r1, r2 = make_compat(s1, s2, axes=0)
        # Axis 0 should be elevated to degree 2
        assert r1.degree[0] == 2  # noqa: PLR2004
        assert r2.degree[0] == 2  # noqa: PLR2004
        # Axis 1 should be unchanged for s1
        assert r1.degree[1] == 2  # noqa: PLR2004


class TestCompatCombined:
    """Test compat with all three stages active simultaneously."""

    def test_degree_and_knots(self) -> None:
        """Test compat with both different degrees and different knots."""
        c1 = _make_curve([0, 0, 1, 1], degree=1)
        c2 = _make_curve([0, 0, 0, 0.5, 1, 1, 1], degree=2)
        r1, r2 = make_compat(c1, c2)
        assert r1.degree == r2.degree
        assert_allclose(r1.space.spaces[0].knots, r2.space.spaces[0].knots)

    def test_three_curves(self) -> None:
        """Test compat with three different curves."""
        c1 = _make_curve([0, 0, 1, 1], degree=1, rank=3)
        c2 = _make_curve([0, 0, 0, 0.5, 1, 1, 1], degree=2, rank=3)
        c3 = _make_curve([0, 0, 0, 0, 0.3, 0.7, 1, 1, 1, 1], degree=3, rank=3)
        r1, r2, r3 = make_compat(c1, c2, c3)
        # All should have degree 3
        assert r1.degree == (3,)
        assert r2.degree == (3,)
        assert r3.degree == (3,)
        # All should share the same knot vector
        assert_allclose(r1.space.spaces[0].knots, r2.space.spaces[0].knots)
        assert_allclose(r2.space.spaces[0].knots, r3.space.spaces[0].knots)
        # Geometry preserved
        t = np.linspace(0, 1, 30)
        assert_allclose(r1.evaluate(t), c1.evaluate(t), atol=1e-12)
        assert_allclose(r2.evaluate(t), c2.evaluate(t), atol=1e-12)
        assert_allclose(r3.evaluate(t), c3.evaluate(t), atol=1e-12)
