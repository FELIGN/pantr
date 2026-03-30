"""Tests for Bspline.slice() and Bspline.boundary() methods."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D, create_uniform_periodic_knots

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
    """Create a 2D quadratic surface (3x3 control points, rank 3)."""
    knots: npt.NDArray[np.float64] = np.array([0, 0, 0, 1, 1, 1], dtype=dtype)
    space = BsplineSpace([BsplineSpace1D(knots, 2), BsplineSpace1D(knots, 2)])
    rng = np.random.default_rng(42)
    ctrl: npt.NDArray[np.float64] = rng.standard_normal((3, 3, 3)).astype(dtype)
    return Bspline(space, ctrl)


def _make_3d_volume(dtype: type = np.float64) -> Bspline:
    """Create a 3D linear volume (2x2x2 control points, rank 3)."""
    knots = [0.0, 0.0, 1.0, 1.0]
    space = BsplineSpace([BsplineSpace1D(knots, 1)] * 3)
    rng = np.random.default_rng(123)
    ctrl: npt.NDArray[np.float64] = rng.standard_normal((2, 2, 2, 3)).astype(dtype)
    return Bspline(space, ctrl)


def _make_c0_surface(dtype: type = np.float64) -> Bspline:
    """Create a surface with C0 knot at u=0.5 (quadratic, 2 patches in u)."""
    knots_u = [0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0]
    knots_v = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    space = BsplineSpace([BsplineSpace1D(knots_u, 2), BsplineSpace1D(knots_v, 2)])
    rng = np.random.default_rng(7)
    ctrl: npt.NDArray[np.float64] = rng.standard_normal((5, 3, 3)).astype(dtype)
    return Bspline(space, ctrl)


def _make_rational_curve(dtype: type = np.float64) -> Bspline:
    """Create a rational quadratic B-spline (quarter circle)."""
    w = np.sqrt(2.0) / 2.0
    knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    ctrl: npt.NDArray[np.float64] = np.array([[1, 0, 1], [w, w, w], [0, 1, 1]], dtype=dtype)
    space = BsplineSpace([BsplineSpace1D(knots, 2)])
    return Bspline(space, ctrl, is_rational=True)


def _make_rational_surface(dtype: type = np.float64) -> Bspline:
    """Create a rational quadratic surface."""
    knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    space = BsplineSpace([BsplineSpace1D(knots, 2), BsplineSpace1D(knots, 2)])
    rng = np.random.default_rng(99)
    ctrl: npt.NDArray[np.float64] = rng.standard_normal((3, 3, 3)).astype(dtype)
    ctrl[:, :, -1] = np.abs(ctrl[:, :, -1]) + 0.5
    return Bspline(space, ctrl, is_rational=True)


def _make_periodic_curve(dtype: type = np.float64) -> Bspline:
    """Create a periodic cubic B-spline curve."""
    knots = create_uniform_periodic_knots(num_intervals=4, degree=3, continuity=2, dtype=dtype)
    space_1d = BsplineSpace1D(knots, 3, periodic=True)
    space = BsplineSpace([space_1d])
    rng = np.random.default_rng(55)
    ctrl: npt.NDArray[np.float64] = rng.standard_normal((space_1d.num_basis, 2)).astype(dtype)
    return Bspline(space, ctrl)


def _make_periodic_surface(dtype: type = np.float64) -> Bspline:
    """Create a surface with one periodic direction."""
    knots_u = create_uniform_periodic_knots(num_intervals=3, degree=2, continuity=1, dtype=dtype)
    space_u = BsplineSpace1D(knots_u, 2, periodic=True)
    space_v = BsplineSpace1D([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 2)
    space = BsplineSpace([space_u, space_v])
    rng = np.random.default_rng(77)
    ctrl: npt.NDArray[np.float64] = rng.standard_normal((space_u.num_basis, 3, 3)).astype(dtype)
    return Bspline(space, ctrl)


# ---------------------------------------------------------------------------
# Tests: Bspline.slice()
# ---------------------------------------------------------------------------


class TestSlice1D:
    """Test slicing a 1D curve (produces a point)."""

    def test_slice_matches_evaluate(self) -> None:
        """Slicing a curve at u should match evaluate([u])."""
        crv = _make_1d_curve()
        for u in [0.0, 0.25, 0.5, 0.75, 1.0]:
            pt_slice = crv.slice(0, u)
            pt_eval = crv.evaluate(np.array([u]))
            assert isinstance(pt_slice, np.ndarray)
            np.testing.assert_allclose(pt_slice, pt_eval, atol=1e-14)

    def test_slice_returns_ndarray(self) -> None:
        """Slicing a 1D B-spline returns an ndarray, not a Bspline."""
        crv = _make_1d_curve()
        result = crv.slice(0, 0.5)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)

    def test_slice_at_boundary_knot(self) -> None:
        """Slicing at a clamped boundary should return the endpoint control point."""
        crv = _make_1d_curve()
        pt_start = crv.slice(0, 0.0)
        pt_end = crv.slice(0, 1.0)
        assert isinstance(pt_start, np.ndarray)
        assert isinstance(pt_end, np.ndarray)
        np.testing.assert_allclose(pt_start, crv.control_points[0], atol=1e-15)
        np.testing.assert_allclose(pt_end, crv.control_points[-1], atol=1e-15)


class TestSlice2D:
    """Test slicing a 2D surface (produces a curve)."""

    def test_slice_returns_bspline(self) -> None:
        """Slicing a 2D surface returns a 1D Bspline."""
        srf = _make_2d_surface()
        result = srf.slice(0, 0.5)
        assert isinstance(result, Bspline)
        assert result.dim == 1

    def test_slice_axis0_matches_evaluate(self) -> None:
        """Slicing axis 0 then evaluating should match direct 2D evaluate."""
        srf = _make_2d_surface()
        u, v = 0.3, 0.7
        crv = srf.slice(0, u)
        assert isinstance(crv, Bspline)
        pt_slice = crv.evaluate(np.array([v]))
        pt_direct = srf.evaluate(np.array([[u, v]]))
        np.testing.assert_allclose(pt_slice, pt_direct, atol=1e-13)

    def test_slice_axis1_matches_evaluate(self) -> None:
        """Slicing axis 1 then evaluating should match direct 2D evaluate."""
        srf = _make_2d_surface()
        u, v = 0.4, 0.6
        crv = srf.slice(1, v)
        assert isinstance(crv, Bspline)
        pt_slice = crv.evaluate(np.array([u]))
        pt_direct = srf.evaluate(np.array([[u, v]]))
        np.testing.assert_allclose(pt_slice, pt_direct, atol=1e-13)


class TestSlice3D:
    """Test slicing a 3D volume."""

    def test_volume_to_surface(self) -> None:
        """Slicing a volume produces a surface."""
        vol = _make_3d_volume()
        srf = vol.slice(2, 0.5)
        assert isinstance(srf, Bspline)
        expected_dim = 2
        assert srf.dim == expected_dim

    def test_composable_slice_matches_evaluate(self) -> None:
        """vol.slice(2,w).slice(1,v).slice(0,u) matches vol.evaluate([u,v,w])."""
        vol = _make_3d_volume()
        u, v, w = 0.2, 0.5, 0.8
        srf = vol.slice(2, w)
        assert isinstance(srf, Bspline)
        crv = srf.slice(1, v)
        assert isinstance(crv, Bspline)
        pt_slice = crv.slice(0, u)
        pt_direct = vol.evaluate(np.array([[u, v, w]]))
        assert isinstance(pt_slice, np.ndarray)
        np.testing.assert_allclose(pt_slice, pt_direct, atol=1e-13)

    def test_slice_different_axes(self) -> None:
        """Slicing different axes first should give same final point."""
        vol = _make_3d_volume()
        u, v, w = 0.3, 0.6, 0.4
        # Slice axis 0 first, then 0 (was axis 1), then 0 (was axis 2).
        srf1 = vol.slice(0, u)
        assert isinstance(srf1, Bspline)
        crv1 = srf1.slice(0, v)
        assert isinstance(crv1, Bspline)
        pt1 = crv1.slice(0, w)
        # Slice in reverse order.
        srf2 = vol.slice(2, w)
        assert isinstance(srf2, Bspline)
        crv2 = srf2.slice(1, v)
        assert isinstance(crv2, Bspline)
        pt2 = crv2.slice(0, u)
        pt_direct = vol.evaluate(np.array([[u, v, w]]))
        assert isinstance(pt1, np.ndarray)
        assert isinstance(pt2, np.ndarray)
        np.testing.assert_allclose(pt1, pt_direct, atol=1e-13)
        np.testing.assert_allclose(pt2, pt_direct, atol=1e-13)


class TestSliceC0Knot:
    """Test the C0 knot optimization (direct control point lookup)."""

    def test_c0_knot_matches_evaluate(self) -> None:
        """Slicing at a C0 knot should still give correct results."""
        srf = _make_c0_surface()
        crv = srf.slice(0, 0.5)
        assert isinstance(crv, Bspline)
        v = 0.4
        pt_slice = crv.evaluate(np.array([v]))
        pt_direct = srf.evaluate(np.array([[0.5, v]]))
        np.testing.assert_allclose(pt_slice, pt_direct, atol=1e-13)


class TestSliceRational:
    """Test slicing rational (NURBS) B-splines."""

    def test_rational_1d_projects_correctly(self) -> None:
        """Slicing a rational 1D curve returns projected physical coordinates."""
        crv = _make_rational_curve()
        pt = crv.slice(0, 0.5)
        assert isinstance(pt, np.ndarray)
        pt_eval = crv.evaluate(np.array([0.5]))
        np.testing.assert_allclose(pt, pt_eval, atol=1e-14)

    def test_rational_2d_preserves_rationality(self) -> None:
        """Slicing a rational 2D surface returns a rational 1D curve."""
        srf = _make_rational_surface()
        crv = srf.slice(0, 0.5)
        assert isinstance(crv, Bspline)
        assert crv.is_rational

    def test_rational_2d_matches_evaluate(self) -> None:
        """Slicing a rational surface then evaluating matches direct evaluation."""
        srf = _make_rational_surface()
        u, v = 0.3, 0.7
        crv = srf.slice(0, u)
        assert isinstance(crv, Bspline)
        pt_slice = crv.evaluate(np.array([v]))
        pt_direct = srf.evaluate(np.array([[u, v]]))
        np.testing.assert_allclose(pt_slice, pt_direct, atol=1e-12)


class TestSlicePeriodic:
    """Test slicing periodic B-splines."""

    def test_periodic_1d_matches_evaluate(self) -> None:
        """Slicing a periodic 1D curve matches evaluate."""
        crv = _make_periodic_curve()
        domain = crv.space.spaces[0].domain
        for t in np.linspace(float(domain[0]), float(domain[1]), 7):
            pt_slice = crv.slice(0, t)
            pt_eval = crv.evaluate(np.array([t]))
            assert isinstance(pt_slice, np.ndarray)
            np.testing.assert_allclose(pt_slice, pt_eval, atol=1e-13)

    def test_periodic_surface_slice_matches_evaluate(self) -> None:
        """Slicing a periodic surface along the periodic direction matches evaluate."""
        srf = _make_periodic_surface()
        domain_u = srf.space.spaces[0].domain
        u = float(domain_u[0]) + 0.3 * (float(domain_u[1]) - float(domain_u[0]))
        v = 0.6
        crv = srf.slice(0, u)
        assert isinstance(crv, Bspline)
        pt_slice = crv.evaluate(np.array([v]))
        pt_direct = srf.evaluate(np.array([[u, v]]))
        np.testing.assert_allclose(pt_slice, pt_direct, atol=1e-12)


class TestSliceDtype:
    """Test that slice preserves dtype."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_slice_preserves_dtype(self, dtype: type) -> None:
        """Output dtype matches input dtype."""
        srf = _make_2d_surface(dtype=dtype)
        crv = srf.slice(0, 0.5)
        assert isinstance(crv, Bspline)
        assert crv.dtype == dtype

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_slice_1d_preserves_dtype(self, dtype: type) -> None:
        """1D slice output array has matching dtype."""
        crv = _make_1d_curve(dtype=dtype)
        pt = crv.slice(0, 0.5)
        assert isinstance(pt, np.ndarray)
        assert pt.dtype == dtype


class TestSliceErrors:
    """Test error handling for slice."""

    def test_axis_out_of_range(self) -> None:
        """Axis out of range raises ValueError."""
        crv = _make_1d_curve()
        with pytest.raises(ValueError, match="axis must be in"):
            crv.slice(1, 0.5)

    def test_axis_negative(self) -> None:
        """Negative axis raises ValueError."""
        crv = _make_1d_curve()
        with pytest.raises(ValueError, match="axis must be in"):
            crv.slice(-1, 0.5)

    def test_value_out_of_domain(self) -> None:
        """Value outside domain raises ValueError."""
        crv = _make_1d_curve()
        with pytest.raises(ValueError, match="outside the domain"):
            crv.slice(0, 2.0)


# ---------------------------------------------------------------------------
# Tests: Bspline.boundary()
# ---------------------------------------------------------------------------


class TestBoundary:
    """Test boundary extraction."""

    def test_boundary_1d_start(self) -> None:
        """Boundary at side=0 returns the start control point."""
        crv = _make_1d_curve()
        pt = crv.boundary(0, 0)
        assert isinstance(pt, np.ndarray)
        np.testing.assert_allclose(pt, crv.control_points[0], atol=1e-15)

    def test_boundary_1d_end(self) -> None:
        """Boundary at side=1 returns the end control point."""
        crv = _make_1d_curve()
        pt = crv.boundary(0, 1)
        assert isinstance(pt, np.ndarray)
        np.testing.assert_allclose(pt, crv.control_points[-1], atol=1e-15)

    def test_boundary_2d_matches_evaluate(self) -> None:
        """Boundary curve of a surface matches evaluation at domain boundary."""
        srf = _make_2d_surface()
        crv_start = srf.boundary(0, 0)
        crv_end = srf.boundary(0, 1)
        assert isinstance(crv_start, Bspline)
        assert isinstance(crv_end, Bspline)

        v = 0.4
        pt_start = crv_start.evaluate(np.array([v]))
        pt_end = crv_end.evaluate(np.array([v]))
        domain = srf.space.spaces[0].domain
        pt_start_direct = srf.evaluate(np.array([[float(domain[0]), v]]))
        pt_end_direct = srf.evaluate(np.array([[float(domain[1]), v]]))
        np.testing.assert_allclose(pt_start, pt_start_direct, atol=1e-13)
        np.testing.assert_allclose(pt_end, pt_end_direct, atol=1e-13)

    def test_boundary_invalid_side(self) -> None:
        """Invalid side raises ValueError."""
        crv = _make_1d_curve()
        with pytest.raises(ValueError, match="side must be 0 or 1"):
            crv.boundary(0, 2)

    def test_boundary_axis_out_of_range(self) -> None:
        """Axis out of range raises ValueError."""
        crv = _make_1d_curve()
        with pytest.raises(ValueError, match="axis must be in"):
            crv.boundary(1, 0)
