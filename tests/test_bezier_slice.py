"""Tests for Bezier.slice() and Bezier.boundary() methods."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bezier import Bezier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_1d_curve(dtype: type = np.float64) -> Bezier:
    """Create a 1D cubic Bézier curve (4 control points, rank 2)."""
    ctrl: npt.NDArray[np.float64] = np.array([[0, 0], [1, 3], [4, 3], [5, 0]], dtype=dtype)
    return Bezier(ctrl)


def _make_2d_surface(dtype: type = np.float64) -> Bezier:
    """Create a 2D quadratic Bézier surface (3x3 control points, rank 3)."""
    rng = np.random.default_rng(42)
    ctrl: npt.NDArray[np.float64] = rng.standard_normal((3, 3, 3)).astype(dtype)
    return Bezier(ctrl)


def _make_3d_volume(dtype: type = np.float64) -> Bezier:
    """Create a 3D linear Bézier volume (2x2x2 control points, rank 3)."""
    rng = np.random.default_rng(123)
    ctrl: npt.NDArray[np.float64] = rng.standard_normal((2, 2, 2, 3)).astype(dtype)
    return Bezier(ctrl)


def _make_rational_curve(dtype: type = np.float64) -> Bezier:
    """Create a rational quadratic Bézier (quarter circle)."""
    w = np.sqrt(2.0) / 2.0
    ctrl: npt.NDArray[np.float64] = np.array([[1, 0, 1], [w, w, w], [0, 1, 1]], dtype=dtype)
    return Bezier(ctrl, is_rational=True)


def _make_rational_surface(dtype: type = np.float64) -> Bezier:
    """Create a rational quadratic Bézier surface."""
    rng = np.random.default_rng(99)
    ctrl: npt.NDArray[np.float64] = rng.standard_normal((3, 3, 3)).astype(dtype)
    ctrl[:, :, -1] = np.abs(ctrl[:, :, -1]) + 0.5
    return Bezier(ctrl, is_rational=True)


# ---------------------------------------------------------------------------
# Tests: Bezier.slice()
# ---------------------------------------------------------------------------


class TestSlice1D:
    """Test slicing a 1D curve (produces a point)."""

    def test_slice_matches_evaluate(self) -> None:
        """Slicing a curve at u should match evaluate([u])."""
        crv = _make_1d_curve()
        for u in [0.0, 0.25, 0.5, 0.75, 1.0]:
            pt_slice = crv.slice(0, u)
            pt_eval = crv.evaluate(np.array([u])).squeeze()
            assert isinstance(pt_slice, np.ndarray)
            np.testing.assert_allclose(pt_slice, pt_eval, atol=1e-14)

    def test_slice_returns_ndarray(self) -> None:
        """Slicing a 1D Bézier returns an ndarray, not a Bezier."""
        crv = _make_1d_curve()
        result = crv.slice(0, 0.5)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)

    def test_slice_at_boundary(self) -> None:
        """Slicing at boundaries returns the endpoint control points."""
        crv = _make_1d_curve()
        pt_start = crv.slice(0, 0.0)
        pt_end = crv.slice(0, 1.0)
        assert isinstance(pt_start, np.ndarray)
        assert isinstance(pt_end, np.ndarray)
        np.testing.assert_allclose(pt_start, crv.control_points[0], atol=1e-15)
        np.testing.assert_allclose(pt_end, crv.control_points[-1], atol=1e-15)


class TestSlice2D:
    """Test slicing a 2D surface (produces a curve)."""

    def test_slice_returns_bezier(self) -> None:
        """Slicing a 2D Bézier returns a 1D Bezier."""
        srf = _make_2d_surface()
        result = srf.slice(0, 0.5)
        assert isinstance(result, Bezier)
        assert result.dim == 1

    def test_slice_preserves_degree(self) -> None:
        """Slicing a quadratic surface along axis 0 gives a quadratic curve."""
        srf = _make_2d_surface()
        crv = srf.slice(0, 0.5)
        assert isinstance(crv, Bezier)
        expected_degree = (2,)
        assert crv.degree == expected_degree

    def test_slice_axis0_matches_evaluate(self) -> None:
        """Slicing axis 0 then evaluating should match direct 2D evaluate."""
        srf = _make_2d_surface()
        u, v = 0.3, 0.7
        crv = srf.slice(0, u)
        assert isinstance(crv, Bezier)
        pt_slice = crv.evaluate(np.array([v])).squeeze()
        pt_direct = srf.evaluate(np.array([[u, v]])).squeeze()
        np.testing.assert_allclose(pt_slice, pt_direct, atol=1e-13)

    def test_slice_axis1_matches_evaluate(self) -> None:
        """Slicing axis 1 then evaluating should match direct 2D evaluate."""
        srf = _make_2d_surface()
        u, v = 0.4, 0.6
        crv = srf.slice(1, v)
        assert isinstance(crv, Bezier)
        pt_slice = crv.evaluate(np.array([u])).squeeze()
        pt_direct = srf.evaluate(np.array([[u, v]])).squeeze()
        np.testing.assert_allclose(pt_slice, pt_direct, atol=1e-13)


class TestSlice3D:
    """Test slicing a 3D volume."""

    def test_volume_to_surface(self) -> None:
        """Slicing a volume produces a surface."""
        vol = _make_3d_volume()
        srf = vol.slice(2, 0.5)
        assert isinstance(srf, Bezier)
        expected_dim = 2
        assert srf.dim == expected_dim

    def test_composable_slice_matches_evaluate(self) -> None:
        """vol.slice(2,w).slice(1,v).slice(0,u) matches vol.evaluate([u,v,w])."""
        vol = _make_3d_volume()
        u, v, w = 0.2, 0.5, 0.8
        srf = vol.slice(2, w)
        assert isinstance(srf, Bezier)
        crv = srf.slice(1, v)
        assert isinstance(crv, Bezier)
        pt_slice = crv.slice(0, u)
        pt_direct = vol.evaluate(np.array([[u, v, w]])).squeeze()
        assert isinstance(pt_slice, np.ndarray)
        np.testing.assert_allclose(pt_slice, pt_direct, atol=1e-13)

    def test_slice_different_axes(self) -> None:
        """Slicing different axes first should give same final point."""
        vol = _make_3d_volume()
        u, v, w = 0.3, 0.6, 0.4
        srf1 = vol.slice(0, u)
        assert isinstance(srf1, Bezier)
        crv1 = srf1.slice(0, v)
        assert isinstance(crv1, Bezier)
        pt1 = crv1.slice(0, w)
        srf2 = vol.slice(2, w)
        assert isinstance(srf2, Bezier)
        crv2 = srf2.slice(1, v)
        assert isinstance(crv2, Bezier)
        pt2 = crv2.slice(0, u)
        pt_direct = vol.evaluate(np.array([[u, v, w]])).squeeze()
        assert isinstance(pt1, np.ndarray)
        assert isinstance(pt2, np.ndarray)
        np.testing.assert_allclose(pt1, pt_direct, atol=1e-13)
        np.testing.assert_allclose(pt2, pt_direct, atol=1e-13)


class TestSliceRational:
    """Test slicing rational (NURBS) Béziers."""

    def test_rational_1d_projects_correctly(self) -> None:
        """Slicing a rational 1D curve returns projected physical coordinates."""
        crv = _make_rational_curve()
        pt = crv.slice(0, 0.5)
        assert isinstance(pt, np.ndarray)
        pt_eval = crv.evaluate(np.array([0.5])).squeeze()
        np.testing.assert_allclose(pt, pt_eval, atol=1e-14)

    def test_rational_2d_preserves_rationality(self) -> None:
        """Slicing a rational 2D surface returns a rational 1D curve."""
        srf = _make_rational_surface()
        crv = srf.slice(0, 0.5)
        assert isinstance(crv, Bezier)
        assert crv.is_rational

    def test_rational_2d_matches_evaluate(self) -> None:
        """Slicing a rational surface then evaluating matches direct evaluation."""
        srf = _make_rational_surface()
        u, v = 0.3, 0.7
        crv = srf.slice(0, u)
        assert isinstance(crv, Bezier)
        pt_slice = crv.evaluate(np.array([v])).squeeze()
        pt_direct = srf.evaluate(np.array([[u, v]])).squeeze()
        np.testing.assert_allclose(pt_slice, pt_direct, atol=1e-12)


class TestSliceDtype:
    """Test that slice preserves dtype."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_slice_preserves_dtype(self, dtype: type) -> None:
        """Output dtype matches input dtype."""
        srf = _make_2d_surface(dtype=dtype)
        crv = srf.slice(0, 0.5)
        assert isinstance(crv, Bezier)
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

    def test_value_below_zero(self) -> None:
        """Value below 0 raises ValueError."""
        crv = _make_1d_curve()
        with pytest.raises(ValueError, match=r"value must be in \[0, 1\]"):
            crv.slice(0, -0.1)

    def test_value_above_one(self) -> None:
        """Value above 1 raises ValueError."""
        crv = _make_1d_curve()
        with pytest.raises(ValueError, match=r"value must be in \[0, 1\]"):
            crv.slice(0, 1.5)


# ---------------------------------------------------------------------------
# Tests: Bezier.boundary()
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
        assert isinstance(crv_start, Bezier)
        assert isinstance(crv_end, Bezier)

        v = 0.4
        pt_start = crv_start.evaluate(np.array([v])).squeeze()
        pt_end = crv_end.evaluate(np.array([v])).squeeze()
        pt_start_direct = srf.evaluate(np.array([[0.0, v]])).squeeze()
        pt_end_direct = srf.evaluate(np.array([[1.0, v]])).squeeze()
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
