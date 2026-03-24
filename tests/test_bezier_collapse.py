"""Tests for Bezier.collapse_along_axis() method."""

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


def _make_3d_higher_degree(dtype: type = np.float64) -> Bezier:
    """Create a 3D Bézier volume with degrees (2, 3, 4)."""
    rng = np.random.default_rng(456)
    ctrl: npt.NDArray[np.float64] = rng.standard_normal((3, 4, 5, 2)).astype(dtype)
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
# Tests: 1D (no-op case)
# ---------------------------------------------------------------------------


class TestCollapse1D:
    """Test that collapsing a 1D Bézier raises an error."""

    def test_raises_on_1d(self) -> None:
        """Collapsing a 1D Bézier raises ValueError."""
        crv = _make_1d_curve()
        with pytest.raises(ValueError, match="dim >= 2"):
            crv.collapse_along_axis(0, [])


# ---------------------------------------------------------------------------
# Tests: 2D surface
# ---------------------------------------------------------------------------


class TestCollapse2D:
    """Test collapse of a 2D surface to a 1D curve."""

    def test_collapse_axis0_returns_1d(self) -> None:
        """Collapsing along axis 0 produces a 1D Bézier."""
        srf = _make_2d_surface()
        crv = srf.collapse_along_axis(0, [0.5])
        assert isinstance(crv, Bezier)
        assert crv.dim == 1
        assert crv.degree == (srf.degree[0],)

    def test_collapse_axis1_returns_1d(self) -> None:
        """Collapsing along axis 1 produces a 1D Bézier."""
        srf = _make_2d_surface()
        crv = srf.collapse_along_axis(1, [0.5])
        assert isinstance(crv, Bezier)
        assert crv.dim == 1
        assert crv.degree == (srf.degree[1],)

    def test_collapse_axis0_evaluate_matches(self) -> None:
        """Evaluating the collapsed curve matches evaluating the surface."""
        srf = _make_2d_surface()
        v = 0.6
        crv = srf.collapse_along_axis(0, [v])
        for u in [0.0, 0.25, 0.5, 0.75, 1.0]:
            pt_collapse = crv.evaluate(np.array([u], dtype=srf.dtype)).squeeze()
            pt_direct = srf.evaluate(np.array([[u, v]], dtype=srf.dtype)).squeeze()
            np.testing.assert_allclose(pt_collapse, pt_direct, atol=1e-13)

    def test_collapse_axis1_evaluate_matches(self) -> None:
        """Evaluating the collapsed curve matches evaluating the surface."""
        srf = _make_2d_surface()
        u = 0.3
        crv = srf.collapse_along_axis(1, [u])
        for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
            pt_collapse = crv.evaluate(np.array([v], dtype=srf.dtype)).squeeze()
            pt_direct = srf.evaluate(np.array([[u, v]], dtype=srf.dtype)).squeeze()
            np.testing.assert_allclose(pt_collapse, pt_direct, atol=1e-13)

    def test_collapse_at_boundary_zero(self) -> None:
        """Collapsing at boundary 0 matches slicing."""
        srf = _make_2d_surface()
        crv_collapse = srf.collapse_along_axis(0, [0.0])
        crv_slice = srf.slice(1, 0.0)
        assert isinstance(crv_slice, Bezier)
        np.testing.assert_allclose(
            crv_collapse.control_points, crv_slice.control_points, atol=1e-14
        )

    def test_collapse_at_boundary_one(self) -> None:
        """Collapsing at boundary 1 matches slicing."""
        srf = _make_2d_surface()
        crv_collapse = srf.collapse_along_axis(1, [1.0])
        crv_slice = srf.slice(0, 1.0)
        assert isinstance(crv_slice, Bezier)
        np.testing.assert_allclose(
            crv_collapse.control_points, crv_slice.control_points, atol=1e-14
        )


# ---------------------------------------------------------------------------
# Tests: 3D volume
# ---------------------------------------------------------------------------


class TestCollapse3D:
    """Test collapse of a 3D volume to a 1D curve."""

    def test_collapse_axis0(self) -> None:
        """Collapsing along axis 0 matches sequential slicing."""
        vol = _make_3d_volume()
        v, w = 0.4, 0.7
        crv = vol.collapse_along_axis(0, [v, w])
        assert crv.dim == 1
        assert crv.degree == (vol.degree[0],)
        for u in [0.0, 0.5, 1.0]:
            pt_collapse = crv.evaluate(np.array([u], dtype=vol.dtype)).squeeze()
            pt_direct = vol.evaluate(np.array([[u, v, w]], dtype=vol.dtype)).squeeze()
            np.testing.assert_allclose(pt_collapse, pt_direct, atol=1e-13)

    def test_collapse_axis1(self) -> None:
        """Collapsing along axis 1 matches direct evaluation."""
        vol = _make_3d_volume()
        u, w = 0.3, 0.8
        crv = vol.collapse_along_axis(1, [u, w])
        assert crv.dim == 1
        assert crv.degree == (vol.degree[1],)
        for v in [0.0, 0.5, 1.0]:
            pt_collapse = crv.evaluate(np.array([v], dtype=vol.dtype)).squeeze()
            pt_direct = vol.evaluate(np.array([[u, v, w]], dtype=vol.dtype)).squeeze()
            np.testing.assert_allclose(pt_collapse, pt_direct, atol=1e-13)

    def test_collapse_axis2(self) -> None:
        """Collapsing along axis 2 matches direct evaluation."""
        vol = _make_3d_volume()
        u, v = 0.2, 0.6
        crv = vol.collapse_along_axis(2, [u, v])
        assert crv.dim == 1
        assert crv.degree == (vol.degree[2],)
        for w in [0.0, 0.5, 1.0]:
            pt_collapse = crv.evaluate(np.array([w], dtype=vol.dtype)).squeeze()
            pt_direct = vol.evaluate(np.array([[u, v, w]], dtype=vol.dtype)).squeeze()
            np.testing.assert_allclose(pt_collapse, pt_direct, atol=1e-13)


class TestCollapse3DHigherDegree:
    """Test collapse of a 3D Bézier with non-uniform degrees."""

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_collapse_preserves_degree(self, axis: int) -> None:
        """Collapsed curve has the correct degree."""
        vol = _make_3d_higher_degree()
        values = [0.3, 0.7]
        crv = vol.collapse_along_axis(axis, values)
        assert crv.degree == (vol.degree[axis],)

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_collapse_matches_evaluate(self, axis: int) -> None:
        """Collapsed curve evaluation matches full volume evaluation."""
        vol = _make_3d_higher_degree()
        # Fixed values for non-axis directions
        fixed = [0.3, 0.7]
        crv = vol.collapse_along_axis(axis, fixed)
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            pt_collapse = crv.evaluate(np.array([t], dtype=vol.dtype)).squeeze()
            # Build full point
            full_pt = np.empty(3, dtype=vol.dtype)
            val_idx = 0
            for d in range(3):
                if d == axis:
                    full_pt[d] = t
                else:
                    full_pt[d] = fixed[val_idx]
                    val_idx += 1
            pt_direct = vol.evaluate(full_pt.reshape(1, 3)).squeeze()
            np.testing.assert_allclose(pt_collapse, pt_direct, atol=1e-12)


# ---------------------------------------------------------------------------
# Tests: rational Béziers
# ---------------------------------------------------------------------------


class TestCollapseRational:
    """Test collapse on rational Béziers."""

    def test_rational_1d_raises(self) -> None:
        """Collapsing a rational 1D curve raises ValueError."""
        crv = _make_rational_curve()
        with pytest.raises(ValueError, match="dim >= 2"):
            crv.collapse_along_axis(0, [])

    def test_rational_2d_preserves_rationality(self) -> None:
        """Collapsing a rational 2D surface preserves is_rational."""
        srf = _make_rational_surface()
        crv = srf.collapse_along_axis(0, [0.5])
        assert crv.is_rational

    def test_rational_2d_evaluate_matches(self) -> None:
        """Collapsed rational curve evaluation matches surface evaluation."""
        srf = _make_rational_surface()
        v = 0.4
        crv = srf.collapse_along_axis(0, [v])
        for u in [0.0, 0.25, 0.5, 0.75, 1.0]:
            pt_collapse = crv.evaluate(np.array([u], dtype=srf.dtype)).squeeze()
            pt_direct = srf.evaluate(np.array([[u, v]], dtype=srf.dtype)).squeeze()
            np.testing.assert_allclose(pt_collapse, pt_direct, atol=1e-12)


# ---------------------------------------------------------------------------
# Tests: dtype preservation
# ---------------------------------------------------------------------------


class TestCollapseDtype:
    """Test that collapse preserves dtype."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_preserves_dtype(self, dtype: type) -> None:
        """Output dtype matches input dtype."""
        srf = _make_2d_surface(dtype=dtype)
        crv = srf.collapse_along_axis(0, [0.5])
        assert crv.dtype == dtype

    def test_float32_values_cast_and_correct(self) -> None:
        """float64 values passed to a float32 Bézier produce correct float32 output."""
        srf = _make_2d_surface(dtype=np.float32)
        v = 0.3  # float64 literal
        crv = srf.collapse_along_axis(0, [v])
        assert crv.dtype == np.float32
        # Evaluation should still match the surface within float32 tolerance.
        for u in [0.0, 0.25, 0.5, 0.75, 1.0]:
            pt_collapse = crv.evaluate(np.array([u], dtype=np.float32)).squeeze()
            pt_direct = srf.evaluate(np.array([[u, v]], dtype=np.float32)).squeeze()
            np.testing.assert_allclose(pt_collapse, pt_direct, atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


class TestCollapseErrors:
    """Test error handling for collapse_along_axis."""

    def test_axis_out_of_range(self) -> None:
        """Axis out of range raises ValueError."""
        srf = _make_2d_surface()
        with pytest.raises(ValueError, match="axis must be in"):
            srf.collapse_along_axis(2, [0.5])

    def test_axis_negative(self) -> None:
        """Negative axis raises ValueError."""
        srf = _make_2d_surface()
        with pytest.raises(ValueError, match="axis must be in"):
            srf.collapse_along_axis(-1, [0.5])

    def test_wrong_values_length(self) -> None:
        """Wrong number of values raises ValueError."""
        srf = _make_2d_surface()
        with pytest.raises(ValueError, match="values must have length"):
            srf.collapse_along_axis(0, [0.3, 0.5])

    def test_value_below_zero(self) -> None:
        """Value below 0 raises ValueError."""
        srf = _make_2d_surface()
        with pytest.raises(ValueError, match="All values must be in"):
            srf.collapse_along_axis(0, [-0.1])

    def test_value_above_one(self) -> None:
        """Value above 1 raises ValueError."""
        srf = _make_2d_surface()
        with pytest.raises(ValueError, match="All values must be in"):
            srf.collapse_along_axis(0, [1.5])
