"""Tests for Bezier.split() and the refactored Bezier.restrict()."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bezier import Bezier, create_from_bspline

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
# Tests: Bezier.split()
# ---------------------------------------------------------------------------


class TestSplit1D:
    """Test splitting a 1D Bézier curve."""

    def test_split_returns_two_beziers(self) -> None:
        """Split returns a tuple of two Béziers with same degree."""
        crv = _make_1d_curve()
        left, right = crv.split(0, 0.5)
        assert isinstance(left, Bezier)
        assert isinstance(right, Bezier)
        assert left.degree == crv.degree
        assert right.degree == crv.degree

    def test_split_continuity(self) -> None:
        """Left(1) == right(0) at the split point."""
        crv = _make_1d_curve()
        for u in [0.1, 0.3, 0.5, 0.7, 0.9]:
            left, right = crv.split(0, u)
            pt_left = left.evaluate(np.array([1.0])).squeeze()
            pt_right = right.evaluate(np.array([0.0])).squeeze()
            np.testing.assert_allclose(pt_left, pt_right, atol=1e-14)

    def test_split_matches_original(self) -> None:
        """Evaluations on the halves match the original curve."""
        crv = _make_1d_curve()
        u_split = 0.4
        left, right = crv.split(0, u_split)

        # Left half: [0, u_split] mapped to [0, 1].
        pts_orig_left = np.linspace(0, u_split, 20)
        pts_mapped_left = pts_orig_left / u_split
        vals_orig = crv.evaluate(pts_orig_left)
        vals_left = left.evaluate(pts_mapped_left)
        np.testing.assert_allclose(vals_left, vals_orig, atol=1e-13)

        # Right half: [u_split, 1] mapped to [0, 1].
        pts_orig_right = np.linspace(u_split, 1.0, 20)
        pts_mapped_right = (pts_orig_right - u_split) / (1.0 - u_split)
        vals_orig = crv.evaluate(pts_orig_right)
        vals_right = right.evaluate(pts_mapped_right)
        np.testing.assert_allclose(vals_right, vals_orig, atol=1e-13)

    def test_split_endpoints(self) -> None:
        """Left starts at original start, right ends at original end."""
        crv = _make_1d_curve()
        left, right = crv.split(0, 0.6)
        np.testing.assert_allclose(
            left.evaluate(np.array([0.0])).squeeze(),
            crv.evaluate(np.array([0.0])).squeeze(),
            atol=1e-14,
        )
        np.testing.assert_allclose(
            right.evaluate(np.array([1.0])).squeeze(),
            crv.evaluate(np.array([1.0])).squeeze(),
            atol=1e-14,
        )

    def test_split_degree_0(self) -> None:
        """Splitting a degree-0 curve produces identical constants."""
        crv = Bezier(np.array([[3.0, 4.0]]))
        left, right = crv.split(0, 0.5)
        np.testing.assert_allclose(left.control_points, crv.control_points)
        np.testing.assert_allclose(right.control_points, crv.control_points)


class TestSplit2D:
    """Test splitting a 2D Bézier surface."""

    def test_split_preserves_degree_and_dim(self) -> None:
        """Split preserves parametric dimension and degrees."""
        srf = _make_2d_surface()
        left, right = srf.split(0, 0.5)
        assert left.dim == srf.dim
        assert right.dim == srf.dim
        assert left.degree == srf.degree
        assert right.degree == srf.degree

    def test_split_dir0_matches_original(self) -> None:
        """Splitting along direction 0 matches original evaluations."""
        srf = _make_2d_surface()
        u_split = 0.3
        left, right = srf.split(0, u_split)

        # Test points along the split boundary.
        n = 15
        v_vals = np.linspace(0, 1, n)

        # Original at u=u_split.
        pts_orig = np.column_stack([np.full(n, u_split), v_vals])
        vals_orig = srf.evaluate(pts_orig)

        # Left at u=1.
        pts_left = np.column_stack([np.ones(n), v_vals])
        vals_left = left.evaluate(pts_left)
        np.testing.assert_allclose(vals_left, vals_orig, atol=1e-13)

        # Right at u=0.
        pts_right = np.column_stack([np.zeros(n), v_vals])
        vals_right = right.evaluate(pts_right)
        np.testing.assert_allclose(vals_right, vals_orig, atol=1e-13)

    def test_split_dir1_matches_original(self) -> None:
        """Splitting along direction 1 matches original evaluations."""
        srf = _make_2d_surface()
        v_split = 0.6
        left, right = srf.split(1, v_split)

        n = 15
        u_vals = np.linspace(0, 1, n)

        # Original at v=v_split.
        pts_orig = np.column_stack([u_vals, np.full(n, v_split)])
        vals_orig = srf.evaluate(pts_orig)

        # Left at v=1.
        pts_left = np.column_stack([u_vals, np.ones(n)])
        vals_left = left.evaluate(pts_left)
        np.testing.assert_allclose(vals_left, vals_orig, atol=1e-13)

        # Right at v=0.
        pts_right = np.column_stack([u_vals, np.zeros(n)])
        vals_right = right.evaluate(pts_right)
        np.testing.assert_allclose(vals_right, vals_orig, atol=1e-13)


class TestSplit3D:
    """Test splitting a 3D Bézier volume."""

    def test_split_preserves_degree_and_dim(self) -> None:
        """Split preserves parametric dimension and degrees."""
        vol = _make_3d_volume()
        left, right = vol.split(2, 0.5)
        assert left.dim == vol.dim
        assert right.dim == vol.dim
        assert left.degree == vol.degree
        assert right.degree == vol.degree

    def test_split_volume_continuity(self) -> None:
        """Split of a 3D volume preserves continuity at the split plane."""
        vol = _make_3d_volume()
        u_split = 0.4
        left, _right = vol.split(0, u_split)

        n = 10
        rng = np.random.default_rng(77)
        vw = rng.uniform(0, 1, (n, 2))

        # Original at u=u_split.
        pts_orig = np.column_stack([np.full(n, u_split), vw])
        vals_orig = vol.evaluate(pts_orig)

        # Left at u=1.
        pts_left = np.column_stack([np.ones(n), vw])
        vals_left = left.evaluate(pts_left)
        np.testing.assert_allclose(vals_left, vals_orig, atol=1e-13)


class TestSplitRational:
    """Test splitting rational Bézier curves and surfaces."""

    def test_split_rational_curve(self) -> None:
        """Splitting a rational curve preserves rational flag and continuity."""
        crv = _make_rational_curve()
        left, right = crv.split(0, 0.5)
        assert left.is_rational
        assert right.is_rational
        assert left.rank == crv.rank
        assert right.rank == crv.rank

        # Continuity at split point.
        pt_left = left.evaluate(np.array([1.0])).squeeze()
        pt_right = right.evaluate(np.array([0.0])).squeeze()
        np.testing.assert_allclose(pt_left, pt_right, atol=1e-14)

    def test_split_rational_matches_original(self) -> None:
        """Split of a rational curve matches original evaluations."""
        crv = _make_rational_curve()
        u_split = 0.3
        left, _right = crv.split(0, u_split)

        pts_orig = np.linspace(0, u_split, 15)
        pts_mapped = pts_orig / u_split
        vals_orig = crv.evaluate(pts_orig)
        vals_left = left.evaluate(pts_mapped)
        np.testing.assert_allclose(vals_left, vals_orig, atol=1e-13)

    def test_split_rational_surface(self) -> None:
        """Splitting a rational surface preserves rational flag."""
        srf = _make_rational_surface()
        left, right = srf.split(0, 0.5)
        assert left.is_rational
        assert right.is_rational


# ---------------------------------------------------------------------------
# Tests: Bezier.split() error cases
# ---------------------------------------------------------------------------


class TestSplitErrors:
    """Test that split raises errors for invalid inputs."""

    def test_direction_too_large(self) -> None:
        """Direction >= dim raises ValueError."""
        crv = _make_1d_curve()
        with pytest.raises(ValueError, match="direction"):
            crv.split(1, 0.5)

    def test_direction_negative(self) -> None:
        """Negative direction raises ValueError."""
        crv = _make_1d_curve()
        with pytest.raises(ValueError, match="direction"):
            crv.split(-1, 0.5)

    def test_value_at_zero(self) -> None:
        """Value at 0 raises ValueError (must be strictly inside)."""
        crv = _make_1d_curve()
        with pytest.raises(ValueError, match="strictly inside"):
            crv.split(0, 0.0)

    def test_value_at_one(self) -> None:
        """Value at 1 raises ValueError (must be strictly inside)."""
        crv = _make_1d_curve()
        with pytest.raises(ValueError, match="strictly inside"):
            crv.split(0, 1.0)


# ---------------------------------------------------------------------------
# Tests: Bezier.restrict() (refactored)
# ---------------------------------------------------------------------------


class TestRestrictRefactored:
    """Test the refactored restrict (de Casteljau, no Bspline round-trip)."""

    def test_restrict_1d_matches_bspline_roundtrip(self) -> None:
        """New restrict matches the old Bspline-based restrict."""
        crv = _make_1d_curve()
        restricted = crv.restrict((0.2, 0.7))

        # Reference via Bspline round-trip.
        bspl = crv.to_bspline()
        ref = create_from_bspline(bspl.restrict((0.2, 0.7)))
        np.testing.assert_allclose(restricted.control_points, ref.control_points, atol=1e-13)

    def test_restrict_2d_matches_bspline_roundtrip(self) -> None:
        """New restrict on 2D surface matches old Bspline-based restrict."""
        srf = _make_2d_surface()
        restricted = srf.restrict([(0.1, 0.8), (0.2, 0.9)])

        bspl = srf.to_bspline()
        ref = create_from_bspline(bspl.restrict([(0.1, 0.8), (0.2, 0.9)]))
        np.testing.assert_allclose(restricted.control_points, ref.control_points, atol=1e-12)

    def test_restrict_partial_direction(self) -> None:
        """Restrict with None in one direction only restricts the other."""
        srf = _make_2d_surface()
        restricted = srf.restrict([(0.3, 0.7), None])

        bspl = srf.to_bspline()
        ref = create_from_bspline(bspl.restrict([(0.3, 0.7), None]))
        np.testing.assert_allclose(restricted.control_points, ref.control_points, atol=1e-13)

    def test_restrict_evaluations_match(self) -> None:
        """Restricted Bézier evaluations match the original on the sub-domain."""
        crv = _make_1d_curve()
        a, b = 0.25, 0.75
        restricted = crv.restrict((a, b))

        # Evaluate original on [a, b].
        pts_orig = np.linspace(a, b, 30)
        vals_orig = crv.evaluate(pts_orig)

        # Evaluate restricted on [0, 1] (reparametrized).
        pts_new = (pts_orig - a) / (b - a)
        vals_new = restricted.evaluate(pts_new)

        np.testing.assert_allclose(vals_new, vals_orig, atol=1e-13)

    def test_restrict_rational(self) -> None:
        """Restricting a rational Bézier preserves rational flag."""
        crv = _make_rational_curve()
        restricted = crv.restrict((0.2, 0.8))
        assert restricted.is_rational

        # Check evaluations.
        a, b = 0.2, 0.8
        pts_orig = np.linspace(a, b, 15)
        vals_orig = crv.evaluate(pts_orig)
        pts_new = (pts_orig - a) / (b - a)
        vals_new = restricted.evaluate(pts_new)
        np.testing.assert_allclose(vals_new, vals_orig, atol=1e-13)

    def test_restrict_full_domain_raises(self) -> None:
        """Restricting to the full domain raises ValueError."""
        crv = _make_1d_curve()
        with pytest.raises(ValueError, match="full domain"):
            crv.restrict((0.0, 1.0))

    def test_restrict_invalid_bounds(self) -> None:
        """Invalid bounds (lower >= upper) raises ValueError."""
        crv = _make_1d_curve()
        with pytest.raises(ValueError, match="strictly less"):
            crv.restrict((0.5, 0.3))

    def test_restrict_out_of_domain(self) -> None:
        """Bounds outside [0, 1] raises ValueError."""
        crv = _make_1d_curve()
        with pytest.raises(ValueError, match="within"):
            crv.restrict((-0.1, 0.5))
