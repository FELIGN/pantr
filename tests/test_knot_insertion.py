"""Tests for B-spline knot insertion (insert_knots and subdivide)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pantr._bspline_space_factory import create_uniform_periodic_knot_vector
from pantr.bspline import Bspline
from pantr.bspline_space_1D import BsplineSpace1D
from pantr.bspline_space_nd import BsplineSpace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_1d_bspline(
    knots: list[float],
    degree: int,
    ctrl: list[list[float]],
    is_rational: bool = False,
) -> Bspline:
    """Create a 1D Bspline from lists."""
    space_1d = BsplineSpace1D(knots, degree)
    space = BsplineSpace([space_1d])
    return Bspline(space, np.array(ctrl, dtype=np.float64), is_rational=is_rational)


def _eval_pts_1d(
    bspline: Bspline, n: int = 50
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float32 | np.float64]]:
    """Return (pts, values) on a dense grid for a 1D Bspline."""
    lo, hi = bspline.space.spaces[0].domain
    pts = np.linspace(float(lo), float(hi), n, dtype=np.float64)
    vals = bspline.evaluate(pts)
    return pts, vals


def _eval_pts_2d(
    bspline: Bspline, nu: int = 15, nv: int = 15
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float32 | np.float64]]:
    """Return (pts, values) on a grid for a 2D Bspline."""
    lo_u, hi_u = bspline.space.spaces[0].domain
    lo_v, hi_v = bspline.space.spaces[1].domain
    us = np.linspace(float(lo_u), float(hi_u), nu, dtype=np.float64)
    vs = np.linspace(float(lo_v), float(hi_v), nv, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs, indexing="ij")
    pts = np.column_stack([uu.ravel(), vv.ravel()])
    vals = bspline.evaluate(pts)
    return pts, vals


# ---------------------------------------------------------------------------
# BsplineSpace1D.insert_knots
# ---------------------------------------------------------------------------


class TestBsplineSpace1DInsertKnots:
    """Test BsplineSpace1D.insert_knots."""

    def test_insert_single_knot_updates_knot_vector(self) -> None:
        """Inserting one knot adds it to the knot vector."""
        space = BsplineSpace1D([0, 0, 0, 1, 2, 2, 2], 2)
        new_space = space.insert_knots([0.5])
        assert np.any(np.isclose(new_space.knots, 0.5))

    def test_insert_single_knot_increases_num_basis(self) -> None:
        """Inserting one new knot adds one basis function."""
        space = BsplineSpace1D([0, 0, 0, 1, 2, 2, 2], 2)
        new_space = space.insert_knots([0.5])
        assert new_space.num_basis == space.num_basis + 1

    def test_insert_multiple_knots(self) -> None:
        """Inserting k knots increases num_basis by k."""
        space = BsplineSpace1D([0, 0, 0, 1, 1, 1], 2)
        new_space = space.insert_knots([0.25, 0.5, 0.75])
        assert new_space.num_basis == space.num_basis + 3

    def test_insert_existing_knot_raises_multiplicity(self) -> None:
        """Inserting a knot that would exceed degree+1 multiplicity raises ValueError."""
        # [0,0,0,1,1,1]: degree 2, knot 0 already has multiplicity 3 = degree+1
        space = BsplineSpace1D([0, 0, 0, 1, 1, 1], 2)
        with pytest.raises(ValueError, match="multiplicity"):
            space.insert_knots([0.0])

    def test_insert_knot_at_multiplicity_limit(self) -> None:
        """Inserting a knot that reaches (but does not exceed) degree+1 is allowed."""
        # [0,0,0,0.5,1,1,1]: interior knot 0.5 has multiplicity 1; inserting once more → 2
        space = BsplineSpace1D([0, 0, 0, 0.5, 1, 1, 1], 2)
        new_space = space.insert_knots([0.5])
        assert np.sum(np.isclose(new_space.knots, 0.5)) == 2  # noqa: PLR2004
        # Inserting 0.5 once more reaches the maximum multiplicity degree+1=3.
        new_space2 = new_space.insert_knots([0.5])
        assert np.sum(np.isclose(new_space2.knots, 0.5)) == 3  # noqa: PLR2004

    def test_insert_repeated_knots_in_one_call(self) -> None:
        """Inserting [0.5, 0.5] in one call raises multiplicity from 1 to degree+1=3."""
        # degree=2, so max multiplicity = 3; starting from 1, insert [0.5, 0.5] → 3
        space = BsplineSpace1D([0, 0, 0, 0.5, 1, 1, 1], 2)
        new_space = space.insert_knots([0.5, 0.5])
        assert np.sum(np.isclose(new_space.knots, 0.5)) == 3  # noqa: PLR2004

    def test_insert_out_of_domain_raises(self) -> None:
        """Values outside the domain raise ValueError."""
        space = BsplineSpace1D([0, 0, 0, 1, 1, 1], 2)
        with pytest.raises(ValueError, match="domain"):
            space.insert_knots([-0.5])
        with pytest.raises(ValueError, match="domain"):
            space.insert_knots([1.5])

    def test_insert_empty_raises(self) -> None:
        """Inserting an empty array raises ValueError."""
        space = BsplineSpace1D([0, 0, 0, 1, 1, 1], 2)
        with pytest.raises(ValueError, match="empty"):
            space.insert_knots([])

    def test_degree_preserved(self) -> None:
        """Degree is preserved after knot insertion."""
        space = BsplineSpace1D([0, 0, 0, 1, 2, 2, 2], 2)
        new_space = space.insert_knots([0.7])
        assert new_space.degree == space.degree


# ---------------------------------------------------------------------------
# BsplineSpace1D.subdivide
# ---------------------------------------------------------------------------


class TestBsplineSpace1DSubdivide:
    """Test BsplineSpace1D.subdivide."""

    def test_subdivide_n_less_than_2_raises(self) -> None:
        """Subdivide with n < 2 (including 1, 0, and negative values) raises ValueError."""
        space = BsplineSpace1D([0, 0, 0, 1, 1, 1], 2)
        for bad in (1, 0, -1, -5):
            with pytest.raises(ValueError, match="n_subdivisions"):
                space.subdivide(bad)

    def test_subdivide_2_single_span(self) -> None:
        """subdivide(2) on a single-span knot vector inserts the midpoint."""
        space = BsplineSpace1D([0, 0, 0, 1, 1, 1], 2)
        new_space = space.subdivide(2)
        assert np.any(np.isclose(new_space.knots, 0.5))

    def test_subdivide_3_single_span(self) -> None:
        """subdivide(3) on a single-span knot vector inserts two equidistant points."""
        space = BsplineSpace1D([0, 0, 0, 1, 1, 1], 2)
        new_space = space.subdivide(3)
        knots = new_space.knots
        assert any(np.isclose(knots, 1 / 3))
        assert any(np.isclose(knots, 2 / 3))

    def test_subdivide_2_multi_span(self) -> None:
        """subdivide(2) on a multi-span knot vector inserts midpoints in each span."""
        space = BsplineSpace1D([0, 0, 0, 1, 2, 2, 2], 2)
        new_space = space.subdivide(2)
        # Unique domain knots were [0, 1, 2]; midpoints 0.5 and 1.5 should appear.
        assert any(np.isclose(new_space.knots, 0.5))
        assert any(np.isclose(new_space.knots, 1.5))

    def test_subdivide_regularity_default_gives_multiplicity_1(self) -> None:
        """Default regularity=degree-1 inserts each knot once (multiplicity 1)."""
        space = BsplineSpace1D([0, 0, 0, 1, 1, 1], 2)
        new_space = space.subdivide(2)  # regularity defaults to degree-1=1
        assert np.sum(np.isclose(new_space.knots, 0.5)) == 1

    def test_subdivide_regularity_0_gives_multiplicity_degree(self) -> None:
        """regularity=0 inserts each knot degree times (C^0 continuity)."""
        # degree=2, regularity=0 → repeat=2
        space = BsplineSpace1D([0, 0, 0, 1, 1, 1], 2)
        new_space = space.subdivide(2, regularity=0)
        assert np.sum(np.isclose(new_space.knots, 0.5)) == 2  # noqa: PLR2004

    def test_subdivide_regularity_minus1_gives_multiplicity_degree_plus1(self) -> None:
        """regularity=-1 inserts each knot degree+1 times (discontinuous, C^{-1})."""
        # degree=2, regularity=-1 → repeat=3
        space = BsplineSpace1D([0, 0, 0, 1, 1, 1], 2)
        new_space = space.subdivide(2, regularity=-1)
        assert np.sum(np.isclose(new_space.knots, 0.5)) == 3  # noqa: PLR2004

    def test_subdivide_regularity_out_of_range_raises(self) -> None:
        """Regularity outside [-1, degree-1] raises ValueError."""
        space = BsplineSpace1D([0, 0, 0, 1, 1, 1], 2)
        with pytest.raises(ValueError, match="regularity"):
            space.subdivide(2, regularity=2)  # degree-1=1, so 2 is out of range
        with pytest.raises(ValueError, match="regularity"):
            space.subdivide(2, regularity=-2)


# ---------------------------------------------------------------------------
# Bspline.insert_knots — 1D non-rational
# ---------------------------------------------------------------------------


class TestBsplineInsertKnots1DNonRational:
    """Test Bspline.insert_knots for 1D non-rational B-splines."""

    def test_1d_shorthand_flat_array(self) -> None:
        """A flat 1D array is accepted for a 1D Bspline."""
        bspline = _make_1d_bspline([0, 0, 0, 1, 1, 1], 2, [[0, 0], [0.5, 1], [1, 0]])
        new_bs = bspline.insert_knots(np.array([0.5]))
        assert new_bs.space.spaces[0].num_basis == bspline.space.spaces[0].num_basis + 1

    def test_1d_geometry_preserved_after_insertion(self) -> None:
        """Geometry (evaluated values) is preserved after knot insertion."""
        bspline = _make_1d_bspline(
            [0, 0, 0, 1, 2, 2, 2],
            2,
            [[0.0, 0.0], [1.0, 1.5], [2.0, 0.5], [3.0, 1.0]],
        )
        new_bs = bspline.insert_knots([0.5, 1.5])

        pts = np.linspace(0.0, 2.0, 60, dtype=np.float64)
        old_vals = bspline.evaluate(pts)
        new_vals = new_bs.evaluate(pts)
        np.testing.assert_allclose(new_vals, old_vals, atol=1e-12)

    def test_1d_list_of_knots(self) -> None:
        """A plain Python list of knots is accepted."""
        bspline = _make_1d_bspline([0, 0, 0, 1, 1, 1], 2, [[0, 0], [0.5, 1], [1, 0]])
        new_bs = bspline.insert_knots([0.25, 0.75])
        assert new_bs.space.spaces[0].num_basis == bspline.space.spaces[0].num_basis + 2

    def test_1d_empty_insertion_raises(self) -> None:
        """Inserting an empty array in 1D raises ValueError."""
        bspline = _make_1d_bspline([0, 0, 0, 1, 1, 1], 2, [[0, 0], [0.5, 1], [1, 0]])
        with pytest.raises(ValueError):
            bspline.insert_knots([])

    def test_1d_out_of_domain_raises(self) -> None:
        """Knots outside the domain raise ValueError."""
        bspline = _make_1d_bspline([0, 0, 0, 1, 1, 1], 2, [[0, 0], [0.5, 1], [1, 0]])
        with pytest.raises(ValueError, match="domain"):
            bspline.insert_knots([1.5])

    def test_1d_multiplicity_exceeded_raises(self) -> None:
        """Exceeding maximum multiplicity raises ValueError."""
        bspline = _make_1d_bspline([0, 0, 0, 1, 1, 1], 2, [[0, 0], [0.5, 1], [1, 0]])
        with pytest.raises(ValueError, match="multiplicity"):
            bspline.insert_knots([0.0])

    def test_1d_knot_already_present_increases_multiplicity(self) -> None:
        """Inserting a knot that already exists increases its multiplicity."""
        bspline = _make_1d_bspline(
            [0, 0, 0, 0.5, 1, 1, 1],
            2,
            [[0, 0], [0.25, 1], [0.75, 1], [1, 0]],
        )
        new_bs = bspline.insert_knots([0.5])
        knots = new_bs.space.spaces[0].knots
        assert np.sum(np.isclose(knots, 0.5)) == 2  # noqa: PLR2004

    def test_1d_repeated_knots_in_one_call_reaches_max_multiplicity(self) -> None:
        """Inserting [0.5, 0.5] in one call raises multiplicity from 1 to degree+1=3."""
        bspline = _make_1d_bspline(
            [0, 0, 0, 0.5, 1, 1, 1],
            2,
            [[0, 0], [0.25, 1], [0.75, 1], [1, 0]],
        )
        new_bs = bspline.insert_knots([0.5, 0.5])
        knots = new_bs.space.spaces[0].knots
        assert np.sum(np.isclose(knots, 0.5)) == 3  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Bspline.insert_knots — 1D rational (NURBS)
# ---------------------------------------------------------------------------


class TestBsplineInsertKnots1DRational:
    """Test Bspline.insert_knots for 1D rational (NURBS) B-splines."""

    def test_nurbs_geometry_preserved(self) -> None:
        """Geometry of a rational B-spline is preserved after knot insertion.

        Uses a standard quarter-circle NURBS representation (degree 2):
        control points in homogeneous form [wx, wy, w].
        """
        w = np.sqrt(2.0) / 2.0
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        ctrl = np.array([[1.0, 0.0, 1.0], [w, w, w], [0.0, 1.0, 1.0]], dtype=np.float64)
        bspline = _make_1d_bspline(knots, 2, ctrl.tolist(), is_rational=True)

        new_bs = bspline.insert_knots([0.5])

        pts = np.linspace(0.0, 1.0, 40, dtype=np.float64)
        old_vals = bspline.evaluate(pts)
        new_vals = new_bs.evaluate(pts)
        np.testing.assert_allclose(new_vals, old_vals, atol=1e-12)


# ---------------------------------------------------------------------------
# Bspline.insert_knots — multi-dimensional
# ---------------------------------------------------------------------------


class TestBsplineInsertKnotsMultiDim:
    """Test Bspline.insert_knots for multi-dimensional B-splines."""

    def _make_bilinear_surface(self) -> Bspline:
        """Create a simple bilinear tensor-product surface."""
        space_u = BsplineSpace1D([0, 0, 1, 1], 1)  # 2 basis functions
        space_v = BsplineSpace1D([0, 0, 1, 1], 1)  # 2 basis functions
        space = BsplineSpace([space_u, space_v])
        # 2x2 grid of 2D control points → shape (2, 2, 2)
        ctrl = np.array([[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]], dtype=np.float64)
        return Bspline(space, ctrl, is_rational=False)

    def _make_biquadratic_surface(self) -> Bspline:
        """Create a simple biquadratic tensor-product B-spline surface."""
        space_u = BsplineSpace1D([0, 0, 0, 1, 2, 2, 2], 2)  # 4 basis
        space_v = BsplineSpace1D([0, 0, 0, 1, 1, 1], 2)  # 3 basis
        space = BsplineSpace([space_u, space_v])
        rng = np.random.default_rng(42)
        ctrl = rng.standard_normal((4, 3, 2)).astype(np.float64)
        return Bspline(space, ctrl, is_rational=False)

    def test_2d_insert_one_direction_only(self) -> None:
        """Inserting in one direction (None for other) preserves geometry."""
        bspline = self._make_biquadratic_surface()

        new_bs = bspline.insert_knots([np.array([0.5, 1.5]), None])

        _, old_vals = _eval_pts_2d(bspline)
        _, new_vals = _eval_pts_2d(new_bs)
        np.testing.assert_allclose(new_vals, old_vals, atol=1e-12)

    def test_2d_insert_both_directions(self) -> None:
        """Inserting in both directions preserves geometry."""
        bspline = self._make_biquadratic_surface()

        new_bs = bspline.insert_knots([np.array([0.5, 1.5]), np.array([0.5])])

        _, old_vals = _eval_pts_2d(bspline)
        _, new_vals = _eval_pts_2d(new_bs)
        np.testing.assert_allclose(new_vals, old_vals, atol=1e-12)

    def test_2d_wrong_sequence_length_raises(self) -> None:
        """Sequence length != dim raises ValueError."""
        bspline = self._make_bilinear_surface()
        with pytest.raises(ValueError, match="dim"):
            bspline.insert_knots([np.array([0.5])])  # need length 2

    def test_2d_all_none_raises(self) -> None:
        """A sequence of all-None raises ValueError."""
        bspline = self._make_bilinear_surface()
        with pytest.raises(ValueError):
            bspline.insert_knots([None, None])

    def test_2d_all_empty_raises(self) -> None:
        """A sequence of all-empty arrays raises ValueError."""
        bspline = self._make_bilinear_surface()
        with pytest.raises(ValueError):
            bspline.insert_knots([[], []])


# ---------------------------------------------------------------------------
# Bspline.subdivide — 1D
# ---------------------------------------------------------------------------


class TestBsplineSubdivide1D:
    """Test Bspline.subdivide for 1D B-splines."""

    def test_subdivide_n1_raises(self) -> None:
        """subdivide(1) raises ValueError."""
        bspline = _make_1d_bspline([0, 0, 0, 1, 1, 1], 2, [[0, 0], [0.5, 1], [1, 0]])
        with pytest.raises(ValueError):
            bspline.subdivide(1)

    def test_subdivide_n_less_than_1_raises(self) -> None:
        """Subdivide with n < 1 raises ValueError."""
        bspline = _make_1d_bspline([0, 0, 0, 1, 1, 1], 2, [[0, 0], [0.5, 1], [1, 0]])
        with pytest.raises(ValueError, match="n_subdivisions"):
            bspline.subdivide(0)

    def test_subdivide_2_geometry_preserved(self) -> None:
        """subdivide(2) inserts midpoints and preserves geometry."""
        bspline = _make_1d_bspline(
            [0, 0, 0, 1, 2, 2, 2],
            2,
            [[0.0, 0.0], [1.0, 1.5], [2.0, 0.5], [3.0, 1.0]],
        )
        new_bs = bspline.subdivide(2)

        knots = new_bs.space.spaces[0].knots
        assert any(np.isclose(knots, 0.5))
        assert any(np.isclose(knots, 1.5))

        pts = np.linspace(0.0, 2.0, 60, dtype=np.float64)
        old_vals = bspline.evaluate(pts)
        new_vals = new_bs.evaluate(pts)
        np.testing.assert_allclose(new_vals, old_vals, atol=1e-12)

    def test_subdivide_3_geometry_preserved(self) -> None:
        """subdivide(3) splits each span into 3 and preserves geometry."""
        bspline = _make_1d_bspline([0, 0, 0, 1, 1, 1], 2, [[0, 0], [0.5, 1], [1, 0]])
        new_bs = bspline.subdivide(3)

        pts = np.linspace(0.0, 1.0, 60, dtype=np.float64)
        old_vals = bspline.evaluate(pts)
        new_vals = new_bs.evaluate(pts)
        np.testing.assert_allclose(new_vals, old_vals, atol=1e-12)

    def test_subdivide_regularity_0_gives_multiplicity_degree(self) -> None:
        """regularity=0 inserts each new knot degree times (C^0 continuity)."""
        # degree=2, regularity=0 → repeat=2; midpoint 0.5 inserted twice
        bspline = _make_1d_bspline([0, 0, 0, 1, 1, 1], 2, [[0, 0], [0.5, 1], [1, 0]])
        new_bs = bspline.subdivide(2, regularity=0)
        knots = new_bs.space.spaces[0].knots
        assert np.sum(np.isclose(knots, 0.5)) == 2  # noqa: PLR2004

    def test_subdivide_regularity_minus1_gives_discontinuous(self) -> None:
        """regularity=-1 inserts each new knot degree+1 times (C^{-1}, discontinuous)."""
        # degree=2, regularity=-1 → repeat=3; midpoint 0.5 inserted three times
        bspline = _make_1d_bspline([0, 0, 0, 1, 1, 1], 2, [[0, 0], [0.5, 1], [1, 0]])
        new_bs = bspline.subdivide(2, regularity=-1)
        knots = new_bs.space.spaces[0].knots
        assert np.sum(np.isclose(knots, 0.5)) == 3  # noqa: PLR2004

    def test_subdivide_regularity_out_of_range_raises(self) -> None:
        """Regularity outside valid range raises ValueError."""
        bspline = _make_1d_bspline([0, 0, 0, 1, 1, 1], 2, [[0, 0], [0.5, 1], [1, 0]])
        with pytest.raises(ValueError, match="regularity"):
            bspline.subdivide(2, regularity=2)
        with pytest.raises(ValueError, match="regularity"):
            bspline.subdivide(2, regularity=-2)


# ---------------------------------------------------------------------------
# Bspline.subdivide — multi-dimensional
# ---------------------------------------------------------------------------


class TestBsplineSubdivideMultiDim:
    """Test Bspline.subdivide for multi-dimensional B-splines."""

    def _make_biquadratic_surface(self) -> Bspline:
        """Create a simple biquadratic tensor-product B-spline surface."""
        space_u = BsplineSpace1D([0, 0, 0, 1, 2, 2, 2], 2)
        space_v = BsplineSpace1D([0, 0, 0, 1, 1, 1], 2)
        space = BsplineSpace([space_u, space_v])
        rng = np.random.default_rng(7)
        ctrl = rng.standard_normal((4, 3, 2)).astype(np.float64)
        return Bspline(space, ctrl, is_rational=False)

    def test_subdivide_int_broadcasts_to_all_directions(self) -> None:
        """A single int applies to all directions and preserves geometry."""
        bspline = self._make_biquadratic_surface()
        new_bs = bspline.subdivide(2)

        _, old_vals = _eval_pts_2d(bspline)
        _, new_vals = _eval_pts_2d(new_bs)
        np.testing.assert_allclose(new_vals, old_vals, atol=1e-12)

    def test_subdivide_per_direction_with_none(self) -> None:
        """Per-direction sequence with None skips that direction."""
        bspline = self._make_biquadratic_surface()
        new_bs = bspline.subdivide([2, None])

        # Only u direction was refined.
        assert new_bs.space.spaces[0].num_basis > bspline.space.spaces[0].num_basis
        assert new_bs.space.spaces[1].num_basis == bspline.space.spaces[1].num_basis

        _, old_vals = _eval_pts_2d(bspline)
        _, new_vals = _eval_pts_2d(new_bs)
        np.testing.assert_allclose(new_vals, old_vals, atol=1e-12)

    def test_subdivide_sequence_wrong_length_raises(self) -> None:
        """Sequence length != dim raises ValueError."""
        bspline = self._make_biquadratic_surface()
        with pytest.raises(ValueError, match="dim"):
            bspline.subdivide([2])  # dim == 2, need length 2

    def test_subdivide_all_n1_raises(self) -> None:
        """[1, 1] raises ValueError (at least one direction must be >= 2)."""
        bspline = self._make_biquadratic_surface()
        with pytest.raises(ValueError):
            bspline.subdivide([1, 1])

    def test_subdivide_one_direction_n1_ok(self) -> None:
        """[2, 1] is valid — only u is refined, geometry preserved."""
        bspline = self._make_biquadratic_surface()
        new_bs = bspline.subdivide([2, 1])

        assert new_bs.space.spaces[0].num_basis > bspline.space.spaces[0].num_basis
        assert new_bs.space.spaces[1].num_basis == bspline.space.spaces[1].num_basis

        _, old_vals = _eval_pts_2d(bspline)
        _, new_vals = _eval_pts_2d(new_bs)
        np.testing.assert_allclose(new_vals, old_vals, atol=1e-12)

    def test_subdivide_regularity_multi_dim(self) -> None:
        """Regularity parameter is respected for each active direction."""
        bspline = self._make_biquadratic_surface()
        # degree=2 in both directions; regularity=0 → each knot inserted twice
        new_bs = bspline.subdivide(2, regularity=0)

        knots_u = new_bs.space.spaces[0].knots
        knots_v = new_bs.space.spaces[1].knots
        # u: domain [0,2], midpoint 0.5 and 1.5 should appear twice each
        assert np.sum(np.isclose(knots_u, 0.5)) == 2  # noqa: PLR2004
        # v: domain [0,1], midpoint 0.5 should appear twice
        assert np.sum(np.isclose(knots_v, 0.5)) == 2  # noqa: PLR2004

        _, old_vals = _eval_pts_2d(bspline)
        _, new_vals = _eval_pts_2d(new_bs)
        np.testing.assert_allclose(new_vals, old_vals, atol=1e-12)


# ---------------------------------------------------------------------------
# Periodic flag behaviour after insert_knots / subdivide
# ---------------------------------------------------------------------------


class TestPeriodicInsertKnotsFlag:
    """Document and verify the periodic-flag semantics of insert_knots and subdivide.

    insert_knots and subdivide return a new BsplineSpace1D that is always
    non-periodic (periodic=False).  This is intentional: once interior knots
    are inserted the underlying space is no longer a genuine periodic B-spline
    — the ghost-knot structure that enforces periodicity is broken.
    """

    def test_insert_knots_periodic_loses_periodicity(self) -> None:
        """insert_knots on a periodic space returns a non-periodic space."""
        degree = 2
        knots = create_uniform_periodic_knot_vector(num_intervals=4, degree=degree)
        space = BsplineSpace1D(knots, degree, periodic=True)
        assert space.periodic

        new_space = space.insert_knots([0.125])

        assert not new_space.periodic

    def test_insert_knots_C0_periodic_loses_periodicity(self) -> None:
        """insert_knots on a C^0 periodic space returns a non-periodic space."""
        degree = 2
        knots = create_uniform_periodic_knot_vector(num_intervals=4, degree=degree, continuity=0)
        space = BsplineSpace1D(knots, degree, periodic=True)
        assert space.periodic

        new_space = space.insert_knots([0.125])

        assert not new_space.periodic

    def test_subdivide_periodic_loses_periodicity(self) -> None:
        """Subdivide on a periodic space returns a non-periodic space."""
        degree = 2
        knots = create_uniform_periodic_knot_vector(num_intervals=4, degree=degree)
        space = BsplineSpace1D(knots, degree, periodic=True)
        assert space.periodic

        new_space = space.subdivide(2)

        assert not new_space.periodic
