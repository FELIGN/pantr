"""Tests for B-spline knot insertion (insert_knots and subdivide)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D, create_uniform_periodic_knots
from pantr.bspline._bspline_knot_insertion_core import (
    _compute_oslo_matrix_1d_core,
    _compute_oslo_rows_1d_core,
    _insert_knots_1d_core,
)

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
        assert np.sum(np.isclose(new_space.knots, 0.5)) == 2
        # Inserting 0.5 once more reaches the maximum multiplicity degree+1=3.
        new_space2 = new_space.insert_knots([0.5])
        assert np.sum(np.isclose(new_space2.knots, 0.5)) == 3

    def test_insert_repeated_knots_in_one_call(self) -> None:
        """Inserting [0.5, 0.5] in one call raises multiplicity from 1 to degree+1=3."""
        # degree=2, so max multiplicity = 3; starting from 1, insert [0.5, 0.5] → 3
        space = BsplineSpace1D([0, 0, 0, 0.5, 1, 1, 1], 2)
        new_space = space.insert_knots([0.5, 0.5])
        assert np.sum(np.isclose(new_space.knots, 0.5)) == 3

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
        assert np.sum(np.isclose(new_space.knots, 0.5)) == 2

    def test_subdivide_regularity_minus1_gives_multiplicity_degree_plus1(self) -> None:
        """regularity=-1 inserts each knot degree+1 times (discontinuous, C^{-1})."""
        # degree=2, regularity=-1 → repeat=3
        space = BsplineSpace1D([0, 0, 0, 1, 1, 1], 2)
        new_space = space.subdivide(2, regularity=-1)
        assert np.sum(np.isclose(new_space.knots, 0.5)) == 3

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
        assert np.sum(np.isclose(knots, 0.5)) == 2

    def test_1d_repeated_knots_in_one_call_reaches_max_multiplicity(self) -> None:
        """Inserting [0.5, 0.5] in one call raises multiplicity from 1 to degree+1=3."""
        bspline = _make_1d_bspline(
            [0, 0, 0, 0.5, 1, 1, 1],
            2,
            [[0, 0], [0.25, 1], [0.75, 1], [1, 0]],
        )
        new_bs = bspline.insert_knots([0.5, 0.5])
        knots = new_bs.space.spaces[0].knots
        assert np.sum(np.isclose(knots, 0.5)) == 3


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
        assert np.sum(np.isclose(knots, 0.5)) == 2

    def test_subdivide_regularity_minus1_gives_discontinuous(self) -> None:
        """regularity=-1 inserts each new knot degree+1 times (C^{-1}, discontinuous)."""
        # degree=2, regularity=-1 → repeat=3; midpoint 0.5 inserted three times
        bspline = _make_1d_bspline([0, 0, 0, 1, 1, 1], 2, [[0, 0], [0.5, 1], [1, 0]])
        new_bs = bspline.subdivide(2, regularity=-1)
        knots = new_bs.space.spaces[0].knots
        assert np.sum(np.isclose(knots, 0.5)) == 3

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
        assert np.sum(np.isclose(knots_u, 0.5)) == 2
        # v: domain [0,1], midpoint 0.5 should appear twice
        assert np.sum(np.isclose(knots_v, 0.5)) == 2

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
        knots = create_uniform_periodic_knots(num_intervals=4, degree=degree)
        space = BsplineSpace1D(knots, degree, periodic=True)
        assert space.periodic

        new_space = space.insert_knots([0.125])

        assert not new_space.periodic

    def test_insert_knots_C0_periodic_loses_periodicity(self) -> None:
        """insert_knots on a C^0 periodic space returns a non-periodic space."""
        degree = 2
        knots = create_uniform_periodic_knots(num_intervals=4, degree=degree, continuity=0)
        space = BsplineSpace1D(knots, degree, periodic=True)
        assert space.periodic

        new_space = space.insert_knots([0.125])

        assert not new_space.periodic

    def test_subdivide_periodic_loses_periodicity(self) -> None:
        """Subdivide on a periodic space returns a non-periodic space."""
        degree = 2
        knots = create_uniform_periodic_knots(num_intervals=4, degree=degree)
        space = BsplineSpace1D(knots, degree, periodic=True)
        assert space.periodic

        new_space = space.subdivide(2)

        assert not new_space.periodic


# ---------------------------------------------------------------------------
# Periodic Bspline: insert_knots and subdivide preserve periodicity
# ---------------------------------------------------------------------------


def _make_periodic_bspline(
    num_intervals: int,
    degree: int,
    continuity: int | None = None,
    rank: int = 2,
) -> Bspline:
    """Create a 1D periodic B-spline with sequential control points."""
    knots = create_uniform_periodic_knots(num_intervals, degree, continuity=continuity)
    space_1d = BsplineSpace1D(knots, degree, periodic=True)
    space = BsplineSpace([space_1d])
    rng = np.random.default_rng(42)
    ctrl = rng.random((space.num_total_basis, rank))
    return Bspline(space, ctrl)


class TestPeriodicBsplineInsertKnots:
    """Test that Bspline.insert_knots preserves periodicity and geometry."""

    @pytest.mark.parametrize(
        "degree,continuity",
        [(2, None), (3, None), (3, 1), (3, 0), (2, 0)],
    )
    def test_insert_knots_preserves_periodic(self, degree: int, continuity: int | None) -> None:
        """insert_knots on a periodic Bspline returns a periodic Bspline."""
        bsp = _make_periodic_bspline(4, degree, continuity)
        refined = bsp.insert_knots(np.array([0.125, 0.375]))

        assert refined.space.spaces[0].periodic

    @pytest.mark.parametrize(
        "degree,continuity",
        [(2, None), (3, None), (3, 1), (3, 0), (2, 0)],
    )
    def test_insert_knots_preserves_geometry(self, degree: int, continuity: int | None) -> None:
        """insert_knots on a periodic Bspline preserves geometry."""
        bsp = _make_periodic_bspline(4, degree, continuity)
        refined = bsp.insert_knots(np.array([0.125, 0.375]))

        pts = np.linspace(0.01, 0.99, 50)
        orig = bsp.to_open_bspline().evaluate(pts)
        ref = refined.to_open_bspline().evaluate(pts)
        np.testing.assert_allclose(orig, ref, atol=1e-12)

    def test_insert_knots_multidim_mixed_periodic_open(self) -> None:
        """insert_knots preserves periodicity for mixed periodic/open 2D splines."""
        # Direction 0: periodic, direction 1: open
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

        refined = bsp.insert_knots([np.array([0.125]), np.array([0.25])])

        assert refined.space.spaces[0].periodic
        assert not refined.space.spaces[1].periodic

        pts = rng.random((30, 2))
        pts[:, 0] = pts[:, 0] * 0.98 + 0.01
        orig = bsp.to_open_bspline().evaluate(pts)
        ref = refined.to_open_bspline().evaluate(pts)
        np.testing.assert_allclose(orig, ref, atol=1e-12)

    def test_insert_knots_rational_periodic(self) -> None:
        """insert_knots preserves periodic NURBS geometry."""
        knots = create_uniform_periodic_knots(num_intervals=4, degree=3)
        space_1d = BsplineSpace1D(knots, 3, periodic=True)
        space = BsplineSpace([space_1d])
        n = space.num_total_basis
        rng = np.random.default_rng(42)
        ctrl_h = rng.random((n, 3))  # (x, y, w) homogeneous
        ctrl_h[:, -1] = np.abs(ctrl_h[:, -1]) + 0.5  # positive weights
        bsp = Bspline(space, ctrl_h, is_rational=True)

        refined = bsp.insert_knots(np.array([0.125]))

        assert refined.space.spaces[0].periodic
        assert refined.is_rational

        pts = np.linspace(0.01, 0.99, 50)
        orig = bsp.to_open_bspline().evaluate(pts)
        ref = refined.to_open_bspline().evaluate(pts)
        np.testing.assert_allclose(orig, ref, atol=1e-12)


class TestPeriodicBsplineSubdivide:
    """Test that Bspline.subdivide preserves periodicity and geometry."""

    @pytest.mark.parametrize(
        "degree,continuity",
        [(2, None), (3, None), (3, 1), (2, 0)],
    )
    def test_subdivide_preserves_periodic(self, degree: int, continuity: int | None) -> None:
        """Subdivide on a periodic Bspline returns a periodic Bspline."""
        bsp = _make_periodic_bspline(4, degree, continuity)
        subdivided = bsp.subdivide(2)

        assert subdivided.space.spaces[0].periodic

    @pytest.mark.parametrize(
        "degree,continuity",
        [(2, None), (3, None), (3, 1), (2, 0)],
    )
    def test_subdivide_preserves_geometry(self, degree: int, continuity: int | None) -> None:
        """Subdivide on a periodic Bspline preserves geometry."""
        bsp = _make_periodic_bspline(4, degree, continuity)
        subdivided = bsp.subdivide(2)

        pts = np.linspace(0.01, 0.99, 50)
        orig = bsp.to_open_bspline().evaluate(pts)
        sub = subdivided.to_open_bspline().evaluate(pts)
        np.testing.assert_allclose(orig, sub, atol=1e-12)


# ---------------------------------------------------------------------------
# Banded Oslo rows
# ---------------------------------------------------------------------------

_EPS = float(np.finfo(np.float64).eps)

_OSLO_ATOL_FACTOR = 8.0
"""Safety factor on the agreement between the banded and dense Oslo sweeps.

The two run the same recurrence over the same knot differences in the same
order — the banded one simply skips the entries the dense one computes as exact
zeros — so they agree bit for bit whenever the compiler emits the same
instructions, which is what is observed here. The bound is written in eps rather
than as an equality because the assertion crosses a compilation boundary: the
reference is interpreted NumPy while the kernel is compiled, and a fused
multiply-add contracted by one and not the other moves the last bit without
anything being wrong. Eight eps is the classic forward bound for the three
roundings per entry, with room for that contraction.
"""


def _dense_oslo_reference(
    degree: int,
    old_knots: npt.NDArray[np.float64],
    new_knots: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Build the Oslo matrix by the dense sweep the banded kernel replaced.

    Kept here as an oracle: it walks every column at every order, so it makes no
    assumption about where the discrete B-splines are supported.

    Args:
        degree (int): Polynomial degree.
        old_knots (npt.NDArray[np.float64]): Original knot vector.
        new_knots (npt.NDArray[np.float64]): Refined knot vector.

    Returns:
        npt.NDArray[np.float64]: Dense refinement matrix of shape ``(m+1, n+1)``.
    """
    n = old_knots.shape[0] - degree - 2
    m = new_knots.shape[0] - degree - 2

    previous = np.zeros((m + 1, n + 2))
    current = np.zeros((m + 1, n + 2))

    for i in range(m + 1):
        j = int(np.searchsorted(old_knots, new_knots[i], side="right")) - 1
        previous[i, min(max(j, 0), n)] = 1.0

    for k in range(2, degree + 2):
        current[:] = 0.0
        for i in range(m + 1):
            sik = new_knots[i + k - 1]
            for j in range(n + 1):
                value = 0.0
                denom1 = old_knots[j + k - 1] - old_knots[j]
                if denom1 > 0.0:
                    value += (sik - old_knots[j]) / denom1 * previous[i, j]
                denom2 = old_knots[j + k] - old_knots[j + 1]
                if denom2 > 0.0:
                    value += (old_knots[j + k] - sik) / denom2 * previous[i, j + 1]
                current[i, j] = value
        previous[:] = current

    return previous[:, : n + 1].copy()


def _scatter_oslo_rows(
    alphas: npt.NDArray[np.float64], first_col: npt.NDArray[np.int64], n_cols: int
) -> npt.NDArray[np.float64]:
    """Place banded rows into a dense matrix, dropping columns outside the space.

    Args:
        alphas (npt.NDArray[np.float64]): Bands of shape ``(m+1, degree+1)``.
        first_col (npt.NDArray[np.int64]): Global column of each band's first entry.
        n_cols (int): Number of old control points.

    Returns:
        npt.NDArray[np.float64]: Dense matrix of shape ``(m+1, n_cols)``.
    """
    dense = np.zeros((alphas.shape[0], n_cols))
    for i in range(alphas.shape[0]):
        for offset in range(alphas.shape[1]):
            col = int(first_col[i]) + offset
            if 0 <= col < n_cols:
                dense[i, col] = alphas[i, offset]
    return dense


def _oslo_case(
    degree: int, kind: str, seed: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Build an (old, refined) knot-vector pair of the requested shape.

    Args:
        degree (int): Polynomial degree.
        kind (str): One of ``uniform``, ``non_uniform``, ``multiple``,
            ``single_knot``, ``re_insert``, ``many``.
        seed (int): Seed for the random parts.

    Returns:
        tuple: ``(old_knots, new_knots)`` with ``new_knots`` a superset.
    """
    rng = np.random.default_rng(seed)
    if kind == "non_uniform":
        interior = np.sort(rng.random(5))
    elif kind == "multiple":
        # Interior knots repeated up to multiplicity `degree`.
        interior = np.repeat(np.array([0.25, 0.5, 0.75]), max(1, degree))
    else:
        interior = np.linspace(0.0, 1.0, 6)[1:-1]

    old = np.concatenate([np.zeros(degree + 1), interior, np.ones(degree + 1)])

    if kind == "single_knot":
        inserted = np.array([0.37])
    elif kind == "re_insert":
        # Raise the multiplicity of knots that are already there.
        existing = np.unique(old[degree + 1 : old.shape[0] - degree - 1])
        inserted = np.repeat(existing, max(1, degree - 1)) if existing.size else np.array([0.5])
    elif kind == "many":
        inserted = np.sort(rng.random(37))
    else:
        inserted = np.sort(rng.random(4))

    return old, np.sort(np.concatenate([old, inserted]))


_OSLO_KINDS = ["uniform", "non_uniform", "multiple", "single_knot", "re_insert", "many"]


class TestOsloBandedRows:
    """The banded row recurrence against the dense sweep it replaced."""

    @pytest.mark.parametrize("degree", range(9))
    @pytest.mark.parametrize("kind", _OSLO_KINDS)
    def test_banded_rows_reproduce_the_dense_sweep(self, degree: int, kind: str) -> None:
        """Scattering the bands gives back the dense refinement matrix."""
        old, new = _oslo_case(degree, kind, seed=degree * 10 + len(kind))
        reference = _dense_oslo_reference(degree, old, new)

        alphas, first_col = _compute_oslo_rows_1d_core(degree, old, new)
        scattered = _scatter_oslo_rows(alphas, first_col, reference.shape[1])

        assert alphas.shape == (reference.shape[0], degree + 1)
        assert first_col.shape == (reference.shape[0],)
        np.testing.assert_allclose(scattered, reference, rtol=0.0, atol=_OSLO_ATOL_FACTOR * _EPS)

    @pytest.mark.parametrize("degree", range(9))
    @pytest.mark.parametrize("kind", _OSLO_KINDS)
    def test_dense_wrapper_is_unchanged(self, degree: int, kind: str) -> None:
        """The public dense kernel still returns what the dense sweep returned."""
        old, new = _oslo_case(degree, kind, seed=degree + 3)

        np.testing.assert_allclose(
            _compute_oslo_matrix_1d_core(degree, old, new),
            _dense_oslo_reference(degree, old, new),
            rtol=0.0,
            atol=_OSLO_ATOL_FACTOR * _EPS,
        )

    @pytest.mark.parametrize("degree", range(1, 7))
    @pytest.mark.parametrize("kind", _OSLO_KINDS)
    def test_the_dense_matrix_has_no_nonzeros_outside_the_band(
        self, degree: int, kind: str
    ) -> None:
        """Everything the dense sweep computes outside the band is exactly zero.

        This is the claim the whole change rests on, so it is checked against the
        dense oracle rather than against the banded kernel.
        """
        old, new = _oslo_case(degree, kind, seed=degree * 7 + 1)
        reference = _dense_oslo_reference(degree, old, new)
        _, first_col = _compute_oslo_rows_1d_core(degree, old, new)

        for i in range(reference.shape[0]):
            nonzero = np.flatnonzero(reference[i] != 0.0)
            if nonzero.size:
                assert nonzero.min() >= first_col[i]
                assert nonzero.max() <= first_col[i] + degree

    @pytest.mark.parametrize("degree", range(9))
    @pytest.mark.parametrize("kind", _OSLO_KINDS)
    def test_rows_sum_to_one(self, degree: int, kind: str) -> None:
        """Discrete B-splines form a partition of unity on every row."""
        old, new = _oslo_case(degree, kind, seed=degree * 5 + 2)
        alphas, first_col = _compute_oslo_rows_1d_core(degree, old, new)
        n_cols = old.shape[0] - degree - 1

        sums = _scatter_oslo_rows(alphas, first_col, n_cols).sum(axis=1)

        np.testing.assert_allclose(sums, 1.0, rtol=0.0, atol=_OSLO_ATOL_FACTOR * _EPS)

    @pytest.mark.parametrize("degree", [2, 3, 4, 5])
    def test_periodic_knot_vectors_push_the_band_left_of_the_space(self, degree: int) -> None:
        """A non-clamped knot vector produces bands that start before column 0.

        The clipping in the kernel's consumers is therefore load-bearing, not
        defensive: on these vectors some spans sit inside the first ``degree``
        knots.
        """
        periodic = np.asarray(
            create_uniform_periodic_knots(num_intervals=6, degree=degree), dtype=np.float64
        )
        refined = np.sort(np.concatenate([periodic, np.array([0.05, 0.55])]))

        alphas, first_col = _compute_oslo_rows_1d_core(degree, periodic, refined)
        n_cols = periodic.shape[0] - degree - 1

        assert (first_col < 0).any()
        reference = _dense_oslo_reference(degree, periodic, refined)
        np.testing.assert_allclose(
            _scatter_oslo_rows(alphas, first_col, n_cols),
            reference,
            rtol=0.0,
            atol=_OSLO_ATOL_FACTOR * _EPS,
        )

    @pytest.mark.parametrize("degree", range(1, 7))
    @pytest.mark.parametrize("kind", _OSLO_KINDS)
    def test_insertion_matches_the_dense_matrix_product(self, degree: int, kind: str) -> None:
        """Applying the bands agrees with multiplying by the dense matrix.

        The two sum the same products in a different order, so they are compared
        within the forward bound of a dot product of ``degree + 1`` terms rather
        than by equality.
        """
        rng = np.random.default_rng(degree * 11 + len(kind))
        old, new = _oslo_case(degree, kind, seed=degree + 17)
        ctrl = rng.standard_normal((old.shape[0] - degree - 1, 3))

        banded = _insert_knots_1d_core(degree, old, ctrl, new)
        dense = _dense_oslo_reference(degree, old, new) @ ctrl

        atol = (degree + 1) * _EPS * float(np.max(np.abs(ctrl)))
        np.testing.assert_allclose(banded, dense, rtol=0.0, atol=atol)

    @pytest.mark.parametrize("degree", range(1, 7))
    @pytest.mark.parametrize("rational", [False, True])
    def test_geometry_is_preserved(self, degree: int, rational: bool) -> None:
        """Refinement leaves the mapping unchanged at a thousand random points."""
        rng = np.random.default_rng(degree * 3 + int(rational))
        knots = np.concatenate([np.zeros(degree), np.linspace(0.0, 1.0, 7), np.ones(degree)])
        space = BsplineSpace([BsplineSpace1D(knots, degree)])
        ctrl = rng.random((space.num_total_basis, 4)) + 1.0
        bsp = Bspline(space, ctrl, is_rational=rational)

        refined = bsp.insert_knots(np.sort(rng.random(23)))

        pts = np.sort(rng.random(1000))
        atol = 64.0 * _EPS * float(np.max(np.abs(bsp.evaluate(pts))))
        np.testing.assert_allclose(refined.evaluate(pts), bsp.evaluate(pts), rtol=0.0, atol=atol)
