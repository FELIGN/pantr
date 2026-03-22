"""Tests for B-spline knot removal."""

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_curve_p2() -> Bspline:
    """Create a degree-2 B-spline curve with 4 control points in 3D.

    Knots: [0,0,0,0.5,1,1,1] -> 4 basis functions, degree 2.
    """
    knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
    space = BsplineSpace([BsplineSpace1D(knots, 2)])
    cp = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [3.0, 0.0, 0.0]])
    return Bspline(space, cp)


def _make_curve_p3() -> Bspline:
    """Create a degree-3 B-spline curve with 5 control points in 2D.

    Knots: [0,0,0,0,0.5,1,1,1,1] -> 5 basis, degree 3.
    """
    knots = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
    space = BsplineSpace([BsplineSpace1D(knots, 3)])
    cp = np.array([[0.0, 0.0], [1.0, 3.0], [2.0, 2.0], [3.0, 3.0], [4.0, 0.0]])
    return Bspline(space, cp)


def _make_surface() -> Bspline:
    """Create a 2D B-spline surface with an interior knot in the u-direction.

    Degree (2, 1), rank 2.
    """
    knots_u = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
    knots_v = np.array([0.0, 0.0, 1.0, 1.0])
    space_u = BsplineSpace1D(knots_u, 2)
    space_v = BsplineSpace1D(knots_v, 1)
    space = BsplineSpace([space_u, space_v])
    cp = np.arange(16, dtype=np.float64).reshape(4, 2, 2)
    return Bspline(space, cp)


def _assert_same_geometry(a: Bspline, b: Bspline, atol: float = 1e-12) -> None:
    """Assert two B-splines represent the same geometry via knot insertion.

    Inserts all knots from ``b`` into ``a`` and vice versa to get a common
    representation, then compares control points.
    """
    for i in range(a.dim):
        knots_a = a.space.spaces[i].knots
        knots_b = b.space.spaces[i].knots

        # Knots to insert into a to match b.
        extra_b = _knot_difference(knots_b, knots_a, atol)
        if extra_b.size > 0:
            insert_a: list[npt.NDArray[np.float32 | np.float64] | None] = [None] * a.dim
            insert_a[i] = extra_b
            a = a.insert_knots(insert_a)

        # Knots to insert into b to match a.
        extra_a = _knot_difference(knots_a, knots_b, atol)
        if extra_a.size > 0:
            insert_b: list[npt.NDArray[np.float32 | np.float64] | None] = [None] * b.dim
            insert_b[i] = extra_a
            b = b.insert_knots(insert_b)

    np.testing.assert_allclose(a.control_points, b.control_points, atol=atol)


def _knot_difference(
    knots_a: npt.NDArray[np.float32 | np.float64],
    knots_b: npt.NDArray[np.float32 | np.float64],
    atol: float,
) -> npt.NDArray[np.float32 | np.float64]:
    """Return knots in ``a`` that are not in ``b`` (with multiplicity)."""
    remaining = list(knots_b)
    diff = []
    for val in knots_a:
        found = False
        for j, rem in enumerate(remaining):
            if abs(val - rem) <= atol:
                remaining.pop(j)
                found = True
                break
        if not found:
            diff.append(val)
    return np.array(diff, dtype=knots_a.dtype)


# ===========================================================================
# Round-trip: insert then remove
# ===========================================================================


class TestRoundTrip:
    """Insert knots, then remove them, and verify geometry is preserved."""

    def test_insert_remove_single_knot_p2(self) -> None:
        """Insert then remove a single knot on a degree-2 curve."""
        orig = _make_curve_p2()
        refined = orig.insert_knots(np.array([0.25]))
        reduced = refined.remove_knots(0.25)

        # Knot vectors should match original.
        np.testing.assert_allclose(
            reduced.space.spaces[0].knots, orig.space.spaces[0].knots, atol=1e-14
        )
        # Control points should match original.
        np.testing.assert_allclose(reduced.control_points, orig.control_points, atol=1e-12)

    def test_insert_remove_single_knot_p3(self) -> None:
        """Insert then remove a single knot on a degree-3 curve."""
        orig = _make_curve_p3()
        refined = orig.insert_knots(np.array([0.25]))
        reduced = refined.remove_knots(0.25)

        np.testing.assert_allclose(
            reduced.space.spaces[0].knots, orig.space.spaces[0].knots, atol=1e-14
        )
        np.testing.assert_allclose(reduced.control_points, orig.control_points, atol=1e-12)

    def test_insert_remove_multiple_same_knot(self) -> None:
        """Insert a knot twice, then remove it twice."""
        orig = _make_curve_p3()
        refined = orig.insert_knots(np.array([0.3, 0.3]))
        reduced = refined.remove_knots(0.3, num=2)

        np.testing.assert_allclose(
            reduced.space.spaces[0].knots, orig.space.spaces[0].knots, atol=1e-14
        )
        np.testing.assert_allclose(reduced.control_points, orig.control_points, atol=1e-12)

    def test_insert_remove_multiple_distinct_knots(self) -> None:
        """Insert two different knots, then remove both."""
        orig = _make_curve_p3()
        refined = orig.insert_knots(np.array([0.25, 0.75]))
        reduced = refined.remove_knots(np.array([0.25, 0.75]))

        np.testing.assert_allclose(
            reduced.space.spaces[0].knots, orig.space.spaces[0].knots, atol=1e-14
        )
        np.testing.assert_allclose(reduced.control_points, orig.control_points, atol=1e-12)

    def test_remove_existing_knot_preserves_geometry(self) -> None:
        """Removing an original interior knot still preserves geometry (within tol)."""
        orig = _make_curve_p2()
        reduced = orig.remove_knots(0.5)
        _assert_same_geometry(orig, reduced)


# ===========================================================================
# Partial removal
# ===========================================================================


class TestPartialRemoval:
    """Test removing fewer knots than the full multiplicity."""

    def test_remove_one_of_two(self) -> None:
        """Insert a knot twice, remove only once — multiplicity drops by 1."""
        orig = _make_curve_p3()
        refined = orig.insert_knots(np.array([0.3, 0.3]))

        reduced = refined.remove_knots(0.3, num=1)

        # One copy should remain.
        knots = reduced.space.spaces[0].knots
        count = np.sum(np.isclose(knots, 0.3, atol=1e-12))
        assert count == 1

    def test_num_none_removes_all(self) -> None:
        """num=None should remove as many times as possible."""
        orig = _make_curve_p3()
        refined = orig.insert_knots(np.array([0.3, 0.3]))

        reduced = refined.remove_knots(0.3, num=None)

        knots = reduced.space.spaces[0].knots
        count = np.sum(np.isclose(knots, 0.3, atol=1e-12))
        assert count == 0


# ===========================================================================
# Multi-dimensional
# ===========================================================================


class TestMultiDim:
    """Test knot removal on multi-dimensional B-splines."""

    def test_surface_remove_u_direction(self) -> None:
        """Remove an inserted interior knot from the u-direction of a surface."""
        orig = _make_surface()
        refined = orig.insert_knots([np.array([0.25]), None])
        reduced = refined.remove_knots([np.array([0.25]), None])

        np.testing.assert_allclose(
            reduced.space.spaces[0].knots, orig.space.spaces[0].knots, atol=1e-14
        )
        np.testing.assert_allclose(reduced.control_points, orig.control_points, atol=1e-12)


# ===========================================================================
# Rational (NURBS) support
# ===========================================================================


class TestRational:
    """Test knot removal on rational B-splines (NURBS)."""

    def test_nurbs_round_trip(self) -> None:
        """Insert then remove a knot on a NURBS curve."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        space = BsplineSpace([BsplineSpace1D(knots, 2)])
        # Homogeneous coords: (w*x, w*y, w) — a rational quadratic.
        cp = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 0.5], [2.0, 0.0, 1.0]])
        nurbs = Bspline(space, cp, is_rational=True)

        refined = nurbs.insert_knots(np.array([0.5]))
        reduced = refined.remove_knots(0.5)

        np.testing.assert_allclose(
            reduced.space.spaces[0].knots, nurbs.space.spaces[0].knots, atol=1e-14
        )
        np.testing.assert_allclose(reduced.control_points, nurbs.control_points, atol=1e-12)


# ===========================================================================
# Error handling
# ===========================================================================


class TestErrors:
    """Test error conditions for remove_knots."""

    def test_knot_not_found(self) -> None:
        """Requesting removal of a non-existent knot raises ValueError."""
        b = _make_curve_p2()
        with pytest.raises(ValueError, match="not found"):
            b.remove_knots(0.123)

    def test_boundary_knot(self) -> None:
        """Attempting to remove a boundary knot raises ValueError."""
        b = _make_curve_p2()
        with pytest.raises(ValueError, match="boundary knot"):
            b.remove_knots(0.0)

    def test_boundary_knot_end(self) -> None:
        """Attempting to remove the end boundary knot raises ValueError."""
        b = _make_curve_p2()
        with pytest.raises(ValueError, match="boundary knot"):
            b.remove_knots(1.0)

    def test_empty_knot_values(self) -> None:
        """Empty knot_values array raises ValueError."""
        b = _make_curve_p2()
        with pytest.raises(ValueError, match="non-empty"):
            b.remove_knots(np.array([]))

    def test_dim_mismatch(self) -> None:
        """Wrong number of direction arrays raises ValueError."""
        srf = _make_surface()
        with pytest.raises(ValueError, match="must match dim"):
            srf.remove_knots([np.array([0.5])])  # only 1 direction, need 2

    def test_periodic_not_supported(self) -> None:
        """Periodic B-splines raise ValueError."""
        knots = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        space = BsplineSpace([BsplineSpace1D(knots, 2, periodic=True)])
        cp = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        b = Bspline(space, cp)
        with pytest.raises(ValueError, match="periodic"):
            b.remove_knots(0.5)

    def test_num_zero_raises(self) -> None:
        """num=0 raises ValueError."""
        b = _make_curve_p2()
        with pytest.raises(ValueError, match="positive"):
            b.remove_knots(0.5, num=0)


# ===========================================================================
# Tolerance rejection
# ===========================================================================


class TestToleranceRejection:
    """Test that tight tolerances prevent removals that would distort geometry."""

    def test_tight_tolerance_prevents_removal(self) -> None:
        """A knot that exists in the original cannot be removed with tol=0."""
        b = _make_curve_p2()
        # The knot 0.5 is part of the original spline definition.
        # Removing it changes the geometry, so with tol=0 it should fail to remove.
        result = b.remove_knots(0.5, tol=0.0)
        # With zero tolerance, geometry must be preserved exactly.
        # The knot should still be present (removal was rejected).
        knots = result.space.spaces[0].knots
        count = np.sum(np.isclose(knots, 0.5, atol=1e-14))
        assert count == 1
