"""Tests for the join operation."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pytest
from numpy import typing as npt
from numpy.testing import assert_allclose, assert_array_equal

from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
from pantr.cad import create_bilinear, create_line, join
from pantr.tolerance import get_conservative


def _bezier_patch(control_points: npt.NDArray[np.float64], degree_u: int, degree_v: int) -> Bspline:
    """Build a single-Bezier-span B-spline surface of the given bidegree on ``[0, 1]^2``.

    ``create_bilinear`` only reaches bidegree ``(1, 1)``, whose shared boundary row has
    no interior: every one of its control points is a corner, so it cannot exhibit a
    disagreement that a corner-only check would miss.

    Args:
        control_points (npt.NDArray[np.float64]): Shape ``(degree_u + 1, degree_v + 1, 3)``.
        degree_u (int): Degree in the first parametric direction.
        degree_v (int): Degree in the second.

    Returns:
        Bspline: A clamped, non-rational surface on ``[0, 1]^2``.
    """
    spaces = [
        BsplineSpace1D(np.array([0.0] * (d + 1) + [1.0] * (d + 1)), d) for d in (degree_u, degree_v)
    ]
    return Bspline(BsplineSpace(spaces), np.asarray(control_points, dtype=np.float64))


def _rational_curve(control_points: npt.NDArray[np.float64], degree: int) -> Bspline:
    """Build a clamped rational B-spline curve from homogeneous control points.

    Args:
        control_points (npt.NDArray[np.float64]): Shape ``(degree + 1, rank + 1)``, the
            trailing column being the weight.
        degree (int): Polynomial degree.

    Returns:
        Bspline: A clamped rational curve on ``[0, 1]``.
    """
    knots = np.array([0.0] * (degree + 1) + [1.0] * (degree + 1))
    space = BsplineSpace([BsplineSpace1D(knots, degree)])
    return Bspline(space, np.asarray(control_points, dtype=np.float64), is_rational=True)


class TestJoinCurves:
    """Test joining 1D B-spline curves."""

    def test_join_two_line_segments(self) -> None:
        """Test joining two collinear line segments."""
        c1 = create_line([0, 0, 0], [1, 0, 0])
        c2 = create_line([1, 0, 0], [2, 0, 0])
        result = join(c1, c2, axis=0)
        assert result.dim == 1
        # Evaluate over the full domain
        domain = result.space.spaces[0].domain
        t = np.linspace(float(domain[0]), float(domain[1]), 11)
        pts = result.evaluate(t)
        expected_x = np.linspace(0, 2, 11)
        assert_allclose(pts[:, 0], expected_x, atol=1e-13)
        assert_allclose(pts[:, 1], 0.0, atol=1e-14)
        assert_allclose(pts[:, 2], 0.0, atol=1e-14)

    def test_join_preserves_geometry(self) -> None:
        """Test that join preserves the geometry of both segments."""
        c1 = create_line([0, 0, 0], [1, 1, 0])
        c2 = create_line([1, 1, 0], [3, 0, 0])
        result = join(c1, c2, axis=0)
        domain = result.space.spaces[0].domain
        pt_start = result.evaluate(np.array([float(domain[0])]))
        pt_end = result.evaluate(np.array([float(domain[1])]))
        assert_allclose(pt_start, [0, 0, 0], atol=1e-14)
        assert_allclose(pt_end, [3, 0, 0], atol=1e-14)

    def test_join_c0_at_junction(self) -> None:
        """Test C0 continuity at the junction point."""
        c1 = create_line([0, 0, 0], [1, 0, 0])
        c2 = create_line([1, 0, 0], [1, 1, 0])
        result = join(c1, c2, axis=0)
        # The junction should be at parameter u=1 (end of c1's domain)
        pt = result.evaluate(np.array([1.0]))
        assert_allclose(pt, [1, 0, 0], atol=1e-12)


class TestJoinSurfaces:
    """Test joining 2D B-spline surfaces."""

    def test_join_two_bilinear_patches(self) -> None:
        """Test joining two bilinear patches along axis 0."""
        corners1 = np.array([[[0, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0]]], dtype=np.float64)
        corners2 = np.array([[[1, 0, 0], [1, 1, 0]], [[2, 0, 0], [2, 1, 0]]], dtype=np.float64)
        s1 = create_bilinear(corners1)
        s2 = create_bilinear(corners2)
        result = join(s1, s2, axis=0)
        assert result.dim == 2
        domain_u = result.space.spaces[0].domain
        u_end = float(domain_u[1])
        pts = result.evaluate(
            np.array(
                [
                    [0.0, 0.0],
                    [u_end, 0.0],
                    [0.0, 1.0],
                    [u_end, 1.0],
                ]
            )
        )
        assert_allclose(pts[0], [0, 0, 0], atol=1e-13)
        assert_allclose(pts[1], [2, 0, 0], atol=1e-13)
        assert_allclose(pts[2], [0, 1, 0], atol=1e-13)
        assert_allclose(pts[3], [2, 1, 0], atol=1e-13)

    def test_join_along_axis_1(self) -> None:
        """Test joining two patches along the second axis."""
        corners1 = np.array([[[0, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0]]], dtype=np.float64)
        corners2 = np.array([[[0, 1, 0], [0, 2, 0]], [[1, 1, 0], [1, 2, 0]]], dtype=np.float64)
        s1 = create_bilinear(corners1)
        s2 = create_bilinear(corners2)
        result = join(s1, s2, axis=1)
        domain_v = result.space.spaces[1].domain
        v_end = float(domain_v[1])
        pt = result.evaluate(np.array([[0.5, v_end]]))
        assert_allclose(pt, [0.5, 2, 0], atol=1e-13)


class TestJoinErrors:
    """Test error handling in join."""

    def test_different_dim_raises(self) -> None:
        """Test that different dimensions raises ValueError."""
        crv = create_line([0, 0, 0], [1, 0, 0])
        srf = create_bilinear()
        with pytest.raises(ValueError, match="same dim"):
            join(crv, srf, axis=0)

    def test_axis_out_of_range_raises(self) -> None:
        """Test that out-of-range axis raises ValueError."""
        c1 = create_line([0, 0, 0], [1, 0, 0])
        c2 = create_line([1, 0, 0], [2, 0, 0])
        with pytest.raises(ValueError, match="axis"):
            join(c1, c2, axis=1)


class TestJoinSharedBoundaryConsistency:
    """The two control rows meeting at the junction are not independent data.

    ``join`` fuses two patches by averaging the control row at the end of the first with
    the one at the start of the second.  When those rows agree the average is that same
    row and the join is exact; when they do not, the average is a new row belonging to
    neither input, and the result misses both by half the gap -- silently, and by an
    amount that grows with the mistake rather than staying near roundoff (pantr issue
    310).  ``create_coons_surface`` and ``create_coons_volume`` have always refused the
    same class of input.

    The tolerance is the Coons path's: ``get_conservative(float64)`` times the largest
    absolute coordinate over the two rows being compared.  Its derivation is in
    ``pantr.cad._coons._verify_corners_2d``.
    """

    @staticmethod
    def _patch_pair(bulge: float) -> tuple[Bspline, Bspline]:
        """Two bidegree-``(1, 2)`` patches meeting at ``x = 1``, the second's edge bulged.

        The shared boundary row is the three control points at ``u = 1`` of the first and
        at ``u = 0`` of the second.  ``bulge`` displaces only the *middle* one, so the two
        rows agree at both of their corners for every value of it.

        Args:
            bulge (float): Displacement along *z* of the second patch's middle boundary
                control point.

        Returns:
            tuple[Bspline, Bspline]: The two patches, in join order.
        """
        left = np.array(
            [
                [[0, 0, 0], [0, 1, 0], [0, 2, 0]],
                [[1, 0, 0], [1, 1, 0], [1, 2, 0]],
            ],
            dtype=np.float64,
        )
        right = np.array(
            [
                [[1, 0, 0], [1, 1, bulge], [1, 2, 0]],
                [[2, 0, 0], [2, 1, 0], [2, 2, 0]],
            ],
            dtype=np.float64,
        )
        return _bezier_patch(left, 1, 2), _bezier_patch(right, 1, 2)

    def test_two_segments_that_do_not_meet_are_refused(self) -> None:
        """The reproduction filed with issue 310, hardcoded.

        Two collinear degree-1 segments with a gap of 4 units between them, four times
        the length of either.  Before the check, ``join`` returned a single spline whose
        junction sat at ``[3, 0, 0]``, the midpoint of the two disagreeing endpoints,
        which lies on neither input.
        """
        c1 = create_line([0, 0, 0], [1, 0, 0])
        c2 = create_line([5, 0, 0], [6, 0, 0])
        with pytest.raises(ValueError, match="mismatch"):
            join(c1, c2, axis=0)

    def test_the_message_names_the_axis_the_gap_and_the_tolerance(self) -> None:
        """A refusal has to say what is wrong, not only that something is.

        The tolerance printed must be the one actually compared against, so the message
        is checked against ``get_conservative(float64)`` times the scale of the two rows
        -- here 6, the largest coordinate over both -- rather than against itself.
        """
        c1 = create_line([0, 0, 0], [1, 0, 0])
        c2 = create_line([5, 0, 0], [6, 0, 0])
        with pytest.raises(ValueError) as excinfo:
            join(c1, c2, axis=0)
        message = str(excinfo.value)
        assert "axis 0" in message
        assert f"{4.0:.3e}" in message
        assert f"{get_conservative(np.float64) * 6.0:.3e}" in message

    def test_two_segments_that_meet_still_join(self) -> None:
        """The control of the same reproduction: the check must not cost the good case.

        The junction control point is the average of two bitwise-equal values, which in
        IEEE arithmetic returns that value exactly, so the result is pinned exactly and
        not to a tolerance.
        """
        c1 = create_line([0, 0, 0], [1, 0, 0])
        c2 = create_line([1, 0, 0], [2, 0, 0])
        result = join(c1, c2, axis=0)
        assert_array_equal(
            np.asarray(result.control_points),
            np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64),
        )
        domain = result.space.spaces[0].domain
        ts = np.linspace(float(domain[0]), float(domain[1]), 5)
        pts = np.asarray(result.evaluate(ts), dtype=np.float64)
        for point in ([0, 0, 0], [1, 0, 0], [2, 0, 0]):
            assert np.isclose(pts, point).all(axis=1).any()

    def test_a_mismatch_in_the_interior_of_a_shared_face_is_refused(self) -> None:
        """Agreeing at the corners of the shared face is necessary but not sufficient.

        The two patches meet exactly at both ends of the row they share and disagree by
        half the patch in between, so a check reading only the corners of the junction
        accepts them.  Before the check, the result missed both patches along the whole
        interior of the seam.
        """
        left, right = self._patch_pair(bulge=0.5)
        with pytest.raises(ValueError, match="mismatch"):
            join(left, right, axis=0)

    def test_the_same_patches_join_when_the_shared_face_agrees(self) -> None:
        """The control for the interior case, and the seam is interpolated.

        With the bulge removed the two rows are bitwise equal, so the averaged row is
        exactly theirs and the joined surface restricts to the shared boundary curve.
        """
        left, right = self._patch_pair(bulge=0.0)
        result = join(left, right, axis=0)
        assert result.dim == 2
        u_junction = float(left.space.spaces[0].domain[1])
        v = np.linspace(0.0, 1.0, 9)
        params = np.column_stack([np.full_like(v, u_junction), v])
        got = np.asarray(result.evaluate(params), dtype=np.float64)
        want = np.asarray(left.evaluate(params), dtype=np.float64)
        assert_allclose(got, want, atol=1e-14)

    def test_a_mismatch_in_the_interior_of_a_shared_face_is_refused_on_axis_1(self) -> None:
        """The check has to reach the seam whichever axis carries it.

        Same construction transposed: two bidegree-``(2, 1)`` patches meeting along
        ``v``, agreeing at both corners of the shared row and bulging in between.
        """
        left, right = (b.permute_directions([1, 0]) for b in self._patch_pair(bulge=0.5))
        with pytest.raises(ValueError, match="mismatch"):
            join(left, right, axis=1)

    def test_a_degree_mismatch_on_the_join_axis_is_accepted(self) -> None:
        """Elevating one side to the common degree must not manufacture a disagreement.

        Degree elevation leaves a clamped endpoint control point where it is, so the two
        readings of a genuinely shared junction stay equal and no tolerance is spent.
        Nothing in the suite covered a join across degrees before.
        """
        c1 = create_line([0, 0, 0], [1, 0, 0])
        c2 = create_line([1, 0, 0], [2, 0, 0]).elevate_degree(2)
        result = join(c1, c2, axis=0)
        domain = result.space.spaces[0].domain
        ends = np.asarray(result.evaluate(np.array([float(d) for d in domain])), dtype=np.float64)
        assert_allclose(ends, [[0, 0, 0], [2, 0, 0]], atol=1e-14)

    def test_a_knot_mismatch_on_a_non_join_axis_is_accepted(self) -> None:
        """Making the non-join axes compatible must not manufacture one either.

        The second patch carries an extra knot in ``v``, so ``make_compat`` refines both
        to a common space before the rows are compared.  That runs the same insertion on
        coefficients that already agree, so whatever it rounds it rounds identically on
        both sides.
        """
        left, right = self._patch_pair(bulge=0.0)
        right = right.insert_knots([None, np.array([0.5])])
        result = join(left, right, axis=0)
        assert result.dim == 2
        u_junction = float(left.space.spaces[0].domain[1])
        v = np.linspace(0.0, 1.0, 9)
        params = np.column_stack([np.full_like(v, u_junction), v])
        got = np.asarray(result.evaluate(params), dtype=np.float64)
        want = np.asarray(left.evaluate(params), dtype=np.float64)
        assert_allclose(got, want, atol=1e-14)


class TestJoinBoundaryToleranceScale:
    """The junction check must reach the same verdict at every model scale.

    Its tolerance is ``4096 * eps`` times the largest absolute coordinate over the two
    rows compared, so a last-bit disagreement is accepted on a metre-scale part measured
    in microns, and a gap of one part per million of the model is refused on a micron-
    scale one.  A fixed absolute constant can do neither.
    """

    @staticmethod
    def _segments(scale: float, shift: float = 0.0) -> tuple[Bspline, Bspline]:
        """Two collinear segments of length ``scale``, the second's start displaced.

        Args:
            scale (float): Length of each segment, and so the model size.
            shift (float): Displacement of the junction as the second segment sees it.

        Returns:
            tuple[Bspline, Bspline]: The two segments, in join order.
        """
        s = scale
        return (
            create_line([0, 0, 0], [s, 0, 0]),
            create_line([s + shift, 0, 0], [2 * s, 0, 0]),
        )

    @pytest.mark.parametrize("scale", [1.0e-6, 1.0e-3, 1.0, 1.0e3, 1.0e6, 1.0e9])
    def test_a_one_ulp_junction_mismatch_is_accepted_at_every_scale(self, scale: float) -> None:
        """A geometrically perfect join must never be refused for a last-bit disagreement.

        At ``scale >= 1e6`` one ulp of the coordinate already exceeds ``1e-12``, which is
        the direction an absolute constant fails in.
        """
        shift = float(np.nextafter(scale, np.inf) - scale)
        result = join(*self._segments(scale, shift), axis=0)
        assert result.dim == 1

    @pytest.mark.parametrize(
        "scale", [1.0e-9, 1.0e-8, 1.0e-7, 1.0e-6, 1.0e-3, 1.0, 1.0e3, 1.0e6, 1.0e9]
    )
    def test_a_relative_junction_gap_is_refused_at_every_scale(self, scale: float) -> None:
        """And this is the direction an absolute constant fails in on a small model.

        A gap of one part per million of the model is a real modelling mistake at any
        scale; graded against a fixed ``1e-12`` it would be accepted from ``scale = 1e-6``
        down.
        """
        with pytest.raises(ValueError, match="mismatch"):
            join(*self._segments(scale, shift=1.0e-6 * scale), axis=0)

    def test_the_scale_comes_from_both_rows_not_from_the_control_point_under_test(self) -> None:
        """A seam touching the origin is graded like the rest of the seam.

        The control point at the origin supplies no magnitude of its own.  Reading the
        scale off that point alone would grade it against zero and demand bitwise
        agreement there, while allowing ``4096 * eps * s`` everywhere else on the same
        row.  The displacement here is ``1e-9`` on a model spanning ``1e6``, i.e. ``1e-15``
        relative -- roundoff, not a modelling error.
        """
        s = 1.0e6
        left = np.array(
            [
                [[0, 0, 0], [0, s, 0]],
                [[0, 0, 0], [s, s, 0]],
            ],
            dtype=np.float64,
        )
        right = np.array(
            [
                [[1.0e-9, 0, 0], [s, s, 0]],
                [[s, 0, 0], [2 * s, s, 0]],
            ],
            dtype=np.float64,
        )
        result = join(_bezier_patch(left, 1, 1), _bezier_patch(right, 1, 1), axis=0)
        assert result.dim == 2


class TestJoinRationalBoundary:
    """A rational join compares homogeneous control points, as the Coons edge check does.

    ``join`` averages the homogeneous rows -- weights included -- so those are what has
    to agree for the average to be the row itself.  Two rows that project to the same
    points under different weights are therefore refused: the resulting fused row is not
    either of them, which is exactly the failure the check exists to stop.  The gap it
    reports is then a coefficient gap and not a distance, since turning one into the
    other needs a lower bound on the weights that this check does not compute.
    """

    def test_a_rational_join_with_agreeing_homogeneous_rows_is_accepted(self) -> None:
        """Non-unit weights are fine as long as both sides carry the same ones."""
        c1 = _rational_curve(np.array([[0, 0, 0, 1], [2, 0, 0, 2]], dtype=np.float64), 1)
        c2 = _rational_curve(np.array([[2, 0, 0, 2], [3, 0, 0, 1]], dtype=np.float64), 1)
        result = join(c1, c2, axis=0)
        assert result.is_rational
        domain = result.space.spaces[0].domain
        ends = np.asarray(result.evaluate(np.array([float(d) for d in domain])), dtype=np.float64)
        assert_allclose(ends, [[0, 0, 0], [3, 0, 0]], atol=1e-14)

    def test_a_rational_join_whose_junction_weights_disagree_is_refused(self) -> None:
        """The two rows project to the same point and are still refused.

        ``[2, 0, 0, 2]`` and ``[1, 0, 0, 1]`` are the same point of space carried at
        different weights.  Averaging them gives ``[1.5, 0, 0, 1.5]``, which projects
        back to that same point, but the *weight* at the junction is now neither input's,
        so the curve on both sides of the seam is not the curve either input described.
        """
        c1 = _rational_curve(np.array([[0, 0, 0, 1], [2, 0, 0, 2]], dtype=np.float64), 1)
        c2 = _rational_curve(np.array([[1, 0, 0, 1], [3, 0, 0, 1]], dtype=np.float64), 1)
        with pytest.raises(ValueError, match="mismatch"):
            join(c1, c2, axis=0)

    def test_a_mixed_rational_and_polynomial_join_is_accepted_at_unit_weight(self) -> None:
        """Promotion gives the polynomial side unit weights, which the rational side has.

        This is the ordinary mixed join, and it must keep working: the check runs after
        ``_promote_to_rational``, so both rows are homogeneous by the time they meet.
        """
        c1 = _rational_curve(np.array([[0, 0, 0, 1], [1, 0, 0, 1]], dtype=np.float64), 1)
        c2 = create_line([1, 0, 0], [2, 0, 0])
        result = join(c1, c2, axis=0)
        assert result.is_rational
        domain = result.space.spaces[0].domain
        ends = np.asarray(result.evaluate(np.array([float(d) for d in domain])), dtype=np.float64)
        assert_allclose(ends, [[0, 0, 0], [2, 0, 0]], atol=1e-14)

    def test_a_mixed_join_whose_rational_side_carries_a_non_unit_weight_is_refused(self) -> None:
        """And the documented consequence of comparing homogeneously.

        The rational curve ends at the same point of space the polynomial one starts
        from, but carries weight 2 there while promotion gives the polynomial side
        weight 1.  Refusing is the same verdict ``_verify_edges_3d`` reaches on the same
        input, and the message reports a coefficient gap rather than a distance.
        """
        c1 = _rational_curve(np.array([[0, 0, 0, 1], [2, 0, 0, 2]], dtype=np.float64), 1)
        c2 = create_line([1, 0, 0], [2, 0, 0])
        with pytest.raises(ValueError, match=re.compile("mismatch")):
            join(c1, c2, axis=0)


class TestJoinPrecision:
    """A float32 model must survive a join, in both its knots and its control points."""

    @staticmethod
    def _line(start: Sequence[float], end: Sequence[float], dtype: npt.DTypeLike) -> Bspline:
        """Build a clamped degree-1 curve of the given dtype.

        ``create_line`` is float64-only, so a float32 curve has to be built directly.

        Args:
            start (Sequence[float]): First control point.
            end (Sequence[float]): Second control point.
            dtype (npt.DTypeLike): Floating-point dtype of both the space and the points.

        Returns:
            Bspline: A degree-1 curve on ``[0, 1]``.
        """
        knots = np.array([0.0, 0.0, 1.0, 1.0], dtype=dtype)
        space = BsplineSpace([BsplineSpace1D(knots, 1)])
        return Bspline(space, np.asarray([start, end], dtype=dtype))

    def test_a_float32_join_keeps_float32(self) -> None:
        """Joining two float32 curves must produce a float32 result.

        It used to raise ``The control points must have the same dtype as the B-spline
        space``, and had nothing to do with the seam, which here agrees bitwise: the
        junction knots were built from a Python float, so the merged knot vector promoted
        to float64 while the control points stayed float32.
        """
        c1 = self._line([0, 0, 0], [1, 0, 0], np.float32)
        c2 = self._line([1, 0, 0], [2, 0, 0], np.float32)
        result = join(c1, c2, axis=0)
        assert result.control_points.dtype == np.float32
        assert result.space.spaces[0].dtype == np.float32

    @pytest.mark.parametrize("increment", [1, 2, 3])
    def test_a_float32_join_needing_degree_elevation_keeps_float32(self, increment: int) -> None:
        """The dtype has to survive what ``_prepare_for_join`` runs, not only the merge.

        The fix reads the incoming knots' dtype, so it is only right if ``elevate_degree``
        preserves float32 too.  At ``increment >= 1`` the joined degree exceeds 1, which
        also puts the result through ``_try_remove_junction_knots``.
        """
        c1 = self._line([0, 0, 0], [1, 0, 0], np.float32)
        c2 = self._line([1, 0, 0], [2, 0, 0], np.float32).elevate_degree(increment)
        result = join(c1, c2, axis=0)
        assert result.control_points.dtype == np.float32
        assert result.space.spaces[0].dtype == np.float32

    def test_a_float32_join_needing_knot_insertion_keeps_float32(self) -> None:
        """And the same for a knot mismatch on the join axis."""
        c1 = self._line([0, 0, 0], [1, 0, 0], np.float32)
        c2 = self._line([1, 0, 0], [2, 0, 0], np.float32).insert_knots(
            np.array([0.5], dtype=np.float32)
        )
        result = join(c1, c2, axis=0)
        assert result.control_points.dtype == np.float32
