"""Tests for CAD primitive functions."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pantr.cad import create_bilinear, create_circle, create_line, create_trilinear

_RANK_3D = 3


class TestLine:
    """Test the line primitive."""

    def test_default_line(self) -> None:
        """Test default line from origin to (1, 0, 0)."""
        crv = create_line()
        assert crv.dim == 1
        assert crv.degree == (1,)
        assert crv.rank == _RANK_3D
        assert not crv.is_rational
        assert_allclose(crv.control_points[0], [0.0, 0.0, 0.0])
        assert_allclose(crv.control_points[1], [1.0, 0.0, 0.0])

    def test_custom_3d_endpoints(self) -> None:
        """Test line with explicit 3D endpoints."""
        crv = create_line([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert_allclose(crv.control_points[0], [1.0, 2.0, 3.0])
        assert_allclose(crv.control_points[1], [4.0, 5.0, 6.0])

    def test_2d_endpoints_zero_padded(self) -> None:
        """Test that 2D endpoints are zero-padded to 3D."""
        crv = create_line([0, 0], [1, 1])
        assert crv.rank == _RANK_3D
        assert_allclose(crv.control_points[0], [0.0, 0.0, 0.0])
        assert_allclose(crv.control_points[1], [1.0, 1.0, 0.0])

    def test_1d_endpoints(self) -> None:
        """Test that 1D endpoints are zero-padded to 3D."""
        crv = create_line([0], [5])
        assert_allclose(crv.control_points[0], [0.0, 0.0, 0.0])
        assert_allclose(crv.control_points[1], [5.0, 0.0, 0.0])

    def test_evaluate_midpoint(self) -> None:
        """Test that midpoint evaluates to the average of endpoints."""
        crv = create_line([0, 0, 0], [2, 4, 6])
        mid = crv.evaluate(np.array([0.5]))
        assert_allclose(mid, [1.0, 2.0, 3.0])

    def test_evaluate_endpoints(self) -> None:
        """Test evaluation at parameter boundaries."""
        crv = create_line([1, 2, 3], [4, 5, 6])
        pts = crv.evaluate(np.array([0.0, 1.0]))
        assert_allclose(pts[0], [1.0, 2.0, 3.0])
        assert_allclose(pts[1], [4.0, 5.0, 6.0])

    def test_knot_vector(self) -> None:
        """Test that the knot vector is [0, 0, 1, 1]."""
        crv = create_line()
        assert_allclose(crv.space.spaces[0].knots, [0.0, 0.0, 1.0, 1.0])

    def test_point_too_long_raises(self) -> None:
        """Test that a point with more than 3 coordinates raises."""
        with pytest.raises(ValueError, match="at most 3"):
            create_line([1, 2, 3, 4], [0, 0, 0])


class TestBilinear:
    """Test the bilinear surface primitive."""

    def test_default_bilinear(self) -> None:
        """Test default unit square surface."""
        srf = create_bilinear()
        assert srf.dim == 2
        assert srf.degree == (1, 1)
        assert srf.rank == _RANK_3D
        assert not srf.is_rational

    def test_default_corners(self) -> None:
        """Test that the default corners form a unit square in XY."""
        srf = create_bilinear()
        cp = srf.control_points
        assert_allclose(cp[0, 0], [-0.5, -0.5, 0.0])
        assert_allclose(cp[1, 0], [+0.5, -0.5, 0.0])
        assert_allclose(cp[0, 1], [-0.5, +0.5, 0.0])
        assert_allclose(cp[1, 1], [+0.5, +0.5, 0.0])

    def test_custom_corners(self) -> None:
        """Test bilinear surface with custom 3D corners."""
        corners = np.array(
            [
                [[0, 0, 0], [0, 1, 0]],
                [[1, 0, 0], [1, 1, 1]],
            ],
            dtype=np.float64,
        )
        srf = create_bilinear(corners)
        assert_allclose(srf.control_points[1, 1], [1.0, 1.0, 1.0])

    def test_2d_corners_padded(self) -> None:
        """Test that 2D corners are zero-padded to 3D."""
        corners = np.array(
            [
                [[0, 0], [0, 1]],
                [[1, 0], [1, 1]],
            ],
            dtype=np.float64,
        )
        srf = create_bilinear(corners)
        assert srf.rank == _RANK_3D
        assert_allclose(srf.control_points[1, 1], [1.0, 1.0, 0.0])

    def test_evaluate_center(self) -> None:
        """Test evaluation at the center of the default patch."""
        srf = create_bilinear()
        pts = srf.evaluate(np.array([[0.5, 0.5]]))
        assert_allclose(pts, [0.0, 0.0, 0.0], atol=1e-15)

    def test_evaluate_corners(self) -> None:
        """Test evaluation at the four parametric corners."""
        corners = np.array(
            [
                [[0, 0, 0], [0, 2, 0]],
                [[3, 0, 0], [3, 2, 0]],
            ],
            dtype=np.float64,
        )
        srf = create_bilinear(corners)
        pts = srf.evaluate(
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ]
            )
        )
        assert_allclose(pts[0], [0.0, 0.0, 0.0])
        assert_allclose(pts[1], [3.0, 0.0, 0.0])
        assert_allclose(pts[2], [0.0, 2.0, 0.0])
        assert_allclose(pts[3], [3.0, 2.0, 0.0])

    def test_wrong_shape_raises(self) -> None:
        """Test that corners with wrong shape raises ValueError."""
        with pytest.raises(ValueError, match="shape"):
            create_bilinear(np.ones((3, 2, 3)))

    def test_rank_too_large_raises(self) -> None:
        """Test that corners with rank > 3 raises ValueError."""
        with pytest.raises(ValueError, match="rank"):
            create_bilinear(np.ones((2, 2, 4)))

    def test_knot_vectors(self) -> None:
        """Test that knot vectors are [0, 0, 1, 1] in each direction."""
        srf = create_bilinear()
        for sp in srf.space.spaces:
            assert_allclose(sp.knots, [0.0, 0.0, 1.0, 1.0])


class TestTrilinear:
    """Test the trilinear volume primitive."""

    def test_default_trilinear(self) -> None:
        """Test default unit cube volume."""
        vol = create_trilinear()
        assert vol.dim == _RANK_3D
        assert vol.degree == (1, 1, 1)
        assert vol.rank == _RANK_3D
        assert not vol.is_rational

    def test_default_corners(self) -> None:
        """Test that the default corners form a unit cube."""
        vol = create_trilinear()
        cp = vol.control_points
        assert_allclose(cp[0, 0, 0], [-0.5, -0.5, -0.5])
        assert_allclose(cp[1, 1, 1], [+0.5, +0.5, +0.5])
        assert_allclose(cp[1, 0, 0], [+0.5, -0.5, -0.5])
        assert_allclose(cp[0, 1, 0], [-0.5, +0.5, -0.5])

    def test_custom_corners(self) -> None:
        """Test trilinear volume with custom corners."""
        corners = np.zeros((2, 2, 2, 3), dtype=np.float64)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    corners[i, j, k] = [i, j, k]
        vol = create_trilinear(corners)
        assert_allclose(vol.control_points[0, 0, 0], [0.0, 0.0, 0.0])
        assert_allclose(vol.control_points[1, 1, 1], [1.0, 1.0, 1.0])

    def test_evaluate_center(self) -> None:
        """Test evaluation at the center of the default cube."""
        vol = create_trilinear()
        pts = vol.evaluate(np.array([[0.5, 0.5, 0.5]]))
        assert_allclose(pts, [0.0, 0.0, 0.0], atol=1e-15)

    def test_evaluate_corners(self) -> None:
        """Test evaluation at the eight parametric corners."""
        corners = np.zeros((2, 2, 2, 3), dtype=np.float64)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    corners[i, j, k] = [float(i), float(j), float(k)]
        vol = create_trilinear(corners)
        pts = vol.evaluate(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            )
        )
        assert_allclose(pts[0], [0.0, 0.0, 0.0])
        assert_allclose(pts[1], [1.0, 0.0, 0.0])
        assert_allclose(pts[2], [0.0, 1.0, 0.0])
        assert_allclose(pts[3], [0.0, 0.0, 1.0])
        assert_allclose(pts[4], [1.0, 1.0, 1.0])

    def test_2d_corners_padded(self) -> None:
        """Test that 2D corners are zero-padded to 3D."""
        corners = np.ones((2, 2, 2, 2), dtype=np.float64)
        vol = create_trilinear(corners)
        assert vol.rank == _RANK_3D
        assert_allclose(vol.control_points[0, 0, 0, 2], 0.0)

    def test_wrong_shape_raises(self) -> None:
        """Test that corners with wrong shape raises ValueError."""
        with pytest.raises(ValueError, match="shape"):
            create_trilinear(np.ones((2, 2, 3, 3)))

    def test_rank_too_large_raises(self) -> None:
        """Test that corners with rank > 3 raises ValueError."""
        with pytest.raises(ValueError, match="rank"):
            create_trilinear(np.ones((2, 2, 2, 4)))

    def test_knot_vectors(self) -> None:
        """Test that knot vectors are [0, 0, 1, 1] in each direction."""
        vol = create_trilinear()
        for sp in vol.space.spaces:
            assert_allclose(sp.knots, [0.0, 0.0, 1.0, 1.0])


class TestCircle:
    """Test the circle / circular arc primitive."""

    def test_full_circle_properties(self) -> None:
        """Test that the default full circle has the expected structure."""
        crv = create_circle()
        assert crv.dim == 1
        assert crv.degree == (_RANK_3D - 1,)  # degree 2
        assert crv.rank == _RANK_3D
        assert crv.is_rational

    def test_full_circle_control_point_count(self) -> None:
        """Test full circle has 9 control points (4 spans)."""
        crv = create_circle()
        assert crv.control_points.shape[0] == 9

    def test_full_circle_knot_structure(self) -> None:
        """Test full circle knot vector: 4 spans, double interior knots."""
        crv = create_circle()
        sp = crv.space.spaces[0]
        knots = sp.knots
        # [0,0,0, 0.25,0.25, 0.5,0.5, 0.75,0.75, 1,1,1]
        expected = np.array([0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1])
        assert_allclose(knots, expected)

    def test_full_circle_evaluate_cardinal_points(self) -> None:
        """Test full circle evaluates to correct cardinal points."""
        crv = create_circle()
        pts = crv.evaluate(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
        assert_allclose(pts[0], [1, 0, 0], atol=1e-14)
        assert_allclose(pts[1], [0, 1, 0], atol=1e-14)
        assert_allclose(pts[2], [-1, 0, 0], atol=1e-14)
        assert_allclose(pts[3], [0, -1, 0], atol=1e-14)
        assert_allclose(pts[4], [1, 0, 0], atol=1e-14)

    def test_full_circle_points_on_circle(self) -> None:
        """Test that evaluated points lie on the unit circle."""
        crv = create_circle()
        t = np.linspace(0, 1, 50)
        pts = crv.evaluate(t)
        radii = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        assert_allclose(radii, 1.0, atol=1e-14)
        assert_allclose(pts[:, 2], 0.0, atol=1e-14)

    def test_radius(self) -> None:
        """Test circle with custom radius."""
        r = 3.5
        crv = create_circle(radius=r)
        t = np.linspace(0, 1, 30)
        pts = crv.evaluate(t)
        radii = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        assert_allclose(radii, r, atol=1e-14)

    def test_quarter_arc_structure(self) -> None:
        """Test arc <= 90 deg has 1 span, 3 control points, no interior knots."""
        crv = create_circle(angle=np.pi / 2)
        assert crv.control_points.shape[0] == _RANK_3D  # 3 control points
        expected_knots = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
        assert_allclose(crv.space.spaces[0].knots, expected_knots)

    def test_quarter_arc_endpoints(self) -> None:
        """Test quarter arc from 0 to pi/2."""
        crv = create_circle(angle=np.pi / 2)
        pts = crv.evaluate(np.array([0.0, 1.0]))
        assert_allclose(pts[0], [1, 0, 0], atol=1e-14)
        assert_allclose(pts[1], [0, 1, 0], atol=1e-14)

    def test_half_circle_structure(self) -> None:
        """Test arc <= 180 deg has 2 spans, 5 control points, 1 double knot."""
        crv = create_circle(angle=np.pi)
        assert crv.control_points.shape[0] == 5
        expected_knots = np.array([0, 0, 0, 0.5, 0.5, 1, 1, 1])
        assert_allclose(crv.space.spaces[0].knots, expected_knots)

    def test_three_quarter_arc_structure(self) -> None:
        """Test arc <= 270 deg has 3 spans, 7 control points, 2 double knots."""
        crv = create_circle(angle=3 * np.pi / 2)
        assert crv.control_points.shape[0] == 7

    def test_arc_points_on_circle(self) -> None:
        """Test that arc points lie on the circle."""
        crv = create_circle(radius=2.0, angle=np.pi)
        t = np.linspace(0, 1, 30)
        pts = crv.evaluate(t)
        radii = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        assert_allclose(radii, 2.0, atol=1e-13)

    def test_angle_tuple(self) -> None:
        """Test arc with (start, end) angle tuple."""
        crv = create_circle(radius=2, angle=(np.pi / 2, -np.pi / 2))
        pts = crv.evaluate(np.array([0.0, 0.5, 1.0]))
        assert_allclose(pts[0], [0, 2, 0], atol=1e-14)
        assert_allclose(pts[1], [2, 0, 0], atol=1e-14)
        assert_allclose(pts[2], [0, -2, 0], atol=1e-14)

    def test_center_translation(self) -> None:
        """Test that center translates the circle."""
        crv = create_circle(radius=1, center=(1, 2, 0))
        pt = crv.evaluate(np.array([0.0]))
        assert_allclose(pt, [2, 2, 0], atol=1e-14)

    def test_center_2d(self) -> None:
        """Test center with 2D coordinates (zero-padded to 3D)."""
        crv = create_circle(radius=3, center=2, angle=np.pi / 2)
        pts = crv.evaluate(np.array([0.0, 1.0]))
        assert_allclose(pts[0], [5, 0, 0], atol=1e-14)
        assert_allclose(pts[1], [2, 3, 0], atol=1e-14)

    def test_negative_sweep(self) -> None:
        """Test arc with negative sweep (clockwise)."""
        crv = create_circle(angle=-np.pi / 2)
        pts = crv.evaluate(np.array([0.0, 1.0]))
        assert_allclose(pts[0], [1, 0, 0], atol=1e-14)
        assert_allclose(pts[1], [0, -1, 0], atol=1e-14)
