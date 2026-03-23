"""Tests for CAD primitive functions: line, bilinear, trilinear."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pantr.cad import bilinear, line, trilinear

_RANK_3D = 3


class TestLine:
    """Test the line primitive."""

    def test_default_line(self) -> None:
        """Test default line from origin to (1, 0, 0)."""
        crv = line()
        assert crv.dim == 1
        assert crv.degree == (1,)
        assert crv.rank == _RANK_3D
        assert not crv.is_rational
        assert_allclose(crv.control_points[0], [0.0, 0.0, 0.0])
        assert_allclose(crv.control_points[1], [1.0, 0.0, 0.0])

    def test_custom_3d_endpoints(self) -> None:
        """Test line with explicit 3D endpoints."""
        crv = line([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert_allclose(crv.control_points[0], [1.0, 2.0, 3.0])
        assert_allclose(crv.control_points[1], [4.0, 5.0, 6.0])

    def test_2d_endpoints_zero_padded(self) -> None:
        """Test that 2D endpoints are zero-padded to 3D."""
        crv = line([0, 0], [1, 1])
        assert crv.rank == _RANK_3D
        assert_allclose(crv.control_points[0], [0.0, 0.0, 0.0])
        assert_allclose(crv.control_points[1], [1.0, 1.0, 0.0])

    def test_1d_endpoints(self) -> None:
        """Test that 1D endpoints are zero-padded to 3D."""
        crv = line([0], [5])
        assert_allclose(crv.control_points[0], [0.0, 0.0, 0.0])
        assert_allclose(crv.control_points[1], [5.0, 0.0, 0.0])

    def test_evaluate_midpoint(self) -> None:
        """Test that midpoint evaluates to the average of endpoints."""
        crv = line([0, 0, 0], [2, 4, 6])
        mid = crv.evaluate(np.array([0.5]))
        assert_allclose(mid, [1.0, 2.0, 3.0])

    def test_evaluate_endpoints(self) -> None:
        """Test evaluation at parameter boundaries."""
        crv = line([1, 2, 3], [4, 5, 6])
        pts = crv.evaluate(np.array([0.0, 1.0]))
        assert_allclose(pts[0], [1.0, 2.0, 3.0])
        assert_allclose(pts[1], [4.0, 5.0, 6.0])

    def test_knot_vector(self) -> None:
        """Test that the knot vector is [0, 0, 1, 1]."""
        crv = line()
        assert_allclose(crv.space.spaces[0].knots, [0.0, 0.0, 1.0, 1.0])

    def test_point_too_long_raises(self) -> None:
        """Test that a point with more than 3 coordinates raises."""
        with pytest.raises(ValueError, match="at most 3"):
            line([1, 2, 3, 4], [0, 0, 0])


class TestBilinear:
    """Test the bilinear surface primitive."""

    def test_default_bilinear(self) -> None:
        """Test default unit square surface."""
        srf = bilinear()
        assert srf.dim == 2  # noqa: PLR2004
        assert srf.degree == (1, 1)
        assert srf.rank == _RANK_3D
        assert not srf.is_rational

    def test_default_corners(self) -> None:
        """Test that the default corners form a unit square in XY."""
        srf = bilinear()
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
        srf = bilinear(corners)
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
        srf = bilinear(corners)
        assert srf.rank == _RANK_3D
        assert_allclose(srf.control_points[1, 1], [1.0, 1.0, 0.0])

    def test_evaluate_center(self) -> None:
        """Test evaluation at the center of the default patch."""
        srf = bilinear()
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
        srf = bilinear(corners)
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
            bilinear(np.ones((3, 2, 3)))

    def test_rank_too_large_raises(self) -> None:
        """Test that corners with rank > 3 raises ValueError."""
        with pytest.raises(ValueError, match="rank"):
            bilinear(np.ones((2, 2, 4)))

    def test_knot_vectors(self) -> None:
        """Test that knot vectors are [0, 0, 1, 1] in each direction."""
        srf = bilinear()
        for sp in srf.space.spaces:
            assert_allclose(sp.knots, [0.0, 0.0, 1.0, 1.0])


class TestTrilinear:
    """Test the trilinear volume primitive."""

    def test_default_trilinear(self) -> None:
        """Test default unit cube volume."""
        vol = trilinear()
        assert vol.dim == _RANK_3D
        assert vol.degree == (1, 1, 1)
        assert vol.rank == _RANK_3D
        assert not vol.is_rational

    def test_default_corners(self) -> None:
        """Test that the default corners form a unit cube."""
        vol = trilinear()
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
        vol = trilinear(corners)
        assert_allclose(vol.control_points[0, 0, 0], [0.0, 0.0, 0.0])
        assert_allclose(vol.control_points[1, 1, 1], [1.0, 1.0, 1.0])

    def test_evaluate_center(self) -> None:
        """Test evaluation at the center of the default cube."""
        vol = trilinear()
        pts = vol.evaluate(np.array([[0.5, 0.5, 0.5]]))
        assert_allclose(pts, [0.0, 0.0, 0.0], atol=1e-15)

    def test_evaluate_corners(self) -> None:
        """Test evaluation at the eight parametric corners."""
        corners = np.zeros((2, 2, 2, 3), dtype=np.float64)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    corners[i, j, k] = [float(i), float(j), float(k)]
        vol = trilinear(corners)
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
        vol = trilinear(corners)
        assert vol.rank == _RANK_3D
        assert_allclose(vol.control_points[0, 0, 0, 2], 0.0)

    def test_wrong_shape_raises(self) -> None:
        """Test that corners with wrong shape raises ValueError."""
        with pytest.raises(ValueError, match="shape"):
            trilinear(np.ones((2, 2, 3, 3)))

    def test_rank_too_large_raises(self) -> None:
        """Test that corners with rank > 3 raises ValueError."""
        with pytest.raises(ValueError, match="rank"):
            trilinear(np.ones((2, 2, 2, 4)))

    def test_knot_vectors(self) -> None:
        """Test that knot vectors are [0, 0, 1, 1] in each direction."""
        vol = trilinear()
        for sp in vol.space.spaces:
            assert_allclose(sp.knots, [0.0, 0.0, 1.0, 1.0])
