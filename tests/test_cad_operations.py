"""Tests for CAD operations: extrude and ruled."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pantr.cad import create_bilinear, create_circle, create_line, create_ruled, extrude

_RANK_3D = 3


class TestExtrude:
    """Test the extrude operation."""

    def test_line_to_surface(self) -> None:
        """Test extruding a line into a surface."""
        crv = create_line([0, 0, 0], [1, 0, 0])
        srf = extrude(crv, [0, 0, 1])
        assert srf.dim == 2  # noqa: PLR2004
        assert srf.rank == _RANK_3D
        assert not srf.is_rational

    def test_extrude_degree(self) -> None:
        """Test that the new direction has degree 1."""
        crv = create_line([0, 0, 0], [1, 0, 0])
        srf = extrude(crv, [0, 1, 0])
        assert srf.degree == (1, 1)

    def test_extrude_evaluate_corners(self) -> None:
        """Test evaluation at the four corners of an extruded line."""
        crv = create_line([0, 0, 0], [2, 0, 0])
        srf = extrude(crv, [0, 3, 0])
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
        assert_allclose(pts[0], [0, 0, 0], atol=1e-14)
        assert_allclose(pts[1], [2, 0, 0], atol=1e-14)
        assert_allclose(pts[2], [0, 3, 0], atol=1e-14)
        assert_allclose(pts[3], [2, 3, 0], atol=1e-14)

    def test_extrude_circle_to_cylinder(self) -> None:
        """Test extruding a circle produces a cylinder."""
        crv = create_circle(radius=2.0)
        cyl = extrude(crv, [0, 0, 5])
        assert cyl.dim == 2  # noqa: PLR2004
        assert cyl.is_rational
        # Evaluate on a grid and check all points lie on the cylinder
        u = np.linspace(0, 1, 20)
        v = np.array([0.0, 0.5, 1.0])
        params = np.array([[ui, vi] for ui in u for vi in v])
        pts = cyl.evaluate(params)
        radii = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        assert_allclose(radii, 2.0, atol=1e-13)
        # z should range from 0 to 5
        assert_allclose(pts[params[:, 1] == 0.0, 2], 0.0, atol=1e-14)
        assert_allclose(pts[params[:, 1] == 1.0, 2], 5.0, atol=1e-13)

    def test_extrude_surface_to_volume(self) -> None:
        """Test extruding a surface into a volume."""
        srf = create_bilinear()
        vol = extrude(srf, [0, 0, 1])
        assert vol.dim == _RANK_3D
        assert vol.degree == (1, 1, 1)

    def test_extrude_volume_raises(self) -> None:
        """Test that extruding a volume raises ValueError."""
        vol = extrude(create_bilinear(), [0, 0, 1])
        with pytest.raises(ValueError, match="dim"):
            extrude(vol, [0, 0, 1])

    def test_extrude_2d_displacement(self) -> None:
        """Test that 2D displacement is zero-padded to 3D."""
        crv = create_line([0, 0, 0], [1, 0, 0])
        srf = extrude(crv, [0, 1])
        pt = srf.evaluate(np.array([[0.5, 1.0]]))
        assert_allclose(pt, [0.5, 1.0, 0.0], atol=1e-14)


class TestRuled:
    """Test the ruled surface/volume construction."""

    def test_ruled_two_lines(self) -> None:
        """Test ruled surface between two parallel lines."""
        c1 = create_line([0, 0, 0], [1, 0, 0])
        c2 = create_line([0, 1, 0], [1, 1, 0])
        srf = create_ruled(c1, c2)
        assert srf.dim == 2  # noqa: PLR2004
        assert srf.degree == (1, 1)
        assert not srf.is_rational

    def test_ruled_evaluate_corners(self) -> None:
        """Test evaluation at the four parametric corners."""
        c1 = create_line([0, 0, 0], [3, 0, 0])
        c2 = create_line([0, 2, 0], [3, 2, 0])
        srf = create_ruled(c1, c2)
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
        assert_allclose(pts[0], [0, 0, 0], atol=1e-14)
        assert_allclose(pts[1], [3, 0, 0], atol=1e-14)
        assert_allclose(pts[2], [0, 2, 0], atol=1e-14)
        assert_allclose(pts[3], [3, 2, 0], atol=1e-14)

    def test_ruled_annulus(self) -> None:
        """Test ruled surface between two circles forms an annulus."""
        inner = create_circle(radius=1.0)
        outer = create_circle(radius=2.0)
        annulus = create_ruled(inner, outer)
        assert annulus.dim == 2  # noqa: PLR2004
        assert annulus.is_rational
        # Evaluate and check radii
        u = np.linspace(0, 1, 30)
        params_inner = np.column_stack([u, np.zeros_like(u)])
        params_outer = np.column_stack([u, np.ones_like(u)])
        pts_inner = annulus.evaluate(params_inner)
        pts_outer = annulus.evaluate(params_outer)
        radii_inner = np.sqrt(pts_inner[:, 0] ** 2 + pts_inner[:, 1] ** 2)
        radii_outer = np.sqrt(pts_outer[:, 0] ** 2 + pts_outer[:, 1] ** 2)
        assert_allclose(radii_inner, 1.0, atol=1e-13)
        assert_allclose(radii_outer, 2.0, atol=1e-13)

    def test_ruled_different_degrees(self) -> None:
        """Test that ruled makes inputs compatible (different degrees)."""
        c1 = create_line([0, 0, 0], [1, 0, 0])  # degree 1
        c2 = create_circle(angle=np.pi / 2)  # degree 2
        srf = create_ruled(c1, c2)
        # Both should have been elevated to degree 2
        assert srf.degree[0] == 2  # noqa: PLR2004

    def test_ruled_mixed_rational(self) -> None:
        """Test that mixing rational and non-rational promotes correctly."""
        c1 = create_line([1, 0, 0], [2, 0, 0])  # non-rational
        c2 = create_circle(radius=1.5, angle=np.pi / 4)  # rational
        srf = create_ruled(c1, c2)
        assert srf.is_rational

    def test_ruled_different_dim_raises(self) -> None:
        """Test that different dimensions raises ValueError."""
        crv = create_line([0, 0, 0], [1, 0, 0])
        srf = create_bilinear()
        with pytest.raises(ValueError, match="same dim"):
            create_ruled(crv, srf)

    def test_ruled_volume_from_surfaces(self) -> None:
        """Test ruled volume between two surfaces."""
        corners1 = np.array([[[0, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0]]], dtype=np.float64)
        corners2 = np.array([[[0, 0, 1], [0, 1, 1]], [[1, 0, 1], [1, 1, 1]]], dtype=np.float64)
        s1 = create_bilinear(corners1)
        s2 = create_bilinear(corners2)
        vol = create_ruled(s1, s2)
        assert vol.dim == _RANK_3D
        assert vol.degree == (1, 1, 1)
        # Center should be at (0.5, 0.5, 0.5)
        pt = vol.evaluate(np.array([[0.5, 0.5, 0.5]]))
        assert_allclose(pt, [0.5, 0.5, 0.5], atol=1e-14)
