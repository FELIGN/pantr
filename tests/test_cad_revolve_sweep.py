"""Tests for CAD operations: revolve and sweep."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
from pantr.cad import create_bilinear, create_circle, create_line, extrude, revolve, sweep

_RANK_3D = 3


class TestRevolve:
    """Test the revolve operation."""

    def test_line_to_annulus(self) -> None:
        """Test revolving a radial line produces a full annulus."""
        crv = create_line([1, 0, 0], [2, 0, 0])
        srf = revolve(crv, point=0, axis=2)
        assert srf.dim == 2  # noqa: PLR2004
        assert srf.is_rational

    def test_revolve_points_on_cylinder(self) -> None:
        """Test revolving a vertical line around Z produces a cylinder."""
        crv = create_line([1, 0, 0], [1, 0, 3])
        srf = revolve(crv, point=0, axis=2)
        # Evaluate on a grid
        u = np.linspace(0, 1, 10)
        v = np.linspace(0, 1, 20)
        params = np.array([[ui, vi] for ui in u for vi in v])
        pts = srf.evaluate(params)
        radii = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        assert_allclose(radii, 1.0, atol=1e-12)

    def test_revolve_quarter_turn(self) -> None:
        """Test revolving a line by pi/2 around Z."""
        crv = create_line([1, 0, 0], [2, 0, 0])
        srf = revolve(crv, point=0, axis=2, angle=np.pi / 2)
        # At v=0 should be on +X axis, at v=1 on +Y axis
        pt_start = srf.evaluate(np.array([[0.5, 0.0]]))
        pt_end = srf.evaluate(np.array([[0.5, 1.0]]))
        r = 1.5  # midpoint of [1, 2]
        assert_allclose(pt_start, [r, 0, 0], atol=1e-12)
        assert_allclose(pt_end, [0, r, 0], atol=1e-12)

    def test_revolve_around_y_axis(self) -> None:
        """Test revolving around the Y axis."""
        crv = create_line([1, 0, 0], [1, 0, 1])
        srf = revolve(crv, point=0, axis=1, angle=np.pi / 2)
        # At v=1, (x, z) should rotate to (z, -x) for Y-axis rotation
        pt = srf.evaluate(np.array([[0.0, 1.0]]))
        assert_allclose(pt, [0, 0, -1], atol=1e-12)

    def test_revolve_with_center(self) -> None:
        """Test revolving around an offset point."""
        crv = create_line([3, 0, 0], [3, 0, 1])
        srf = revolve(crv, point=[2, 0, 0], axis=2)
        # At v=0, should be at original position
        pt = srf.evaluate(np.array([[0.0, 0.0]]))
        assert_allclose(pt, [3, 0, 0], atol=1e-12)
        # All points should be distance 1 from (2, 0, z)
        u = np.linspace(0, 1, 5)
        v = np.linspace(0, 1, 20)
        params = np.array([[ui, vi] for ui in u for vi in v])
        pts = srf.evaluate(params)
        radii = np.sqrt((pts[:, 0] - 2) ** 2 + pts[:, 1] ** 2)
        assert_allclose(radii, 1.0, atol=1e-12)

    def test_revolve_negative_angle(self) -> None:
        """Test revolving with a negative angle (clockwise)."""
        crv = create_line([1, 0, 0], [2, 0, 0])
        srf = revolve(crv, point=0, axis=2, angle=-np.pi / 2)
        pt = srf.evaluate(np.array([[0.5, 1.0]]))
        r = 1.5
        assert_allclose(pt, [0, -r, 0], atol=1e-12)

    def test_revolve_angle_tuple(self) -> None:
        """Test revolving with (start, end) angle tuple."""
        crv = create_line([1, 0, 0], [2, 0, 0])
        srf = revolve(crv, point=0, axis=2, angle=(np.pi / 2, np.pi))
        # At v=0, should be at angle pi/2 (on +Y axis)
        pt0 = srf.evaluate(np.array([[0.5, 0.0]]))
        r = 1.5
        assert_allclose(pt0, [0, r, 0], atol=1e-12)
        # At v=1, should be at angle pi (on -X axis)
        pt1 = srf.evaluate(np.array([[0.5, 1.0]]))
        assert_allclose(pt1, [-r, 0, 0], atol=1e-12)

    def test_revolve_dim_too_large_raises(self) -> None:
        """Test that revolving a volume raises ValueError."""
        vol = extrude(create_bilinear(), [0, 0, 1])
        with pytest.raises(ValueError, match="dim"):
            revolve(vol, point=0, axis=2)

    def test_revolve_wrong_rank_raises(self) -> None:
        """Test that rank != 3 raises ValueError."""
        knots = np.array([0.0, 0.0, 1.0, 1.0])
        sp = BsplineSpace([BsplineSpace1D(knots, 1)])
        crv = Bspline(sp, np.array([[0.0, 0.0], [1.0, 1.0]]))
        with pytest.raises(ValueError, match="rank"):
            revolve(crv, point=0, axis=2)


class TestSweep:
    """Test the translational sweep operation."""

    def test_sweep_line_along_line(self) -> None:
        """Test sweeping a line along a line produces a bilinear surface."""
        section = create_line([0, 0, 0], [1, 0, 0])
        trajectory = create_line([0, 0, 0], [0, 1, 0])
        srf = sweep(section, trajectory)
        assert srf.dim == 2  # noqa: PLR2004
        assert srf.rank == _RANK_3D
        # Evaluate corners
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
        assert_allclose(pts[1], [1, 0, 0], atol=1e-14)
        assert_allclose(pts[2], [0, 1, 0], atol=1e-14)
        assert_allclose(pts[3], [1, 1, 0], atol=1e-14)

    def test_sweep_preserves_section_geometry(self) -> None:
        """Test that at trajectory start the section geometry is preserved."""
        section = create_circle(radius=2.0, angle=np.pi / 2)
        trajectory = create_line([0, 0, 0], [0, 0, 5])
        srf = sweep(section, trajectory)
        # At v=0 (trajectory start), should match the section
        u = np.linspace(0, 1, 20)
        params_v0 = np.column_stack([u, np.zeros_like(u)])
        pts = srf.evaluate(params_v0)
        section_pts = section.evaluate(u)
        assert_allclose(pts, section_pts, atol=1e-13)

    def test_sweep_trajectory_offset(self) -> None:
        """Test that the trajectory adds an offset."""
        section = create_line([0, 0, 0], [1, 0, 0])
        trajectory = create_line([0, 0, 0], [0, 0, 3])
        srf = sweep(section, trajectory)
        pt = srf.evaluate(np.array([[0.5, 1.0]]))
        assert_allclose(pt, [0.5, 0, 3], atol=1e-14)

    def test_sweep_section_dim2_raises_for_dim3(self) -> None:
        """Test that section dim > 2 raises ValueError."""
        vol = extrude(create_bilinear(), [0, 0, 1])
        traj = create_line([0, 0, 0], [1, 0, 0])
        with pytest.raises(ValueError, match=r"section\.dim"):
            sweep(vol, traj)

    def test_sweep_trajectory_dim2_raises(self) -> None:
        """Test that trajectory dim != 1 raises ValueError."""
        sec = create_line([0, 0, 0], [1, 0, 0])
        traj = create_bilinear()
        with pytest.raises(ValueError, match=r"trajectory\.dim"):
            sweep(sec, traj)
