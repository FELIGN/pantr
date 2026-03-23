"""Tests for derived CAD primitives: rectangle, disk, cylinder."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from pantr.cad import cylinder, disk, rectangle

_RANK_3D = 3


class TestRectangle:
    """Test the rectangle primitive."""

    def test_default_rectangle(self) -> None:
        """Test default unit rectangle."""
        rect = rectangle()
        assert rect.dim == 1
        assert rect.rank == _RANK_3D
        assert rect.degree == (1,)
        assert not rect.is_rational

    def test_closed_curve(self) -> None:
        """Test that the rectangle is a closed curve."""
        rect = rectangle()
        domain = rect.space.spaces[0].domain
        pt_start = rect.evaluate(np.array([float(domain[0])]))
        pt_end = rect.evaluate(np.array([float(domain[1])]))
        assert_allclose(pt_start, pt_end, atol=1e-14)

    def test_custom_rectangle(self) -> None:
        """Test rectangle with custom corner, width, height."""
        rect = rectangle(corner=[1, 2, 0], width=3, height=4)
        domain = rect.space.spaces[0].domain
        t = np.linspace(float(domain[0]), float(domain[1]), 5)
        pts = rect.evaluate(t)
        # First point = corner, last = same (closed)
        assert_allclose(pts[0], [1, 2, 0], atol=1e-14)
        assert_allclose(pts[-1], [1, 2, 0], atol=1e-14)
        # Second point should be corner + (3, 0, 0)
        assert_allclose(pts[1], [4, 2, 0], atol=1e-14)

    def test_rectangle_four_sides(self) -> None:
        """Test that the rectangle visits all four corners."""
        rect = rectangle(corner=[0, 0, 0], width=2, height=3)
        domain = rect.space.spaces[0].domain
        t = np.linspace(float(domain[0]), float(domain[1]), 5)
        pts = rect.evaluate(t)
        assert_allclose(pts[0], [0, 0, 0], atol=1e-14)
        assert_allclose(pts[1], [2, 0, 0], atol=1e-14)
        assert_allclose(pts[2], [2, 3, 0], atol=1e-14)
        assert_allclose(pts[3], [0, 3, 0], atol=1e-14)
        assert_allclose(pts[4], [0, 0, 0], atol=1e-14)


class TestDisk:
    """Test the disk primitive."""

    def test_full_disk(self) -> None:
        """Test a full disk (inner radius = 0)."""
        d = disk(radius_outer=2.0)
        assert d.dim == 2  # noqa: PLR2004
        assert d.is_rational

    def test_annulus(self) -> None:
        """Test an annular sector."""
        ann = disk(radius_inner=1.0, radius_outer=2.0)
        assert ann.dim == 2  # noqa: PLR2004
        assert ann.is_rational

    def test_disk_points_within_radius(self) -> None:
        """Test that disk points lie within the outer radius."""
        d = disk(radius_outer=3.0)
        u = np.linspace(0, 1, 15)
        v = np.linspace(0, 1, 5)
        params = np.array([[ui, vi] for ui in u for vi in v])
        pts = d.evaluate(params)
        radii = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        assert np.all(radii <= 3.0 + 1e-12)

    def test_annulus_inner_boundary(self) -> None:
        """Test that annulus inner boundary has correct radius."""
        ann = disk(radius_inner=1.0, radius_outer=2.0)
        u = np.linspace(0, 1, 20)
        params_inner = np.column_stack([u, np.zeros_like(u)])
        pts = ann.evaluate(params_inner)
        radii = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        assert_allclose(radii, 1.0, atol=1e-13)

    def test_annulus_outer_boundary(self) -> None:
        """Test that annulus outer boundary has correct radius."""
        ann = disk(radius_inner=1.0, radius_outer=2.0)
        u = np.linspace(0, 1, 20)
        params_outer = np.column_stack([u, np.ones_like(u)])
        pts = ann.evaluate(params_outer)
        radii = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        assert_allclose(radii, 2.0, atol=1e-13)

    def test_disk_with_center(self) -> None:
        """Test disk with offset center."""
        d = disk(radius_outer=1.0, center=[5, 0, 0])
        pt = d.evaluate(np.array([[0.0, 1.0]]))
        # At v=1 (outer), u=0: should be at (6, 0, 0)
        assert_allclose(pt, [6, 0, 0], atol=1e-13)

    def test_disk_with_angle(self) -> None:
        """Test partial disk (sector)."""
        d = disk(radius_outer=1.0, angle=np.pi / 2)
        assert d.dim == 2  # noqa: PLR2004


class TestCylinder:
    """Test the cylinder primitive."""

    def test_default_cylinder(self) -> None:
        """Test default cylinder."""
        cyl = cylinder()
        assert cyl.dim == 2  # noqa: PLR2004
        assert cyl.is_rational

    def test_cylinder_radius(self) -> None:
        """Test that cylinder points lie on the correct radius."""
        r = 2.5
        cyl = cylinder(radius=r, height=3.0)
        u = np.linspace(0, 1, 20)
        v = np.array([0.0, 0.5, 1.0])
        params = np.array([[ui, vi] for ui in u for vi in v])
        pts = cyl.evaluate(params)
        radii = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        assert_allclose(radii, r, atol=1e-13)

    def test_cylinder_height(self) -> None:
        """Test that cylinder z-range matches height."""
        cyl = cylinder(radius=1.0, height=7.0)
        # At v=0, z should be 0; at v=1, z should be 7
        pt_bottom = cyl.evaluate(np.array([[0.0, 0.0]]))
        pt_top = cyl.evaluate(np.array([[0.0, 1.0]]))
        assert_allclose(pt_bottom[2], 0.0, atol=1e-14)
        assert_allclose(pt_top[2], 7.0, atol=1e-13)

    def test_cylinder_with_center(self) -> None:
        """Test cylinder with offset center."""
        cyl = cylinder(radius=1.0, height=1.0, center=[3, 0, 0])
        pt = cyl.evaluate(np.array([[0.0, 0.0]]))
        assert_allclose(pt, [4, 0, 0], atol=1e-14)

    def test_cylinder_partial_angle(self) -> None:
        """Test partial cylinder (angular sector)."""
        cyl = cylinder(radius=1.0, height=1.0, angle=np.pi / 2)
        assert cyl.dim == 2  # noqa: PLR2004
