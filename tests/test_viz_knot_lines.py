"""Tests for knot line computation and control point visualization."""

from __future__ import annotations

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from pantr.bezier import Bezier  # noqa: E402
from pantr.bspline import (  # noqa: E402
    Bspline,
    BsplineSpace,
    BsplineSpace1D,
    create_uniform_open,
)
from pantr.viz import control_points_mesh, control_polygon_mesh, knot_lines_meshes  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def bspline_curve() -> Bspline:
    """Quadratic B-spline curve in 3D with 3 elements."""
    space1d = BsplineSpace1D([0, 0, 0, 1, 2, 3, 3, 3], 2)
    space = BsplineSpace([space1d])
    cp = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.0, 0.5, 0.0],
            [1.5, 1.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    return Bspline(space, cp)


@pytest.fixture()
def bspline_surface() -> Bspline:
    """Biquadratic B-spline surface in 3D."""
    knots_u = create_uniform_open(3, 2, domain=(0.0, 1.0))
    knots_v = create_uniform_open(2, 2, domain=(0.0, 1.0))
    s_u = BsplineSpace1D(knots_u, 2)
    s_v = BsplineSpace1D(knots_v, 2)
    space = BsplineSpace([s_u, s_v])
    rng = np.random.default_rng(42)
    cp = rng.standard_normal((*space.num_basis, 3))
    return Bspline(space, cp)


@pytest.fixture()
def bezier_curve() -> Bezier:
    """Quadratic Bézier curve in 3D."""
    cp = np.array([[0.0, 0.0, 0.0], [0.5, 1.0, 0.0], [1.0, 0.0, 0.0]])
    return Bezier(cp)


@pytest.fixture()
def rational_bezier_curve() -> Bezier:
    """Rational quadratic Bézier (quarter circle)."""
    w = np.sqrt(2.0) / 2.0
    cp = np.array([[1.0, 0.0, 1.0], [w, w, w], [0.0, 1.0, 1.0]])
    return Bezier(cp, is_rational=True)


# ---------------------------------------------------------------------------
# Knot lines
# ---------------------------------------------------------------------------


class TestKnotLinesCurve:
    """Tests for knot line computation on 1D B-splines."""

    def test_returns_list(self, bspline_curve: Bspline) -> None:
        meshes = knot_lines_meshes(bspline_curve)
        assert isinstance(meshes, list)
        assert len(meshes) == 1  # single PolyData with all knot points

    def test_knot_point_count(self, bspline_curve: Bspline) -> None:
        meshes = knot_lines_meshes(bspline_curve)
        kp = meshes[0]
        # 3 elements → 2 interior knots
        n_interior = 2
        assert kp.n_points == n_interior

    def test_single_element_no_interior_knots(self) -> None:
        """Single element B-spline has no interior knots."""
        space1d = BsplineSpace1D([0, 0, 0, 1, 1, 1], 2)
        space = BsplineSpace([space1d])
        cp = np.array([[0.0, 0.0, 0.0], [0.5, 1.0, 0.0], [1.0, 0.0, 0.0]])
        bspline = Bspline(space, cp)
        meshes = knot_lines_meshes(bspline)
        assert meshes[0].n_points == 0


class TestKnotLinesSurface:
    """Tests for knot line computation on 2D B-splines."""

    def test_returns_list_of_grids(self, bspline_surface: Bspline) -> None:
        meshes = knot_lines_meshes(bspline_surface)
        assert isinstance(meshes, list)
        assert len(meshes) > 0

    def test_knot_line_count(self, bspline_surface: Bspline) -> None:
        meshes = knot_lines_meshes(bspline_surface)
        # 3 elements in u → 2 interior knots; 2 elements in v → 1 interior knot
        n_expected = 2 + 1
        assert len(meshes) == n_expected

    def test_knot_lines_are_curves(self, bspline_surface: Bspline) -> None:
        meshes = knot_lines_meshes(bspline_surface)
        for mesh in meshes:
            assert mesh.n_cells > 0


# ---------------------------------------------------------------------------
# Control points
# ---------------------------------------------------------------------------


class TestControlPointsMesh:
    """Tests for control point point cloud generation."""

    def test_bezier_point_count(self, bezier_curve: Bezier) -> None:
        mesh = control_points_mesh(bezier_curve)
        n_expected = 3
        assert mesh.n_points == n_expected

    def test_bspline_point_count(self, bspline_curve: Bspline) -> None:
        mesh = control_points_mesh(bspline_curve)
        n_expected = 5
        assert mesh.n_points == n_expected

    def test_rational_uses_projected_coords(
        self,
        rational_bezier_curve: Bezier,
    ) -> None:
        mesh = control_points_mesh(rational_bezier_curve)
        # First control point is (1, 0) with weight 1 → projected (1, 0, 0)
        np.testing.assert_allclose(mesh.points[0], [1.0, 0.0, 0.0], atol=1e-12)

    def test_3d_points_padded(self, bezier_curve: Bezier) -> None:
        mesh = control_points_mesh(bezier_curve)
        # z should be 0 for this planar curve
        np.testing.assert_array_equal(mesh.points[:, 2], 0.0)


# ---------------------------------------------------------------------------
# Control polygon
# ---------------------------------------------------------------------------


class TestControlPolygonMesh:
    """Tests for control polygon wireframe generation."""

    def test_curve_has_lines(self, bezier_curve: Bezier) -> None:
        mesh = control_polygon_mesh(bezier_curve)
        assert mesh.n_lines > 0

    def test_curve_line_count(self, bezier_curve: Bezier) -> None:
        mesh = control_polygon_mesh(bezier_curve)
        # Single polyline through 3 control points
        assert mesh.n_lines == 1

    def test_surface_has_lines(self, bspline_surface: Bspline) -> None:
        mesh = control_polygon_mesh(bspline_surface)
        assert mesh.n_lines > 0

    def test_bspline_curve_polygon(self, bspline_curve: Bspline) -> None:
        mesh = control_polygon_mesh(bspline_curve)
        n_pts = 5
        assert mesh.n_points == n_pts
        assert mesh.n_lines == 1  # single polyline
