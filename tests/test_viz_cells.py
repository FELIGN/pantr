"""Tests for Bézier/B-spline to pyvista UnstructuredGrid conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from pantr.bezier import Bezier  # noqa: E402
from pantr.bspline import (  # noqa: E402
    Bspline,
    BsplineSpace,
    BsplineSpace1D,
    create_uniform_periodic_knots,
)
from pantr.cad import create_circle, create_disk  # noqa: E402
from pantr.viz import save, to_pyvista  # noqa: E402
from pantr.viz._vtk_cells import (  # noqa: E402
    VTK_BEZIER_CURVE,
    VTK_BEZIER_HEXAHEDRON,
    VTK_BEZIER_QUADRILATERAL,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def bezier_curve_2d() -> Bezier:
    """Quadratic Bézier curve in 2D."""
    cp = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]])
    return Bezier(cp)


@pytest.fixture()
def bezier_curve_3d() -> Bezier:
    """Cubic Bézier curve in 3D."""
    cp = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    return Bezier(cp)


@pytest.fixture()
def bezier_surface_3d() -> Bezier:
    """Biquadratic Bézier surface in 3D."""
    cp = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            cp[i, j] = [i * 0.5, j * 0.5, 0.1 * i * j]
    return Bezier(cp)


@pytest.fixture()
def bezier_scalar_2d() -> Bezier:
    """Bilinear scalar Bézier (dim=2, rank=1)."""
    cp = np.array([[[0.0], [1.0]], [[2.0], [3.0]]])
    return Bezier(cp)


@pytest.fixture()
def anisotropic_surface_3d() -> Bezier:
    """Bézier surface with direction-dependent degree (2 in u, 1 in v)."""
    cp = np.zeros((3, 2, 3))
    for i, u in enumerate((0.0, 0.5, 1.0)):
        for j, v in enumerate((0.0, 1.0)):
            cp[i, j] = [u, v, np.sin(np.pi * u)]
    return Bezier(cp)


@pytest.fixture()
def bspline_curve_3d() -> Bspline:
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
def rational_bezier_curve() -> Bezier:
    """Rational quadratic Bézier (quarter circle)."""
    w = np.sqrt(2.0) / 2.0
    cp = np.array([[1.0, 0.0, 1.0], [w, w, w], [0.0, 1.0, 1.0]])
    return Bezier(cp, is_rational=True)


# ---------------------------------------------------------------------------
# Bézier → pyvista
# ---------------------------------------------------------------------------


class TestBezierToPyvista:
    """Tests for converting Bézier objects to pyvista grids."""

    def test_curve_2d_cell_type(self, bezier_curve_2d: Bezier) -> None:
        grid = to_pyvista(bezier_curve_2d)
        assert grid.n_cells == 1
        assert grid.celltypes[0] == VTK_BEZIER_CURVE

    def test_curve_2d_point_count(self, bezier_curve_2d: Bezier) -> None:
        grid = to_pyvista(bezier_curve_2d)
        n_expected = 3
        assert grid.n_points == n_expected

    def test_curve_2d_z_is_zero(self, bezier_curve_2d: Bezier) -> None:
        grid = to_pyvista(bezier_curve_2d)
        np.testing.assert_array_equal(grid.points[:, 2], 0.0)

    def test_curve_3d_cell_type(self, bezier_curve_3d: Bezier) -> None:
        grid = to_pyvista(bezier_curve_3d)
        assert grid.n_cells == 1
        assert grid.celltypes[0] == VTK_BEZIER_CURVE
        n_expected = 4
        assert grid.n_points == n_expected

    def test_surface_cell_type(self, bezier_surface_3d: Bezier) -> None:
        grid = to_pyvista(bezier_surface_3d)
        assert grid.n_cells == 1
        assert grid.celltypes[0] == VTK_BEZIER_QUADRILATERAL
        n_expected = 9
        assert grid.n_points == n_expected

    def test_rational_has_weights(self, rational_bezier_curve: Bezier) -> None:
        grid = to_pyvista(rational_bezier_curve)
        assert "RationalWeights" in grid.point_data
        weights = grid.point_data["RationalWeights"]
        n_expected = 3
        assert len(weights) == n_expected
        assert weights[0] == pytest.approx(1.0)

    def test_scalar_2d_has_point_data(self, bezier_scalar_2d: Bezier) -> None:
        grid = to_pyvista(bezier_scalar_2d)
        assert "scalar" in grid.point_data
        assert grid.n_cells == 1
        assert grid.celltypes[0] == VTK_BEZIER_QUADRILATERAL

    def test_scalar_2d_custom_name(self, bezier_scalar_2d: Bezier) -> None:
        grid = to_pyvista(bezier_scalar_2d, scalar_name="temperature")
        assert "temperature" in grid.point_data
        assert "scalar" not in grid.point_data


# ---------------------------------------------------------------------------
# Anisotropic 2D higher-order cells (regression)
# ---------------------------------------------------------------------------


class TestRationalAndAnisotropicRendering:
    """Rational and anisotropic-degree cells tessellate with their exact geometry.

    VTK's surface tessellator silently disables rational evaluation
    (``RationalWeights``) and direction-dependent degree evaluation
    (``HigherOrderDegrees``) whenever those arrays are the *active scalars* —
    which is exactly what pyvista's ``grid.data[name] = ...`` setter makes them.
    ``to_pyvista`` attaches them via ``AddArray`` (the dedicated attribute slot)
    instead, so both are honored and no degree elevation is needed.
    """

    def test_attributes_not_active_scalars(self, anisotropic_surface_3d: Bezier) -> None:
        grid = to_pyvista(anisotropic_surface_3d)
        cell_data = grid.GetCellData()
        assert cell_data.GetHigherOrderDegrees() is not None
        active = cell_data.GetScalars()
        # The degrees array must NOT be the active scalars (that disables it).
        assert active is None or active.GetName() != "HigherOrderDegrees"

    def test_anisotropic_degree_preserved(self, anisotropic_surface_3d: Bezier) -> None:
        # No elevation: the cell keeps its true (2, 1) order.
        grid = to_pyvista(anisotropic_surface_3d)
        cell = grid.GetCell(0)
        assert (cell.GetOrder(0), cell.GetOrder(1)) == anisotropic_surface_3d.degree
        np.testing.assert_array_equal(grid.cell_data["HigherOrderDegrees"][0], [2.0, 1.0, 0.0])

    def test_anisotropic_surface_renders_curved(self, anisotropic_surface_3d: Bezier) -> None:
        # z = sin(pi u) peaks at the degree-2 Bézier midpoint 0.5; a flat
        # (bilinear) fallback would give max z == 0.
        grid = to_pyvista(anisotropic_surface_3d)
        surf = grid.extract_surface(nonlinear_subdivision=4, algorithm="dataset_surface")
        assert surf.points[:, 2].max() > 0.4

    def test_rational_disk_renders_within_radius(self) -> None:
        # A correct unit disk stays within radius 1; ignoring the weights would
        # reach the corner control point at radius √2.
        disk = create_disk(radius_outer=1.0)
        grid = to_pyvista(disk)
        point_data = grid.GetPointData()
        assert point_data.GetRationalWeights() is not None
        # The weights must NOT be the active scalars (that disables rational eval).
        active = point_data.GetScalars()
        assert active is None or active.GetName() != "RationalWeights"
        surf = grid.extract_surface(nonlinear_subdivision=4, algorithm="dataset_surface")
        r = np.hypot(surf.points[:, 0], surf.points[:, 1])
        assert r.max() <= 1.0 + 1e-6

    def test_rational_circle_renders_on_circle(self) -> None:
        # Every tessellated point of a unit-circle curve must lie on the circle
        # (radius exactly 1); ignoring the weights bulges it to ~1.06.
        arc = create_circle(radius=1.0, angle=(0.0, 2.0 * np.pi))
        grid = to_pyvista(arc)
        surf = grid.extract_surface(nonlinear_subdivision=5, algorithm="dataset_surface")
        r = np.hypot(surf.points[:, 0], surf.points[:, 1])
        np.testing.assert_allclose(r, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# B-spline → pyvista
# ---------------------------------------------------------------------------


class TestBsplineToPyvista:
    """Tests for converting B-spline objects to pyvista grids."""

    def test_curve_multiple_cells(self, bspline_curve_3d: Bspline) -> None:
        grid = to_pyvista(bspline_curve_3d)
        n_expected_cells = 3
        assert grid.n_cells == n_expected_cells
        assert all(ct == VTK_BEZIER_CURVE for ct in grid.celltypes)

    def test_curve_point_count(self, bspline_curve_3d: Bspline) -> None:
        grid = to_pyvista(bspline_curve_3d)
        # 3 cells x 3 points each (degree 2)
        n_expected = 9
        assert grid.n_points == n_expected

    def test_periodic_bspline(self) -> None:
        """Periodic B-spline should convert without error."""
        knots = create_uniform_periodic_knots(4, 2, domain=(0.0, 1.0))
        space1d = BsplineSpace1D(knots, 2, periodic=True)
        space = BsplineSpace([space1d])
        n_basis = space1d.num_basis
        cp = np.zeros((n_basis, 3))
        for i in range(n_basis):
            angle = 2 * np.pi * i / n_basis
            cp[i] = [np.cos(angle), np.sin(angle), 0.0]
        bspline = Bspline(space, cp)
        grid = to_pyvista(bspline)
        assert grid.n_cells > 0
        assert all(ct == VTK_BEZIER_CURVE for ct in grid.celltypes)


# ---------------------------------------------------------------------------
# Scalar fields
# ---------------------------------------------------------------------------


class TestScalarFields:
    """Tests for scalar field visualization."""

    def test_1d_scalar_elevation(self) -> None:
        """dim=1 rank=1: should produce line plot (t, f(t), 0)."""
        cp = np.array([[0.0], [1.0], [0.0]])
        bezier = Bezier(cp)
        grid = to_pyvista(bezier)
        # Points should have nonzero y (elevation)
        assert grid.points[:, 1].max() > 0.0
        # z should be zero
        np.testing.assert_array_equal(grid.points[:, 2], 0.0)

    def test_2d_scalar_flat_default(self) -> None:
        """dim=2 rank=1: default should be flat (z=0)."""
        cp = np.array([[[1.0], [2.0]], [[3.0], [4.0]]])
        bezier = Bezier(cp)
        grid = to_pyvista(bezier, elevation=False)
        np.testing.assert_array_equal(grid.points[:, 2], 0.0)
        assert "scalar" in grid.point_data

    def test_2d_scalar_elevation(self) -> None:
        """dim=2 rank=1: elevation=True should use scalar as z."""
        cp = np.array([[[1.0], [2.0]], [[3.0], [4.0]]])
        bezier = Bezier(cp)
        grid = to_pyvista(bezier, elevation=True)
        assert grid.points[:, 2].max() > 0.0


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


class TestSave:
    """Tests for VTK file export."""

    def test_save_vtu(self, bezier_curve_3d: Bezier, tmp_path: Path) -> None:
        path = tmp_path / "test.vtu"
        save(bezier_curve_3d, path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_save_vtk(self, bezier_surface_3d: Bezier, tmp_path: Path) -> None:
        path = tmp_path / "test.vtk"
        save(bezier_surface_3d, path)
        assert path.exists()
        assert path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Expected Bspline or Bezier"):
            to_pyvista("not a geometry")  # type: ignore[arg-type]

    def test_bezier_volume(self) -> None:
        """Trilinear Bézier volume."""
        cp = np.zeros((2, 2, 2, 3))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    cp[i, j, k] = [float(i), float(j), float(k)]
        bezier = Bezier(cp)
        grid = to_pyvista(bezier)
        assert grid.n_cells == 1
        assert grid.celltypes[0] == VTK_BEZIER_HEXAHEDRON
        n_expected = 8
        assert grid.n_points == n_expected
