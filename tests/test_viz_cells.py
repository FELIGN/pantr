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
