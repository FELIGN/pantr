"""Tests for Scene class and plot() convenience function."""

from __future__ import annotations

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
pv.OFF_SCREEN = True

from pantr.bezier import Bezier  # noqa: E402
from pantr.bspline import (  # noqa: E402
    Bspline,
    BsplineSpace,
    BsplineSpace1D,
)
from pantr.viz import Scene  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def bezier_curve() -> Bezier:
    """Quadratic Bézier curve in 3D."""
    cp = np.array([[0.0, 0.0, 0.0], [0.5, 1.0, 0.0], [1.0, 0.0, 0.0]])
    return Bezier(cp)


@pytest.fixture()
def bezier_surface() -> Bezier:
    """Bilinear Bézier surface in 3D."""
    cp = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.5]],
        ]
    )
    return Bezier(cp)


@pytest.fixture()
def bspline_curve() -> Bspline:
    """Quadratic B-spline curve in 3D."""
    space1d = BsplineSpace1D([0, 0, 0, 1, 2, 2, 2], 2)
    space = BsplineSpace([space1d])
    cp = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.5, 1.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    return Bspline(space, cp)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


class TestScene:
    """Tests for the Scene class."""

    def test_empty_scene(self) -> None:
        scene = Scene()
        plotter = scene.to_plotter()
        assert plotter is not None

    def test_single_geometry(self, bezier_curve: Bezier) -> None:
        scene = Scene()
        result = scene.add(bezier_curve, color="blue")
        assert result is scene  # chaining

        plotter = scene.to_plotter()
        assert plotter is not None

    def test_multiple_geometries(
        self,
        bezier_curve: Bezier,
        bezier_surface: Bezier,
    ) -> None:
        scene = Scene()
        scene.add(bezier_curve, color="red")
        scene.add(bezier_surface, color="blue")
        plotter = scene.to_plotter()
        assert plotter is not None

    def test_with_control_polygon(self, bezier_curve: Bezier) -> None:
        scene = Scene()
        scene.add(bezier_curve, show_control_polygon=True)
        plotter = scene.to_plotter()
        assert plotter is not None

    def test_with_knot_lines(self, bspline_curve: Bspline) -> None:
        scene = Scene()
        scene.add(bspline_curve, show_knot_lines=True)
        plotter = scene.to_plotter()
        assert plotter is not None

    def test_chaining(
        self,
        bezier_curve: Bezier,
        bezier_surface: Bezier,
    ) -> None:
        plotter = (
            Scene().add(bezier_curve, color="red").add(bezier_surface, color="blue").to_plotter()
        )
        assert plotter is not None

    def test_scalar_field(self) -> None:
        """Scene with a scalar Bézier field."""
        cp = np.array([[[0.0], [1.0]], [[2.0], [3.0]]])
        scalar_bez = Bezier(cp)
        scene = Scene()
        scene.add(scalar_bez, scalar_bar=True)
        plotter = scene.to_plotter()
        assert plotter is not None

    def test_mixed_dimensions(
        self,
        bezier_curve: Bezier,
        bezier_surface: Bezier,
    ) -> None:
        """Curves and surfaces in the same scene."""
        scene = Scene()
        scene.add(bezier_curve, color="red")
        scene.add(bezier_surface, color="blue", opacity=0.5)
        plotter = scene.to_plotter()
        assert plotter is not None


# ---------------------------------------------------------------------------
# Bspline.plot() / Bezier.plot() convenience methods
# ---------------------------------------------------------------------------


class TestPlotMethods:
    """Tests for the convenience plot() methods on geometry classes."""

    def test_bezier_has_plot(self, bezier_curve: Bezier) -> None:
        assert hasattr(bezier_curve, "plot")

    def test_bspline_has_plot(self, bspline_curve: Bspline) -> None:
        assert hasattr(bspline_curve, "plot")
