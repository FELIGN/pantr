"""Tests for THB-spline visualization (to_pyvista, knot lines, control nets, Scene)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
pv.OFF_SCREEN = True

from pantr.bezier import Bezier  # noqa: E402
from pantr.bspline import (  # noqa: E402
    BsplineSpace,
    BsplineSpace1D,
    THBSpline,
    THBSplineSpace,
    create_uniform_space,
    quasi_interpolate_thb_spline,
)
from pantr.grid import hierarchical_grid, uniform_grid  # noqa: E402
from pantr.viz import (  # noqa: E402
    Scene,
    control_polygon_mesh,
    knot_lines_meshes,
    plot,
    save,
    to_pyvista,
)
from pantr.viz._vtk_cells import (  # noqa: E402
    VTK_BEZIER_CURVE,
    VTK_BEZIER_HEXAHEDRON,
    VTK_BEZIER_QUADRILATERAL,
    _thb_bezier_patches,
)

_VTK_TYPE_BY_DIM = {1: VTK_BEZIER_CURVE, 2: VTK_BEZIER_QUADRILATERAL, 3: VTK_BEZIER_HEXAHEDRON}

# ---------------------------------------------------------------------------
# Builders / fixtures
# ---------------------------------------------------------------------------


def _graded_space(dim: int, degree: int = 2, n: int = 4) -> THBSplineSpace:
    """A truncated THB space: refine the lower corner block, leaving coarse cells.

    Produces a genuinely hierarchical (2-level) space with active functions on
    both levels — the case that exercises the per-cell extraction and the
    per-level control net.
    """
    root = create_uniform_space([degree] * dim, [n] * dim)
    grid = hierarchical_grid(uniform_grid([[0.0, 1.0]] * dim, n), 2)
    grid.refine(0, [0] * dim, [n // 2] * dim)
    return THBSplineSpace(root, grid)


def _single_level_space(dim: int, degree: int = 2, n: int = 3) -> THBSplineSpace:
    """A THB space with no refinement (a single-level tensor-product space)."""
    root = create_uniform_space([degree] * dim, [n] * dim)
    grid = hierarchical_grid(uniform_grid([[0.0, 1.0]] * dim, n), 2)
    return THBSplineSpace(root, grid)


def _scalar_thb(space: THBSplineSpace, seed: int = 0) -> THBSpline:
    """A scalar (rank-1) THB spline with random coefficients."""
    rng = np.random.default_rng(seed)
    return THBSpline(space, rng.standard_normal(space.num_total_basis))


def _geometric_thb(space: THBSplineSpace, rank: int, seed: int = 0) -> THBSpline:
    """A geometric (rank >= 2) THB spline with random control points."""
    rng = np.random.default_rng(seed)
    return THBSpline(space, rng.standard_normal((space.num_total_basis, rank)))


# ---------------------------------------------------------------------------
# Rendering: to_pyvista / save
# ---------------------------------------------------------------------------


class TestTHBToPyvista:
    """THB rendering via per-active-cell Bézier decomposition."""

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_scalar_cell_and_point_counts(self, dim: int) -> None:
        space = _graded_space(dim)
        thb = _scalar_thb(space)
        ug = to_pyvista(thb)
        n_cells = space.grid.num_cells
        assert ug.n_cells == n_cells
        assert ug.n_points == n_cells * 3**dim  # (degree + 1)**dim, degree=2
        assert np.all(ug.celltypes == _VTK_TYPE_BY_DIM[dim])
        assert "scalar" in ug.point_data

    @pytest.mark.parametrize(("dim", "rank"), [(1, 2), (2, 3), (3, 3)])
    def test_geometric_has_no_scalar_array(self, dim: int, rank: int) -> None:
        space = _graded_space(dim)
        ug = to_pyvista(_geometric_thb(space, rank))
        assert ug.n_cells == space.grid.num_cells
        assert "scalar" not in ug.point_data

    def test_custom_scalar_name(self) -> None:
        ug = to_pyvista(_scalar_thb(_graded_space(2)), scalar_name="temperature")
        assert "temperature" in ug.point_data
        assert "scalar" not in ug.point_data

    @pytest.mark.parametrize(("dim", "rank"), [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 3)])
    def test_decomposition_is_exact(self, dim: int, rank: int) -> None:
        """Each cell's Bézier patch reproduces THBSpline.evaluate to machine precision."""
        space = _graded_space(dim)
        thb = _scalar_thb(space) if rank == 1 else _geometric_thb(space, rank)
        patches, n_per = _thb_bezier_patches(thb)
        rng = np.random.default_rng(99)
        max_err = 0.0
        for cid, bern in patches:
            lo, hi = space.grid.cell_bounds(cid)
            bez = Bezier(bern.reshape(*n_per, rank))
            local = rng.random((4, dim))
            glob = lo + local * (hi - lo)
            ref = np.asarray(thb.evaluate(glob)).reshape(4, rank)
            got = np.asarray(bez.evaluate(local.ravel() if dim == 1 else local)).reshape(4, rank)
            max_err = max(max_err, float(np.max(np.abs(got - ref))))
        assert max_err < 1e-12

    def test_single_level_renders(self) -> None:
        """A THB space with no refinement still renders one cell per grid cell."""
        space = _single_level_space(2)
        assert space.num_levels == 1
        ug = to_pyvista(_scalar_thb(space))
        assert ug.n_cells == space.grid.num_cells

    def test_elevation_lifts_scalar(self) -> None:
        """elevation=True puts the value in z; the flat default keeps z = 0."""
        thb = _scalar_thb(_graded_space(2))
        flat = to_pyvista(thb, elevation=False)
        lifted = to_pyvista(thb, elevation=True)
        np.testing.assert_allclose(flat.points[:, 2], 0.0)
        assert np.any(np.abs(lifted.points[:, 2]) > 1e-6)

    def test_unsupported_dimension_raises(self) -> None:
        root = BsplineSpace([BsplineSpace1D([0, 0, 1, 1], 1) for _ in range(4)])
        grid = hierarchical_grid(uniform_grid([[0.0, 1.0]] * 4, 1), 2)
        thb = THBSpline(THBSplineSpace(root, grid), np.zeros(2**4))
        with pytest.raises(ValueError, match="parametric dimension"):
            to_pyvista(thb)

    def test_save_writes_file(self, tmp_path: Path) -> None:
        out = tmp_path / "thb.vtu"
        save(_scalar_thb(_graded_space(2)), out)
        assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Knot lines (active-cell boundaries)
# ---------------------------------------------------------------------------


class TestTHBKnotLines:
    """Active-cell boundary knot lines."""

    def test_curve_knot_points(self) -> None:
        thb = _scalar_thb(_graded_space(1))
        meshes = knot_lines_meshes(thb)
        assert len(meshes) == 1
        assert meshes[0].n_points > 0  # interior cell boundaries exist in a graded mesh

    @pytest.mark.parametrize(("dim", "per_cell"), [(2, 4), (3, 12)])
    def test_boundary_curve_counts(self, dim: int, per_cell: int) -> None:
        space = _graded_space(dim)
        meshes = knot_lines_meshes(_scalar_thb(space))
        assert len(meshes) == 1
        mesh = meshes[0]
        assert mesh.n_cells == per_cell * space.grid.num_cells
        assert np.all(mesh.celltypes == VTK_BEZIER_CURVE)

    def test_geometric_surface_boundaries(self) -> None:
        space = _graded_space(2)
        mesh = knot_lines_meshes(_geometric_thb(space, 3))[0]
        assert mesh.n_cells == 4 * space.grid.num_cells


# ---------------------------------------------------------------------------
# Control polygon (per-level nets)
# ---------------------------------------------------------------------------


class TestTHBControlPolygon:
    """Per-level control nets tagged by level."""

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_point_count_and_level_tags(self, dim: int) -> None:
        space = _graded_space(dim)
        poly = control_polygon_mesh(_scalar_thb(space))
        assert poly.n_points == space.num_total_basis
        assert "level" in poly.point_data
        levels = np.asarray(poly.point_data["level"])
        per_level = [int((levels == lvl).sum()) for lvl in range(space.num_levels)]
        assert per_level == list(space.num_basis_per_level)

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_edges_stay_within_level(self, dim: int) -> None:
        poly = control_polygon_mesh(_scalar_thb(_graded_space(dim)))
        levels = np.asarray(poly.point_data["level"])
        if poly.n_lines:
            endpoints = poly.lines.reshape(-1, 3)[:, 1:]
            assert np.all(levels[endpoints[:, 0]] == levels[endpoints[:, 1]])

    @pytest.mark.parametrize("dim", [1, 2])
    def test_linear_reproduction_places_points_on_graph(self, dim: int) -> None:
        """A THB reproducing f = x0 has control points on the graph value == x0.

        This exercises coefficient preservation: truncated coarse functions keep
        their own level's Greville node, so a per-level net at (greville, value)
        lies exactly on the plane ``value = x0``.
        """
        space = _graded_space(dim)
        thb = quasi_interpolate_thb_spline(lambda p: p[:, 0], space)
        pts = np.asarray(control_polygon_mesh(thb).points)
        np.testing.assert_allclose(pts[:, dim], pts[:, 0], atol=1e-12)

    def test_geometric_control_net(self) -> None:
        space = _graded_space(2)
        poly = control_polygon_mesh(_geometric_thb(space, 3))
        assert poly.n_points == space.num_total_basis
        assert "level" in poly.point_data


# ---------------------------------------------------------------------------
# Scene / plot integration
# ---------------------------------------------------------------------------


class TestTHBScene:
    """THB splines integrate with Scene and plot()."""

    def test_scalar_scene_with_overlays(self) -> None:
        thb = _scalar_thb(_graded_space(2))
        scene = Scene()
        result = scene.add(thb, show_knot_lines=True, show_control_polygon=True, elevation=True)
        assert result is scene  # chaining
        assert scene.to_plotter() is not None

    def test_geometric_surface_scene(self) -> None:
        thb = _geometric_thb(_graded_space(2), 3)
        scene = Scene().add(thb, show_knot_lines=True, show_control_polygon=True)
        assert scene.to_plotter() is not None

    def test_plot_convenience(self) -> None:
        assert plot(_scalar_thb(_graded_space(2)), elevation=True) is not None
