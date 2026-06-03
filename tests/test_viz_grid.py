"""Tests for ``pantr.viz.grid_to_pyvista``."""

from __future__ import annotations

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from pantr.grid import TensorProductGrid, uniform_grid  # noqa: E402
from pantr.viz import grid_to_pyvista  # noqa: E402

_VTK_LINE = 3
_VTK_QUAD = 9
_VTK_HEXAHEDRON = 12


def test_2d_quads() -> None:
    """A 2-D grid exports one quad per cell with 4 vertices each."""
    g = uniform_grid([[0.0, 4.0], [0.0, 3.0]], [4, 3])
    ug = grid_to_pyvista(g)
    assert ug.n_cells == 12  # noqa: PLR2004
    assert ug.n_points == 12 * 4
    assert np.all(ug.celltypes == _VTK_QUAD)


def test_3d_hexes() -> None:
    """A 3-D grid exports one hexahedron per cell with 8 vertices each."""
    g = uniform_grid([[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]], 2)
    ug = grid_to_pyvista(g)
    assert ug.n_cells == 8  # noqa: PLR2004
    assert ug.n_points == 8 * 8
    assert np.all(ug.celltypes == _VTK_HEXAHEDRON)


def test_1d_lines() -> None:
    """A 1-D grid exports one line per cell."""
    g = TensorProductGrid([[0.0, 1.0, 3.0, 6.0]])
    ug = grid_to_pyvista(g)
    assert ug.n_cells == 3  # noqa: PLR2004
    assert np.all(ug.celltypes == _VTK_LINE)


def test_cell_bounds_recovered() -> None:
    """Each exported cell's point bounds match the grid cell box."""
    g = TensorProductGrid([[0.0, 2.0, 5.0], [0.0, 4.0]])
    ug = grid_to_pyvista(g)
    for cid in range(g.num_cells):
        lo, hi = g.cell_bounds(cid)
        cell = ug.get_cell(cid)
        pts = np.asarray(cell.points)
        np.testing.assert_allclose(pts[:, :2].min(axis=0), lo)
        np.testing.assert_allclose(pts[:, :2].max(axis=0), hi)


def test_cell_data_attachable() -> None:
    """An array indexed by cell id attaches directly as cell_data."""
    g = uniform_grid([[0.0, 3.0], [0.0, 2.0]], [3, 2])
    ug = grid_to_pyvista(g)
    ug.cell_data["loc"] = np.arange(g.num_cells)
    assert ug.cell_data["loc"].tolist() == list(range(g.num_cells))


def test_unsupported_ndim_raises() -> None:
    """grid_to_pyvista rejects ndim outside {1, 2, 3}."""
    g = uniform_grid([[0.0, 1.0]] * 4, 1)
    with pytest.raises(ValueError, match="ndim"):
        grid_to_pyvista(g)
