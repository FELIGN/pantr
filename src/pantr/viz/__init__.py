"""Optional visualization module for PaNTr using pyvista.

Provides conversion of B-spline, Bézier, and THB-spline geometries to pyvista
``UnstructuredGrid`` objects with native VTK Bézier cell types, interactive
visualization, and export to VTK file formats for ParaView. A
:class:`~pantr.bspline.THBSpline` is decomposed into one VTK Bézier cell per
active cell of its hierarchical grid.

This module requires ``pyvista`` (optional dependency). Install with::

    pip install pantr[viz]

Main exports:

- :func:`to_pyvista`: Convert a B-spline / Bézier / THB-spline to a pyvista
  ``UnstructuredGrid``.
- :func:`save`: Export a geometry to a VTK file.
- :func:`plot`: Quick interactive visualization of one or more geometries.
- :class:`Scene`: Composable multi-geometry visualization scene.
- :func:`control_polygon_mesh`: Control polygon (points + wireframe); a per-level
  control net coloured by level for THB-splines.
- :func:`knot_lines_meshes`: Knot line meshes for B-splines; active-cell
  boundaries for THB-splines.
- :func:`grid_to_pyvista`: Convert a :class:`pantr.grid.Grid` to an ``UnstructuredGrid``.
"""

from __future__ import annotations

from ._control_points import control_polygon_mesh
from ._grid import grid_to_pyvista
from ._knot_lines import knot_lines_meshes
from ._scene import Scene, plot
from ._vtk_cells import save, to_pyvista

__all__ = [
    "Scene",
    "control_polygon_mesh",
    "grid_to_pyvista",
    "knot_lines_meshes",
    "plot",
    "save",
    "to_pyvista",
]
