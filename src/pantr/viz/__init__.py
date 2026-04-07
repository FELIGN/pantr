"""Optional visualization module for PaNTr using pyvista.

Provides conversion of B-spline and Bézier geometries to pyvista
``UnstructuredGrid`` objects with native VTK Bézier cell types, interactive
visualization, and export to VTK file formats for ParaView.

This module requires ``pyvista`` (optional dependency). Install with::

    pip install pantr[viz]

Main exports:

- :func:`to_pyvista`: Convert a geometry to a pyvista ``UnstructuredGrid``.
- :func:`save`: Export a geometry to a VTK file.
- :func:`plot`: Quick interactive visualization of one or more geometries.
- :class:`Scene`: Composable multi-geometry visualization scene.
- :func:`control_polygon_mesh`: Control polygon (points + wireframe).
- :func:`knot_lines_meshes`: Knot line meshes for B-splines.
"""

from __future__ import annotations

from ._control_points import control_polygon_mesh
from ._knot_lines import knot_lines_meshes
from ._scene import Scene, plot
from ._vtk_cells import save, to_pyvista
from ._vtk_lagrange import implicit_to_pyvista

__all__ = [
    "Scene",
    "control_polygon_mesh",
    "implicit_to_pyvista",
    "knot_lines_meshes",
    "plot",
    "save",
    "to_pyvista",
]
