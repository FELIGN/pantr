"""Optional visualization module for PaNTr using pyvista.

Provides conversion of B-spline and Bézier geometries to pyvista
``UnstructuredGrid`` objects with native VTK Bézier cell types, and
export to VTK file formats for ParaView.

This module requires ``pyvista`` (optional dependency). Install with::

    pip install pantr[viz]

Main exports:

- :func:`to_pyvista`: Convert a geometry to a pyvista ``UnstructuredGrid``.
- :func:`save`: Export a geometry to a VTK file.
"""

from __future__ import annotations

from ._vtk_cells import save, to_pyvista

__all__ = [
    "save",
    "to_pyvista",
]
