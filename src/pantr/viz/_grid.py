"""Export a :class:`pantr.grid.Grid` to a pyvista ``UnstructuredGrid``.

Builds one VTK cell per grid cell -- a line (1-D), quad (2-D), or hexahedron
(3-D) -- from the cell's axis-aligned ``(lo, hi)`` corners. Vertices are emitted
per cell (not shared between neighbours), which keeps the export independent of
the grid's internal structure and therefore works for any :class:`Grid`
subclass. Cell ordering matches :meth:`pantr.grid.Grid.iter_cells`, so an array
indexed by cell id can be attached directly as ``grid.cell_data``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np

from ._lazy_import import _import_pyvista

if TYPE_CHECKING:
    import numpy.typing as npt
    import pyvista as pv

    from ..grid import Grid

# VTK cell-type codes, re-declared so this module needs no ``vtk`` import.
_VTK_LINE: Final[int] = 3
_VTK_QUAD: Final[int] = 9
_VTK_HEXAHEDRON: Final[int] = 12

# Per-dimension (vtk cell type, corner-sign table). Each row of the sign table
# selects lo (0) or hi (1) on each axis for one vertex, in VTK vertex order.
_CORNER_SIGNS: Final[dict[int, tuple[int, npt.NDArray[np.int64]]]] = {
    1: (_VTK_LINE, np.array([[0], [1]], dtype=np.int64)),
    2: (_VTK_QUAD, np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.int64)),
    3: (
        _VTK_HEXAHEDRON,
        np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.int64,
        ),
    ),
}


def grid_to_pyvista(grid: Grid) -> pv.UnstructuredGrid:
    """Convert a 1-D, 2-D, or 3-D :class:`pantr.grid.Grid` to a pyvista grid.

    Args:
        grid (Grid): The grid to export. Must have ``ndim in {1, 2, 3}``.

    Returns:
        pyvista.UnstructuredGrid: A grid of lines (1-D), quads (2-D), or
        hexahedra (3-D), one cell per grid cell, in :meth:`~pantr.grid.Grid.iter_cells`
        order. Points are padded to 3-D (``z = 0``, and ``y = 0`` in 1-D).

    Raises:
        ValueError: If ``grid.ndim`` is not 1, 2, or 3.
        ImportError: If pyvista is not installed.
    """
    if grid.ndim not in _CORNER_SIGNS:
        raise ValueError(f"grid_to_pyvista supports ndim in {{1, 2, 3}}; got ndim={grid.ndim}.")
    pv = _import_pyvista()
    ndim = grid.ndim
    n = grid.num_cells
    vtk_type, signs = _CORNER_SIGNS[ndim]
    n_verts = signs.shape[0]

    cell_lo = np.empty((n, ndim), dtype=np.float64)
    cell_hi = np.empty((n, ndim), dtype=np.float64)
    for cid in range(n):
        lo, hi = grid.cell_bounds(cid)
        cell_lo[cid] = lo
        cell_hi[cid] = hi

    # Per-cell vertices, interleaved so point id == cid * n_verts + vertex.
    points = np.zeros((n * n_verts, 3), dtype=np.float64)
    for v in range(n_verts):
        for d in range(ndim):
            points[v::n_verts, d] = cell_hi[:, d] if signs[v, d] else cell_lo[:, d]

    # Connectivity in pyvista's prepended-count format: [n_verts, v0, v1, ...].
    base = (np.arange(n, dtype=np.int64) * n_verts)[:, np.newaxis]
    conn = np.empty((n, n_verts + 1), dtype=np.int64)
    conn[:, 0] = n_verts
    conn[:, 1:] = base + np.arange(n_verts, dtype=np.int64)
    cell_types = np.full(n, vtk_type, dtype=np.uint8)
    return pv.UnstructuredGrid(conn.ravel(), cell_types, points)


__all__ = ["grid_to_pyvista"]
