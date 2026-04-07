"""Convert implicit reparameterization results to pyvista UnstructuredGrids.

Uses VTK higher-order Lagrange cell types (``VTK_LAGRANGE_CURVE``,
``VTK_LAGRANGE_QUADRILATERAL``, ``VTK_LAGRANGE_HEXAHEDRON``) which share
the same point ordering convention as the Bézier cell types already used
by pantr (corners → edges → faces → interior).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy import typing as npt

from ._lazy_import import _import_pyvista
from ._vtk_ordering import vtk_ordering

if TYPE_CHECKING:
    import pyvista as pv

    from ..bezier.implicit._implicit_quad import ReparamResult

# VTK cell type constants for Lagrange cells.
VTK_LAGRANGE_CURVE = 68
VTK_LAGRANGE_QUADRILATERAL = 70
VTK_LAGRANGE_HEXAHEDRON = 72

_VTK_LAGRANGE_CELL_TYPE_BY_DIM = {
    1: VTK_LAGRANGE_CURVE,
    2: VTK_LAGRANGE_QUADRILATERAL,
    3: VTK_LAGRANGE_HEXAHEDRON,
}

_MAX_PHYSICAL_DIM = 3
"""Maximum physical dimension for VTK coordinates."""


def implicit_to_pyvista(result: ReparamResult) -> pv.UnstructuredGrid:
    """Convert a :class:`~pantr.bezier.implicit.ReparamResult` to pyvista.

    Creates a ``pyvista.UnstructuredGrid`` with VTK Lagrange cell types.
    Points in parametric space are embedded in 3D by zero-padding.

    Args:
        result: Reparameterization result from
            :meth:`~pantr.bezier.implicit.ImplicitQuadrature.volume_reparam`
            or
            :meth:`~pantr.bezier.implicit.ImplicitQuadrature.surface_reparam`.

    Returns:
        pv.UnstructuredGrid: Grid with Lagrange cells.

    Raises:
        ImportError: If pyvista is not installed.
        ValueError: If the cell dimension is not 1, 2, or 3.
    """
    pv_mod = _import_pyvista()

    cell_dim = result.cell_dim
    if cell_dim not in _VTK_LAGRANGE_CELL_TYPE_BY_DIM:
        raise ValueError(f"Unsupported cell dimension {cell_dim}.")

    cell_type = _VTK_LAGRANGE_CELL_TYPE_BY_DIM[cell_dim]
    degree = tuple(result.q - 1 for _ in range(cell_dim))
    ordering = vtk_ordering(degree)
    ppc = result.pts_per_cell

    # Embed in 3D.
    pts_3d = _embed_in_3d(result.points, result.dim)

    # Build connectivity.
    cells, cell_types = _build_connectivity(
        result.n_cells,
        ppc,
        ordering,
        cell_type,
    )

    return pv_mod.UnstructuredGrid(cells, cell_types, pts_3d)  # type: ignore[no-any-return]


def _embed_in_3d(
    points: npt.NDArray[np.float64],
    dim: int,
) -> npt.NDArray[np.float64]:
    """Embed parametric-space points in 3D by zero-padding.

    Args:
        points: Points of shape ``(n, dim)``.
        dim: Parametric dimension.

    Returns:
        NDArray[float64]: Points of shape ``(n, 3)``.
    """
    n = len(points)
    pts_3d = np.zeros((n, _MAX_PHYSICAL_DIM), dtype=np.float64)
    pts_3d[:, :dim] = points
    return pts_3d


def _build_connectivity(
    n_cells: int,
    pts_per_cell: int,
    ordering: npt.NDArray[Any],
    cell_type: int,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.uint8]]:
    """Build VTK cell connectivity and type arrays.

    Args:
        n_cells: Number of cells.
        pts_per_cell: Points per cell.
        ordering: VTK point ordering permutation.
        cell_type: VTK cell type constant.

    Returns:
        tuple: ``(cells, cell_types)`` arrays for pyvista.
    """
    # Each cell entry: [n_pts, idx_0, idx_1, ..., idx_{n-1}]
    entry_len = pts_per_cell + 1
    cells = np.empty(n_cells * entry_len, dtype=np.intp)

    for i in range(n_cells):
        base = i * entry_len
        pt_offset = i * pts_per_cell
        cells[base] = pts_per_cell
        cells[base + 1 : base + entry_len] = pt_offset + ordering

    cell_types = np.full(n_cells, cell_type, dtype=np.uint8)
    return cells, cell_types
