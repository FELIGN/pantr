"""Convert implicit reparameterization results to pyvista UnstructuredGrids.

Each tensor-product Lagrange cell is linearly tessellated into VTK quads
(2D cells) or hexahedra (3D cells) by connecting adjacent nodes.  For
surface curves the nodes are connected into a VTK polyline.  This
approach is simple, robust at any polynomial degree, and avoids the
Runge-phenomenon artefacts that plague VTK higher-order cells with
equispaced nodes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy import typing as npt

from ._lazy_import import _import_pyvista

if TYPE_CHECKING:
    import pyvista as pv

    from ..bezier.implicit._implicit_quad import ReparamResult

_MAX_PHYSICAL_DIM = 3
"""Maximum physical dimension for VTK coordinates."""

# VTK linear cell types used for tessellation.
_VTK_LINE = 3
_VTK_QUAD = 9
_VTK_HEXAHEDRON = 12


def implicit_to_pyvista(result: ReparamResult) -> pv.UnstructuredGrid:
    """Convert a :class:`~pantr.bezier.implicit.ReparamResult` to pyvista.

    Each tensor-product Lagrange cell is linearly tessellated: adjacent
    nodes are connected into VTK quads (for 2D cells), hexahedra (for 3D
    cells), or line segments (for 1D curves).

    Args:
        result: Reparameterization result from
            :meth:`~pantr.bezier.implicit.ImplicitQuadrature.volume_reparam`
            or
            :meth:`~pantr.bezier.implicit.ImplicitQuadrature.surface_reparam`.

    Returns:
        pv.UnstructuredGrid: Tessellated grid.

    Raises:
        ImportError: If pyvista is not installed.
        ValueError: If the cell dimension is not 1, 2, or 3.
    """
    pv_mod = _import_pyvista()

    cell_dim = result.cell_dim
    q = result.q

    # Embed in 3D.
    pts_3d = _embed_in_3d(result.points, result.dim)

    if cell_dim == 1:
        cells, cell_types = _tessellate_curves(result.n_cells, q)
    elif cell_dim == 2:  # noqa: PLR2004
        cells, cell_types = _tessellate_quads(result.n_cells, q)
    elif cell_dim == 3:  # noqa: PLR2004
        cells, cell_types = _tessellate_hexes(result.n_cells, q)
    else:
        raise ValueError(f"Unsupported cell dimension {cell_dim}.")

    return pv_mod.UnstructuredGrid(cells, cell_types, pts_3d)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Tessellation helpers
# ---------------------------------------------------------------------------


def _tessellate_curves(
    n_cells: int,
    q: int,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.uint8]]:
    """Tessellate 1D Lagrange curves into VTK line segments.

    Each degree-(q-1) curve produces ``q - 1`` line segments.

    Args:
        n_cells: Number of Lagrange curves.
        q: Nodes per curve.

    Returns:
        tuple: ``(cells, cell_types)`` arrays for pyvista.
    """
    n_segs = (q - 1) * n_cells
    # Each line: [2, p0, p1]
    cells = np.empty(n_segs * 3, dtype=np.intp)
    idx = 0
    for c in range(n_cells):
        base = c * q
        for i in range(q - 1):
            cells[idx] = 2
            cells[idx + 1] = base + i
            cells[idx + 2] = base + i + 1
            idx += 3

    cell_types = np.full(n_segs, _VTK_LINE, dtype=np.uint8)
    return cells, cell_types


def _tessellate_quads(
    n_cells: int,
    q: int,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.uint8]]:
    """Tessellate 2D Lagrange quads into VTK linear quads.

    Each ``q x q`` tensor-product cell produces ``(q-1)^2`` linear quads.
    The tensor-product layout is row-major: index ``i*q + j`` where *i*
    varies slowest.

    Args:
        n_cells: Number of Lagrange quads.
        q: Nodes per direction.

    Returns:
        tuple: ``(cells, cell_types)`` arrays for pyvista.
    """
    n_sub = (q - 1) * (q - 1)
    n_total = n_sub * n_cells
    # Each quad: [4, p0, p1, p2, p3]
    cells = np.empty(n_total * 5, dtype=np.intp)
    idx = 0
    ppc = q * q
    for c in range(n_cells):
        base = c * ppc
        for i in range(q - 1):
            for j in range(q - 1):
                # Four corners of the sub-quad in TP order.
                p00 = base + i * q + j
                p01 = base + i * q + (j + 1)
                p10 = base + (i + 1) * q + j
                p11 = base + (i + 1) * q + (j + 1)
                cells[idx] = 4
                cells[idx + 1] = p00
                cells[idx + 2] = p10
                cells[idx + 3] = p11
                cells[idx + 4] = p01
                idx += 5

    cell_types = np.full(n_total, _VTK_QUAD, dtype=np.uint8)
    return cells, cell_types


def _tessellate_hexes(
    n_cells: int,
    q: int,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.uint8]]:
    """Tessellate 3D Lagrange hexes into VTK linear hexahedra.

    Each ``q x q x q`` tensor-product cell produces ``(q-1)^3`` linear
    hexahedra.  The tensor-product layout is row-major: index
    ``i*q*q + j*q + k`` where *i* varies slowest.

    Args:
        n_cells: Number of Lagrange hexes.
        q: Nodes per direction.

    Returns:
        tuple: ``(cells, cell_types)`` arrays for pyvista.
    """
    n_sub = (q - 1) ** 3
    n_total = n_sub * n_cells
    # Each hex: [8, p000, p100, p110, p010, p001, p101, p111, p011]
    cells = np.empty(n_total * 9, dtype=np.intp)
    idx = 0
    ppc = q * q * q
    qq = q * q
    for c in range(n_cells):
        base = c * ppc
        for i in range(q - 1):
            for j in range(q - 1):
                for k in range(q - 1):
                    p000 = base + i * qq + j * q + k
                    p100 = base + (i + 1) * qq + j * q + k
                    p110 = base + (i + 1) * qq + (j + 1) * q + k
                    p010 = base + i * qq + (j + 1) * q + k
                    p001 = base + i * qq + j * q + (k + 1)
                    p101 = base + (i + 1) * qq + j * q + (k + 1)
                    p111 = base + (i + 1) * qq + (j + 1) * q + (k + 1)
                    p011 = base + i * qq + (j + 1) * q + (k + 1)
                    cells[idx] = 8
                    cells[idx + 1] = p000
                    cells[idx + 2] = p100
                    cells[idx + 3] = p110
                    cells[idx + 4] = p010
                    cells[idx + 5] = p001
                    cells[idx + 6] = p101
                    cells[idx + 7] = p111
                    cells[idx + 8] = p011
                    idx += 9

    cell_types = np.full(n_total, _VTK_HEXAHEDRON, dtype=np.uint8)
    return cells, cell_types


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
