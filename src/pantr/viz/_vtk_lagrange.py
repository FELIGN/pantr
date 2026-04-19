"""Visualization helpers for implicit reparameterization and quadrature.

- :func:`implicit_to_pyvista`: convert reparameterization cells to pyvista.
- :func:`quadrature_to_pyvista`: convert quadrature points to a pyvista
  point cloud with weight scalars (optional normal arrows).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy import typing as npt

from ._lazy_import import _import_pyvista

if TYPE_CHECKING:
    import pyvista as pv
    from ocelat.algoim._implicit_quad import ReparamResult, SurfQuadResult, VolQuadResult

_MAX_PHYSICAL_DIM = 3
"""Maximum physical dimension for VTK coordinates."""

_VTK_LINE = 3
"""VTK cell type ID for a line segment."""

_VTK_QUAD = 9
"""VTK cell type ID for a linear quadrilateral."""

_VTK_HEXAHEDRON = 12
"""VTK cell type ID for a linear hexahedron."""


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

    Each q-node Lagrange curve is tessellated into ``q - 1`` line segments.

    Args:
        n_cells: Number of Lagrange curves.
        q: Nodes per curve.

    Returns:
        tuple[npt.NDArray[np.intp], npt.NDArray[np.uint8]]: ``(cells,
            cell_types)`` arrays for pyvista.
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
        tuple[npt.NDArray[np.intp], npt.NDArray[np.uint8]]: ``(cells,
            cell_types)`` arrays for pyvista.
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
                # Four corners in VTK quad winding order.
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
        tuple[npt.NDArray[np.intp], npt.NDArray[np.uint8]]: ``(cells,
            cell_types)`` arrays for pyvista.
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


# ---------------------------------------------------------------------------
# Quadrature point visualization
# ---------------------------------------------------------------------------


def quadrature_to_pyvista(
    quad_result: VolQuadResult | SurfQuadResult,
    *,
    show_normals: bool = False,
    normal_scale: float = 0.02,
) -> pv.PolyData | tuple[pv.PolyData, pv.PolyData]:
    """Convert quadrature results to pyvista for visualization.

    Creates a ``pv.PolyData`` point cloud with a ``"weight"`` scalar
    array suitable for rendering as coloured spheres.  For surface
    quadrature, optionally produces a second ``PolyData`` with normal
    arrows.

    Usage example::

        grid = quadrature_to_pyvista(iq.volume_quad(q=4))
        plotter.add_mesh(grid, scalars="weight",
                         render_points_as_spheres=True, point_size=8)

    For surface quadrature with normals::

        pts_mesh, arrows = quadrature_to_pyvista(
            iq.surface_quad(q=4), show_normals=True)
        plotter.add_mesh(pts_mesh, scalars="weight",
                         render_points_as_spheres=True, point_size=8)
        plotter.add_mesh(arrows, color="blue")

    Args:
        quad_result: Output of
            :meth:`~pantr.bezier.implicit.ImplicitQuadrature.volume_quad`
            (2-tuple) or
            :meth:`~pantr.bezier.implicit.ImplicitQuadrature.surface_quad`
            (3-tuple).
        show_normals: If ``True`` and *quad_result* is a 3-tuple (surface),
            return an additional ``PolyData`` with arrow glyphs for the
            normal weights.
        normal_scale: Length scale for normal arrows (fraction of domain).

    Returns:
        pv.PolyData | tuple[pv.PolyData, pv.PolyData]: Point cloud with
            ``"weight"`` scalar data.  When *show_normals* is ``True``
            and *quad_result* is a surface result, returns
            ``(point_cloud, arrows)`` instead.

    Raises:
        ImportError: If pyvista is not installed.
    """
    pv_mod = _import_pyvista()

    is_surface = len(quad_result) == 3  # noqa: PLR2004
    points = quad_result[0]
    weights = quad_result[1]

    dim = points.shape[1] if points.ndim == 2 else 1  # noqa: PLR2004
    pts_3d = _embed_in_3d(points, dim)

    cloud = pv_mod.PolyData(pts_3d)
    cloud.point_data["weight"] = weights

    if not (show_normals and is_surface):
        return cloud  # type: ignore[return-value]

    # Surface normals: build arrow glyphs.
    normal_weights = quad_result[2]  # type: ignore[misc]
    normals_3d = _embed_in_3d(normal_weights, dim)

    cloud["normal"] = normals_3d
    arrows = cloud.glyph(
        orient="normal",
        scale="weight",
        factor=normal_scale,
    )
    # Remove normal data from the point cloud (not needed for rendering).
    del cloud.point_data["normal"]

    return cloud, arrows  # type: ignore[return-value]
