"""Knot line visualization for B-spline and THB-spline geometries.

Knot lines are the element (knot-span) boundaries drawn on the rendered
geometry/field:

- **dim=1 (curves)**: knot *points* — evaluate the curve at each interior knot.
- **dim=2/3 (surfaces/volumes)**: the **cell boundaries** of the VTK Bézier-cell
  decomposition. Because Bézier extraction yields one VTK cell per element, the
  boundary edges of those cells *are* the knot lines.

For dim ≥ 2 the boundaries are obtained by tessellating each Bézier cell at the
**same subdivision level the surface uses** and extracting the tessellated cell
boundary with :meth:`pyvista.DataSet.extract_feature_edges`. The resulting edges
therefore share the surface's facet vertices exactly: they lie *on* the rendered
surface (no facet-vs-curve mismatch) and VTK's coincident-topology resolution
keeps them from z-fighting. This applies uniformly to B-splines (element
boundaries) and :class:`~pantr.bspline.THBSpline` (active-cell boundaries).

For a 1D :class:`~pantr.bspline.THBSpline` the analogue is the interior
active-cell endpoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy import typing as npt

from ._common import _MAX_PHYSICAL_DIM, _pad_points_to_3d
from ._lazy_import import _import_pyvista
from ._vtk_cells import to_pyvista

if TYPE_CHECKING:
    import pyvista as pv

    from ..bspline import Bspline, THBSpline

# Matches the scene's default; callers (e.g. ``Scene``) pass the level actually
# used for the surface so the extracted edges coincide with the rendered facets.
_DEFAULT_TESSELLATION_LEVEL = 4


def _get_interior_knots(
    bspline: Bspline,
) -> list[npt.NDArray[np.float32 | np.float64]]:
    """Get unique interior knot values for each parametric direction.

    Args:
        bspline: Input B-spline geometry.

    Returns:
        list: One array per direction containing interior (non-boundary)
        unique knot values.
    """
    interior_knots: list[npt.NDArray[np.float32 | np.float64]] = []
    for d in range(bspline.dim):
        sp1d = bspline.space.spaces[d]
        unique_knots, _ = sp1d.get_unique_knots_and_multiplicity(in_domain=True)
        # Exclude boundary knots (first and last)
        n_boundary = 2
        if len(unique_knots) > n_boundary:
            interior_knots.append(unique_knots[1:-1])
        else:
            interior_knots.append(np.array([], dtype=unique_knots.dtype))
    return interior_knots


def _knot_points_curve(bspline: Bspline) -> pv.PolyData:
    """Compute knot points for a 1D B-spline curve.

    Evaluates the curve at each interior knot value and returns a point cloud.

    Args:
        bspline: A 1D B-spline curve.

    Returns:
        pv.PolyData: Point cloud of knot locations on the curve.
    """
    pv = _import_pyvista()
    interior_knots = _get_interior_knots(bspline)
    knot_vals = interior_knots[0]

    if len(knot_vals) == 0:
        return pv.PolyData()  # type: ignore[no-any-return]

    # Evaluate curve at knot values (1D points are flat arrays)
    pts_param = knot_vals.astype(np.float64)
    pts_phys = bspline.evaluate(pts_param)

    # evaluate() returns (rank,) for a single point, (n, rank) for multiple
    if pts_phys.ndim == 1:
        pts_phys = pts_phys.reshape(1, -1)

    pts_3d = _pad_points_to_3d(pts_phys, bspline.rank)
    return pv.PolyData(pts_3d)  # type: ignore[no-any-return]


def _thb_knot_points(thb: THBSpline) -> pv.PolyData:
    """Compute knot points for a 1D THB spline (interior active-cell endpoints).

    Collects the interior cell boundaries of the hierarchical grid and evaluates
    the THB spline there. Scalar fields are placed at ``(t, f(t))``.

    Args:
        thb: A 1D THB spline.

    Returns:
        pv.PolyData: Point cloud of interior cell-boundary locations.
    """
    pv = _import_pyvista()
    grid = thb.space.grid
    coords: set[float] = set()
    for cid in range(grid.num_cells):
        lo, hi = grid.cell_bounds(cid)
        coords.add(round(float(lo[0]), 12))
        coords.add(round(float(hi[0]), 12))
    interior = sorted(coords)[1:-1]  # drop the two domain boundaries
    if not interior:
        return pv.PolyData()  # type: ignore[no-any-return]

    params = np.array(interior, dtype=np.float64).reshape(-1, 1)
    values = np.asarray(thb.evaluate(params), dtype=np.float64)
    if thb.rank == 1:
        pts = np.column_stack([params[:, 0], values.ravel()])  # (t, f(t))
        pts_3d = _pad_points_to_3d(pts, 2)
    else:
        pts_3d = _pad_points_to_3d(values.reshape(len(interior), thb.rank), thb.rank)
    return pv.PolyData(pts_3d)  # type: ignore[no-any-return]


def _cell_boundary_edges(
    grid: pv.UnstructuredGrid, tessellation_level: int, dim: int
) -> pv.PolyData:
    """Extract the per-cell tessellated boundary edges of a Bézier-cell grid.

    Each Bézier cell is tessellated at *tessellation_level* (matching the
    surface) and its boundary edges are extracted; the union over cells is the
    knot-line grid. Because the edges are taken from the cells' own tessellation
    they share the surface's facet vertices exactly, so they lie on the rendered
    surface without floating or z-fighting.

    For a surface (``dim == 2``) the boundary is the perimeter of each cell's
    tessellated patch. For a volume (``dim == 3``) the cell tessellates to a
    closed solid, so the cell wireframe is recovered as the *feature* edges of
    its outer surface instead.

    Args:
        grid: A :func:`~pantr.viz.to_pyvista` grid of VTK Bézier cells.
        tessellation_level: Non-linear subdivision level, equal to the surface's.
        dim: Parametric dimension (2 or 3).

    Returns:
        pv.PolyData: Merged boundary-edge polylines (empty if the grid has no
        cells).
    """
    pv_mod = _import_pyvista()
    level = max(int(tessellation_level), 1)
    is_volume = dim == _MAX_PHYSICAL_DIM
    edge_meshes: list[pv.PolyData] = []
    for cid in range(grid.n_cells):
        tess = grid.extract_cells([cid]).tessellate(max_n_subdivide=level)
        # A surface patch already exposes its perimeter to extract_feature_edges;
        # a volume must first be reduced to its bounding surface.
        src = tess.extract_surface(algorithm="geometry") if is_volume else tess
        edges = src.extract_feature_edges(
            boundary_edges=not is_volume,
            feature_edges=is_volume,
            manifold_edges=False,
            non_manifold_edges=False,
        )
        if edges.n_cells:
            edge_meshes.append(edges)

    if not edge_meshes:
        return pv_mod.PolyData()  # type: ignore[no-any-return]
    if len(edge_meshes) == 1:
        return edge_meshes[0]
    return edge_meshes[0].merge(edge_meshes[1:])  # type: ignore[no-any-return]


def knot_lines_meshes(
    geom: Bspline | THBSpline,
    *,
    tessellation_level: int = _DEFAULT_TESSELLATION_LEVEL,
    elevation: bool = False,
) -> list[pv.PolyData | pv.UnstructuredGrid]:
    """Compute knot line meshes for a B-spline or THB-spline geometry.

    - **dim=1**: a single ``PolyData`` point cloud of interior knot locations.
    - **dim=2/3**: a single ``PolyData`` of the Bézier cells' boundary edges
      (element boundaries for a B-spline, active-cell boundaries for a THB
      spline), tessellated at *tessellation_level* so the edges coincide with a
      surface rendered at the same level.

    Args:
        geom: Input B-spline or THB-spline geometry (dim 1, 2, or 3).
        tessellation_level: Non-linear subdivision level for the cell-boundary
            edges (dim ≥ 2). Pass the same level used to render the surface so
            the edges lie exactly on the rendered facets. Ignored for dim=1.
        elevation: For a dim=2 scalar field (``rank == 1``), use the value as a
            spatial coordinate so the boundaries lie on the elevated field. A
            dim=1 scalar field is always drawn as the graph ``(t, f(t))``;
            ignored for vector-valued geometries.

    Returns:
        list[pv.PolyData | pv.UnstructuredGrid]: Knot line meshes.

    Raises:
        ImportError: If pyvista is not installed.
        ValueError: If the parametric dimension is not 1, 2, or 3.
    """
    from ..bspline import THBSpline as THBSplineCls  # noqa: PLC0415

    _import_pyvista()  # ensure pyvista is available

    dim = geom.dim
    if dim == 1:
        if isinstance(geom, THBSplineCls):
            return [_thb_knot_points(geom)]
        return [_knot_points_curve(geom)]
    if dim not in (2, _MAX_PHYSICAL_DIM):
        raise ValueError(f"Unsupported parametric dimension {dim}.")

    grid = to_pyvista(geom, elevation=elevation)
    return [_cell_boundary_edges(grid, tessellation_level, dim)]
