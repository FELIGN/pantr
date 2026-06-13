"""Knot line visualization for B-spline and THB-spline geometries.

For a B-spline, knot lines are the images of iso-parametric lines at interior
knot values:

- **dim=1 (curves)**: knot *points* — evaluate the curve at each interior knot.
- **dim=2 (surfaces)**: knot *curves* — slice the surface at each interior knot
  in each direction, yielding iso-parametric curves.
- **dim=3 (volumes)**: knot *surfaces* — slice the volume at each interior knot
  in each direction, yielding iso-parametric surfaces.

For a :class:`~pantr.bspline.THBSpline` the analogue is the **active-cell
boundaries** of its hierarchical grid, drawn on the rendered geometry/field: the
endpoints of each active cell (dim=1) or the boundary Bézier curves of each
active cell's patch (dim=2/3). Finer cells therefore yield denser boundaries.

Each lower-dimensional slice is converted to a pyvista mesh via
:func:`~pantr.viz.to_pyvista`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy import typing as npt

from ._common import _MAX_PHYSICAL_DIM, _pad_points_to_3d
from ._lazy_import import _import_pyvista
from ._vtk_cells import (
    VTK_BEZIER_CURVE,
    _thb_bezier_patches,
    _thb_patch_coords,
    to_pyvista,
)
from ._vtk_ordering import vtk_ordering_curve

if TYPE_CHECKING:
    import pyvista as pv

    from ..bspline import Bspline, THBSpline


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

    Evaluates the curve at each interior knot value and returns
    a point cloud.

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


def _knot_slices(bspline: Bspline, n_directions: int) -> list[pv.UnstructuredGrid]:
    """Slice a B-spline at every interior knot along the first ``n_directions`` axes.

    Shared by 2D surfaces (``n_directions=2``, yielding iso-parametric curves)
    and 3D volumes (``n_directions=3``, yielding iso-parametric surfaces); each
    slice is converted to a VTK Bézier cell mesh.

    Args:
        bspline: A 2D B-spline surface or 3D B-spline volume.
        n_directions: Number of parametric directions to slice along.

    Returns:
        list[pv.UnstructuredGrid]: One grid per knot slice.
    """
    interior_knots = _get_interior_knots(bspline)
    grids: list[pv.UnstructuredGrid] = []

    for direction in range(n_directions):
        for knot_val in interior_knots[direction]:
            grids.append(to_pyvista(bspline.slice(direction, float(knot_val))))  # type: ignore[arg-type]

    return grids


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


def _boundary_curve_lines(
    pts_nd: npt.NDArray[np.float64], dim: int
) -> list[npt.NDArray[np.float64]]:
    """Extract the boundary Bézier-curve control points of one patch.

    Args:
        pts_nd: Patch control points embedded in 3D, shape ``(*n_per, 3)``.
        dim: Parametric dimension (2 or 3).

    Returns:
        list[NDArray[float64]]: One ``(m, 3)`` control-point array per boundary
        edge — 4 edges for a surface patch, 12 for a volume patch.
    """
    ends = (0, -1)
    if dim == 2:  # noqa: PLR2004
        return (
            [pts_nd[:, j, :] for j in ends]  # u-edges at v = 0, 1
            + [pts_nd[i, :, :] for i in ends]  # v-edges at u = 0, 1
        )
    return (
        [pts_nd[:, j, k, :] for j in ends for k in ends]  # u-edges
        + [pts_nd[i, :, k, :] for i in ends for k in ends]  # v-edges
        + [pts_nd[i, j, :, :] for i in ends for j in ends]  # w-edges
    )


def _curves_to_grid(lines: list[npt.NDArray[np.float64]]) -> pv.UnstructuredGrid:
    """Assemble boundary control-point lines into one VTK Bézier-curve grid.

    Args:
        lines: Control-point arrays, one ``(m, 3)`` per boundary curve.

    Returns:
        pv.UnstructuredGrid: A grid of ``VTK_BEZIER_CURVE`` cells.
    """
    pv = _import_pyvista()
    if not lines:
        return pv.UnstructuredGrid()  # type: ignore[no-any-return]
    all_pts: list[npt.NDArray[np.float64]] = []
    conn: list[int] = []
    offset = 0
    for line in lines:
        m = line.shape[0]
        all_pts.append(line[vtk_ordering_curve(m - 1)])
        conn.append(m)
        conn.extend(range(offset, offset + m))
        offset += m
    cell_types = np.full(len(lines), VTK_BEZIER_CURVE, dtype=np.uint8)
    return pv.UnstructuredGrid(  # type: ignore[no-any-return]
        np.array(conn, dtype=np.intp), cell_types, np.vstack(all_pts)
    )


def _thb_knot_lines(thb: THBSpline, elevation: bool) -> list[pv.PolyData | pv.UnstructuredGrid]:
    """Compute active-cell-boundary knot lines for a THB spline.

    Args:
        thb: Input THB spline (dim 1, 2, or 3).
        elevation: For scalar fields with dim ≤ 2, use the value as a spatial
            coordinate so the boundaries lie on the elevated field.

    Returns:
        list[pv.PolyData | pv.UnstructuredGrid]: A single mesh holding the
        active-cell boundaries.

    Raises:
        ValueError: If the parametric dimension is not 1, 2, or 3.
    """
    _import_pyvista()  # ensure pyvista is available
    dim, rank = thb.dim, thb.rank
    if dim == 1:
        return [_thb_knot_points(thb)]
    if dim not in (2, _MAX_PHYSICAL_DIM):
        raise ValueError(f"Unsupported parametric dimension {dim}.")

    grid = thb.space.grid
    patches, n_per = _thb_bezier_patches(thb)
    lines: list[npt.NDArray[np.float64]] = []
    for cid, bern in patches:
        pts, _ = _thb_patch_coords(bern, n_per, grid.cell_bounds(cid), rank, dim, elevation)
        lines.extend(_boundary_curve_lines(pts.reshape(*n_per, _MAX_PHYSICAL_DIM), dim))
    return [_curves_to_grid(lines)]


def knot_lines_meshes(
    geom: Bspline | THBSpline, *, elevation: bool = False
) -> list[pv.PolyData | pv.UnstructuredGrid]:
    """Compute knot line meshes for a B-spline or THB-spline geometry.

    For a B-spline, returns one mesh per knot line (or point, or surface)
    depending on the parametric dimension:

    - **dim=1**: single ``PolyData`` point cloud of knot locations.
    - **dim=2**: list of ``UnstructuredGrid`` iso-parametric curves.
    - **dim=3**: list of ``UnstructuredGrid`` iso-parametric surfaces.

    For a :class:`~pantr.bspline.THBSpline`, returns a single mesh holding the
    active-cell boundaries (a ``PolyData`` point cloud in dim=1, a
    ``UnstructuredGrid`` of boundary Bézier curves in dim=2/3).

    Args:
        geom: Input B-spline or THB-spline geometry (dim 1, 2, or 3).
        elevation: For THB scalar fields with dim ≤ 2, use the value as a spatial
            coordinate so the boundaries lie on the elevated field. Ignored for
            B-splines.

    Returns:
        list[pv.PolyData | pv.UnstructuredGrid]: Knot line meshes.

    Raises:
        ImportError: If pyvista is not installed.
        ValueError: If the parametric dimension is not 1, 2, or 3.
    """
    from ..bspline import THBSpline as THBSplineCls  # noqa: PLC0415

    if isinstance(geom, THBSplineCls):
        return _thb_knot_lines(geom, elevation)

    _import_pyvista()  # ensure pyvista is available

    dim = geom.dim
    if dim == 1:
        return [_knot_points_curve(geom)]
    if dim == 2:  # noqa: PLR2004
        return _knot_slices(geom, 2)
    if dim == _MAX_PHYSICAL_DIM:
        return _knot_slices(geom, _MAX_PHYSICAL_DIM)
    raise ValueError(f"Unsupported parametric dimension {dim}.")
