"""Knot line visualization for B-spline geometries.

Knot lines are the images of iso-parametric lines at interior knot values:

- **dim=1 (curves)**: knot *points* — evaluate the curve at each interior knot.
- **dim=2 (surfaces)**: knot *curves* — slice the surface at each interior knot
  in each direction, yielding iso-parametric curves.
- **dim=3 (volumes)**: knot *surfaces* — slice the volume at each interior knot
  in each direction, yielding iso-parametric surfaces.

Each lower-dimensional slice is converted to a pyvista mesh via
:func:`~pantr.viz.to_pyvista`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy import typing as npt

from ._lazy_import import _import_pyvista
from ._vtk_cells import to_pyvista

if TYPE_CHECKING:
    import pyvista as pv

    from ..bspline import Bspline

_MAX_PHYSICAL_DIM = 3
"""Maximum physical dimension for VTK coordinates."""


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

    rank = bspline.rank
    n_pts = len(knot_vals)
    pts_3d = np.zeros((n_pts, _MAX_PHYSICAL_DIM), dtype=np.float64)
    pts_3d[:, :rank] = pts_phys[:, :rank]
    return pv.PolyData(pts_3d)  # type: ignore[no-any-return]


def _knot_lines_surface(bspline: Bspline) -> list[pv.UnstructuredGrid]:
    """Compute knot lines for a 2D B-spline surface.

    Slices the surface at each interior knot in each direction, producing
    iso-parametric curves that are converted to VTK Bézier curve cells.

    Args:
        bspline: A 2D B-spline surface.

    Returns:
        list[pv.UnstructuredGrid]: One grid per knot line (iso-curve).
    """
    interior_knots = _get_interior_knots(bspline)
    grids: list[pv.UnstructuredGrid] = []

    n_directions = 2
    for direction in range(n_directions):
        for knot_val in interior_knots[direction]:
            curve = bspline.slice(direction, float(knot_val))
            grids.append(to_pyvista(curve))  # type: ignore[arg-type]

    return grids


def _knot_surfaces_volume(bspline: Bspline) -> list[pv.UnstructuredGrid]:
    """Compute knot surfaces for a 3D B-spline volume.

    Slices the volume at each interior knot in each direction, producing
    iso-parametric surfaces that are converted to VTK Bézier quad cells.

    Args:
        bspline: A 3D B-spline volume.

    Returns:
        list[pv.UnstructuredGrid]: One grid per knot surface.
    """
    interior_knots = _get_interior_knots(bspline)
    grids: list[pv.UnstructuredGrid] = []

    for direction in range(_MAX_PHYSICAL_DIM):
        for knot_val in interior_knots[direction]:
            surface = bspline.slice(direction, float(knot_val))
            grids.append(to_pyvista(surface))  # type: ignore[arg-type]

    return grids


def knot_lines_meshes(bspline: Bspline) -> list[pv.PolyData | pv.UnstructuredGrid]:
    """Compute knot line meshes for a B-spline geometry.

    Returns one mesh per knot line (or point, or surface) depending on the
    parametric dimension:

    - **dim=1**: single ``PolyData`` point cloud of knot locations.
    - **dim=2**: list of ``UnstructuredGrid`` iso-parametric curves.
    - **dim=3**: list of ``UnstructuredGrid`` iso-parametric surfaces.

    Args:
        bspline: Input B-spline geometry (dim 1, 2, or 3).

    Returns:
        list[pv.PolyData | pv.UnstructuredGrid]: Knot line meshes.

    Raises:
        ImportError: If pyvista is not installed.
        ValueError: If the parametric dimension is not 1, 2, or 3.
    """
    _import_pyvista()  # ensure pyvista is available

    dim = bspline.dim
    if dim == 1:
        return [_knot_points_curve(bspline)]
    if dim == 2:  # noqa: PLR2004
        return list(_knot_lines_surface(bspline))
    if dim == _MAX_PHYSICAL_DIM:
        return list(_knot_surfaces_volume(bspline))
    raise ValueError(f"Unsupported parametric dimension {dim}.")
