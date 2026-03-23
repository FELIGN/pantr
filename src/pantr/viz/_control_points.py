"""Control point and control polygon visualization helpers.

Provides functions to create pyvista meshes for:

- **Control points**: a point cloud rendered as small spheres.
- **Control polygon**: a wireframe connecting adjacent control points along
  each parametric direction.

Rational geometries always use projected (Euclidean) coordinates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy import typing as npt

from ._lazy_import import _import_pyvista

if TYPE_CHECKING:
    import pyvista as pv

    from ..bezier import Bezier
    from ..bspline import Bspline

_MAX_PHYSICAL_DIM = 3
"""Maximum physical dimension for VTK coordinates."""


def _get_euclidean_control_points(
    geom: Bspline | Bezier,
) -> tuple[npt.NDArray[np.float64], tuple[int, ...], int]:
    """Extract Euclidean control points from a geometry, padding to 3D.

    For rational geometries, projects to Euclidean space by dividing by
    the homogeneous weight.

    Args:
        geom: Input B-spline or Bézier geometry.

    Returns:
        tuple: ``(points_3d, grid_shape, rank)`` where *points_3d* has shape
        ``(n_pts, 3)``, *grid_shape* is the tensor-product shape
        (e.g. ``(5,)`` for a curve with 5 CPs), and *rank* is the geometric
        output rank.
    """
    cp = np.asarray(geom.control_points, dtype=np.float64)
    grid_shape = cp.shape[:-1]
    rank = geom.rank
    n_pts = int(np.prod(grid_shape))
    flat = cp.reshape(n_pts, -1)

    if geom.is_rational:
        weights = flat[:, -1:]
        coords = flat[:, :-1] / weights
    else:
        coords = flat

    pts_3d = np.zeros((n_pts, _MAX_PHYSICAL_DIM), dtype=np.float64)
    pts_3d[:, :rank] = coords[:, :rank]
    return pts_3d, grid_shape, rank


def control_points_mesh(geom: Bspline | Bezier) -> pv.PolyData:
    """Create a point cloud mesh of control points for visualization.

    Control points are projected to Euclidean space for rational geometries
    and padded to 3D.

    Args:
        geom: Input B-spline or Bézier geometry.

    Returns:
        pv.PolyData: Point cloud suitable for rendering as glyphs (spheres).

    Raises:
        ImportError: If pyvista is not installed.
    """
    pv = _import_pyvista()
    pts_3d, _, _ = _get_euclidean_control_points(geom)
    return pv.PolyData(pts_3d)  # type: ignore[no-any-return]


def control_polygon_mesh(geom: Bspline | Bezier) -> pv.PolyData:
    """Create a wireframe mesh of the control polygon.

    Connects adjacent control points along each parametric direction:

    - **dim=1**: a single polyline through all control points.
    - **dim=2**: a grid of lines along both parametric directions.
    - **dim=3**: edges of the 3D control point lattice along all three
      parametric directions.

    Args:
        geom: Input B-spline or Bézier geometry.

    Returns:
        pv.PolyData: Wireframe mesh with line cells.

    Raises:
        ImportError: If pyvista is not installed.
    """
    pv = _import_pyvista()
    pts_3d, grid_shape, _ = _get_euclidean_control_points(geom)
    dim = len(grid_shape)
    lines = _build_polygon_lines(grid_shape, dim)

    if not lines:
        return pv.PolyData(pts_3d)  # type: ignore[no-any-return]

    # pyvista line format: [n_pts_in_line, idx0, idx1, ..., n_pts_in_line, ...]
    line_cells = np.concatenate(lines)
    poly = pv.PolyData(pts_3d)
    poly.lines = line_cells
    return poly  # type: ignore[no-any-return]


def _build_polygon_lines(
    grid_shape: tuple[int, ...],
    dim: int,
) -> list[npt.NDArray[Any]]:
    """Build line connectivity arrays for the control polygon.

    For each parametric direction, creates polylines connecting control
    points that are adjacent in that direction.

    Args:
        grid_shape: Tensor-product shape of the control point grid.
        dim: Parametric dimension.

    Returns:
        list: Line cell arrays in pyvista format.
    """
    lines: list[npt.NDArray[Any]] = []
    strides = _compute_strides(grid_shape)

    for direction in range(dim):
        n_along = grid_shape[direction]
        if n_along < 2:  # noqa: PLR2004
            continue
        # Iterate over all "fibers" in this direction
        for base_idx in np.ndindex(*_fiber_iteration_shape(grid_shape, direction)):
            # Compute flat indices along this fiber
            fiber_start = sum(base_idx[d] * strides[d] for d in range(dim) if d != direction)
            fiber_indices = np.array(
                [fiber_start + i * strides[direction] for i in range(n_along)],
                dtype=np.intp,
            )
            # pyvista polyline: [n_pts, idx0, idx1, ...]
            cell = np.empty(n_along + 1, dtype=np.intp)
            cell[0] = n_along
            cell[1:] = fiber_indices
            lines.append(cell)

    return lines


def _compute_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Compute row-major strides for a given shape.

    Args:
        shape: Array shape.

    Returns:
        tuple[int, ...]: Strides (in elements, not bytes).
    """
    strides: list[int] = []
    stride = 1
    for s in reversed(shape):
        strides.append(stride)
        stride *= s
    return tuple(reversed(strides))


def _fiber_iteration_shape(
    shape: tuple[int, ...],
    direction: int,
) -> tuple[int, ...]:
    """Get the iteration shape for fibers along a given direction.

    Returns a shape tuple with the given direction set to 1, suitable
    for ``np.ndindex`` to iterate over all fibers perpendicular to
    that direction.

    Args:
        shape: Full grid shape.
        direction: Direction along which to iterate.

    Returns:
        tuple[int, ...]: Modified shape with ``shape[direction] = 1``.
    """
    return tuple(1 if d == direction else s for d, s in enumerate(shape))
