"""Control polygon visualization helpers.

Provides a function to create the control polygon mesh: a wireframe connecting
adjacent control points along each parametric direction, with the control
points themselves included as vertices.

Rational geometries always use projected (Euclidean) coordinates.

For a :class:`~pantr.bspline.THBSpline` the mesh is a *per-level* control net:
each hierarchy level's active control points are connected by that level's own
tensor-grid adjacency and tagged with a ``"level"`` point-data array (so callers
can colour by level). Scalar fields (``rank == 1``) are drawn as a graph -- each
active control point at ``(greville, value)`` using its own level's Greville
abscissa (value as elevation, for dim <= 2); geometric splines (``rank >= 2``)
use the physical control-point position. Placing each scalar control point at its
own level's Greville abscissa is well-defined because THB truncation *preserves
coefficients* (Giannelli-Jüttler-Speleers): truncated coarse functions keep their
own level's Greville node, so the values shown are the genuine THB basis
coefficients and the geometry lies within the per-level convex hull.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy import typing as npt

from ._common import _MAX_PHYSICAL_DIM, _pad_points_to_3d, _project_homogeneous
from ._lazy_import import _import_pyvista

if TYPE_CHECKING:
    import pyvista as pv

    from ..bezier import Bezier
    from ..bspline import Bspline, THBSpline


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
    coords, _ = _project_homogeneous(cp.reshape(n_pts, -1), geom.is_rational)
    pts_3d = _pad_points_to_3d(coords, rank)
    return pts_3d, grid_shape, rank


def _thb_control_polygon(thb: THBSpline) -> pv.PolyData:
    """Build the per-level control net of a THB spline (tagged by level).

    For each hierarchy level, every active control point is placed at that
    level's tensor-Greville point (geometric ``rank >= 2`` uses the physical
    control point; scalar ``rank == 1`` uses ``(greville, value)`` for dim ≤ 2)
    and connected to its active neighbours along each parametric direction. The
    returned mesh carries a ``"level"`` point-data array for colouring.

    Args:
        thb: Input THB spline.

    Returns:
        pv.PolyData: Per-level control points (vertices), within-level adjacency
        line cells, and a ``"level"`` point-data array.
    """
    from ..bspline import get_greville_abscissae  # noqa: PLC0415

    pv = _import_pyvista()
    space = thb.space
    dim, rank = thb.dim, thb.rank
    cp = np.asarray(thb.control_points, dtype=np.float64)
    cp2d = cp.reshape(cp.shape[0], -1)  # (n_dofs, rank)
    counts = space.num_basis_per_level

    points: list[npt.NDArray[np.float64]] = []
    levels: list[int] = []
    segments: list[tuple[int, int]] = []
    dof_offset = 0
    for level in range(space.num_levels):
        active = np.asarray(space.active_function_indices(level), dtype=np.int64)
        level_space = space.level_space(level)
        shape = tuple(int(n) for n in level_space.num_basis)
        greville = (
            [np.asarray(get_greville_abscissae(level_space.spaces[d])) for d in range(dim)]
            if rank == 1
            else []
        )
        base = len(points)
        flat_to_local = {int(flat): base + k for k, flat in enumerate(active)}
        for k, flat in enumerate(active):
            midx = np.unravel_index(int(flat), shape)
            pos = np.zeros(_MAX_PHYSICAL_DIM, dtype=np.float64)
            if rank == 1:
                for d in range(dim):
                    pos[d] = greville[d][midx[d]]
                if dim < _MAX_PHYSICAL_DIM:
                    pos[dim] = cp2d[dof_offset + k, 0]  # value as elevation
            else:
                pos = _pad_points_to_3d(cp2d[dof_offset + k : dof_offset + k + 1], rank)[0]
            points.append(pos)
            levels.append(level)
        for k, flat in enumerate(active):
            midx = np.unravel_index(int(flat), shape)
            for d in range(dim):
                neighbour = list(midx)
                neighbour[d] += 1
                if neighbour[d] < shape[d]:
                    nbr_flat = int(np.ravel_multi_index(tuple(neighbour), shape))
                    j = flat_to_local.get(nbr_flat)
                    if j is not None:
                        segments.append((base + k, j))
        dof_offset += counts[level]

    pts_arr = np.array(points, dtype=np.float64) if points else np.zeros((0, _MAX_PHYSICAL_DIM))
    poly = pv.PolyData(pts_arr)
    if segments:
        cells = np.empty((len(segments), 3), dtype=np.intp)
        cells[:, 0] = 2
        cells[:, 1:] = np.array(segments, dtype=np.intp)
        poly.lines = cells.ravel()
    poly.point_data["level"] = np.array(levels, dtype=np.int64)
    return poly  # type: ignore[no-any-return]


def control_polygon_mesh(geom: Bspline | Bezier | THBSpline) -> pv.PolyData:
    """Create the control polygon mesh with points and connecting wireframe.

    For a B-spline or Bézier, the mesh contains the control points as vertices
    and polylines connecting adjacent control points along each parametric
    direction:

    - **dim=1**: a single polyline through all control points.
    - **dim=2**: a grid of lines along both parametric directions.
    - **dim=3**: edges of the 3D control point lattice along all three
      parametric directions.

    Rational geometries use projected (Euclidean) coordinates.

    For a :class:`~pantr.bspline.THBSpline`, the mesh is a *per-level* control
    net: each level's active control points (at that level's Greville abscissae)
    connected by that level's tensor-grid adjacency, with a ``"level"``
    point-data array for colouring. Scalar fields are drawn as a graph
    ``(greville, value)`` for dim ≤ 2.

    Args:
        geom: Input B-spline, Bézier, or THB-spline geometry.

    Returns:
        pv.PolyData: Mesh with control points as vertices and line cells
        forming the polygon wireframe (plus a ``"level"`` array for THB).

    Raises:
        ImportError: If pyvista is not installed.
    """
    from ..bspline import THBSpline as THBSplineCls  # noqa: PLC0415

    if isinstance(geom, THBSplineCls):
        return _thb_control_polygon(geom)

    pv = _import_pyvista()
    pts_3d, grid_shape, _ = _get_euclidean_control_points(geom)
    dim = len(grid_shape)
    lines = _build_polygon_lines(grid_shape, dim)

    if not lines:
        return pv.PolyData(pts_3d)  # type: ignore[no-any-return]

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
