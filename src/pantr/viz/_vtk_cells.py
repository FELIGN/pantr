"""Convert Bézier and B-spline objects to pyvista UnstructuredGrids.

Implements the core pipeline:
``Bspline/Bezier → open form → Bézier decomposition → VTK Bézier cells``

Uses native VTK higher-order Bézier cell types (``VTK_BEZIER_CURVE``,
``VTK_BEZIER_QUADRILATERAL``, ``VTK_BEZIER_HEXAHEDRON``) which render exact
polynomial geometry without tessellation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy import typing as npt

from ._lazy_import import _import_pyvista
from ._vtk_ordering import vtk_ordering

if TYPE_CHECKING:
    import pyvista as pv

    from ..bezier import Bezier
    from ..bspline import Bspline

# VTK cell type constants for Bézier cells.
VTK_BEZIER_CURVE = 75
VTK_BEZIER_QUADRILATERAL = 77
VTK_BEZIER_HEXAHEDRON = 79

_VTK_CELL_TYPE_BY_DIM = {
    1: VTK_BEZIER_CURVE,
    2: VTK_BEZIER_QUADRILATERAL,
    3: VTK_BEZIER_HEXAHEDRON,
}

_MAX_PHYSICAL_DIM = 3
"""Maximum physical dimension for VTK coordinates."""


def _get_bezier_patches(
    geom: Bspline | Bezier,
) -> tuple[npt.NDArray[np.object_], bool]:
    """Extract Bézier patches from a geometry object.

    For a ``Bspline``, converts to open form if needed and decomposes into
    Bézier patches.  For a ``Bezier``, wraps it in an object array.

    Args:
        geom: Input geometry (Bspline or Bezier).

    Returns:
        tuple: ``(patches, is_rational)`` where *patches* is an object array
        of :class:`~pantr.bezier.Bezier` objects and *is_rational* is a bool.
    """
    from ..bezier import Bezier as BezierCls  # noqa: PLC0415
    from ..bspline import Bspline as BsplineCls  # noqa: PLC0415

    if isinstance(geom, BsplineCls):
        patches = geom.to_beziers()
        return patches, geom.is_rational
    if isinstance(geom, BezierCls):
        arr = np.empty((1,) * geom.dim, dtype=object)
        arr.flat[0] = geom
        return arr, geom.is_rational
    raise TypeError(f"Expected Bspline or Bezier, got {type(geom).__name__}")


@dataclass
class _PatchGeometry:
    """Intermediate representation of a single Bézier patch for VTK assembly."""

    points_3d: npt.NDArray[np.float64]
    """Control points in 3D VTK ordering, shape ``(n_pts, 3)``."""

    weights: npt.NDArray[np.float64] | None
    """Rational weights in VTK ordering, shape ``(n_pts,)``, or ``None``."""

    scalars: npt.NDArray[np.float64] | None
    """Scalar values in VTK ordering, shape ``(n_pts,)``, or ``None``."""


def _flatten_and_project(
    cp: npt.NDArray[np.floating[Any]],
    is_rational: bool,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None]:
    """Flatten control points and project rational ones to Euclidean space.

    Args:
        cp: Control points with shape ``(*degrees_plus_1, rank_or_rank_plus_1)``.
        is_rational: Whether the last coordinate is a homogeneous weight.

    Returns:
        tuple: ``(coords, weights)`` where *coords* has shape ``(n_pts, rank)``
        and *weights* is ``(n_pts,)`` or ``None``.
    """
    n_pts = int(np.prod(cp.shape[:-1]))
    flat_cp = cp.reshape(n_pts, -1).astype(np.float64)

    if is_rational:
        weights = flat_cp[:, -1].copy()
        coords = flat_cp[:, :-1] / weights[:, np.newaxis]
        return coords, weights
    return flat_cp, None


def _embed_scalar_field(
    scalar_vals: npt.NDArray[np.float64],
    parametric_coords: npt.NDArray[np.float64],
    dim: int,
    elevation: bool,
) -> npt.NDArray[np.float64]:
    """Embed a scalar field in 3D space for VTK rendering.

    Args:
        scalar_vals: Scalar values, shape ``(n_pts,)``.
        parametric_coords: Flat parametric coordinates, shape ``(n_pts, dim)``.
        dim: Parametric dimension.
        elevation: Use scalar as a spatial coordinate.

    Returns:
        NDArray[float64]: Points in 3D, shape ``(n_pts, 3)``.
    """
    n_pts = len(scalar_vals)
    pts_3d = np.zeros((n_pts, _MAX_PHYSICAL_DIM), dtype=np.float64)

    if elevation:
        # dim=1: (t, f(t), 0);  dim=2: (u, v, f(u,v))
        pts_3d[:, :dim] = parametric_coords
        pts_3d[:, min(dim, _MAX_PHYSICAL_DIM - 1)] = scalar_vals
    else:
        pts_3d[:, :dim] = parametric_coords
    return pts_3d


def _build_parametric_greville_coords(
    geom: Bspline | Bezier,
    bezier_index: tuple[int, ...],
) -> npt.NDArray[np.float64]:
    """Build parametric coordinates for control points of a Bézier patch.

    For each parametric direction, creates uniformly spaced points within
    the knot span corresponding to this Bézier patch.

    Args:
        geom: The parent geometry (Bspline or Bezier).
        bezier_index: Multi-index of this Bézier patch within the decomposition.

    Returns:
        NDArray[float64]: Array of shape ``(n_pts, dim)`` with parametric
        coordinates for each control point (already flattened).
    """
    from ..bezier import Bezier as BezierCls  # noqa: PLC0415
    from ..bspline import Bspline as BsplineCls  # noqa: PLC0415

    if isinstance(geom, BezierCls):
        dim = geom.dim
        degree = geom.degree
        grids_1d = [np.linspace(0.0, 1.0, degree[d] + 1) for d in range(dim)]
    else:
        assert isinstance(geom, BsplineCls)
        dim = geom.dim
        space = geom.space
        degree = geom.degree
        grids_1d = []
        for d in range(dim):
            sp1d = space.spaces[d]
            unique_knots, _ = sp1d.get_unique_knots_and_multiplicity(in_domain=True)
            t0 = float(unique_knots[bezier_index[d]])
            t1 = float(unique_knots[bezier_index[d] + 1])
            grids_1d.append(np.linspace(t0, t1, degree[d] + 1))

    mesh = np.meshgrid(*grids_1d, indexing="ij")
    coords = np.stack(mesh, axis=-1)
    n_pts = int(np.prod(coords.shape[:-1]))
    result: npt.NDArray[np.float64] = coords.reshape(n_pts, -1).astype(np.float64)
    return result


def _process_patch(  # noqa: PLR0913
    bezier: Bezier,
    geom: Bspline | Bezier,
    bezier_index: tuple[int, ...],
    is_rational: bool,
    rank: int,
    dim: int,
    elevation: bool,
    ordering: npt.NDArray[np.intp],
) -> _PatchGeometry:
    """Convert a single Bézier patch to VTK-ordered 3D geometry.

    Args:
        bezier: The Bézier patch to convert.
        geom: The parent geometry (for parametric coordinate computation).
        bezier_index: Multi-index of the patch within the decomposition.
        is_rational: Whether the geometry is rational.
        rank: Output rank (excluding weight).
        dim: Parametric dimension.
        elevation: Use scalar value as spatial coordinate.
        ordering: VTK point ordering permutation.

    Returns:
        _PatchGeometry: Processed patch with 3D points, weights, and scalars.
    """
    cp = bezier.control_points
    coords, weights = _flatten_and_project(cp, is_rational)
    scalars: npt.NDArray[np.float64] | None = None

    if rank == 1:
        scalar_vals = coords[:, 0].copy()
        param_coords = _build_parametric_greville_coords(geom, bezier_index)
        pts_3d = _embed_scalar_field(scalar_vals, param_coords, dim, elevation)
        scalars = scalar_vals[ordering]
    else:
        n_pts = coords.shape[0]
        pts_3d = np.zeros((n_pts, _MAX_PHYSICAL_DIM), dtype=np.float64)
        pts_3d[:, :rank] = coords

    return _PatchGeometry(
        points_3d=pts_3d[ordering],
        weights=weights[ordering] if weights is not None else None,
        scalars=scalars,
    )


def to_pyvista(
    geom: Bspline | Bezier,
    *,
    scalar_name: str = "scalar",
    elevation: bool = False,
) -> pv.UnstructuredGrid:
    """Convert a B-spline or Bézier geometry to a pyvista UnstructuredGrid.

    Uses native VTK Bézier cell types for exact polynomial rendering.
    Periodic/unclamped B-splines are automatically converted to open form.

    For scalar fields (``rank == 1``):

    - **dim=1**: always displayed as a line plot ``(t, f(t), 0)``.
    - **dim=2**: by default a flat color map on ``(u, v, 0)``; set
      ``elevation=True`` for ``(u, v, f(u,v))``.
    - **dim=3**: color map on ``(u, v, w)``.

    Args:
        geom: Input B-spline or Bézier geometry.
        scalar_name: Name for the scalar point data array when ``rank == 1``.
        elevation: For scalar fields with dim ≤ 2, use the scalar value as
            a spatial coordinate instead of a flat color map.  Ignored when
            ``rank > 1`` or ``dim == 1`` (which always uses elevation).

    Returns:
        pv.UnstructuredGrid: PyVista unstructured grid with VTK Bézier cells.

    Raises:
        ImportError: If pyvista is not installed.
        TypeError: If *geom* is not a ``Bspline`` or ``Bezier``.
        ValueError: If the parametric dimension is not 1, 2, or 3.
    """
    pv = _import_pyvista()

    patches, is_rational = _get_bezier_patches(geom)
    from ..bezier import Bezier as BezierCls  # noqa: PLC0415

    first_patch: BezierCls = patches.flat[0]  # type: ignore[assignment]
    dim, rank, degree = first_patch.dim, first_patch.rank, first_patch.degree

    if dim not in _VTK_CELL_TYPE_BY_DIM:
        raise ValueError(f"Unsupported parametric dimension {dim}.")

    cell_type = _VTK_CELL_TYPE_BY_DIM[dim]
    ordering = vtk_ordering(degree)
    n_pts_per_cell = len(ordering)
    effective_elevation = elevation or (rank == 1 and dim == 1)

    patch_data = [
        _process_patch(
            patches[idx],
            geom,
            idx,
            is_rational,
            rank,
            dim,
            effective_elevation,
            ordering,
        )
        for idx in np.ndindex(patches.shape)
    ]

    return _assemble_grid(
        pv,
        patch_data,
        cell_type,
        n_pts_per_cell,
        is_rational=is_rational,
        rank=rank,
        scalar_name=scalar_name,
    )


def _assemble_grid(  # noqa: PLR0913
    pv: Any,  # noqa: ANN401
    patch_data: list[_PatchGeometry],
    cell_type: int,
    n_pts_per_cell: int,
    *,
    is_rational: bool,
    rank: int,
    scalar_name: str,
) -> pv.UnstructuredGrid:
    """Assemble processed patches into a pyvista UnstructuredGrid.

    Args:
        pv: The pyvista module.
        patch_data: List of processed patch geometries.
        cell_type: VTK cell type constant.
        n_pts_per_cell: Number of points per cell.
        is_rational: Whether to attach rational weights.
        rank: Output rank of the geometry.
        scalar_name: Name for scalar point data.

    Returns:
        pv.UnstructuredGrid: Assembled grid with cell data and point arrays.
    """
    all_points = [p.points_3d for p in patch_data]
    cells: list[npt.NDArray[np.intp]] = []
    point_offset = 0

    for _ in patch_data:
        conn = np.empty(n_pts_per_cell + 1, dtype=np.intp)
        conn[0] = n_pts_per_cell
        conn[1:] = np.arange(point_offset, point_offset + n_pts_per_cell)
        cells.append(conn)
        point_offset += n_pts_per_cell

    points = np.vstack(all_points)
    cell_array = np.concatenate(cells)
    cell_type_array = np.full(len(patch_data), cell_type, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cell_array, cell_type_array, points)

    if is_rational:
        weight_arrays = [p.weights for p in patch_data if p.weights is not None]
        if weight_arrays:
            grid.point_data["RationalWeights"] = np.concatenate(weight_arrays)

    if rank == 1:
        scalar_arrays = [p.scalars for p in patch_data if p.scalars is not None]
        if scalar_arrays:
            grid.point_data[scalar_name] = np.concatenate(scalar_arrays)

    return grid  # type: ignore[no-any-return]


def save(
    geom: Bspline | Bezier,
    filename: str | Path,
    *,
    scalar_name: str = "scalar",
    elevation: bool = False,
) -> None:
    """Export a B-spline or Bézier geometry to a VTK file.

    Converts the geometry to VTK Bézier cells and saves using pyvista.
    The file format is inferred from the extension (``.vtu`` recommended,
    ``.vtk`` for legacy format).

    ParaView ≥ 5.10 renders VTK Bézier cells natively with exact geometry.

    Args:
        geom: Input B-spline or Bézier geometry.
        filename: Output file path. Extension determines format.
        scalar_name: Name for scalar point data when ``rank == 1``.
        elevation: For scalar fields with dim ≤ 2, use scalar as
            spatial coordinate.

    Raises:
        ImportError: If pyvista is not installed.
    """
    grid = to_pyvista(geom, scalar_name=scalar_name, elevation=elevation)
    grid.save(str(filename))
