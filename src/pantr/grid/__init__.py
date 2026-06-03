"""Structured cell grids for the PaNTr stack.

This package provides a small, performance-conscious grid layer: a partition of
a parametric domain into cells with *implicit* (computed, not stored)
connectivity. It is the shared grid abstraction consumed by immersed / unfitted
discretizations, B-spline knot-span grids, and hierarchical refinement grids.

The :class:`TensorProductGrid` is deliberately low-footprint: it stores only the
per-axis breakpoint arrays and a little metadata, computing cell bounds,
neighbours, and ids on demand. The only ``O(num_cells)`` structure -- a
:class:`BVH` spatial index -- is built lazily on the first
:meth:`~Grid.query_aabb`, so a grid used purely as a B-spline's knot grid stays
proportional to its breakpoints, not its cell count.

Main exports:

- :class:`Grid`: abstract base class defining the grid contract and supplying
  axis-aligned box defaults for facets, neighbours, reference maps, batch point
  location, and spatial queries.
- :class:`TensorProductGrid`: concrete tensor-product grid of axis-aligned boxes
  with per-axis breakpoints and row-major (C-order) cell ids.
- :class:`HierarchicalGrid`: hierarchical grid with a fixed per-direction
  subdivision factor; active cells stored as rectangular blocks per level.
- :func:`uniform_grid`: build a uniform grid on a bounding box.
- :func:`tensor_product_grid`: build the knot-span grid of a
  :class:`pantr.bspline.BsplineSpace`.
- :func:`hierarchical_grid`: build a :class:`HierarchicalGrid` from a root
  :class:`TensorProductGrid` and a subdivision factor.
- :class:`BVH`: bounding-volume hierarchy over cell AABBs, backing
  :meth:`Grid.query_aabb`.
- :class:`CellTags`, :class:`FacetTags`: sparse, dolfinx-style named tag
  registries for cells and facets.
"""

from __future__ import annotations

from ._bvh import BVH
from ._grid import Grid
from ._hierarchical_grid import HierarchicalGrid, hierarchical_grid
from ._tags import CellTags, FacetTags
from ._tensor_product_grid import TensorProductGrid, tensor_product_grid, uniform_grid

__all__ = [
    "BVH",
    "CellTags",
    "FacetTags",
    "Grid",
    "HierarchicalGrid",
    "TensorProductGrid",
    "hierarchical_grid",
    "tensor_product_grid",
    "uniform_grid",
]
