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
- :class:`GridRestriction`: result of :meth:`Grid.restrict` -- a windowed
  sub-grid plus local-to-global cell index maps.
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
- :func:`cell_quadrature`: map a :class:`pantr.quad.QuadratureRule` from the
  unit cube onto a grid's cells (per-cell points and weights).
- :func:`overlay`: the coarsest :class:`TensorProductGrid` refining two input
  tensor-product grids (union of per-axis breakpoints on their domain overlap).
- :class:`Partition`: a per-cell owner assignment for distributing a grid's cells.
- :func:`partition_grid`: split a grid into ``n_parts`` rank subdomains (native,
  dependency-free; the ``"block"`` Cartesian backend).
"""

from __future__ import annotations

from ._bvh import BVH
from ._cell_quadrature import cell_quadrature
from ._grid import Grid, GridRestriction
from ._hierarchical_grid import HierarchicalGrid, hierarchical_grid
from ._overlay import overlay
from ._partition import Partition
from ._partition_grid import partition_grid
from ._tags import CellTags, FacetTags
from ._tensor_product_grid import TensorProductGrid, tensor_product_grid, uniform_grid

__all__ = [
    "BVH",
    "CellTags",
    "FacetTags",
    "Grid",
    "GridRestriction",
    "HierarchicalGrid",
    "Partition",
    "TensorProductGrid",
    "cell_quadrature",
    "hierarchical_grid",
    "overlay",
    "partition_grid",
    "tensor_product_grid",
    "uniform_grid",
]
