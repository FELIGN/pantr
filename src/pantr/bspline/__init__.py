"""B-spline geometric objects, spaces, and knot vector factories.

This package consolidates the B-spline API:

- :class:`Bspline`: parametric B-spline curves, surfaces, and volumes.
- :class:`BsplineSpace1D`: 1D B-spline space (knot vector + degree).
- :class:`BsplineSpace`: multi-dimensional tensor-product B-spline spaces.
- :func:`create_uniform_open_knots`, :func:`create_uniform_periodic_knots`,
  :func:`create_cardinal_knots`: knot vector construction helpers.
- :func:`create_uniform_space`: convenience factory for tensor-product spaces.
- :func:`get_greville_abscissae`, :func:`create_greville_lattice`: Greville point
  utilities.
- :func:`interpolate_bspline`, :func:`fit_bspline`,
  :func:`l2_project_bspline`: approximation functions.
- :func:`create_from_bezier`: create a B-spline from a Bézier.
- :class:`SpanwiseElementExtraction`: lazy tensor-product change-of-basis
  operator across B-spline elements (Bézier/Lagrange/cardinal targets).
- :class:`ExtractionStructView`, :func:`make_struct_view`: Numba-passable
  bundle of a :class:`SpanwiseElementExtraction`'s compact storage for
  downstream ``@njit`` code.
- :class:`THBSplineSpace`: hierarchical B-spline space (truncated /
  non-truncated) on a :class:`pantr.grid.HierarchicalGrid`.
- :class:`MultiLevelExtraction`: per-element multi-level (Bézier) extraction
  operators for a :class:`THBSplineSpace`.
"""

from ._bspline import Bspline, create_from_bezier
from ._bspline_interpolate import fit_bspline, interpolate_bspline, l2_project_bspline
from ._bspline_space_1d import BsplineSpace1D
from ._bspline_space_factory import (
    create_cardinal_knots,
    create_greville_lattice,
    create_uniform_open_knots,
    create_uniform_periodic_knots,
    create_uniform_space,
    get_greville_abscissae,
)
from ._bspline_space_nd import BsplineSpace
from ._thb_spline_space import THBSplineSpace
from .multilevel_extraction import MultiLevelExtraction
from .spanwise_element_extraction import (
    ExtractionStructView,
    SpanwiseElementExtraction,
    make_struct_view,
)

__all__ = [
    "Bspline",
    "BsplineSpace",
    "BsplineSpace1D",
    "ExtractionStructView",
    "MultiLevelExtraction",
    "SpanwiseElementExtraction",
    "THBSplineSpace",
    "create_cardinal_knots",
    "create_from_bezier",
    "create_greville_lattice",
    "create_uniform_open_knots",
    "create_uniform_periodic_knots",
    "create_uniform_space",
    "fit_bspline",
    "get_greville_abscissae",
    "interpolate_bspline",
    "l2_project_bspline",
    "make_struct_view",
]
