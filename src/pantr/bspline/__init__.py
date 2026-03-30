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
"""

from ._bspline import Bspline
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

__all__ = [
    "Bspline",
    "BsplineSpace",
    "BsplineSpace1D",
    "create_cardinal_knots",
    "create_greville_lattice",
    "create_uniform_open_knots",
    "create_uniform_periodic_knots",
    "create_uniform_space",
    "fit_bspline",
    "get_greville_abscissae",
    "interpolate_bspline",
    "l2_project_bspline",
]
