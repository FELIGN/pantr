"""B-spline geometric objects, spaces, and knot vector factories.

This package consolidates the B-spline API:

- :class:`Bspline`: parametric B-spline curves, surfaces, and volumes.
- :class:`BsplineSpace1D`: 1D B-spline space (knot vector + degree).
- :class:`BsplineSpace`: multi-dimensional tensor-product B-spline spaces.
- :func:`create_uniform_open`, :func:`create_uniform_periodic`,
  :func:`create_cardinal`: knot vector construction helpers.
- :func:`create_uniform_space`: convenience factory for tensor-product spaces.
- :func:`greville_abscissae`, :func:`greville_lattice`: Greville point
  utilities.
"""

from ._bspline import Bspline
from ._bspline_space_1d import BsplineSpace1D
from ._bspline_space_factory import (
    create_cardinal,
    create_uniform_open,
    create_uniform_periodic,
    create_uniform_space,
    greville_abscissae,
    greville_lattice,
)
from ._bspline_space_nd import BsplineSpace

__all__ = [
    "Bspline",
    "BsplineSpace",
    "BsplineSpace1D",
    "create_cardinal",
    "create_uniform_open",
    "create_uniform_periodic",
    "create_uniform_space",
    "greville_abscissae",
    "greville_lattice",
]
