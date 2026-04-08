"""High-order quadrature on domains implicitly defined by multivariate polynomials.

Implements the dimension-reduction algorithm of Saye (JCP 2022) entirely in
Numba nopython mode for near-C++ performance. The algorithm recasts implicitly
defined geometry as the graph of a multi-valued height function and applies
recursive dimension reduction down to one-dimensional quadrature.

The algorithm has two phases:

1. **Build phase**: Given tensor-product Bernstein polynomials defining the
   implicit geometry, construct a dimension-reduction hierarchy. This is done
   once per set of polynomials and can be reused for different quadrature orders.

2. **Construction phase**: Given the hierarchy and a quadrature order *q*,
   generate quadrature points and weights. Supports volume integrals
   (over {phi < 0}) and surface integrals (over {phi = 0}).

Main exports:

- :class:`ImplicitQuadrature` -- build + query interface.
"""

from pantr.bezier.implicit._convert_core import (
    monomial_to_bernstein_2d,
    monomial_to_bernstein_3d,
)
from pantr.bezier.implicit._implicit_quad import (
    ImplicitQuadrature,
    QuadStrategy,
    ReparamResult,
    SurfQuadResult,
    VolQuadResult,
)

__all__ = [
    "ImplicitQuadrature",
    "QuadStrategy",
    "ReparamResult",
    "SurfQuadResult",
    "VolQuadResult",
    "monomial_to_bernstein_2d",
    "monomial_to_bernstein_3d",
]
