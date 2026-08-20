"""Quadrature rules and evaluation grid helpers for 1D integration.

This module provides:

- 1D quadrature rules on ``[0, 1]``: trapezoidal (equispaced), Gauss-Legendre,
  Gauss-Lobatto-Legendre, Chebyshev-Gauss (1st and 2nd kind).
- :class:`PointsLattice`: a multi-dimensional tensor-product evaluation grid.
- :func:`create_lagrange_points_lattice`: factory for Lagrange-node lattices.
- :class:`QuadratureRule`: a d-dimensional quadrature rule on the unit cube,
  with :func:`tensor_product_quadrature` and :func:`gauss_legendre_quadrature`
  factories; the reference rule consumed by :func:`pantr.grid.cell_quadrature`.
"""

from ._lattice import PointsLattice, create_lagrange_points_lattice
from ._rule_nd import QuadratureRule, gauss_legendre_quadrature, tensor_product_quadrature
from ._rules import (
    _TANH_SINH_DECAY_FACTOR as _TANH_SINH_DECAY_FACTOR,
)
from ._rules import (
    _generate_tanh_sinh as _generate_tanh_sinh,
)
from ._rules import (
    _scale_and_cast_nodes_and_weights as _scale_and_cast_nodes_and_weights,
)
from ._rules import (
    _tanh_sinh_min_gap as _tanh_sinh_min_gap,
)
from ._rules import (
    _validate_n_pts_and_dtype as _validate_n_pts_and_dtype,
)
from ._rules import (
    get_chebyshev_gauss_1st_kind_1d,
    get_chebyshev_gauss_2nd_kind_1d,
    get_gauss_legendre_1d,
    get_gauss_lobatto_legendre_1d,
    get_modified_chebyshev_nodes_1d,
    get_tanh_sinh_1d,
    get_trapezoidal_1d,
)

# Rebind __module__ so pickles and repr() report the public path, not the
# private submodule they are actually defined in.  Both classes are pickled
# by mpi4py for collective calls (see pantr.mpi._l2, pantr.mpi._collocation).
PointsLattice.__module__ = __name__
QuadratureRule.__module__ = __name__

__all__ = [
    "PointsLattice",
    "QuadratureRule",
    "create_lagrange_points_lattice",
    "gauss_legendre_quadrature",
    "get_chebyshev_gauss_1st_kind_1d",
    "get_chebyshev_gauss_2nd_kind_1d",
    "get_gauss_legendre_1d",
    "get_gauss_lobatto_legendre_1d",
    "get_modified_chebyshev_nodes_1d",
    "get_tanh_sinh_1d",
    "get_trapezoidal_1d",
    "tensor_product_quadrature",
]
