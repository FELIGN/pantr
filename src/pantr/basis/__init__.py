"""Basis function evaluation for various polynomial bases.

This package provides high-level functions for tabulating basis functions
(Bernstein, Lagrange, cardinal B-spline, Legendre) in 1D and multi-dimensional
settings. It handles input normalization and dispatches to specialized Layer 2
and Layer 3 implementations.

- :class:`LagrangeVariant`: Lagrange polynomial node variants.
- :func:`tabulate_bernstein_1d`, :func:`tabulate_cardinal_bspline_1d`,
  :func:`tabulate_lagrange_1d`, :func:`tabulate_legendre_1d`: 1D tabulation.
- :func:`tabulate_bernstein`, :func:`tabulate_cardinal_bspline`,
  :func:`tabulate_lagrange`: multi-dimensional tabulation.
"""

from ._basis_tabulate import (
    LagrangeVariant,
    tabulate_bernstein,
    tabulate_bernstein_1d,
    tabulate_cardinal_bspline,
    tabulate_cardinal_bspline_1d,
    tabulate_lagrange,
    tabulate_lagrange_1d,
    tabulate_legendre_1d,
)

__all__ = [
    "LagrangeVariant",
    "tabulate_bernstein",
    "tabulate_bernstein_1d",
    "tabulate_cardinal_bspline",
    "tabulate_cardinal_bspline_1d",
    "tabulate_lagrange",
    "tabulate_lagrange_1d",
    "tabulate_legendre_1d",
]
