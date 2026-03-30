"""Bezier geometric objects, approximation functions, and Bernstein polynomial root finding.

This module provides :class:`Bezier`, which stores control points to represent
a parametric Bezier curve, surface, or volume. Degree is derived from the
control point array shape. Evaluation and manipulation methods use direct
Bernstein algorithms implemented in ``_bezier_core``, ``_bezier_eval``,
``_bezier_derivative``, ``_bezier_degree``, and ``_bezier_product``.

- :func:`interpolate_bezier`, :func:`fit_bezier`: approximation functions.

Root-finding exports:

- :func:`find_roots` -- find all roots (single or batch, auto-dispatch).
- :func:`find_monotone_root` -- fast solver for monotone polynomials (single or batch).
"""

from ._bezier import Bezier
from ._bezier_interpolate import fit_bezier, interpolate_bezier
from ._root_finding import (
    find_monotone_root,
    find_roots,
)

__all__ = [
    "Bezier",
    "find_monotone_root",
    "find_roots",
    "fit_bezier",
    "interpolate_bezier",
]
