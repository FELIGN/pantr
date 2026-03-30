"""Bezier geometric objects and approximation functions.

This module provides :class:`Bezier`, which stores control points to represent
a parametric Bezier curve, surface, or volume. Degree is derived from the
control point array shape. Evaluation and manipulation methods use direct
Bernstein algorithms implemented in ``_bezier_core``, ``_bezier_eval``,
``_bezier_derivative``, ``_bezier_degree``, and ``_bezier_product``.

- :func:`interpolate_bezier`, :func:`fit_bezier`: approximation functions.
"""

from ._bezier import Bezier
from ._bezier_interpolate import fit_bezier, interpolate_bezier

__all__ = ["Bezier", "fit_bezier", "interpolate_bezier"]
