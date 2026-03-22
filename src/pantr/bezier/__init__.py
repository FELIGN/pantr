"""Bezier geometric objects: the Bezier class.

This module provides :class:`Bezier`, which stores control points to represent
a parametric Bezier curve, surface, or volume. Degree is derived from the
control point array shape. Evaluation and manipulation methods use direct
Bernstein algorithms implemented in ``_bezier_core``, ``_bezier_eval``,
``_bezier_derivative``, ``_bezier_degree``, and ``_bezier_product``.
"""

from ._bezier import Bezier

__all__ = ["Bezier"]
