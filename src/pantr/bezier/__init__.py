"""Bezier geometric objects and Bernstein polynomial root finding.

This module provides :class:`Bezier`, which stores control points to represent
a parametric Bezier curve, surface, or volume. Degree is derived from the
control point array shape. Evaluation and manipulation methods use direct
Bernstein algorithms implemented in ``_bezier_core``, ``_bezier_eval``,
``_bezier_derivative``, ``_bezier_degree``, and ``_bezier_product``.

Root-finding exports:

- :func:`find_roots` -- find all roots (single polynomial, auto-dispatch).
- :func:`find_roots_batch` -- find roots of many same-degree polynomials.
- :func:`solve_monotone_root` -- fast solver for a single monotone polynomial.
- :func:`solve_monotone_root_batch` -- batch-parallel monotone solver.
"""

from ._bezier import Bezier
from ._root_finding import (
    find_roots,
    find_roots_batch,
    solve_monotone_root,
    solve_monotone_root_batch,
)

__all__ = [
    "Bezier",
    "find_roots",
    "find_roots_batch",
    "solve_monotone_root",
    "solve_monotone_root_batch",
]
