"""Constructive geometry for B-spline curves, surfaces, and volumes.

This package provides CAD-style functions for creating and combining
B-spline geometric objects:

- **Primitives**: :func:`line`, :func:`circle`, :func:`bilinear`,
  :func:`trilinear`.
- **Compatibility**: :func:`compat`.
- **Operations**: :func:`extrude`, :func:`ruled`.
"""

from ._compat import compat
from ._operations import extrude, ruled
from ._primitives import bilinear, circle, line, trilinear
from ._validation import _PHYSICAL_DIM, _pad_to_3d, _promote_to_rational

__all__ = [
    "_PHYSICAL_DIM",
    "_pad_to_3d",
    "_promote_to_rational",
    "bilinear",
    "circle",
    "compat",
    "extrude",
    "line",
    "ruled",
    "trilinear",
]
