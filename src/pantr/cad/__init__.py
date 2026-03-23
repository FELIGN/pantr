"""Constructive geometry for B-spline curves, surfaces, and volumes.

This package provides CAD-style functions for creating and combining
B-spline geometric objects:

- **Primitives**: :func:`line`, :func:`circle`, :func:`bilinear`,
  :func:`trilinear`.
- **Derived primitives**: :func:`rectangle`, :func:`disk`,
  :func:`cylinder`.
- **Compatibility**: :func:`compat`.
- **Operations**: :func:`extrude`, :func:`revolve`, :func:`ruled`,
  :func:`sweep`.
- **Coons blending**: :func:`coons_surface`, :func:`coons_volume`.
- **Assembly**: :func:`join`.
"""

from ._compat import compat
from ._coons import coons_surface, coons_volume
from ._derived import cylinder, disk, rectangle
from ._join import join
from ._operations import extrude, revolve, ruled, sweep
from ._primitives import bilinear, circle, line, trilinear
from ._validation import _PHYSICAL_DIM, _pad_to_3d, _promote_to_rational

__all__ = [
    "_PHYSICAL_DIM",
    "_pad_to_3d",
    "_promote_to_rational",
    "bilinear",
    "circle",
    "compat",
    "coons_surface",
    "coons_volume",
    "cylinder",
    "disk",
    "extrude",
    "join",
    "line",
    "rectangle",
    "revolve",
    "ruled",
    "sweep",
    "trilinear",
]
