"""Constructive geometry for B-spline curves, surfaces, and volumes.

This package provides CAD-style functions for creating and combining
B-spline geometric objects:

- **Primitives**: :func:`create_line`, :func:`create_circle`,
  :func:`create_bilinear`, :func:`create_trilinear`.
- **Derived primitives**: :func:`create_rectangle`, :func:`create_disk`,
  :func:`create_cylinder`.
- **Compatibility**: :func:`make_compat`.
- **Operations**: :func:`extrude`, :func:`revolve`, :func:`create_ruled`,
  :func:`sweep`.
- **Coons blending**: :func:`create_coons_surface`, :func:`create_coons_volume`.
- **Assembly**: :func:`join`.
"""

from ._compat import make_compat
from ._coons import create_coons_surface, create_coons_volume
from ._derived import create_cylinder, create_disk, create_rectangle
from ._join import join
from ._operations import create_ruled, extrude, revolve, sweep
from ._primitives import create_bilinear, create_circle, create_line, create_trilinear

__all__ = [
    "create_bilinear",
    "create_circle",
    "create_coons_surface",
    "create_coons_volume",
    "create_cylinder",
    "create_disk",
    "create_line",
    "create_rectangle",
    "create_ruled",
    "create_trilinear",
    "extrude",
    "join",
    "make_compat",
    "revolve",
    "sweep",
]
