"""Multipatch topology and geometry for B-spline patches.

Describes how separate :class:`~pantr.bspline.Bspline` patches meet, which is the
spline-side knowledge a solver needs before it can unify degrees of freedom across
patch boundaries. Purely parametric and geometric: nothing here knows about
assembly, dof numbering, or parallel distribution.

Main exports:

- :class:`Interface`: an immutable record of one conforming face-to-face contact,
  carrying the tangential axis correspondence and orientations.
- :class:`MultiPatch`: a validated collection of patches with their interfaces,
  detected automatically when not supplied.
- :func:`detect_interfaces`: find the conforming interfaces among a set of patches.
- :func:`match_face_cps`: pair the flat control-point ids of two faces under a
  given axis correspondence (index arithmetic only).
"""

from ._detect import detect_interfaces
from ._interface import Interface
from ._match import match_face_cps
from ._multi_patch import MultiPatch

__all__ = [
    "Interface",
    "MultiPatch",
    "detect_interfaces",
    "match_face_cps",
]
