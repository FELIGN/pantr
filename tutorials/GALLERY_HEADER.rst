:orphan:

Tutorials
=========

A guided, hands-on path through PaNTr. Each tutorial is a standalone, runnable script
in ``tutorials/`` whose output (plots and interactive 3-D scenes) is rendered below by
executing it. Read them in order the first time -- each builds on the last -- or jump to
whichever topic you need.

Pair the tutorials with :doc:`/guide/concepts` (the data model and vocabulary) and the
rest of the User Guide for the conceptual reference behind each step.

Running them yourself::

    pip install "pantr[viz]"          # the 3-D tutorials need the viz extra
    python tutorials/01_first_bspline.py

The 3-D scenes are interactive -- drag to rotate, scroll to zoom. The plotting scripts
use Numba kernels; on a fresh process you may hit a Numba background-warmup threading
error, in which case run with JIT disabled (the documentation build does this
automatically)::

    NUMBA_DISABLE_JIT=1 python tutorials/05_approximation.py

The path, in order:

#. **Your first B-spline** -- spaces, control points, evaluation, derivatives, an exact
   NURBS circle (:mod:`pantr.bspline`).
#. **Visualizing geometries** -- ``plot`` / ``Scene``, control polygons, knot lines,
   scalar fields, VTK export (:mod:`pantr.viz`).
#. **Knot operations & Bézier extraction** -- knot insertion, degree elevation, and the
   element-local Bézier pieces (:mod:`pantr.bspline`).
#. **Constructive CAD modeling** -- primitives plus extrude / revolve / ruled, assembled
   (:mod:`pantr.cad`).
#. **Approximation** -- interpolation, L2 projection, quasi-interpolation, and
   convergence (:mod:`pantr.bspline`).
#. **Polynomial bases & change of basis** -- Bernstein / Lagrange / Legendre and the
   matrices between them (:mod:`pantr.basis`, :mod:`pantr.change_basis`).
#. **Bézier patches & Bernstein root finding** -- a Bézier surface, root finding, a
   curve-line intersection (:mod:`pantr.bezier`).
#. **THB-splines** -- adaptive local refinement on a hierarchical mesh
   (:mod:`pantr.bspline`, :mod:`pantr.grid`).
#. **Grids & quadrature** -- ``cell_quadrature`` integration, a BVH query, grid
   rendering (:mod:`pantr.grid`, :mod:`pantr.quad`, :mod:`pantr.geometry`).
#. **Affine transformations** -- translation / rotation / scaling / shear and
   composition (:mod:`pantr.transform`).
