# Core concepts

This page is the mental model behind PaNTr. Read it once and the rest of the library —
the API reference, the {doc}`tutorials </tutorials/index>`, and the topic guides — falls
into place. Everything here is illustrated end-to-end in
{doc}`/tutorials/01_first_bspline`.

## Spaces vs. geometry

PaNTr keeps a strict separation between a **function space** (a basis) and a
**geometry** (a map built from that basis). This is the same split that underpins
isogeometric analysis, and it is worth internalizing:

| Object | What it is | Holds |
|---|---|---|
| {class}`~pantr.bspline.BsplineSpace1D` | a **1-D** spline space | one knot vector + a degree |
| {class}`~pantr.bspline.BsplineSpace` | a **tensor-product** space | a tuple of 1-D spaces |
| {class}`~pantr.bspline.Bspline` | a **geometry** | a space + control points |

A space defines *which functions exist* (the basis) and the parametric dimension; it
holds no shape. A geometry pairs that basis with coefficients (control points) to give
an actual curve, surface, or volume:

```python
import numpy as np
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D

# A quadratic 1-D space on the knot vector [0,0,0,1,2,3,3,3]
line_space = BsplineSpace1D([0, 0, 0, 1, 2, 3, 3, 3], 2)

# A 1-direction tensor-product space (a curve space)
space = BsplineSpace([line_space])

# Geometry = space + control points
control_points = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, -1.0], [3.0, 1.5], [4.0, 0.0]])
curve = Bspline(space, control_points)
```

The same space can carry many different geometries (just swap the control points), and
the same control points mean different things in different spaces. Keeping the two
apart is what makes refinement, change of basis, and assembly clean.

## Parametric dimension (`dim`) vs. embedding rank (`rank`)

Two independent integers describe every geometry:

- **`dim`** — the number of parametric directions, i.e. how many 1-D spaces the
  tensor-product space has. ``dim == 1`` is a curve, ``2`` a surface, ``3`` a volume.
- **`rank`** — the dimension of the values the map produces (its codomain).

They are orthogonal — a surface (``dim == 2``) can live in the plane (``rank == 2``) or
in space (``rank == 3``), and a ``rank == 1`` geometry is a **scalar field** over its
parametric domain:

| `dim` | `rank` | Meaning |
|---|---|---|
| 1 | 2 or 3 | planar / spatial **curve** |
| 2 | 2 or 3 | planar / spatial **surface** |
| 3 | 3 | **volume** |
| 1, 2, 3 | 1 | a **scalar field** ``f(u)`` / ``f(u, v)`` / ``f(u, v, w)`` |

{attr}`~pantr.bspline.Bspline.dim` and {attr}`~pantr.bspline.Bspline.rank` report these.

## Control points

The control points are the coefficients of the geometry in its space's basis. They are
stored as a single array of shape ``(*num_basis, rank)``: one coefficient vector per
basis function, laid out on the tensor-product grid of basis functions
({attr}`~pantr.bspline.BsplineSpace.num_basis` gives the per-direction counts). The
curve does **not** pass through its control points; it lies in their convex hull and is
pulled toward them — the *control net*.

{attr}`~pantr.bspline.Bspline.control_points` returns the array (read-only). The natural
parameter value attached to each control point is its **Greville abscissa**
({func}`~pantr.bspline.get_greville_abscissae`), which is where interpolation places its
data by default (see {doc}`/tutorials/05_approximation`).

## Rational geometry (NURBS)

Polynomials cannot represent a circle or any other conic exactly; *rational* polynomials
can. A rational B-spline (a NURBS) attaches a positive **weight** to each control point
and evaluates the weighted average.

PaNTr stores NURBS in the standard **homogeneous (projective)** representation: the
control-point array carries one extra trailing coordinate, the weight ``w``, and the
spatial coordinates are pre-multiplied by it (``[w·x, w·y, …, w]``). Evaluation maps the
homogeneous B-spline and divides by the weight, so {meth}`~pantr.bspline.Bspline.evaluate`
returns ordinary Euclidean points. Consequences worth remembering:

- {attr}`~pantr.bspline.Bspline.is_rational` is ``True``; the stored
  {attr}`~pantr.bspline.Bspline.control_points` include the weight column
  (shape ``(*num_basis, rank + 1)``), but {attr}`~pantr.bspline.Bspline.rank`
  **excludes** it (i.e. ``rank`` is the geometric dimension without the weight).
- {func}`~pantr.cad.create_circle` (and the conic-based CAD operations) produce exact
  rational quadratics — see {doc}`/tutorials/01_first_bspline` and {doc}`/guide/cad`.

## The parametric domain

A geometry is a map from a box-shaped **parametric domain** into ``rank``-space. The
domain of each direction is ``[first knot, last knot]``;
{attr}`~pantr.bspline.BsplineSpace.domain` returns the per-axis bounds, and
{class}`pantr.geometry.AABB` is the axis-aligned box primitive PaNTr uses for both
parametric domains and grid-cell bounds.

{meth}`~pantr.bspline.Bspline.evaluate` accepts either a plain ``(npts, dim)`` array of
parameters or a {class}`~pantr.quad.PointsLattice` (a tensor product of per-axis
parameter vectors), which is the efficient way to sample on a grid.

## Continuity comes from knot multiplicity

Inside a knot span a B-spline is a polynomial of the given degree; across an interior
knot of multiplicity ``m`` it is ``C^{p-m}`` continuous. Repeating a knot lowers
smoothness; a knot of multiplicity ``p+1`` (as at the clamped ends of an *open* knot
vector) breaks the geometry into independent pieces. This is the lever behind knot
insertion, Bézier extraction, and adaptive refinement — all covered in
{doc}`/guide/spaces-knots`.

## Bézier and THB-splines

Two specializations reuse the same model:

- A {class}`~pantr.bezier.Bezier` is a single polynomial patch in Bernstein form — the
  ``dim``/``rank``/control-point vocabulary is identical, but there are no interior
  knots (one element). Every element of a B-spline *is* a Bézier patch, recovered with
  {meth}`~pantr.bspline.Bspline.to_beziers`.
- A {class}`~pantr.bspline.THBSplineSpace` is a hierarchy of nested tensor-product
  spaces for **adaptive local refinement**; its API mirrors
  {class}`~pantr.bspline.BsplineSpace` (see {doc}`/tutorials/08_thb_adaptive_refinement`).

## Library shape

The serial core ({mod}`pantr.bspline`, {mod}`pantr.bezier`, {mod}`pantr.basis`,
{mod}`pantr.cad`, {mod}`pantr.grid`, {mod}`pantr.quad`, …) depends only on NumPy, SciPy,
and Numba. Two heavier capabilities are **opt-in extras** and are never imported by the
core:

- {mod}`pantr.viz` (the ``viz`` extra) — PyVista/VTK rendering and export
  ({doc}`/guide/visualization`).
- {mod}`pantr.mpi` (the ``mpi`` extra) — MPI-distributed spaces
  ({doc}`/guide/distributed`).

Geometric predicates that need a floating-point tolerance (knot-multiplicity tests,
endpoint detection) draw it from {mod}`pantr.tolerance`, so the whole library shares one
consistent notion of "equal".

Continue with {doc}`/tutorials/index` to build something, or {doc}`/guide/spaces-knots`
for knot vectors, refinement, and element extraction.
