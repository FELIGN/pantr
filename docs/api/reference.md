# API reference

Complete signatures for every public module, class, and function. If you are new to
PaNTr, start with {doc}`/guide/concepts` for the data model and the {doc}`/tutorials/index`
for worked examples; this page is the exhaustive contract.

## Module map

| Module | Purpose | Learn more |
|---|---|---|
| {mod}`pantr.bspline` | B-spline / NURBS / THB spaces and geometry; knots, refinement, extraction | {doc}`/guide/spaces-knots` |
| {mod}`pantr.basis` | tabulate 1-D polynomial bases (Bernstein, Lagrange, Legendre, cardinal) | {doc}`/tutorials/06_polynomial_bases` |
| {mod}`pantr.change_basis` | exact change-of-basis matrices between those bases | {doc}`/tutorials/06_polynomial_bases` |
| {mod}`pantr.bezier` | single-patch Bézier geometry and Bernstein root finding | {doc}`/tutorials/07_bezier_and_roots` |
| {mod}`pantr.cad` | constructive geometry: primitives + operations | {doc}`/guide/cad` |
| {mod}`pantr.transform` | affine transforms acting exactly on control points | {doc}`/tutorials/10_transforms` |
| {mod}`pantr.geometry` | the {class}`~pantr.geometry.AABB` box primitive | {doc}`/guide/concepts` |
| {mod}`pantr.grid` | structured / hierarchical cell grids, BVH, tags | {doc}`/tutorials/09_grids_and_quadrature` |
| {mod}`pantr.quad` | quadrature rules and point lattices | {doc}`/tutorials/09_grids_and_quadrature` |
| {mod}`pantr.tolerance` | shared floating-point tolerance policy | {doc}`/guide/concepts` |
| {mod}`pantr.mpi` | optional MPI-distributed spaces (`mpi` extra) | {doc}`/guide/distributed` |
| {mod}`pantr.viz` | optional PyVista/VTK rendering & export (`viz` extra) | {doc}`/guide/visualization` |

## Splines & bases

```{eval-rst}
.. automodule:: pantr.bspline
   :members:
   :show-inheritance:

.. automodule:: pantr.basis
   :members:
   :show-inheritance:

.. automodule:: pantr.change_basis
   :members:
   :show-inheritance:

.. automodule:: pantr.bezier
   :members:
   :show-inheritance:
```

## Geometry & CAD

```{eval-rst}
.. automodule:: pantr.cad
   :members:
   :show-inheritance:

.. automodule:: pantr.transform
   :members:
   :show-inheritance:

.. automodule:: pantr.geometry
   :members:
   :show-inheritance:
```

## Grids & quadrature

```{eval-rst}
.. automodule:: pantr.grid
   :members:
   :show-inheritance:

.. automodule:: pantr.quad
   :members:
   :show-inheritance:

.. automodule:: pantr.tolerance
   :members:
   :show-inheritance:
```

## Parallelism & visualization

Thread-pool control for the Numba kernels (see {doc}`/guide/parallelism`):

```{eval-rst}
.. autofunction:: pantr.get_num_threads
.. autofunction:: pantr.set_num_threads
.. autofunction:: pantr.num_threads
```

```{eval-rst}
.. automodule:: pantr.mpi
   :members:
   :show-inheritance:

.. automodule:: pantr.viz
   :members:
   :show-inheritance:
```
