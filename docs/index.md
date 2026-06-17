# PaNTr

**PaNTr** (Polynomial And NURBS Toolkit) is a pure-Python library for geometric
modeling and numerical analysis with B-splines, NURBS, Bézier patches, and truncated
hierarchical B-splines. It is built on [NumPy](https://numpy.org), [SciPy](https://scipy.org), and [Numba](https://numba.pydata.org) — typed
across its public API and JIT-compiled for multi-core throughput — and targets
isogeometric analysis, CAD-style modeling, and scientific computing.

```python
import numpy as np
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D

space = BsplineSpace([BsplineSpace1D([0, 0, 0, 1, 2, 3, 3, 3], 2)])  # quadratic curve space
curve = Bspline(space, np.array([[0, 0], [1, 2], [2, -1], [3, 1], [4, 0]], dtype=float))
points = curve.evaluate(np.linspace(0, 3, 50))  # (50, 2)
```

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Getting started
:link: getting-started
:link-type: doc

Install PaNTr (and the optional extras) and run your first example.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc

A guided, runnable path from your first spline to adaptive refinement.
:::

:::{grid-item-card} User guide
:link: guide/concepts
:link-type: doc

The data model and topic-by-topic explanations behind the API.
:::

:::{grid-item-card} API reference
:link: api/reference
:link-type: doc

Every module, class, and function, with full signatures.
:::

::::

## What's inside

- **B-spline & NURBS spaces** — univariate and tensor-product
  {class}`~pantr.bspline.BsplineSpace`, exact rational geometry, evaluation and
  derivatives, knot insertion/removal, degree elevation, splitting.
- **Bézier toolkit** — Bernstein/Bézier curves and patches, composition, products,
  degree reduction, and Bernstein-polynomial root finding ({mod}`pantr.bezier`).
- **Truncated hierarchical B-splines** — {class}`~pantr.bspline.THBSplineSpace` for
  adaptive local refinement, mirroring the tensor-product API.
- **Constructive geometry** — lines, arcs, disks, cylinders, extrude, revolve, sweep,
  ruled and Coons surfaces/volumes ({mod}`pantr.cad`).
- **Grids & quadrature** — tensor-product and hierarchical grids, BVH spatial queries,
  cell/facet tags, and quadrature rules ({mod}`pantr.grid`, {mod}`pantr.quad`).
- **Optional MPI & visualization** — distribute spaces across ranks
  ({mod}`pantr.mpi`) and render exact higher-order geometry through
  [PyVista](https://docs.pyvista.org) and [VTK](https://vtk.org) ({mod}`pantr.viz`).

```{toctree}
:caption: Getting started
:maxdepth: 1
:hidden:

getting-started
```

```{toctree}
:caption: Tutorials
:maxdepth: 1
:hidden:

tutorials/index
```

```{toctree}
:caption: User guide
:maxdepth: 2
:hidden:

guide/concepts
guide/spaces-knots
guide/cad
guide/visualization
guide/parallelism
guide/distributed
```

```{toctree}
:caption: Reference
:maxdepth: 1
:hidden:

api/reference
changelog
```
