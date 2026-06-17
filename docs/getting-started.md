# Getting started

## Installation

PaNTr requires Python 3.11–3.14. The serial core depends only on NumPy, SciPy, Numba,
and threadpoolctl:

```bash
pip install pantr
```

### Optional features

Extra capabilities are opt-in via extras, so a plain install stays lightweight:

| Extra | Enables | Pulls in |
|---|---|---|
| `viz` | visualization & VTK export ({mod}`pantr.viz`) | `pyvista` |
| `mpi` | distributed spaces ({mod}`pantr.mpi`) | `mpi4py` (needs an MPI library) |
| `metis` | METIS graph-partitioning backend | `pymetis` |

```bash
pip install "pantr[viz]"        # e.g. to run the tutorials' 3-D scenes
pip install "pantr[mpi,viz]"    # several extras at once
```

The serial core never imports {mod}`pantr.mpi`, so it behaves identically with or
without the `mpi` extra.

## Your first spline

A geometry is a **function space** plus **control points**:

```python
import numpy as np
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D

# A quadratic 1-D space on the knot vector [0, 0, 0, 1, 2, 3, 3, 3]
space = BsplineSpace([BsplineSpace1D([0, 0, 0, 1, 2, 3, 3, 3], 2)])

# Five 2-D control points -> a planar curve (dim == 1, rank == 2)
control_points = np.array(
    [[0.0, 0.0], [1.0, 2.0], [2.0, -1.0], [3.0, 1.0], [4.0, 0.0]]
)
curve = Bspline(space, control_points)

# Evaluate at 50 parameters spanning the domain [0, 3]
points = curve.evaluate(np.linspace(0.0, 3.0, 50))  # shape (50, 2)
```

If `BsplineSpace1D`, `BsplineSpace`, and the `dim`/`rank` distinction are unfamiliar,
read {doc}`/guide/concepts` first — it is short and explains the whole data model.

## Next steps

- **Learn by doing** — work through the {doc}`/tutorials/index`, a runnable path from
  this first curve to NURBS, CAD modeling, approximation, and adaptive refinement.
- **Understand the model** — the {doc}`/guide/concepts` and {doc}`/guide/spaces-knots`
  pages cover spaces, knot vectors, continuity, and representation changes.
- **Visualize** — render geometries interactively or export them to VTK with
  {doc}`/guide/visualization` (needs the `viz` extra).
- **Look something up** — every symbol is in the {doc}`/api/reference`.
