# Getting started

## Installation

PaNTr requires Python 3.11–3.14. The serial core depends only on NumPy, SciPy, Numba,
and threadpoolctl:

```bash
pip install pantr
```

### Optional features

A plain `pip install pantr` already includes **every** PaNTr module, including
{mod}`pantr.viz` and {mod}`pantr.mpi`. These two don't do anything on their own — each
needs a third-party *backend* library, and activates automatically as soon as that
library is importable. Until then, calling into the module raises a clear error; nothing
else is affected.

The extras below install **no extra PaNTr code** — they are a convenience that pulls in
the backend for you. Installing the backend yourself is exactly equivalent:

| Capability | Module | Backend | Install (extra *or* direct) |
|---|---|---|---|
| Visualization & VTK export | {mod}`pantr.viz` | `pyvista` | `pip install "pantr[viz]"` · `pip install pyvista` |
| Distributed (MPI) spaces | {mod}`pantr.mpi` | `mpi4py` (+ an MPI library) | `pip install "pantr[mpi]"` |
| METIS partitioning backend | {func}`pantr.bspline.partition_graph` | `pymetis` | `pip install "pantr[metis]"` |

```bash
pip install "pantr[viz]"        # enable visualization (installs pyvista)
pip install "pantr[mpi,viz]"    # several backends at once
```

The serial core never imports {mod}`pantr.mpi`, so a plain `pip install pantr` behaves
identically whether or not these backends are present.

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
