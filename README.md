# PaNTr

Polynomial and NURBS Toolkit (**PaNTr**) is a pure Python 3.11–3.14 library for geometric modeling and numerical analysis using [NumPy](https://numpy.org), [SciPy](https://scipy.org), and [Numba](https://numba.pydata.org).

## Features

- **B-spline & NURBS spaces** — univariate and tensor-product `BsplineSpace`,
  exact rational (NURBS) geometry, evaluation and derivatives, knot
  insertion/removal, degree elevation, and splitting.
- **Bézier toolkit** — Bernstein/Bézier curves and patches, composition,
  products, degree reduction, and Bernstein-polynomial root finding.
- **Truncated hierarchical B-splines** — `THBSplineSpace` with local
  refinement, mirroring the tensor-product API.
- **Constructive geometry (`pantr.cad`)** — lines, circles and arcs, disks,
  cylinders, extrusion, revolution, sweep, ruled and Coons surfaces/volumes.
- **Structured grids (`pantr.grid`)** — tensor-product and hierarchical grids,
  BVH spatial queries, [dolfinx](https://github.com/FEniCS/dolfinx)-style
  cell/facet tags, and cell quadrature.
- **Quadrature and change of basis** — Gauss–Legendre and tensor-product
  rules, plus exact change of basis operators between Bernstein, Lagrange,
  monomial, and cardinal B-spline bases.
- **Fast and typed** — Numba-JIT kernels parallelized over CPU cores, with
  strict type hints across the public API.
- **Dependency-gated MPI and visualization** — `pantr.mpi` distributes spaces across
  ranks once [`mpi4py`](https://github.com/mpi4py/mpi4py) is installed, and `pantr.viz`
  renders exact higher-order geometry through [PyVista](https://docs.pyvista.org) /
  [VTK](https://vtk.org) once PyVista is installed. Both modules ship with PaNTr; only
  their backends are optional.

## Installation

```bash
pip install pantr
```

Requires Python 3.11–3.14. The serial core depends only on NumPy, SciPy,
Numba, and [threadpoolctl](https://github.com/joblib/threadpoolctl).

To install the latest development version from source:

```bash
git clone https://github.com/FELIGN/pantr.git
cd pantr
pip install .
```

### Optional features

Every install of PaNTr includes **all** of its modules — `pantr.viz` and `pantr.mpi`
among them. Those two stay dormant until their third-party backend is importable, and
activate automatically once it is; until then, calling into them raises a clear,
actionable error (nothing else is affected). The "extras" below add **no PaNTr code** —
they are just a convenience that installs the backend for you, so `pip install
"pantr[viz]"` and `pip install pyvista` are equivalent.

| Capability | Module | Backend it needs | Install (extra or direct) |
|---|---|---|---|
| Visualization & VTK export | `pantr.viz` | [`pyvista`](https://docs.pyvista.org) (+ [VTK](https://vtk.org)) | `pip install "pantr[viz]"` · or `pip install pyvista` |
| Distributed (MPI) spaces | `pantr.mpi` | [`mpi4py`](https://github.com/mpi4py/mpi4py) (+ an MPI library) | `pip install "pantr[mpi]"` |
| METIS partitioning backend | `pantr.bspline.partition_graph` | [`pymetis`](https://github.com/inducer/pymetis) | `pip install "pantr[metis]"` |

The serial core (`pantr.grid`, `pantr.bspline`, …) never imports `pantr.mpi`, so a plain
`pip install pantr` behaves identically with or without these backends. Contributors can
grab everything — including the docs toolchain ([Sphinx](https://www.sphinx-doc.org)) —
with `pip install -e ".[dev]"`.

## Development

```bash
pip install -e ".[dev]"   # includes all optional feature extras
```

## License

PaNTr is licensed under the MIT License. See `LICENSE` for details.
