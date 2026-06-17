# PaNTr

Polynomial and NURBS Toolkit (**PaNTr**) is a pure Python 3.10–3.14 library for geometric modeling and numerical analysis using [NumPy](https://numpy.org), [SciPy](https://scipy.org), and [Numba](https://numba.pydata.org).

## Features

- **B-spline & NURBS spaces** — univariate and tensor-product `BsplineSpace`,
  exact rational (NURBS) geometry, evaluation and derivatives, knot
  insertion/removal, degree elevation, and splitting.
- **Bézier toolkit** — Bernstein/Bézier curves and patches, composition,
  products, degree reduction, and Bernstein-polynomial root finding.
- **Truncated hierarchical B-splines** — `THBSplineSpace` with local
  refinement, mirroring the tensor-product API.
- **Constructive geometry (`pantr.cad`)** — lines, circles and arcs, disks,
  cylinders, extrude, revolve, sweep, ruled and Coons surfaces/volumes.
- **Structured grids (`pantr.grid`)** — tensor-product and hierarchical grids,
  BVH spatial queries, [dolfinx](https://github.com/FEniCS/dolfinx)-style
  cell/facet tags, and cell quadrature.
- **Quadrature and change of basis** — Gauss–Legendre and tensor-product
  rules, plus exact change of basis operators between Bernstein, Lagrange,
  monomial, and cardinal B-spline bases.
- **Fast and typed** — Numba-JIT kernels parallelized over CPU cores, with
  strict type hints across the public API.
- **Optional MPI and visualization** — distribute spaces across ranks
  (`pantr.mpi`) and render exact higher-order geometry through
  [PyVista](https://docs.pyvista.org) and [VTK](https://vtk.org) (`pantr.viz`).

## Installation

```bash
pip install pantr
```

Requires Python 3.10–3.14. The serial core depends only on NumPy, SciPy,
Numba, and [threadpoolctl](https://github.com/joblib/threadpoolctl).

To install the latest development version from source:

```bash
git clone https://github.com/pantolin/pantr.git
cd pantr
pip install .
```

### Optional features

The serial core (`pantr.grid`, `pantr.bspline`, ...) has no optional dependencies.
Extra features are opt-in via extras:

| Extra | Enables | Pulls in |
|---|---|---|
| `mpi` | distributed spaces (`pantr.mpi`) | [`mpi4py`](https://github.com/mpi4py/mpi4py) (needs an MPI library) |
| `metis` | [METIS](https://github.com/KarypisLab/METIS) graph partitioning backend | [`pymetis`](https://github.com/inducer/pymetis) |
| `viz` | visualization (`pantr.viz`) | [`pyvista`](https://docs.pyvista.org) ([VTK](https://vtk.org)) |
| `docs` | building the documentation | [Sphinx](https://www.sphinx-doc.org) stack |

```bash
pip install "pantr[mpi]"        # e.g. distributed spaces
pip install "pantr[mpi,viz]"    # several extras at once
```

The serial core never imports `pantr.mpi`, so it works identically with or without
the `mpi` extra.

## Development

```bash
pip install -e ".[dev]"   # includes all optional feature extras
```

## License

PaNTr is licensed under the MIT License. See `LICENSE` for details.
