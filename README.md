# PaNTr

Polynomial and NURBS Toolkit (**PaNTr**) is a pure Python 3.10–3.14 library for geometric modeling and numerical analysis using **NumPy**, **SciPy**, and **Numba**.

## Features

- Precise polynomial and NURBS basis evaluation with strict type hints.
- Vectorized implementations tailored for scientific computing workflows.
- Comprehensive testing with `pytest`, coverage reporting, and type checking.
- Documentation powered by **Sphinx**, **MyST**, and related tooling.

## Installation

```bash
# Install directly from GitHub
pip install git+https://github.com/pantolin/pantr.git

# Or clone and install locally
git clone https://github.com/pantolin/pantr.git
cd pantr
pip install .
```

### Optional features

The serial core (`pantr.grid`, `pantr.bspline`, ...) has no optional dependencies.
Extra features are opt-in via extras:

| Extra | Enables | Pulls in |
|---|---|---|
| `mpi` | distributed spaces (`pantr.mpi`) | `mpi4py` (needs an MPI library) |
| `metis` | METIS graph partitioning backend | `pymetis` |
| `viz` | visualization (`pantr.viz`) | `pyvista` (VTK) |
| `docs` | building the documentation | Sphinx stack |

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
