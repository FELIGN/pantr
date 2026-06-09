# PaNTr

Polynomial and NURBS Toolkit (**PaNTr**) is a pure Python 3.10–3.12 library for geometric modeling and numerical analysis using **NumPy**, **SciPy**, **Matplotlib**, and **Numba**.

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

By default PaNTr depends on `mpi4py` (for the optional `pantr.mpi` distribution
layer), which requires an MPI library at build time. For a serial-only, MPI-free
install, set `PANTR_NO_MPI` when building:

```bash
PANTR_NO_MPI=1 pip install .
```

The serial core (`pantr.grid`, `pantr.bspline`, ...) never imports `pantr.mpi`, so
it works identically with or without MPI.

## Development

```bash
pip install -e ".[dev,docs]"
```

## License

PaNTr is licensed under the MIT License. See `LICENSE` for details.
