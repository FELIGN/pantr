# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Always run commands in the `pantr` conda environment: `conda activate pantr`
- Always use git worktrees for implementing changes (via the `EnterWorktree` tool or `git worktree add`)

## Commands

```bash
pip install -e ".[dev]"                                              # install with dev deps
pytest --no-cov                                                      # run tests (JIT enabled)
pytest tests/test_basis.py::test_name --no-cov -v                   # single test
pytest tests/ -k "keyword" --no-cov -v                              # filtered tests
NUMBA_DISABLE_JIT=1 pytest --cov=src/pantr --cov-report=xml         # coverage (JIT disabled)
ruff check .                                                         # lint
ruff format .                                                        # format
mypy --config-file mypy.ini src tests                               # type check
```

> Default pytest (via `pytest.ini`) adds `--cov`, which slows things down and requires ≥85% coverage. Always pass `--no-cov` during development.

## Architecture

**PaNTr** is a polynomial and NURBS toolkit for geometric modeling and numerical analysis (Python 3.10–3.12).

### Layers

The library is organized in three strict layers. Each layer has a well-defined responsibility and never duplicates work from the layer below.

**Layer 1 — Public API** (`basis.py`, `bspline_space_1D.py`, `bspline_space_nd.py`, `change_basis.py`, `quad.py`, `tolerance.py`):
- Exposes `tabulate_*_basis()` functions, `BsplineSpace1D`/`BsplineSpace` classes, quadrature helpers, and change-of-basis matrices
- Performs only lightweight validation (e.g. degree ≥ 0, dimension ≥ 1); delegates everything else to Layer 2

**Layer 2 — Implementation helpers** (`_basis_1D.py`, `_bspline_basis_core.py`, `_basis_multidim.py`, `_bspline_extraction.py`, …):
- Does all substantive input validation: shapes, dtypes, domain membership, writability of output arrays
- Allocates or validates `out` arrays, reshapes them as needed, then calls Layer 3 kernels
- Never called directly by users; no Numba inside this layer

**Layer 3 — Numba kernels** (`_basis_core.py`, `_basis_lagrange.py`, `_bspline_basis_core.py` core functions, …):
- Pure computation: Cox–de Boor, de Boor, Lagrange evaluation, etc.
- Decorated with `@nb_jit(nopython=True, cache=True, parallel=True)` and `prange` for multi-core throughput
- **No input validation whatsoever** — docstrings explicitly state this. All correctness guarantees come from Layer 2.

Other private modules: `_numba_compat.py` (Numba shim), `_basis_utils.py` (shared validation helpers), `__init__.py` (async Numba warmup at import time).

### Input validation policy

- Validation lives exclusively in **Layer 2**. Layer 1 checks only trivial preconditions; Layer 3 checks nothing.
- `_basis_utils.py` provides reusable validators (`_validate_out_array_1D`, `_validate_out_array_3d_float`, etc.) that check shape, dtype, and writability before any kernel call.
- Integer point arrays are normalized to float64 in Layer 2 before reaching kernels.

### `out` parameter convention (NumPy style)

Public functions and Layer 2 helpers accept an optional `out` argument for the result array:

- If `out=None`, Layer 2 allocates a fresh array with the correct shape and dtype.
- If `out` is provided, Layer 2 validates its shape, dtype, and writability before use.
- Some functions expose multiple output arguments (e.g. `out_basis` + `out_first_basis` in `BsplineSpace1D.tabulate_basis()`).
- Kernels always receive a pre-validated, correctly shaped array — they write directly into it with no further checks.

### Performance notes

- Change-of-basis matrices and unique knots are cached to avoid recomputation across calls
- Basis kernels use `parallel=True` + `prange` for multi-core evaluation

## Code conventions

- Strict mypy (`strict=True`); all public and private functions must be fully typed
- Ruff with Google-style docstrings, line length 100, target Python 3.10
- Warnings are treated as errors in pytest
- Layer 3 kernels run under Numba `nopython=True`: only use NumPy operations supported in that mode; unsupported calls cause a hard compile error or silent object-mode fallback
