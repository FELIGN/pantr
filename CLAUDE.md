# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Non-negotiable rules

- **Never run `git push` without first running the full check suite** (ruff, mypy, pytest, docs build). This applies every time — new PRs, review fixes, hotfixes, everything. Run `pre-pr-checks` skill or the commands in the Commands section below.
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
lint-imports                                                        # import boundaries (core must not import pantr.mpi)
NUMBA_DISABLE_JIT=1 make docs SPHINXOPTS="-W --keep-going -j auto"  # docs build (matches CI)
pip install -e ".[dev]"                                             # full dev env (pulls all optional extras)
pip install "pantr[mpi]"                                            # opt in to MPI (pantr.mpi + mpi4py)
PANTR_RUN_MPI=1 mpiexec -n 2 python -m pytest tests/mpi/ --no-cov   # MPI smoke tests (needs mpi4py + MPI launcher)
```

> `tests/mpi/` holds real-MPI smoke tests; they are **skipped** unless `PANTR_RUN_MPI` is set
> (and run under `mpiexec`). The default `pytest` run collects and skips them. CI runs them in a
> dedicated `mpi-tests` job (installs OpenMPI, builds `mpi4py` from source, `mpiexec -n {2,3}`).

> Default pytest (via `pytest.ini`) adds `--cov`, which slows things down and requires ≥85% coverage. Always pass `--no-cov` during development.

## Commit conventions

Use **conventional commits** with the format:

```
<type>(<scope>): <imperative summary>
```

- **Types**: `feat`, `fix`, `refactor`, `test`, `docs`, `style`, `perf`, `chore`
- **Scope**: the module or area affected (e.g., `bspline`, `basis`, `quad`, `docs`)
- First line: imperative mood, lowercase, no trailing period
- One logical change per commit — do not bundle unrelated changes
- Branch names follow the same convention: `<type>/<short-kebab-description>`

## Architecture

**PaNTr** is a polynomial and NURBS toolkit for geometric modeling and numerical analysis (Python 3.10–3.14).

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

### Optional MPI layer (`pantr.mpi`)

`pantr.mpi` hosts the optional MPI-parallel distribution code (and the dolfinx bridge). It is kept strictly separate from the serial core:

- **The serial core never imports `pantr.mpi`.** This is enforced by an import-linter contract (`make import-lint`, run in CI) and a grimp-based test. New core modules are covered automatically by the test.
- **MPI imports are lazy.** `import pantr.mpi` succeeds even without `mpi4py`; only `pantr.mpi.require_mpi()` imports it (raising a clear error if absent).
- **`mpi4py` is an opt-in dependency**, declared in the `mpi` extra. A plain `pip install pantr` is serial-only and MPI-free; `pip install "pantr[mpi]"` adds `mpi4py` (and needs an MPI library). The `dev` extra includes `pantr[mpi]`, so contributor installs always get it.

Other private modules: `_numba_compat.py` (Numba shim), `_basis_utils.py` (shared validation helpers), `__init__.py` (async Numba warmup at import time).

### Input validation policy

- Validation lives exclusively in **Layer 2**. Layer 1 checks only trivial preconditions; Layer 3 checks nothing.
- `_basis_utils.py` provides reusable validators (`_validate_out_array`, `_allocate_or_validate_out`) that check shape, dtype, and writability before any kernel call.
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

## Documentation guidelines

All code must use **Google-style docstrings** (enforced by Ruff's `pydocstyle` rule set).

### What to document

| Symbol | Required |
|---|---|
| Module / package `__init__.py` | Yes — multi-line summary + bullet list of main exports |
| Class | Yes — summary + `Attributes:` for every instance variable |
| Public function / method | Yes — full `Args:`, `Returns:`, `Raises:` |
| Private function / method (`_foo`) | Yes — same as public; private doesn't mean undocumented |
| Property | Yes — one-line summary + `Returns:` with type and description |
| Class attribute annotation | Yes — inline docstring (`"""…"""` on the line below) or in class `Attributes:` |
| Type alias | Yes — one-line docstring describing the alias |
| Numba kernel (Layer 3) | Yes — full docstring; add a `Note:` stating "No input validation is performed" |

### Required sections per symbol

**Modules** — opening summary paragraph; bullet list of key exports when helpful.

**Classes**:
```
"""Short summary.

Longer description if needed.

Attributes:
    attr_name (type): Description.
"""
```

**Functions / methods**:
```
"""Short summary (imperative mood, ≤ 1 line).

Optional extended description.

Args:
    param_name (type): Description. Defaults to X.

Returns:
    type: Description. Omit if return type is None.

Raises:
    ExceptionType: When/why it is raised.

Example:
    >>> call_example()
    expected_output
"""
```

**Properties** — include `Returns:` even though there are no `Args:`:
```python
@property
def foo(self) -> int:
    """Get the foo value.

    Returns:
        int: Description.
    """
```

### Layer-specific rules

- **Layer 1 (public API)**: Full docstrings with examples where useful. Docstring is the user-facing contract.
- **Layer 2 (implementation helpers)**: Full docstrings. Note any shape/dtype assumptions not enforced by the function itself.
- **Layer 3 (Numba kernels)**: Full docstrings. Always include a `Note:` section:

  ```
  Note:
      Inputs are assumed to be correct (no validation performed).
      For general use, call <Layer2Counterpart> instead.
  ```

### Style rules

- First line: imperative mood, ≤ 100 characters, no trailing period.
- Wrap long lines at 100 characters (matching `line-length` in `ruff.toml`).
- Use backticks for code references: `` `out` ``, `` `np.float64` ``, `` :class:`BsplineSpace1D` ``.
- Do not repeat the function signature in the docstring body.
- Type annotations in `Args:` / `Returns:` should match the function signature exactly.
