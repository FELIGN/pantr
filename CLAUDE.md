# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Non-negotiable rules

- **Never run `git push` without first running the full check suite** (ruff, ruff format, mypy, import-lint, pytest, docs build). This applies every time — new PRs, review fixes, hotfixes, everything. `make pre-pull-request` runs all of it; the individual commands are in the Commands section below.
- **Run every check on the whole repo, not on single files.** Checking one file while skipping a repo-wide check has let real failures through.
- Always run commands in the `pantr` conda environment, as `conda run -n pantr <command>`. Do not use `conda activate` — shell state does not persist between tool calls, so an `activate` in one call is gone by the next.
- Always use git worktrees for implementing changes (via the `EnterWorktree` tool or `git worktree add`). `pytest.ini` sets `pythonpath = src`, so tests import the worktree's own source without a per-worktree reinstall.

## Commands

All of these assume the `pantr` env, i.e. prefix with `conda run -n pantr`.

```bash
make pre-pull-request                                       # the full check suite CI runs
pytest                                                      # run tests (JIT enabled, no coverage)
make doctest                                                # run the docstring examples under src/pantr
pytest tests/test_basis.py::test_name -v                    # single test
pytest tests/ -k "keyword" -v                               # filtered tests
pytest -m "not slow"                                        # skip the slow-marked tests
NUMBA_DISABLE_JIT=1 pytest --cov=src/pantr --cov-report=xml # coverage (JIT disabled)
ruff check .                                                # lint
ruff format .                                               # format
mypy --config-file mypy.ini src tests                       # type check
PYTHONPATH=src lint-imports                                 # import boundaries (core must not import pantr.mpi)
NUMBA_DISABLE_JIT=1 make docs SPHINXOPTS="-W --keep-going -j auto"  # docs build (matches CI)
pip install -e ".[dev]"                                     # full dev env (pulls all optional extras)
pip install "pantr[mpi]"                                    # opt in to MPI (pantr.mpi + mpi4py)
PANTR_RUN_MPI=1 mpiexec -n 2 python -m pytest tests/mpi/    # MPI smoke tests (needs mpi4py + MPI launcher)
```

> `tests/mpi/` holds real-MPI smoke tests; they are **skipped** unless `PANTR_RUN_MPI` is set
> (and run under `mpiexec`). The default `pytest` run collects and skips them. CI runs them in a
> dedicated `mpi-tests` job (installs OpenMPI, builds `mpi4py` from source, `mpiexec -n {2,3}`).

> Coverage is **opt-in**: `pytest.ini`'s `addopts` is only `--strict-config --strict-markers`, so a
> plain `pytest` run carries no coverage overhead and `--no-cov` is unnecessary. The `coverage.toml`
> threshold (`fail_under = 85`) applies only when `--cov` is passed explicitly, which the `coverage`
> make target and the CI `tests` job do.

## Local green is not CI green

The local suite passing is necessary but not sufficient. Known traps:

- **mypy runs on a Python matrix (3.11, 3.13, 3.14)** while the installed *numpy stub* version is
  whatever is local. Be defensive with numpy typing: never a bare `np.ndarray`, and wrap the result
  of `np.einsum` / `np.tensordot` with `np.asarray(..., dtype=np.float64)` where the dtype matters —
  those return `Any` in some stub versions and the annotation silently degrades.
- **Local numpy *behavior* also differs from the CI matrix, not just the stubs.** Anything IEEE 754
  leaves unspecified can differ by numpy version, and the test matrix will find it. Measured:
  `np.minimum(-0.0, 0.0)` returns `-0.0` on numpy 2.4.6 but `+0.0` on the 3.14 job, so a test
  asserting the sign of a min/max tie passes locally and fails there. Never assert that sign — and
  note the tie cannot be forced, since `-0.0` and `+0.0` always compare equal. To exercise a
  signed-zero path deterministically, give *both* operands the same `-0.0`: the minimum of two
  equal values is that value on any implementation.
- **`fail-fast` makes one failing matrix leg look like three.** When a leg genuinely fails, its
  siblings are canceled, and `gh pr checks` prints cancellations as `fail` too. Before concluding
  there are N defects, check each job's steps
  (`gh api repos/FELIGN/pantr/actions/jobs/<id> --jq '.steps[]'`) and find the one that actually
  failed. This is a different trap from the `cancel-in-progress` one below, which comes from
  pushing twice.
- **The CI test job is headless.** pyvista/VTK tests must never call `.show()` or `pantr.viz.plot()`
  (`src/pantr/viz/_scene.py:292`) — they force a render and segfault without a display. Build the
  scene with `Scene.to_plotter()` (`src/pantr/viz/_scene.py:187`) instead. The coverage run has
  `NUMBA_DISABLE_JIT=1`, and one segfault there takes down the whole run.
- **A missing optional dependency skips tests silently.** If `pyvista` or `mpi4py` is absent
  locally, the tests that need them skip without complaint — install them before trusting a local
  green.
- **A new Layer 2 entry point over `parallel=True` kernels must call
  `pantr._numba_compat.wait_for_jit_warmup()` first**, as `basis/_basis_1D.py`,
  `bspline/_bspline_roots.py` and `bezier/_find_roots.py` do. `__init__.py` compiles on a
  background thread, and Numba's default workqueue layer is not safe against a concurrent parallel
  call from another thread: the process **aborts** (`Fatal Python error: Aborted`) rather than
  raising, taking the whole run with it. Three things hide it from a local green — a full-file run
  usually gives the warmup time to finish during the first test's own compilation, a warm Numba
  cache shortens the window to nothing, and the abort needs a *fast* test that reaches kernels
  early. `make test` is `pytest -n auto`, so every xdist worker imports and starts immediately,
  which is exactly the racing pattern. Measured: a new 60-case `locate` sweep aborted 4 of 4 runs
  as the first thing in a process, serially and under `-n 4`, and 0 of 4 with the barrier. **Run a
  new kernel-heavy test file on its own** (`pytest tests/test_x.py -k ...`) before trusting it, and
  test the barrier by asserting it is *called* — no in-process test can observe the race, since the
  barrier is a once-per-process event.
- **`pythonpath = src` covers pytest only.** Anything that imports `pantr` outside the pytest
  process — `lint-imports`, a test that spawns a subprocess — resolves the *installed* package
  instead. Two consequences: those commands need `PYTHONPATH=src` to see the current worktree, and a
  stale editable install (one pointing at a deleted worktree) makes them fail with
  `ModuleNotFoundError` while the rest of the suite passes. Repair it with
  `pip install -e ".[dev]"` **from the main checkout**, never from a worktree.
- **The local suite cannot see a result that depends on the BLAS implementation.** macOS
  links numpy against Accelerate while CI runs OpenBLAS, so any quantity round-off
  dominates can differ between them by orders of magnitude. Measured here: the relative
  error of one degree-13 matrix inverse was `9.56` locally and `0.0335` on CI, and the
  local sequence was not even monotone in degree, which is the giveaway that the number is
  noise rather than error. A threshold fitted to a measurement taken on one machine is not
  a threshold. Derive the bound from exact arithmetic (conditioning, epsilon, degree) and
  it transfers; measure it and CI will disagree with you.
- **`PYTHONPATH=src` relative is not safe under `conda run`.** It resolves against the
  working directory `conda run` gives the process, which can be the main checkout rather
  than your worktree, so the command silently exercises the *installed* package instead of
  the branch — seen as 21 phantom failures against code that was already fixed. Use
  `PYTHONPATH="$(pwd)/src"`. Plain `pytest` is unaffected, since `pytest.ini`'s
  `pythonpath = src` resolves against the rootdir.
- **CI is the finish line, not the push.** After creating a PR, watch it (`gh pr checks <n> --watch`)
  and do not report the work as done until every required check passes. Avoid pushing twice in quick
  succession: `ci.yaml` sets `cancel-in-progress: true`, so a second push cancels the first run's
  jobs and the canceled run reads as a failure. Confirm a job was *canceled* before reacting to it.

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

## Workflow

This repository has **no local skills of its own**. The development process comes from the shared
toolkit in `~/.claude` (agents, commands, and skills), and this file supplies the project-specific
facts that toolkit discovers — the check suite above, the layer architecture below, the conventions.
Do not reintroduce a repo-local workflow skill: it drifts from the shared one and then silently
contradicts it.

Pick the entry point by the shape of the work:

| Situation | Use |
|---|---|
| A feature from a loose idea | `/forge` — map, interview, architecture, plan, build, review, spec |
| An open GitHub issue | `/implement-ticket` |
| Several unblocked issues at once | `/run-wave` — clusters them by file footprint |
| Something is broken | Just say what broke; the `fix` skill fires on its own (reproduce, pin with a failing test *before* the fix, fix, record as a numbered `B` with its own commit) |
| Ready to commit | The `commit` skill — it discovers this project's toolchain from the `Makefile` and this file |
| Review before merging | `/light-review` by default; `/deep-review` when the change carries new numerical content or new public surface |
| Converged research code needs cleaning up | `/formalize` |
| Turning a finding into a ticket | The `file-bug` skill. **Never `gh issue create` by hand.** |

Specialist agents are dispatched by asking in plain language ("run the tolerance-hunter over this
file", "have a scout map the extraction layer"). The ones that come up most here: `engineer` for
judgment-heavy work, `implementer` for pinned execution, `scout` for a code map, `tolerance-hunter`
for any new tolerance, `test-writer` and `test-architect` for the suite, `verifier` for a claim that
one run or quote would settle.

One project-specific rule the shared toolkit cannot know: **`lepard`, a separate and not-yet-public
consumer, imports pantr *private* symbols** (kernels and helpers under a leading underscore). Before
deleting or reshaping any symbol, public or private, check that repository for consumers — pantr's
own CI cannot see breakage there.

## Architecture

**PaNTr** is a polynomial and NURBS toolkit for geometric modeling and numerical analysis (Python 3.11–3.14).

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
- Ruff with Google-style docstrings, line length 100, target Python 3.11
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
