# Contributing to PaNTr

Thanks for your interest in contributing to PaNTr. This guide covers the
development setup and the conventions the project follows.

## Development setup

PaNTr targets Python 3.11–3.14. Clone the repository and install the package
with all development and optional-feature dependencies:

```bash
git clone https://github.com/FELIGN/pantr.git
cd pantr
pip install -e ".[dev]"
pre-commit install
```

The `dev` extra pulls in every optional feature (`mpi`, `metis`, `viz`, `docs`)
so the full test suite runs with nothing silently skipped.

## Running the checks

All of the following must pass before a pull request is merged (CI enforces
them):

```bash
ruff check .                             # lint
ruff format --check .                    # formatting
mypy --config-file mypy.ini src tests    # static types (strict)
lint-imports                             # import-boundary contracts
pytest --no-cov                          # tests (JIT enabled)
```

Build the documentation the same way CI does:

```bash
NUMBA_DISABLE_JIT=1 make docs SPHINXOPTS="-W --keep-going -j auto"
```

> The default `pytest` run adds coverage (`--cov`), which is slower and requires
> ≥85% coverage. Pass `--no-cov` during development. To run the coverage gate
> locally: `NUMBA_DISABLE_JIT=1 pytest --cov=src/pantr --cov-report=xml`.

The optional MPI smoke tests under `tests/mpi/` are skipped unless
`PANTR_RUN_MPI` is set and run under a launcher:

```bash
PANTR_RUN_MPI=1 mpiexec -n 2 python -m pytest tests/mpi/ --no-cov
```

## Code conventions

- **Types**: strict mypy; all public and private functions must be fully typed.
- **Docstrings**: Google-style, enforced by Ruff's `pydocstyle` rules. Private
  functions are documented too; Numba kernels carry a `Note:` stating that no
  input validation is performed.
- **Style**: Ruff, line length 100, target Python 3.11.
- **Layering**: the library is organized in three strict layers — public API,
  implementation helpers (all input validation), and Numba kernels (no
  validation). Review the architecture section of the documentation before
  adding code.

## Commits and branches

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <imperative summary>
```

- **Types**: `feat`, `fix`, `refactor`, `test`, `docs`, `style`, `perf`, `chore`.
- **Scope**: the affected module (e.g. `bspline`, `bezier`, `grid`, `docs`).
- One logical change per commit; branch names follow the same convention
  (`<type>/<short-kebab-description>`).

Changes are merged via pull requests, not direct pushes to `main`.

## Releasing

Releases are automated: pushing a `v*` tag publishes to PyPI (via trusted
publishing) and cuts a GitHub Release. The full maintainer runbook — version
bump, changelog, tagging, and the one-time PyPI / Read the Docs setup — is in
[`RELEASING.md`](RELEASING.md).

## License

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
