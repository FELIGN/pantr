.PHONY: help test doctest coverage clean install ruff-lint ruff-format ruff-format-check type-check import-lint pre-pull-request docs

help:
	@echo "Commands:"
	@echo "  test      : run the test suite."
	@echo "  doctest   : run the docstring examples in src/pantr."
	@echo "  coverage  : generate a coverage report."
	@echo "  clean     : remove build artifacts."
	@echo "  install   : install project with dev extras."
	@echo "  ruff-lint : run Ruff linter."
	@echo "  ruff-format : check Ruff formatting changing files."
	@echo "  ruff-format-check : check Ruff formatting without changing files."
	@echo "  type-check: run mypy static type checker."
	@echo "  import-lint: check import boundaries (core must not import pantr.mpi)."
	@echo "  docs      : build the documentation."
	@echo "  pre-pull-request: run lint, format, format check, type check, import lint, tests, coverage, and docs."

# Run the test suite with Numba JIT enabled
test:
	pytest -n auto

# Run the docstring examples shipped in the package sources. Kept out of `test`
# because `testpaths = tests` in pytest.ini deliberately excludes src/, and a plain
# `pytest` run should stay the fast inner loop. No coverage and no xdist: the whole
# set runs in under a second, and worker startup would dominate.
#
# NUMBA_DISABLE_JIT=1 is deliberate. With JIT off, `prange` is `range` and no Numba
# threading layer is ever entered, so this run cannot race the background warmup thread
# that pantr/__init__.py starts: the concurrent-compilation abort class is structurally
# absent here, not merely improbable. It is also 28x faster (0.9s against 24s cold),
# which matters now that `pre-pull-request` depends on it.
#
# Historical note, so nobody "restores" the JIT here: the JIT-enabled run this replaced
# measured 0 aborts in 44 runs (24 cold-cache, 20 warm), but was safe only because
# pytest happens to collect basis/_basis_1D.py early and its examples reach a
# `wait_for_jit_warmup()` call site before any unguarded kernel. Reordering the modules
# or dropping those examples would have removed that protection silently.
#
# What this gives up is the compiled path, which is `test`'s job over the whole suite.
# This target checks that the documentation matches the code, and the values it asserts
# go through np.allclose or .tolist(), neither of which depends on JIT-vs-interpreter
# rounding.
doctest:
	NUMBA_DISABLE_JIT=1 pytest --doctest-modules src/pantr

# Generate an XML coverage report with Numba JIT disabled
coverage:
	COVERAGE_FILE=/tmp/.coverage NUMBA_DISABLE_JIT=1 pytest -m "not slow" --cov=src/pantr --cov-report=term-missing --cov-report=xml

# Remove build artifacts
clean:
	rm -rf .pytest_cache .coverage coverage.xml htmlcov/

# Install project with development dependencies
install:
	python -m pip install --upgrade pip
	pip install -e ".[dev]"

# Ruff linting
ruff-lint:
	ruff check .

# Ruff formatting check (changes performed)
ruff-format:
	ruff format .

# Ruff formatting check (no changes written)
ruff-format-check:
	ruff format --check .

# Static type checking
type-check:
	mypy --config-file mypy.ini src tests scripts

# Import boundary checks: serial core must not import pantr.mpi.
# PYTHONPATH=src pins the analysis to this checkout's source: import-linter resolves
# `pantr` through sys.path, so without it the contract is checked against whatever the
# editable install points at, which in a git worktree is a different tree entirely.
import-lint:
	PYTHONPATH=src lint-imports

# Build documentation
docs:
	$(MAKE) -C docs html SPHINXOPTS="$(SPHINXOPTS)"

# Aggregate target to run all checks before creating a pull request
pre-pull-request: ruff-lint ruff-format ruff-format-check type-check import-lint test doctest coverage docs
