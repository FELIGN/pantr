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
# NUMBA_DISABLE_JIT=1 is deliberate, for two reasons. It removes this run from the
# concurrent-compilation abort class entirely: with JIT off, `prange` is `range` and
# no Numba threading layer is ever entered, so nothing here can race the background
# warmup thread started by pantr/__init__.py. Today the race does not fire (measured:
# 0 aborts in 44 runs, 24 cold-cache and 20 warm), but only because collection order
# happens to reach a `wait_for_jit_warmup()` call site first -- reorder the modules or
# drop those examples and the protection is gone. It also makes the run 28x faster
# (0.9s against 24s cold). The compiled path is what `test` covers; this target checks
# that the documentation matches the code, and the values it asserts go through
# np.allclose or .tolist(), which do not depend on JIT-vs-interpreter rounding.
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
	mypy --config-file mypy.ini src tests

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
