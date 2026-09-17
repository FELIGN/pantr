.PHONY: help test doctest coverage clean install ruff-lint ruff-format ruff-format-check type-check import-lint pre-pull-request docs

# Every target below that imports `pantr` runs through the active interpreter rather
# than through a console script, and the difference is not cosmetic.
#
# A console script carries its own interpreter in its shebang. The venvs used here
# are made with `--system-site-packages`, so pip sees pytest and import-linter
# already satisfied from the shared environment and installs no console script into
# the venv; PATH then falls through to the shared environment's, whose interpreter
# is the one a checkout's editable install has its meta-path finder on. A meta-path
# finder is consulted before `sys.path`, so `PYTHONPATH` cannot override it.
#
# Measured in a worktree: `make doctest` collected `src/pantr` from the worktree
# while importing `pantr` from another checkout, and reported 140 collection errors;
# the same run through the active interpreter reported 83 passed. `import-lint` has
# the identical shape, and has no `python -m` form, so it is invoked by path.
#
# `PYTHON ?= python` so a caller can point it elsewhere.
PYTHON ?= python


# Which `pantr` a target imports is decided by the active interpreter's meta-path
# finders, not by the working directory. An editable install anywhere puts a finder
# on `sys.meta_path`, which is consulted BEFORE `sys.path`, so a checkout's finder
# answers for every interpreter that carries it -- and `PYTHONPATH` cannot override
# it. Running `make` from this directory is therefore not enough to make a target
# describe this directory.
#
# Using the active interpreter rather than a console script's (below) is necessary
# and NOT sufficient: it fixes the case where PATH selects a foreign interpreter,
# and does nothing where the active interpreter is itself carrying another
# checkout's finder -- which is exactly what `conda run -n pantr make doctest`
# does in a worktree.
#
# So the targets that import `pantr` assert it instead of assuming it. A loud
# failure here costs a command; the alternative is `make test` reporting green over
# another checkout, which is what happened before this guard existed.
.PHONY: check-tree
check-tree:
	@root="$$(cd "$(CURDIR)" && pwd -P)"; \
	got="$$($(PYTHON) -c 'import pantr; print(pantr.__file__)' 2>/dev/null)"; \
	if [ -z "$$got" ]; then \
		echo "make: pantr does not import under $$($(PYTHON) -c 'import sys; print(sys.executable)')" >&2; \
		echo "      install this tree first:  pip install -e '.[dev]'" >&2; \
		exit 1; \
	fi; \
	case "$$(cd "$$(dirname "$$got")" && pwd -P)/" in \
		"$$root"/*) ;; \
		*) echo "make: this would describe another checkout, not this one." >&2; \
		   echo "      here:     $$root" >&2; \
		   echo "      imports:  $$got" >&2; \
		   echo "      cause:    the active interpreter carries another checkout's editable install." >&2; \
		   echo "      fix:      activate this tree's own venv, or PYTHON=/path/to/its/python make ..." >&2; \
		   exit 1 ;; \
	esac

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
test: check-tree
	$(PYTHON) -m pytest -n auto

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
doctest: check-tree
	NUMBA_DISABLE_JIT=1 $(PYTHON) -m pytest --doctest-modules src/pantr

# Generate an XML coverage report with Numba JIT disabled
coverage: check-tree
	COVERAGE_FILE=/tmp/.coverage NUMBA_DISABLE_JIT=1 $(PYTHON) -m pytest -m "not slow" --cov=src/pantr --cov-report=term-missing --cov-report=xml

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
import-lint: check-tree
	PYTHONPATH=src $(PYTHON) "$$(command -v lint-imports)"

# Build documentation
docs: check-tree
	$(MAKE) -C docs html SPHINXBUILD="$(PYTHON) -m sphinx" SPHINXOPTS="$(SPHINXOPTS)"

# Aggregate target to run all checks before creating a pull request
pre-pull-request: ruff-lint ruff-format ruff-format-check type-check import-lint test doctest coverage docs
