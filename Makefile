.PHONY: help test coverage clean install ruff-lint ruff-format-check type-check before_push docs

help:
	@echo "Commands:"
	@echo "  test      : run the test suite."
	@echo "  coverage  : generate a coverage report."
	@echo "  clean     : remove build artifacts."
	@echo "  install   : install project with dev extras."
	@echo "  ruff-lint : run Ruff linter."
	@echo "  ruff-format : check Ruff formatting changing files."
	@echo "  ruff-format-check : check Ruff formatting without changing files."
	@echo "  type-check: run mypy static type checker."
	@echo "  docs      : build the documentation."
	@echo "  befor-pr: run lint, format, format check, type check, tests, coverage, and docs."

# Run the test suite with Numba JIT enabled
test:
	pytest --no-cov

# Generate an XML coverage report with Numba JIT disabled (slow tests excluded)
coverage:
	COVERAGE_FILE=/tmp/.coverage NUMBA_DISABLE_JIT=1 pytest -m "not slow" --cov=src/pantr --cov-report=xml

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

# Build documentation
docs:
	$(MAKE) -C docs html SPHINXOPTS="$(SPHINXOPTS)"

# Aggregate target to run all checks before creating a pull request
pre-pull-request: ruff-lint ruff-format ruff-format-check type-check test coverage docs
