---
name: pre-pr-checks
description: Run the full pre-PR validation suite (ruff lint, ruff format check, mypy type checking, pytest tests, and Sphinx documentation build). Use this skill whenever you need to verify code quality before pushing, creating a PR, or when the user asks to "run checks", "validate", "run CI locally", or "make sure everything passes". Also trigger when about to push code or create a pull request — always validate first.
---

# Pre-PR Checks

Run all quality checks that CI would run, and report results. The goal is to catch issues before they reach the remote, saving CI time and avoiding back-and-forth.

## Environment setup

Activate the conda environment first:

```bash
conda activate pantr
```

## Checks to run

Run these checks **sequentially** in this order — later checks are pointless if earlier ones fail on syntax or type errors.

### 1. Ruff lint

```bash
ruff check .
```

If there are auto-fixable issues, fix them with `ruff check --fix .` and re-run to confirm. Stage the fixes.

### 2. Ruff format check

```bash
ruff format --check .
```

If formatting issues are found, run `ruff format .` to fix them and stage the changes.

### 3. mypy type checking

```bash
mypy --config-file mypy.ini src tests
```

If there are type errors, fix them before proceeding. Type errors often indicate real bugs.

### 4. Pytest (without coverage)

```bash
pytest --no-cov -v
```

Use `--no-cov` to keep things fast. All tests must pass.

### 5. Documentation build

```bash
NUMBA_DISABLE_JIT=1 sphinx-build -M html docs/ docs/_build -W --keep-going -j auto
```

The `-W` flag treats warnings as errors (matching CI). `NUMBA_DISABLE_JIT=1` avoids Numba compilation during doc build.

## Reporting

After running all checks, present a clear summary:

```
Pre-PR Checks Summary
---------------------
Ruff lint:    PASS / FAIL (N issues)
Ruff format:  PASS / FAIL (N files)
mypy:         PASS / FAIL (N errors)
pytest:       PASS / FAIL (N passed, M failed)
docs build:   PASS / FAIL (N warnings)
```

If any check failed and you were unable to fix it, explain what went wrong and suggest next steps. If all checks pass, say so clearly — the code is ready to push.

## When auto-fixing

If you fix lint, format, or type issues during this process, commit those fixes in a dedicated commit:

```
style: fix lint and formatting issues
```

Keep these commits separate from feature work so the git history stays clean.
