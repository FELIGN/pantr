"""Guard that the measurement scripts under ``scripts/`` can be run directly.

Several of them import from the ``tests`` package, which lives at the repository root.
Running a file inside ``scripts/`` sets ``sys.path[0]`` to ``scripts/`` **instead of** the
working directory, so a script that imports from ``tests`` has to put the root on the path
itself. One of them did not, and
``python scripts/measure_bezier_fma_bound.py`` died with
``ModuleNotFoundError: No module named 'tests'`` -- while the same module imported fine
under pytest, which has the root on the path for its own reasons. Nothing in the suite
looked at the way the script's own docstring says to run it.

The scripts are discovered rather than listed, so a new one that reaches for the ``tests``
package is covered the day it is added instead of the day someone remembers this file. The
predicate is a source match, so a script reaching that package through
``importlib.import_module`` or ``__import__`` would not be found; none does today.

Why the check is not "run it with ``--help``"
---------------------------------------------

Because that measures the wrong thing here, and expensively. Of the two scripts,
``measure_hierarchical_refinement_parity.py`` has an ``argparse`` parser and exits on
``--help``; ``measure_bezier_fma_bound.py`` has no parser at all and ignores ``argv``, so
``--help`` runs its full contraction-bound measurement over tens of thousands of entries and
exits ``0`` only because that bound holds on this build. A guard written that way costs
seconds per run and would report *"add the repository root to sys.path"* on a day the
numerical bound was exceeded, which is a misattribution rather than a failure.

So the subprocess reproduces the *import* condition and stops there:
:func:`runpy.run_path` puts the script's own directory on the path exactly as direct
execution does, ``-P`` keeps the working directory off it (which is what makes the root
absent, and is the whole condition), and a ``run_name`` other than ``__main__`` leaves the
script's ``if __name__ == "__main__"`` guard unfired. Checked both ways: against the
unfixed script it raises the ``ModuleNotFoundError`` above, and against the fixed one it
returns without doing any of the work.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

_SCRIPTS_DIR = _REPO_ROOT / "scripts"

_IMPORTS_TESTS_PACKAGE = re.compile(r"^\s*(?:from|import)\s+tests\b", re.MULTILINE)


def _scripts_importing_the_tests_package() -> list[pathlib.Path]:
    """Collect the scripts whose source imports the root-level ``tests`` package.

    Returns:
        list[pathlib.Path]: The matching scripts, sorted by name for a stable test id.
    """
    return sorted(
        path
        for path in _SCRIPTS_DIR.glob("*.py")
        if _IMPORTS_TESTS_PACKAGE.search(path.read_text(encoding="utf-8"))
    )


def test_the_guard_has_something_to_guard() -> None:
    # Without this the parametrization below can silently shrink to nothing -- a rename or
    # a moved import would leave a green test covering no script at all.
    assert _scripts_importing_the_tests_package(), (
        f"no script under {_SCRIPTS_DIR.name}/ imports the tests package; either the "
        f"pattern stopped matching or this guard is obsolete"
    )


@pytest.mark.parametrize(
    "script", _scripts_importing_the_tests_package(), ids=lambda path: path.stem
)
def test_the_script_imports_as_a_direct_run_would(script: pathlib.Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-P",
            "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1], run_name='_import_check_')",
            str(script),
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"{script.name} does not import when run the way its docstring says to run it; a "
        f"script that imports the tests package must put the repository root on sys.path "
        f"before doing so\n{result.stderr}"
    )
