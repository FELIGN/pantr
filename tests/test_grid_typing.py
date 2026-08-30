"""Run mypy over ``tests/typing/cases`` and check it reports exactly the marked errors.

``Grid`` is a :class:`typing.Protocol`, so satisfying it is a static property: no
``isinstance`` answers it, and nothing in the rest of the suite can observe a class
failing to. This module is the harness that can.

See ``tests/typing/README.md`` for the marker convention and for why the case modules
are excluded from the repo-wide mypy run.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CASES = _REPO_ROOT / "tests" / "typing" / "cases"

_MARKER = re.compile(r"#\s*expect-error:\s*([\w-]+)\s*$")
_ERROR = re.compile(r"^(?P<path>.+?):(?P<line>\d+): error: .*\[(?P<code>[\w-]+)\]\s*$")


def _expected(path: Path) -> set[tuple[str, int, str]]:
    """Collect the ``# expect-error:`` markers in one case module.

    Args:
        path (Path): The case module.

    Returns:
        set[tuple[str, int, str]]: ``(file name, 1-based line, mypy error code)``.
    """
    out: set[tuple[str, int, str]] = set()
    for lineno, text in enumerate(path.read_text().splitlines(), start=1):
        match = _MARKER.search(text)
        if match is not None:
            out.add((path.name, lineno, match.group(1)))
    return out


def _reported(paths: list[Path]) -> set[tuple[str, int, str]]:
    """Run mypy over the case modules and parse the errors it reports.

    ``MYPYPATH`` is set explicitly because ``pytest.ini``'s ``pythonpath = src`` reaches
    the pytest process only: a subprocess would otherwise resolve the *installed*
    ``pantr`` rather than this worktree's.

    Args:
        paths (list[Path]): The case modules to check.

    Returns:
        set[tuple[str, int, str]]: ``(file name, 1-based line, mypy error code)``.
    """
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--config-file", "mypy.ini", *[str(p) for p in paths]],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env={**os.environ, "MYPYPATH": str(_REPO_ROOT / "src")},
        check=False,
    )
    if result.returncode not in (0, 1):
        pytest.fail(
            f"mypy failed to run (exit {result.returncode}):\n{result.stdout}{result.stderr}"
        )
    out: set[tuple[str, int, str]] = set()
    for text in result.stdout.splitlines():
        match = _ERROR.match(text)
        if match is not None:
            out.add((Path(match.group("path")).name, int(match.group("line")), match.group("code")))
    return out


def test_type_level_cases_report_exactly_the_marked_errors() -> None:
    """What mypy reports and the ``# expect-error:`` markers agree, both ways.

    Set equality rather than containment, so this fails when an expected error stops
    being reported *and* when an unexpected one appears. The marker names the error
    code, which is what stops a case from passing on the wrong error.
    """
    pytest.importorskip("mypy", reason="mypy is in the dev extra; the check suite runs it too")
    cases = sorted(_CASES.glob("*.py"))
    assert cases, "no type-level case modules found"
    assert _reported(cases) == set().union(*(_expected(p) for p in cases))
