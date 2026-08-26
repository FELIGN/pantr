"""Pin the three ways the suite broke under ``NUMBA_DISABLE_JIT=1``.

``make coverage`` runs the whole suite with the JIT off, because coverage.py cannot
trace machine code numba compiled. That configuration is not a second opinion about
the same code: it swaps the Numba oracle for interpreted python, and three separate
things then break. All three were live at once, none was caught by a plain ``pytest``
run, and each shipped for as long as it did because nothing outside ``make coverage``
exercises the configuration at all.

The three, in the order they were diagnosed:

- **A bitwise parity claim compared against an oracle that is not the compiled
  kernel.** ``np.power`` is not pinned by IEEE 754 and numba's disagrees with numpy's
  by an ulp; a falling-factorial accumulator wraps at int64 compiled and grows without
  bound interpreted, which past the overflow is a factor of about seven rather than a
  rounding. `demand_a_compiled_seed` is what skips those claims.
- **A test that reads generated assembly where none is generated.** With the JIT off
  ``njit`` returns the plain python function, which has no ``inspect_asm``, so the
  check that numba emits no fused multiply-add raised ``AttributeError``.
- **An assertion tighter than the precision of its own data.** A ``float32`` root
  asserted to ten decimal places held only because the compiled path happened to land
  on the exact value.

This test spawns one child ``pytest`` over the parity suite and the root finder with
the variable set. It
is the whole point that it runs in the **plain** suite: ``make coverage`` already
catches all of this, and catching it there means catching it in CI, minutes later,
after a full instrumented run, rather than here in seconds.

It is meaningful without the compiled extension. Only the first of the three needs it;
the other two are pinned either way, and the assertion below refuses a child run that
skipped its way to green.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
"""The repository root, which is also pytest's rootdir and so its `pythonpath` anchor."""

_TARGETS: Final[tuple[str, ...]] = (
    "tests/parity",
    "tests/test_root_finding.py",
)
"""What the child runs.

The whole parity directory rather than the six tests that carry a gate today, which
costs about three seconds more and buys the case every reviewer of this machinery
asked about: a kernel added later that seeds with ``pow`` or accumulates an integer,
in a module that currently has nothing to gate. Named as directories, the list needs
no maintenance when that happens; named as tests, it would silently stop covering the
thing it exists for.
"""


def test_the_jit_disabled_configuration_still_passes() -> None:
    """Run the affected suites in a child process with the JIT switched off.

    A child process is not a stylistic choice. ``NUMBA_DISABLE_JIT`` is read once,
    when ``numba.core.config`` is imported, so by the time any test function runs the
    kernels are already compiled or already not. Monkeypatching the variable moves
    what `the_jit_is_disabled` reports and nothing else, which is enough to test a
    guard in isolation and cannot reproduce the configuration itself.

    Raises:
        AssertionError: If the child fails, or reports no passing test.
    """
    if os.environ.get("NUMBA_DISABLE_JIT", "0") == "1":
        pytest.skip("already inside a JIT-disabled run, so the child would be a copy of it")

    child = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", *_TARGETS],
        cwd=_ROOT,
        env={**os.environ, "NUMBA_DISABLE_JIT": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    report = f"\n--- child stdout ---\n{child.stdout}\n--- child stderr ---\n{child.stderr}"

    assert child.returncode == 0, (
        f"the suite does not pass with NUMBA_DISABLE_JIT=1, which is the configuration "
        f"`make coverage` runs in.{report}"
    )
    assert " passed" in child.stdout, (
        f"the child reported nothing passing, so this test asserted nothing. A run that "
        f"skips its way to green is the trap CLAUDE.md names by name.{report}"
    )
