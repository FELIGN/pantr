"""Keep the bounds-checked adversarial sweep alive as a test.

The sweep proper lives in ``tools/adversarial_sweep/`` and exists because pantr's Layer-3
kernels run under Numba ``nopython=True``: there is no bounds check, a negative index
wraps, and int64 overflows untrapped. The whole suite already runs clean under
``NUMBA_BOUNDSCHECK=1``, so everything the sweep finds comes from inputs the suite does
not contain. This test runs the sweep's bounded ``smoke`` profile and fails when it finds
something that is not already pinned by a regression test.

Three things force the shape of this test:

* **It must be a subprocess.** ``NUMBA_BOUNDSCHECK`` is read when Numba is imported and
  applied when a kernel is compiled, so it cannot be turned on from inside a pytest
  process that has already imported pantr.
* **It needs a fresh Numba cache.** The bounds-check flag is *not* part of the cache key,
  so a ``cache=True`` kernel compiled earlier without it is silently reused and the run
  returns a false clean. Every pantr kernel is ``cache=True``. The test therefore hands
  the child a per-run ``NUMBA_CACHE_DIR`` under pytest's ``tmp_path``.
* **It is opt-in.** A fresh cache means recompiling every kernel the sweep touches, which
  costs minutes rather than the seconds a suite run should. Following the precedent of
  ``tests/mpi/``, it is gated on an environment variable and marked ``slow``, so the
  default suite pays nothing and CI runs it in a step of its own.

Run it with::

    PANTR_RUN_SWEEP=1 pytest tests/test_adversarial_sweep.py

The sweep's own canary is what makes a clean result meaningful: it compiles two
deliberately out-of-range ``cache=True`` kernels and requires both to raise before any
case runs. This test asserts the canary line is present, because a sweep that cannot
detect the bug it hunts is worse than no sweep.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_LAUNCHER = _ROOT / "tools" / "sweep.py"

_ENV_FLAG = "PANTR_RUN_SWEEP"
"""Set it to run the sweep. Absent, the test skips and the suite costs nothing."""

_MIN_CASES = 400
"""Floor on the smoke profile's case count.

Guards the failure mode a verdict-based assertion cannot see: a probe module that stops
yielding cases (an import error swallowed by a generator, a profile guard inverted) makes
the sweep pass by running almost nothing. The smoke profile yields a little over 500
cases; this floor leaves room to prune without becoming a maintenance chore.
"""

_KNOWN_FINDINGS = frozenset(
    {
        # Degree-0 cardinal extraction reads out of bounds in
        # `_get_Bspline_cardinal_intervals_1D_core` (`_bspline_knots.py:294`,
        # `lengths[degree - 1]` on an empty array). Logged before this sweep.
        "cardinal_intervals_d0_m1_float64",
        "extraction_build_d0_m1_float64_cardinal",
        # `_degree_elevate_1d_core` returns a knot vector and a control-point array that
        # disagree once an interior knot reaches multiplicity degree + 1. Pinned by
        # `test_sweep_regressions.py::test_degree_elevation_outputs_are_mutually_consistent`.
        "elevate_degree_d0_m1_float64_random",
        "elevate_degree_d1_m2_float64_random",
        # `_de_casteljau_eval_scalar` reads `coeff[0]` with no guard, so an empty
        # coefficient array reads out of bounds. Layer 3 documents that it validates
        # nothing, and no public path reaches it with an empty array, so this is a port
        # note rather than a live bug: in C++ the same read is undefined behavior.
        "de_casteljau_len0_float64",
    }
)
"""Findings the smoke profile is expected to reproduce.

The assertion is containment, not equality: a **new** finding fails the test, while fixing
a known one merely shrinks the set. Each entry is either pinned by an ``xfail(strict=True)``
test in ``test_sweep_regressions.py``, which is what fails when the bug is fixed, or
recorded above as a deliberate contract-boundary probe.
"""


def _run_sweep(cache_dir: pathlib.Path, journal: pathlib.Path) -> subprocess.CompletedProcess[str]:
    """Run the smoke profile in a child process with a fresh Numba cache.

    Args:
        cache_dir (pathlib.Path): Directory for the child's Numba cache. Must not already
            hold a compilation of these kernels, or the bounds check is silently skipped.
        journal (pathlib.Path): File to receive the per-case JSONL records.

    Returns:
        subprocess.CompletedProcess[str]: The finished process, with output captured.
    """
    env = dict(os.environ)
    env["NUMBA_BOUNDSCHECK"] = "1"
    env["NUMBA_CACHE_DIR"] = str(cache_dir)
    # The coverage run sets this globally, and with the JIT off nothing is compiled and
    # the bounds check grades nothing. The sweep's canary refuses to start in that state,
    # so clear it rather than let an inherited value turn the run into a false clean.
    env.pop("NUMBA_DISABLE_JIT", None)
    # Fixed argv, no shell, in-repo launcher.
    return subprocess.run(
        [sys.executable, str(_LAUNCHER), "--profile", "smoke", "--journal", str(journal)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get(_ENV_FLAG) != "1",
    reason=f"set {_ENV_FLAG}=1 to run the bounds-checked adversarial sweep",
)
def test_smoke_sweep_finds_nothing_new(tmp_path: pathlib.Path) -> None:
    cache_dir = tmp_path / "numba-cache"
    cache_dir.mkdir()
    journal = tmp_path / "sweep.jsonl"

    result = _run_sweep(cache_dir, journal)

    # 0 is clean, 1 is findings, 3 is "the harness is unusable" and anything else is a
    # crash or a signal. Only the first two may be interpreted, and the canary line has to
    # be there: a sweep that cannot detect an out-of-bounds access proves nothing.
    assert result.returncode in (0, 1), (
        f"the sweep did not complete (exit {result.returncode}):\n"
        f"{result.stdout[-2000:]}\n{result.stderr[-4000:]}"
    )
    assert "canary OK" in result.stderr, f"the canary did not report:\n{result.stderr[-4000:]}"

    records = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert len(records) >= _MIN_CASES, f"only {len(records)} cases ran, expected >= {_MIN_CASES}"

    found = {r["label"] for r in records if r["verdict"] == "BUG"}
    new = found - _KNOWN_FINDINGS
    details = "\n".join(
        f"  {r['group']}/{r['label']} [{r['kind']}]\n    {r['detail'].splitlines()[0]}"
        for r in records
        if r["verdict"] == "BUG" and r["label"] in new
    )
    assert not new, f"the sweep found {len(new)} finding(s) not in _KNOWN_FINDINGS:\n{details}"
