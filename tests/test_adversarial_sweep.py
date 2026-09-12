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
        # `_degree_elevate_1d_core` returns a knot vector and a control-point array that
        # disagree once an interior knot reaches multiplicity degree + 1. Pinned by
        # `test_sweep_regressions.py::test_degree_elevation_outputs_are_mutually_consistent`.
        "elevate_degree_d0_m1_float64_random",
        "elevate_degree_d1_m2_float64_random",
    }
)
"""Findings the smoke profile is expected to reproduce.

The assertion is containment, not equality: a **new** finding fails the test, while fixing
a known one merely shrinks the set. Every entry must be *tracked*, not merely *seen* --
each is either pinned by an ``xfail(strict=True)`` test in ``test_sweep_regressions.py``,
which is what fails when the bug is fixed, or recorded above as a deliberate
contract-boundary probe. An entry with neither would make this list a way to silence a
finding, which is the one thing it must not become.

Two entries were removed when ``fix(bspline,grid): … (#289)`` landed: the degree-0
cardinal extraction reads out of bounds no longer occur, so the sweep stopped reporting
``cardinal_intervals_d0_m1_float64`` and ``extraction_build_d0_m1_float64_cardinal``.
Containment would have tolerated leaving them, but a stale entry misdescribes the state of
the code.

Six more went the same way when the knot-vector factories were made to refuse
``num_intervals=0``: the six ``create_uniform_{open,periodic}_knots_d{0,1,3}_float64_[0,1]_n0``
cases now decline the input, and the probe asserts that refusal with ``must_reject``.

``de_casteljau_len0_float64`` left for a different reason, and it is worth keeping straight.
Nothing about the kernel changed: an empty coefficient array still reads out of bounds. What
changed is that the kernel now *states* the precondition it was silently assuming, so the case
carries ``out_of_contract`` and the harness grades it as a documented rejection. The entry was
never a bug to fix; it was a contract that had not been written down.
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


# ---------------------------------------------------------------------------
# The harness's own `out_of_contract` flag, unit-tested
# ---------------------------------------------------------------------------
#
# These need no subprocess and no Numba: they exercise the classifier directly, on
# fabricated cases. They exist because the flag's two branches are not both reachable
# from a sweep run -- the launcher re-execs with `NUMBA_BOUNDSCHECK=1` unconditionally,
# so a kernel that overruns always raises and the "returned anyway" branch is never
# taken by today's cases. Untested, that branch would be free to rot.

sys.path.insert(0, str(_ROOT / "tools"))

from adversarial_sweep._core import (  # noqa: E402  -- needs the path insert above
    NUMBA_OOB_MESSAGE,
    Case,
    Verdict,
    classify,
    custom,
    run_case,
)


def _case(
    *,
    must_succeed: bool = False,
    must_reject: bool = False,
    out_of_contract: bool = False,
) -> Case:
    """Build a minimal case whose entry point documents nothing.

    Args:
        must_succeed (bool): Set the "legal by construction" flag. Defaults to False.
        must_reject (bool): Set the "built to be refused" flag. Defaults to False.
        out_of_contract (bool): Set the "outside the stated contract" flag. Defaults to
            False.

    Returns:
        Case: A case over a no-op entry point.
    """

    def entry() -> int:
        """Do nothing.

        Returns:
            int: Zero.
        """
        return 0

    return Case(
        "unit",
        "probe",
        entry,
        lambda: 0,
        must_succeed=must_succeed,
        must_reject=must_reject,
        out_of_contract=out_of_contract,
    )


def test_out_of_contract_turns_a_kernel_overrun_into_a_documented_rejection() -> None:
    """The flag is what reclassifies the bounds-check hit, and only the flag."""
    overrun = IndexError(NUMBA_OOB_MESSAGE)

    verdict, kind, _ = classify(_case(), overrun)
    assert (verdict, kind) == (Verdict.BUG, "numba-oob")

    verdict, kind, _ = classify(_case(out_of_contract=True), overrun)
    assert verdict is Verdict.DOCUMENTED_REJECTION
    assert kind == "out-of-contract:IndexError"


@pytest.mark.parametrize(
    "exc",
    [
        IndexError(NUMBA_OOB_MESSAGE),
        IndexError("index 3 is out of bounds for axis 0 with size 2"),
        ValueError("some message"),
        ZeroDivisionError("division by zero"),
        RuntimeError("anything at all"),
    ],
    ids=lambda e: type(e).__name__ + ("-numba" if str(e) == NUMBA_OOB_MESSAGE else ""),
)
def test_an_out_of_contract_case_is_graded_on_nothing_whatever_it_raises(
    exc: Exception,
) -> None:
    """The branch takes no view on the exception type, deliberately.

    The behaviour of a Layer 3 kernel on out-of-contract input is unspecified, so which
    exception escapes is unspecified too. Grading only ``IndexError`` would smuggle in a
    promise about the failure mode that no kernel makes.
    """
    verdict, kind, _ = classify(_case(out_of_contract=True), exc)
    assert verdict is Verdict.DOCUMENTED_REJECTION
    assert kind == f"out-of-contract:{type(exc).__name__}"


def test_an_out_of_contract_case_that_returns_is_graded_on_nothing() -> None:
    """Its invariants describe a contract the input is outside of, so they do not run."""
    always_fails = custom("never-holds", lambda _result: "this invariant always fails")

    graded = run_case(0, Case("unit", "probe", int, lambda: 0, invariants=(always_fails,)))
    assert graded.verdict is Verdict.BUG

    ungraded = run_case(
        0,
        Case(
            "unit",
            "probe",
            int,
            lambda: 0,
            invariants=(always_fails,),
            out_of_contract=True,
        ),
    )
    assert ungraded.verdict is Verdict.OK
    assert ungraded.kind == "out-of-contract:returned"


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("must_succeed", "must_reject"),
        ("must_succeed", "out_of_contract"),
        ("must_reject", "out_of_contract"),
    ],
)
def test_a_case_may_make_only_one_claim_about_its_input(first: str, second: str) -> None:
    """An input is legal, illegal, or outside the stated contract, never two of them."""
    with pytest.raises(ValueError, match="not two of them"):
        _case(**{first: True, second: True})
