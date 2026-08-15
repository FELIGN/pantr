"""Case model, outcome classification and the runner for the adversarial sweep.

A *case* is one call of one public pantr entry point with one hostile input
combination. The runner executes it, catches everything it throws, and reduces the
outcome to a single :class:`Verdict`:

``OK``
    The call returned and every invariant the case declared held.
``DOCUMENTED_REJECTION``
    The call raised an exception type that the entry point's own docstring lists
    under ``Raises:``. Correct behavior, not a finding.
``UNDOCUMENTED_REJECTION``
    The call raised ``ValueError`` or ``TypeError`` -- the library's validation
    idiom -- but the entry point's ``Raises:`` section does not list it. This is
    almost always a nested Layer-2 validator firing, so it is reported as
    *suspected* rather than as a bug.
``BUG``
    Everything else: an ``IndexError`` from Numba's bounds check, any other
    exception type, a non-finite result from finite inputs, or a declared
    invariant that failed. Also *any* exception at all on a case flagged
    :attr:`Case.must_succeed`, which is how a documented exception type raised for
    an undocumented reason stops passing for a correct rejection.

Only ``BUG`` makes the runner exit non-zero.

The module also owns :func:`assert_boundscheck_active`, the canary that proves the
harness *could* have caught an out-of-bounds access before any result is trusted.
"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import json
import re
import signal
import sys
import traceback
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, NamedTuple, TextIO

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt


class Verdict(enum.IntEnum):
    """Outcome classification of a single swept case, ordered by severity."""

    OK = 0
    """Returned, and every declared invariant held."""

    DOCUMENTED_REJECTION = 1
    """Raised an exception type listed in the entry point's ``Raises:`` section."""

    UNDOCUMENTED_REJECTION = 2
    """Raised ``ValueError``/``TypeError`` that the ``Raises:`` section omits."""

    BUG = 3
    """A finding: bounds-check hit, unexpected exception, or broken invariant."""


NUMBA_OOB_MESSAGE: Final = "index is out of bounds"
"""Exact ``IndexError`` message Numba's bounds check raises (verified on numba 0.65.1).

NumPy's own out-of-range ``IndexError`` messages always name the offending index and
the axis size, so an exact match on this string separates a Numba kernel overrun --
which is silent memory corruption when the bounds check is off -- from an ordinary
Python-level indexing error.
"""

_RAISES_HEADER: Final = re.compile(r"^\s*Raises:\s*$")
_RAISES_ENTRY: Final = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.]*)\s*:")
_SECTION_HEADER: Final = re.compile(
    r"^\s*(Args|Returns|Yields|Note|Notes|Example|Examples|"
    r"Attributes|Warning|Warns|See Also|References):\s*$"
)


class InvariantResult(NamedTuple):
    """Outcome of one invariant check.

    Attributes:
        name (str): Invariant identifier, reported verbatim on failure.
        failure (str | None): ``None`` when the invariant held, otherwise a short
            message describing the violation.
    """

    name: str
    failure: str | None


Invariant = Callable[[Any], InvariantResult]
"""A check applied to a case's return value; returns :class:`InvariantResult`."""


@dataclasses.dataclass(frozen=True, slots=True)
class Case:
    """One hostile call of one entry point.

    Attributes:
        group (str): Sweep group the case belongs to (selectable on the CLI).
        label (str): Human-readable identity, unique within the group.
        entry (Callable[..., Any]): The public symbol under test. Its docstring's
            ``Raises:`` section defines which exceptions count as documented
            rejections; for a class, ``__init__`` is consulted first.
        run (Callable[[], Any]): Thunk performing the call and returning its result.
        params (Mapping[str, Any]): Axis values recorded verbatim in the JSONL
            journal. This is a serialized record payload, not an options bag.
        invariants (tuple[Invariant, ...]): Checks applied to the return value.
        must_succeed (bool): Set it when the input is legal *by construction* -- built
            from a space that constructed cleanly, a point inside the domain, a degree
            increment the docstring accepts. Then **any** exception is a finding,
            whatever the ``Raises:`` section says. Without it, a docstring-driven rule
            cannot see the failure mode where an entry point raises a documented
            exception *type* for an undocumented *reason*: it reads as a correct
            rejection and the sweep says nothing. That is not hypothetical -- it hid an
            internal control-point/knot-vector inconsistency in degree elevation behind
            the ``ValueError`` that documents a negative increment.
        must_reject (bool): The mirror image, for a case built to be *refused* -- a
            malformed knot vector, a negative point count, a coordinate outside the unit
            cube. Then *returning* is the finding. Without it such a case is graded only
            on whether the result is finite, so an entry point that silently started
            accepting nonsense and producing a plausible number would read as ``OK``:
            the input family's intent lives in a comment and nothing checks it.
        finite_inputs (bool): Whether every input is finite. When ``False`` the
            automatic finiteness check on the result is skipped.
        arrays (Mapping[str, npt.NDArray[Any]]): Input arrays to persist under
            ``--dump-npz`` so the C++ port can replay the case as a parity oracle.
            Empty when the case's inputs are fully determined by ``params``.
    """

    group: str
    label: str
    entry: Callable[..., Any]
    run: Callable[[], Any]
    params: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    invariants: tuple[Invariant, ...] = ()
    must_succeed: bool = False
    must_reject: bool = False
    finite_inputs: bool = True
    arrays: Mapping[str, npt.NDArray[Any]] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject a case that claims its input is both legal and illegal.

        Raises:
            ValueError: If both ``must_succeed`` and ``must_reject`` are set.
        """
        if self.must_succeed and self.must_reject:
            raise ValueError(
                f"case {self.group}/{self.label} sets both must_succeed and must_reject"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class Outcome:
    """Classified result of running one :class:`Case`.

    Attributes:
        index (int): Position of the case in the sweep, for ``--start-index``.
        case (Case): The case that produced this outcome.
        verdict (Verdict): Severity classification.
        kind (str): Short machine-readable reason (e.g. ``"numba-oob"``).
        detail (str): One-line human explanation.
        warnings (tuple[str, ...]): Warning categories raised during the call.
    """

    index: int
    case: Case
    verdict: Verdict
    kind: str
    detail: str
    warnings: tuple[str, ...] = ()

    def as_record(self) -> dict[str, Any]:
        """Build the JSONL journal record for this outcome.

        Returns:
            dict[str, Any]: JSON-serializable summary of the case and its verdict.
        """
        return {
            "index": self.index,
            "group": self.case.group,
            "label": self.case.label,
            "entry": _entry_name(self.case.entry),
            "verdict": self.verdict.name,
            "kind": self.kind,
            "detail": self.detail,
            "params": {k: _jsonable(v) for k, v in self.case.params.items()},
            "warnings": list(self.warnings),
        }


class Summary(NamedTuple):
    """Aggregate result of a sweep run.

    Attributes:
        counts (dict[str, int]): Number of cases per :class:`Verdict` name.
        findings (tuple[Outcome, ...]): Every ``BUG`` outcome, in order.
        suspected (tuple[Outcome, ...]): Every ``UNDOCUMENTED_REJECTION`` outcome.
        warned (tuple[Outcome, ...]): Every outcome that emitted a warning while
            *returning* normally. NumPy reports int64 overflow, division by zero and
            invalid operations as ``RuntimeWarning``, so an ``OK`` case that warned is
            the cheapest available lead on a silent numerical fault.
        aborted (str | None): Set when *enumeration* failed, i.e. a probe generator
            raised while building a case rather than while running one. The cases after
            it never execute, so the run is incomplete and its counts mean nothing --
            which is why it gets its own exit code instead of being folded into the
            findings.
    """

    counts: dict[str, int]
    findings: tuple[Outcome, ...]
    suspected: tuple[Outcome, ...]
    warned: tuple[Outcome, ...] = ()
    aborted: str | None = None

    @property
    def total(self) -> int:
        """Total number of cases run.

        Returns:
            int: Sum of all per-verdict counts.
        """
        return sum(self.counts.values())


# ---------------------------------------------------------------------------
# Canary: prove the bounds check is live before trusting any clean result
# ---------------------------------------------------------------------------


class CanaryError(RuntimeError):
    """Raised when the bounds-check canary fails to fire."""


def assert_boundscheck_active() -> str:
    """Verify that Numba's bounds check is live for a ``cache=True`` kernel.

    ``NUMBA_BOUNDSCHECK=1`` is silently ignored when a stale on-disk cache entry
    exists, because the cache key does not include the flag. Every pantr kernel is
    ``cache=True``, so a sweep run against a warm cache returns a false clean. This
    compiles two deliberately out-of-range ``cache=True`` kernels and requires both
    to raise.

    It also refuses to run with ``NUMBA_DISABLE_JIT=1``, and that check is not
    ceremony: with the JIT off, ``njit`` is a no-op, the two probe kernels run as
    plain Python, and NumPy raises ``IndexError`` on its own. The canary would pass
    while nothing had been compiled at all -- the exact false clean it exists to
    prevent. The repository's coverage run does set that variable, so the
    configuration is reachable by accident.

    Returns:
        str: One-line description of the verified configuration.

    Raises:
        CanaryError: If the bounds check is off, if the JIT is disabled, or if either
            deliberate out-of-bounds access fails to raise ``IndexError``.
    """
    import numba  # noqa: PLC0415  -- deferred so the env can be fixed up first

    if not numba.config.BOUNDSCHECK:
        raise CanaryError(
            "numba.config.BOUNDSCHECK is off; run with NUMBA_BOUNDSCHECK=1 and a "
            "fresh NUMBA_CACHE_DIR (see the module docstring for the invocation)."
        )
    if numba.config.DISABLE_JIT:
        raise CanaryError(
            "NUMBA_DISABLE_JIT is set, so no kernel is compiled and the bounds check "
            "grades nothing; the canary below would pass on plain NumPy indexing. "
            "Unset it: this sweep exists to exercise compiled kernels."
        )

    @numba.njit(cache=True)  # type: ignore[misc]
    def _read_past_end(a: npt.NDArray[np.float64]) -> float:
        """Read three elements past the end of ``a``."""
        acc = 0.0
        for i in range(a.size + 3):
            acc += a[i]
        return acc

    @numba.njit(cache=True)  # type: ignore[misc]
    def _read_before_start(a: npt.NDArray[np.float64]) -> float:
        """Index ``a`` with a negative index that wraps past the front."""
        return float(a[-a.size - 2])

    probe = np.ones(4, dtype=np.float64)
    for name, kernel in (("read-past-end", _read_past_end), ("negative-wrap", _read_before_start)):
        try:
            value = kernel(probe)
        except IndexError:
            continue
        raise CanaryError(
            f"canary {name!r} did not raise: returned {value!r}. The bounds check is "
            "not active for cache=True kernels -- most likely a stale Numba cache. "
            "A sweep that cannot detect an out-of-bounds access is worthless."
        )

    return (
        f"canary OK: numba {numba.__version__}, BOUNDSCHECK="
        f"{numba.config.BOUNDSCHECK}, JIT enabled, both deliberate overruns raised IndexError"
    )


# ---------------------------------------------------------------------------
# Documented-rejection lookup
# ---------------------------------------------------------------------------


def documented_exceptions(entry: Callable[..., Any]) -> frozenset[str]:
    """Collect the exception type names listed in an entry point's ``Raises:`` section.

    Google-style docstrings are mandatory in this repository, so the ``Raises:``
    section is the machine-readable contract for which inputs may legitimately be
    refused. For a class, ``__init__``'s docstring is consulted first and the class
    docstring second.

    Args:
        entry (Callable[..., Any]): The public symbol under test.

    Returns:
        frozenset[str]: Bare exception type names, e.g. ``{"ValueError", "TypeError"}``.
    """
    docs: list[str] = []
    init = getattr(entry, "__init__", None)
    if isinstance(entry, type) and init is not None and init.__doc__:
        docs.append(init.__doc__)
    if entry.__doc__:
        docs.append(entry.__doc__)

    names: set[str] = set()
    for doc in docs:
        names |= _parse_raises(doc)
    return frozenset(names)


def _parse_raises(doc: str) -> set[str]:
    """Extract exception names from the ``Raises:`` section of a Google docstring.

    Args:
        doc (str): Docstring text.

    Returns:
        set[str]: Bare exception type names found in the section.
    """
    names: set[str] = set()
    in_section = False
    for line in doc.splitlines():
        if _RAISES_HEADER.match(line):
            in_section = True
            continue
        if not in_section:
            continue
        if _SECTION_HEADER.match(line):
            break
        match = _RAISES_ENTRY.match(line)
        if match is not None:
            names.add(match.group(1).rsplit(".", maxsplit=1)[-1])
    return names


# ---------------------------------------------------------------------------
# Invariant factories
# ---------------------------------------------------------------------------


def finite_result() -> Invariant:
    """Build an invariant requiring every floating-point output to be finite.

    Returns:
        Invariant: Check reporting the first non-finite array found.
    """

    def check(result: Any) -> InvariantResult:  # noqa: ANN401 -- classifies any return value
        for path, arr in _iter_float_arrays(result):
            if arr.size and not np.all(np.isfinite(arr)):
                bad = int(np.count_nonzero(~np.isfinite(arr)))
                return InvariantResult("finite", f"{bad}/{arr.size} non-finite in {path}")
        return InvariantResult("finite", None)

    return check


def partition_of_unity(degree: int, dtype: npt.DTypeLike, *, axis: int = -1) -> Invariant:
    """Build an invariant requiring basis values to sum to one along ``axis``.

    The tolerance is derived rather than tuned. Summing ``n = degree + 1``
    non-negative values whose exact total is one carries a forward error of at most
    ``(n - 1) * eps``; each value itself comes out of a recurrence of depth
    ``degree``, contributing at most ``O(degree) * eps`` more. Their product bounds
    the total by ``(degree + 1) ** 2 * eps``, and an explicit safety factor of 4
    absorbs the recurrence's unmodelled constant. A genuine partition-of-unity
    failure is gross (zero, ``NaN``, or a value like 1.58 from summing a
    non-truncated hierarchical basis), so a conservative bound costs no sensitivity.

    Args:
        degree (int): Polynomial degree of the basis, which sets the term count.
        dtype (npt.DTypeLike): Working precision, which sets machine epsilon.
        axis (int): Axis holding the basis functions. Defaults to ``-1``.

    Returns:
        Invariant: Check reporting the worst deviation from one.
    """
    from pantr.tolerance import get_machine_epsilon  # noqa: PLC0415  -- avoids import cycle

    tol = 4.0 * (degree + 1) ** 2 * get_machine_epsilon(dtype)

    def check(result: Any) -> InvariantResult:  # noqa: ANN401 -- classifies any return value
        arr = np.asarray(result)
        if arr.size == 0:
            return InvariantResult("partition-of-unity", None)
        sums = np.sum(arr, axis=axis)
        worst = float(np.max(np.abs(sums - 1.0)))
        if not np.isfinite(worst) or worst > tol:
            return InvariantResult("partition-of-unity", f"max|sum-1| = {worst:.3e} > {tol:.3e}")
        return InvariantResult("partition-of-unity", None)

    return check


def expected_shape(shape: tuple[int, ...]) -> Invariant:
    """Build an invariant requiring the result to have an exact shape.

    Args:
        shape (tuple[int, ...]): The shape the entry point's own contract promises.

    Returns:
        Invariant: Check reporting the mismatch.
    """

    def check(result: Any) -> InvariantResult:  # noqa: ANN401 -- classifies any return value
        got = np.shape(result)
        if tuple(got) != shape:
            return InvariantResult("shape", f"got {tuple(got)}, contract says {shape}")
        return InvariantResult("shape", None)

    return check


def custom(name: str, predicate: Callable[[Any], str | None]) -> Invariant:
    """Wrap a bespoke predicate as an invariant.

    Args:
        name (str): Invariant identifier reported on failure.
        predicate (Callable[[Any], str | None]): Returns ``None`` when the property
            holds, otherwise a short failure message.

    Returns:
        Invariant: The wrapped check.
    """

    def check(result: Any) -> InvariantResult:  # noqa: ANN401 -- classifies any return value
        return InvariantResult(name, predicate(result))

    return check


def _iter_float_arrays(
    result: Any,  # noqa: ANN401 -- walks an arbitrary return value
    path: str = "result",
) -> Iterator[tuple[str, npt.NDArray[np.floating[Any]]]]:
    """Walk a return value yielding every floating-point array it contains.

    Args:
        result (Any): Value returned by an entry point.
        path (str): Dotted path used in failure messages. Defaults to ``"result"``.

    Yields:
        tuple[str, npt.NDArray[np.floating[Any]]]: Path and array, for each float array
            reached.
    """
    if isinstance(result, np.ndarray):
        if result.dtype.kind == "f":
            yield path, result
        return
    if isinstance(result, str | bytes):
        return
    if isinstance(result, Mapping):
        for key, value in result.items():
            yield from _iter_float_arrays(value, f"{path}[{key!r}]")
        return
    if isinstance(result, Sequence):
        for i, value in enumerate(result):
            yield from _iter_float_arrays(value, f"{path}[{i}]")
        return
    if isinstance(result, float):
        yield path, np.asarray(result, dtype=np.float64)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def classify(case: Case, exc: Exception) -> tuple[Verdict, str, str]:
    """Classify an exception thrown by a case.

    Args:
        case (Case): The case that raised.
        exc (Exception): The exception it raised.

    Returns:
        tuple[Verdict, str, str]: Verdict, machine-readable kind, and detail line.
    """
    name = type(exc).__name__
    message = str(exc).strip()

    if isinstance(exc, IndexError) and message == NUMBA_OOB_MESSAGE:
        return Verdict.BUG, "numba-oob", "Numba bounds check: out-of-range access in a kernel"

    if case.must_succeed:
        return (
            Verdict.BUG,
            f"must-succeed:{name}",
            f"{name} on input that is legal by construction: {message}",
        )

    documented = documented_exceptions(case.entry)
    if name in documented:
        return Verdict.DOCUMENTED_REJECTION, f"documented:{name}", message

    if isinstance(exc, ValueError | TypeError):
        listed = ", ".join(sorted(documented)) or "nothing"
        return (
            Verdict.UNDOCUMENTED_REJECTION,
            f"undocumented:{name}",
            f"{name}: {message} (Raises: lists {listed})",
        )

    return Verdict.BUG, f"unexpected:{name}", f"{name}: {message}"


class CaseTimeout(RuntimeError):
    """Raised when a case exceeds :data:`CASE_TIMEOUT_SECONDS`."""


CASE_TIMEOUT_SECONDS: Final = 30.0
"""Wall-clock budget for **one attempt** at a case; two expiries make it a hang.

Non-termination is one of the outcomes this sweep hunts, and it needs a budget because
the alternative is that the *sweep* hangs and nobody runs it again. That is not
hypothetical: ``Bspline.reduce_degree(1)`` never returns on a degree-1 *periodic* spline,
and stalled a 25952-case run indefinitely.

The value is deliberately **not** load-bearing, because it cannot be: a budget that
separates "slow" from "stuck" by magnitude alone has to be tuned to a machine, and the
first call into any kernel pays LLVM codegen that is far larger than the work itself
(16.6 s against 0.000 s here). :func:`_call_with_budget` does the separating instead, by
retrying once; thirty seconds is then merely a generous allowance for one compile.

**Limitation, stated because it matters for the port:** the alarm is delivered at a Python
bytecode boundary, so it interrupts a hang in a Layer-2 Python loop but **not** one inside
a compiled ``nopython`` kernel, which does not return to the interpreter. A kernel that
spins forever still hangs the sweep, and the progress line printed before each case is
what identifies it.

**The same limitation makes a slow *library* call indistinguishable from a hang, and a
reader must not read a timeout as non-termination.** A long-running call into LAPACK or any
other C extension does not return to the interpreter either, so both attempts run to
completion and the retry that separates "slow" from "stuck" everywhere else cannot separate
them here. Measured: ``Bspline.multiply`` on a degree-62 periodic spline reaches a dense
``numpy.linalg.lstsq`` over the product space at degree 124 and **returns after 78 s**,
having been reported as "did not terminate, on two attempts". It is slow, not stuck. Before
recording a timeout as a hang, re-run the case with a generous budget and a
``faulthandler`` traceback and see where it actually sits.
"""


def _timeout_guard(seconds: float) -> AbstractContextManager[None]:
    """Build a context manager that raises :class:`CaseTimeout` after ``seconds``.

    Falls back to doing nothing where ``SIGALRM`` is unavailable (Windows), since a sweep
    without the guard is still worth running.

    Args:
        seconds (float): Wall-clock budget.

    Returns:
        AbstractContextManager[None]: The guard.
    """
    if not hasattr(signal, "SIGALRM"):
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def guard() -> Iterator[None]:
        def on_alarm(signum: int, frame: object) -> None:
            del signum, frame
            raise CaseTimeout(f"no return after {seconds:g} s")

        previous = signal.signal(signal.SIGALRM, on_alarm)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous)

    return guard()


def _call_with_budget(case: Case) -> Any:  # noqa: ANN401 -- returns whatever the entry point does
    """Run a case's thunk under the timeout, retrying once if the first attempt expires.

    The retry is what makes the budget's absolute value stop mattering, and it is not an
    optimization -- without it the harness reports a *hang* that is really a slow compile.
    The first call into any Numba kernel pays LLVM codegen, and under
    ``NUMBA_BOUNDSCHECK=1`` on a fresh cache that is expensive: measured at **16.6 s** for
    the first ``tabulate_bernstein_1d`` call on the development machine against **0.000 s**
    for the second, a ratio of 8.7e5. Whichever case happens to reach a kernel first is
    charged the whole of it, so on a machine slower at codegen a perfectly healthy case
    blows any fixed budget -- which is exactly what happened on CI, where a degree-0
    Bernstein tabulation was reported as a 30 s hang.

    Retrying separates the two by observation rather than by a constant tuned to one
    machine: compilation is paid once, so the second attempt is fast, while a genuine
    non-termination expires again. The cost is that a real hang takes
    ``2 * CASE_TIMEOUT_SECONDS`` to report, which is the right trade -- hangs are rare and
    a false one is far more expensive than a slow one.

    Args:
        case (Case): The case to run.

    Returns:
        Any: Whatever the entry point returned.

    Raises:
        CaseTimeout: If the call expires twice, i.e. it genuinely does not terminate.
    """
    try:
        with _timeout_guard(CASE_TIMEOUT_SECONDS):
            return case.run()
    except CaseTimeout:
        pass
    with _timeout_guard(CASE_TIMEOUT_SECONDS):
        return case.run()


def run_case(index: int, case: Case) -> Outcome:
    """Execute one case and classify the outcome.

    An invariant that *raises* is caught and reported, not allowed to propagate. A
    predicate usually raises because the probe is wrong, but the whole point of the
    surrounding machinery is that no single case can end the run, and an uncaught
    exception here would do exactly that: the remaining cases never execute and the
    process exit status is indistinguishable from "findings were reported". A case that
    does not terminate is reported the same way, on the terms
    :func:`_call_with_budget` sets.

    Args:
        index (int): Position of the case in the sweep.
        case (Case): The case to run.

    Returns:
        Outcome: The classified result.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = _call_with_budget(case)
        except CaseTimeout as exc:
            return Outcome(
                index,
                case,
                Verdict.BUG,
                "timeout",
                f"the call did not terminate, on two attempts: {exc}",
                _warning_names(caught),
            )
        except Exception as exc:  # a sweep classifies whatever comes out, by design
            verdict, kind, detail = classify(case, exc)
            if verdict is Verdict.BUG:
                detail = f"{detail}\n{_short_traceback(exc)}"
            return Outcome(index, case, verdict, kind, detail, _warning_names(caught))
        if case.must_reject:
            return Outcome(
                index,
                case,
                Verdict.BUG,
                "must-reject:returned",
                f"input built to be refused was accepted, returning {type(result).__name__}",
                _warning_names(caught),
            )
        checks = [*case.invariants]
        if case.finite_inputs:
            checks.append(finite_result())
        for check in checks:
            try:
                outcome = check(result)
            except Exception as exc:  # a predicate must never end the run
                return Outcome(
                    index,
                    case,
                    Verdict.BUG,
                    f"invariant-raised:{type(exc).__name__}",
                    f"an invariant raised instead of returning a verdict: {exc}\n"
                    f"{_short_traceback(exc)}",
                    _warning_names(caught),
                )
            if outcome.failure is not None:
                return Outcome(
                    index,
                    case,
                    Verdict.BUG,
                    f"invariant:{outcome.name}",
                    outcome.failure,
                    _warning_names(caught),
                )
        return Outcome(index, case, Verdict.OK, "ok", "", _warning_names(caught))


def run_sweep(  # noqa: PLR0913 -- five keyword-only knobs, each one CLI flag
    cases: Iterable[Case],
    *,
    journal: TextIO = sys.stdout,
    progress: TextIO = sys.stderr,
    start_index: int = 0,
    max_cases: int | None = None,
    dump_dir: Path | None = None,
) -> Summary:
    """Run a sequence of cases, journaling each one before it executes.

    Each case's identity is written to ``progress`` and flushed *before* the call,
    so a hang or a hard crash (which an unchecked Numba kernel can produce) still
    identifies the case that caused it.

    Failures while *building* a case are caught too, and reported through
    :attr:`Summary.aborted`. A probe generator that raises cannot be resumed -- Python
    closes it -- so the run genuinely ends there; what must not happen is that it ends
    with a bare traceback and an exit status a caller cannot tell apart from "findings
    were reported". This is not hypothetical: an invariant factory that evaluated its
    reference eagerly, at build time rather than in the returned closure, truncated a
    13341-case run silently.

    Args:
        cases (Iterable[Case]): The cases to run, in order.
        journal (TextIO): Text stream receiving one JSON record per case.
        progress (TextIO): Text stream receiving human-readable progress.
        start_index (int): Skip cases before this index, to resume past a crash.
        max_cases (int | None): Stop after this many executed cases.
        dump_dir (Path | None): When set, persist each case's declared input arrays
            plus its float outputs as ``.npz`` for reuse as parity fixtures.

    Returns:
        Summary: Per-verdict counts, the findings, and the abort reason if enumeration
            failed.
    """
    counts: dict[str, int] = {verdict.name: 0 for verdict in Verdict}
    findings: list[Outcome] = []
    suspected: list[Outcome] = []
    warned: list[Outcome] = []
    executed = 0
    aborted: str | None = None

    iterator = iter(cases)
    index = -1
    while True:
        index += 1
        try:
            case = next(iterator)
        except StopIteration:
            break
        except Exception as exc:  # a probe generator must not end the run silently
            aborted = f"{type(exc).__name__} while building case {index}: {exc}"
            print(f"[sweep] ENUMERATION ABORTED: {aborted}", file=progress, flush=True)
            print(f"{_short_traceback(exc, depth=6)}", file=progress, flush=True)
            break
        if index < start_index:
            continue
        if max_cases is not None and executed >= max_cases:
            break
        print(f"[{index}] {case.group}/{case.label}", file=progress, flush=True)
        outcome = run_case(index, case)
        executed += 1
        counts[outcome.verdict.name] += 1
        if outcome.verdict is Verdict.BUG:
            findings.append(outcome)
        elif outcome.verdict is Verdict.UNDOCUMENTED_REJECTION:
            suspected.append(outcome)
        if outcome.verdict is Verdict.OK and outcome.warnings:
            warned.append(outcome)
        print(json.dumps(outcome.as_record()), file=journal, flush=True)
        if dump_dir is not None and case.arrays:
            _dump_case(dump_dir, outcome)

    return Summary(counts, tuple(findings), tuple(suspected), tuple(warned), aborted)


def _dump_case(dump_dir: Path, outcome: Outcome) -> None:
    """Persist a case's input arrays and float outputs as a parity fixture.

    Args:
        dump_dir (Path): Directory to write into; created if absent.
        outcome (Outcome): The executed case.
    """
    dump_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{outcome.case.group}-{outcome.case.label}")
    payload = {f"in_{k}": np.asarray(v) for k, v in outcome.case.arrays.items()}
    np.savez_compressed(dump_dir / f"{outcome.index:06d}-{safe}.npz", **payload)


def _warning_names(caught: Sequence[warnings.WarningMessage]) -> tuple[str, ...]:
    """Reduce captured warnings to their unique category names.

    Args:
        caught (Sequence[warnings.WarningMessage]): Warnings recorded during a call.

    Returns:
        tuple[str, ...]: Sorted unique category names.
    """
    return tuple(sorted({w.category.__name__ for w in caught}))


def _short_traceback(exc: Exception, depth: int = 3) -> str:
    """Format the innermost frames of an exception's traceback.

    Args:
        exc (Exception): The exception to format.
        depth (int): Number of innermost frames to keep. Defaults to 3.

    Returns:
        str: Indented traceback tail.
    """
    frames = traceback.format_tb(exc.__traceback__)[-depth:]
    return "".join(frames).rstrip()


def _entry_name(entry: Callable[..., Any]) -> str:
    """Build a stable dotted name for an entry point.

    Args:
        entry (Callable[..., Any]): The symbol under test.

    Returns:
        str: ``module.qualname``, or ``repr`` when either is unavailable.
    """
    module = getattr(entry, "__module__", "?")
    qualname = getattr(entry, "__qualname__", None) or repr(entry)
    return f"{module}.{qualname}"


def _jsonable(  # noqa: PLR0911 -- a flat type-dispatch chain, one return per case
    value: Any,  # noqa: ANN401 -- any axis value may be recorded
) -> Any:  # noqa: ANN401
    """Coerce a parameter value into something ``json.dumps`` accepts.

    Args:
        value (Any): Recorded axis value.

    Returns:
        Any: JSON-serializable equivalent.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.dtype):
        return value.name
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, str | int | float | bool | None):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return repr(value)
