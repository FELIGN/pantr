"""Parity of the cardinal interval scan: the C++ free function against the oracle.

`pantr.bspline.BsplineSpace1D.get_cardinal_intervals` reaches C++ through
:mod:`pantr.bspline._knots_backend`, the per-area catalogue for the knot
computations, and `pantr::bspline::cardinal_intervals` is what it calls.

## What kind of claim this file makes

**Exact, and no tolerance applies.** The scan returns one boolean per interval, so
there is no quantity to bound: the two backends agree element for element or one of
them is wrong. The tolerance *inside* the scan -- the space's own
:attr:`~pantr.bspline.BsplineSpace1D.tolerance`, which decides whether two knot spans
are the same length -- is not compared here and is not re-derived by either side;
``tests/parity/test_bspline_space_1d.py`` compares it bitwise, which is what makes it
sound to compare only the flags below.

The oracle is `pantr.bspline._bspline_knots._get_Bspline_cardinal_intervals_1D_impl`,
called directly rather than through the accessor, so a defect in the dispatch cannot
make the comparison vacuous by running the oracle on both sides.

## The independent accuracy check

Agreement with the oracle is all this file establishes, and
``design/backend_parity.md`` asks for more. The independent check for this quantity
is in ``tests/parity/test_bspline_space_1d.py``'s
``test_the_clamped_uniform_closed_form``, which asserts the flags of a clamped
uniform mesh against a closed form derived by hand rather than against either
implementation, over the whole ``(degree, num_intervals)`` grid. The C++ unit test
``cpp/tests/test_bspline_knots.cpp`` checks the same closed form and five further
hand-read vectors without any Python in reach.

## Rule 12

Every test that says something about the *binding* takes the ``cpp_backend``
fixture. The tests that state a property of *each* backend take it on the C++
parameter instead, through ``_BACKENDS`` -- taking it on the test would skip the
Python half too, and the Python half is the only thing here that would catch the
oracle regressing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, NamedTuple

import numpy as np
import pytest

from pantr._backend import Backend, use_backend
from pantr.bspline import BsplineSpace1D
from pantr.bspline._bspline_knots import _get_Bspline_cardinal_intervals_1D_impl
from tests._parity_harness import (
    Field,
    assert_object_parity,
    demand_cpp_backend,
    exact_parity,
)

if TYPE_CHECKING:
    from numpy import typing as npt

_BACKENDS: Final = (
    pytest.param(Backend.PYTHON, id="python"),
    pytest.param(Backend.CPP, id="cpp"),
)
"""The two backends, for the tests that state a property of each one separately."""

_DTYPES: Final = (np.float64, np.float32)
"""The two storage formats a space may hold."""

_WHY: Final = (
    "the scan returns one boolean per interval, reached on both sides by the same "
    "multiplicity comparisons and the same knot-span comparisons against the space's "
    "own tolerance, so there is no quantity to bound and a difference is a defect"
)


def _demand_the_extension_if_needed(backend: Backend) -> None:
    """Require the compiled extension, and only for the half that uses it.

    ``tests/parity/test_bspline_space_1d.py`` carries the argument in full: taking
    the ``cpp_backend`` fixture on a test parametrized over both backends skips the
    Python half as well, and the Python half is what would catch the oracle itself
    regressing.

    Args:
        backend (Backend): The backend this case runs under.
    """
    if backend is Backend.CPP:
        demand_cpp_backend()


class _Case(NamedTuple):
    """One space to scan, and what it is in the table for.

    Attributes:
        label (str): What the case is here to exercise.
        knots (list[float]): The knot vector.
        degree (int): The polynomial degree.
        periodic (bool): Whether the space is periodic.
        snap (bool): Whether to merge knots that are the same knot.
    """

    label: str
    knots: list[float]
    degree: int
    periodic: bool
    snap: bool


_CASES: Final = tuple(
    _Case(*case)
    for case in (
        # A mesh with cardinal intervals in the middle and none at the ends, which
        # is the shape the flag exists to report.
        (
            "clamped uniform quadratic",
            [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0],
            2,
            False,
            True,
        ),
        # The multiplicity gate: a repeated interior knot disqualifies the two
        # intervals it bounds whatever the spacing around them is.
        (
            "repeated interior knot",
            [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0, 6.0, 6.0],
            2,
            False,
            True,
        ),
        # The length gate, on its own: every knot is simple, and one span of the
        # wrong length disqualifies itself and the degree-1 intervals either side.
        (
            "one span of the wrong length",
            [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.5, 5.5, 6.5, 7.5, 7.5, 7.5],
            2,
            False,
            True,
        ),
        # Unclamped: the window reaches past the domain on both sides, so knots
        # outside it decide in-domain answers.
        ("unclamped cubic", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0], 3, False, True),
        # Degree 0: the comparison window is empty, so the length condition holds
        # vacuously and only the multiplicity gate decides.
        ("degree zero", [0.0, 0.1, 0.5, 0.9, 1.0], 0, False, True),
        ("degree zero with a repeated knot", [0.0, 0.25, 0.25, 0.5, 1.0], 0, False, False),
        # Degree 1: the window is the interval itself, so the length condition is
        # trivially satisfied and the clamped ends are the only thing that bites.
        ("degree one", [0.0, 0.0, 1.0, 2.5, 3.0, 4.0, 4.0], 1, False, True),
        # A single span: no interval can be cardinal, because both its bounding
        # knots are repeated.
        ("bezier-like", [1.0, 1.0, 1.0, 3.0, 3.0, 3.0], 2, False, True),
        # An interior knot at the maximum multiplicity `degree + 1`.
        (
            "interior knot at maximum multiplicity",
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            2,
            False,
            True,
        ),
        # Periodic: no end is clamped, so every interval reaches the gates on its
        # spacing alone.
        ("periodic uniform", [float(k) for k in range(10)], 2, True, True),
        ("periodic non-uniform", [0.0, 1.0, 2.0, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5], 2, True, True),
        # The span is 100 against coordinates of 1e6, so the tolerance the span
        # comparison uses is set by the coordinates rather than by the span. The
        # mesh is spaced widely enough that float32 resolves it.
        (
            "offset far from the origin",
            [1e6, 1e6, 1e6, 1e6 + 25.0, 1e6 + 50.0, 1e6 + 75.0] + [1e6 + 100.0] * 3,
            2,
            False,
            True,
        ),
        # A domain smaller than one unit, where the tolerance is set by the span.
        (
            "tiny domain",
            [0.0, 0.0, 0.0, 2.5e-7, 5e-7, 7.5e-7, 1e-6, 1e-6, 1e-6],
            2,
            False,
            True,
        ),
        # Clamped on the left and not on the right, so a swapped end index shows.
        ("asymmetric ends", [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 2, False, True),
        # The one vector whose answer separates the scan's window index from the one
        # the definition alone would give. The first knot repeats four times at
        # degree 2 -- `check_last_multiplicity` refuses that only at the *last* knot --
        # so the walk is seeded one place before the class's true last index and every
        # window is read one place to the left. Interval 1 is cardinal by the
        # definition and is reported non-cardinal by both backends, which is the
        # faithfulness claim `cpp/include/pantr/bspline/knots.hpp` argues for; nothing
        # else here would notice a C++ port that "corrected" it.
        (
            "over-clamped first knot",
            [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0],
            2,
            False,
            True,
        ),
        # Snapping off, with two knots closer than the tolerance: the classes the
        # scan walks are then the only thing merging them.
        (
            "snapping off",
            [0.0, 0.0, 0.0, 0.5, 0.5 + 2e-16, 1.0, 1.5, 1.5, 1.5],
            2,
            False,
            False,
        ),
        # Degree 4, which no hand-written case anywhere else reaches.
        (
            "degree four",
            [0.0] * 4 + [float(k) for k in range(11)] + [10.0] * 4,
            4,
            False,
            True,
        ),
        # Degree 5, where the window spans nine knot intervals and reaches four either
        # side of the interval it judges. Twelve intervals rather than eight, because
        # a degree-5 window does not fit inside a domain of eight: every interval
        # would answer False and four single-token mutations of the scan were shown
        # to leave such a case unchanged.
        (
            "degree five",
            [0.0] * 5 + [float(k) for k in range(13)] + [12.0] * 5,
            5,
            False,
            True,
        ),
    )
)
"""One case per structural feature the scan branches on.

The two gates separately and together, degree 0 where the window is empty, degree 1
where it is the interval itself, degree 5 where it is nine spans wide, an unclamped
and a periodic space where no end is repeated, a single-span space, a domain far from
the origin and one smaller than a unit, asymmetric ends, and snapping turned off.
"""


def _build(case: _Case, dtype: npt.DTypeLike) -> BsplineSpace1D:
    """Build one case's space under whichever backend is active.

    Args:
        case (_Case): The case to build.
        dtype (npt.DTypeLike): The storage format.

    Returns:
        BsplineSpace1D: The space.
    """
    return BsplineSpace1D(
        np.asarray(case.knots, dtype=dtype),
        case.degree,
        periodic=case.periodic,
        snap_knots=case.snap,
    )


def _oracle(space: BsplineSpace1D) -> npt.NDArray[np.bool_]:
    """The oracle's answer for a space, reached without the accessor.

    Called on the space's *own* knots, degree and tolerance, so that a space built
    under either backend can be handed to it.

    Args:
        space (BsplineSpace1D): The space to scan.

    Returns:
        npt.NDArray[np.bool_]: One flag per interval.
    """
    return _get_Bspline_cardinal_intervals_1D_impl(space.knots, space.degree, space.tolerance)


@pytest.mark.parametrize("dtype", _DTYPES, ids=["float64", "float32"])
@pytest.mark.parametrize("case", _CASES, ids=[c.label for c in _CASES])
def test_the_cpp_scan_agrees_with_the_oracle(
    cpp_backend: None,
    case: _Case,
    dtype: npt.DTypeLike,
) -> None:
    """The C++ scan reports the same flags as the oracle, element for element.

    The oracle is called directly on the same space rather than through the
    accessor, so a dispatch that silently fell back would compare the oracle with
    itself and this would still fail.
    """
    with use_backend(Backend.CPP):
        space = _build(case, dtype)
        flags = space.get_cardinal_intervals()

    expected = _oracle(space)
    assert flags.dtype == expected.dtype, f"{case.label}: {flags.dtype} against {expected.dtype}"
    np.testing.assert_array_equal(
        flags,
        expected,
        err_msg=f"{case.label} at {np.dtype(dtype).name}: {_WHY}",
    )


@pytest.mark.parametrize("dtype", _DTYPES, ids=["float64", "float32"])
@pytest.mark.parametrize("case", _CASES, ids=[c.label for c in _CASES])
def test_the_two_backends_scan_their_own_space_alike(
    cpp_backend: None,
    case: _Case,
    dtype: npt.DTypeLike,
) -> None:
    """Two spaces built from one knot vector, one per backend, answer alike.

    A second claim rather than a restatement of the one above, which scans a single
    C++ space twice: this one builds the space on each side, so it would also fail
    if the two backends stored knots or derived a tolerance the scan then disagreed
    about.
    """
    with use_backend(Backend.PYTHON):
        python = _build(case, dtype)
    with use_backend(Backend.CPP):
        cpp = _build(case, dtype)

    assert_object_parity(
        py=python,
        cpp=cpp,
        fields=[
            Field(
                "get_cardinal_intervals",
                exact_parity(why=_WHY),
                read=lambda s: s.get_cardinal_intervals(),
            )
        ],
        context=f"BsplineSpace1D.get_cardinal_intervals, {case.label}",
    )


def test_the_cpp_route_is_the_one_that_ran(
    cpp_backend: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The accessor reaches the binding under the C++ backend and not under the other.

    The vacuity guard for every parity test above: a catalogue entry that was
    removed, or a branch that fell back to the oracle, would leave all of them
    passing while measuring nothing. It records calls through
    ``pantr._pantr_cpp`` itself rather than through the adapter, so it also fails if
    the adapter stops calling the binding.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (the extension is present here)

    calls: list[str] = []
    real = _pantr_cpp.bspline_space_cardinal_intervals_1d

    def recording(space: object, *, out: npt.NDArray[np.bool_]) -> None:
        calls.append(type(space).__name__)
        # `object` rather than the two handle classes, which would need the stub
        # imported at module scope; the recorder does not look inside the argument.
        real(space, out=out)  # type: ignore[arg-type]

    monkeypatch.setattr(_pantr_cpp, "bspline_space_cardinal_intervals_1d", recording)

    with use_backend(Backend.CPP):
        cpp_space = BsplineSpace1D([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0], 2)
        cpp_space.get_cardinal_intervals()
    assert calls == ["BsplineSpace1D64"], "the C++ backend did not reach the binding"

    with use_backend(Backend.PYTHON):
        python_space = BsplineSpace1D([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0], 2)
        python_space.get_cardinal_intervals()
    assert calls == ["BsplineSpace1D64"], "the Python backend reached the binding"


@pytest.mark.parametrize("backend", _BACKENDS)
def test_out_is_filled_and_handed_back(backend: Backend) -> None:
    """``out=`` is written in full and returned, under either backend.

    The ``out`` convention is the wrapper's contract and not a backend's, so both
    halves of it are asserted here rather than only the C++ one.
    """
    _demand_the_extension_if_needed(backend)
    with use_backend(backend):
        space = BsplineSpace1D([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0], 2)
        expected = space.get_cardinal_intervals()

        out = np.zeros(space.num_intervals, dtype=np.bool_)
        returned = space.get_cardinal_intervals(out=out)

    assert returned is out
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_a_strided_out_is_filled_too(backend: Backend) -> None:
    """A non-contiguous ``out`` is accepted, under either backend.

    The binding's typed signature refuses one rather than converting it, so the C++
    path buffers and copies back; the numba kernel fills it directly. The difference
    must not be visible here, and nothing else in the suite passes this accessor a
    strided view.
    """
    _demand_the_extension_if_needed(backend)
    with use_backend(backend):
        space = BsplineSpace1D([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0], 2)
        expected = space.get_cardinal_intervals()

        backing = np.zeros(2 * space.num_intervals, dtype=np.bool_)
        view = backing[::2]
        assert not view.flags["C_CONTIGUOUS"]
        returned = space.get_cardinal_intervals(out=view)

    assert returned is view
    np.testing.assert_array_equal(view, expected)
    np.testing.assert_array_equal(backing[1::2], np.zeros(space.num_intervals, dtype=np.bool_))


class _Refusal(NamedTuple):
    """One bad ``out``, and the message both backends owe for it.

    Attributes:
        label (str): What is wrong with the array.
        make (object): Builds the offending array from the interval count.
        pattern (str): The regex the message has matched since before the port.
    """

    label: str
    make: object
    pattern: str


_REFUSALS: Final = (
    _Refusal("wrong shape", lambda n: np.zeros(n + 1, dtype=np.bool_), "Output array has shape"),
    _Refusal("wrong dtype", lambda n: np.zeros(n, dtype=np.int_), "Output array has dtype"),
    _Refusal("read-only", None, "Output array is not writeable"),
    _Refusal(
        "masked",
        lambda n: np.ma.masked_array(np.zeros(n, dtype=np.bool_)),
        "which pantr does not accept",
    ),
)
"""The four refusals of a bad ``out``, three of them pinned by regex before the port.

``tests/test_bspline_space_1D.py`` already asserted the first three for this accessor.
The masked-array one is a ``TypeError`` rather than a ``ValueError`` and is here
because the C++ route would otherwise write straight through the mask into the
caller's own array, which is the divergence
:func:`pantr.basis._basis_utils._reject_masked_array` exists to stop.

The read-only case builds its array in the test body, because a flag has to be
cleared after construction.
"""


@pytest.mark.parametrize("refusal", _REFUSALS, ids=[r.label for r in _REFUSALS])
def test_the_out_refusals_are_identical(cpp_backend: None, refusal: _Refusal) -> None:
    """Both backends refuse a bad ``out`` with the same message, character for character.

    ``out`` is validated above the branch in :mod:`pantr.bspline._knots_backend`, so
    this is a claim about where the check sits rather than about two texts that
    happen to match. The pattern each message is also matched against is the one
    ``tests/test_bspline_space_1D.py`` has asserted since before the port, except for
    the masked-array case, which is new here.
    """
    expected_error: type[Exception] = TypeError if refusal.label == "masked" else ValueError
    knots = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0]
    messages = []
    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            space = BsplineSpace1D(knots, 2)
            if refusal.make is None:
                out = np.zeros(space.num_intervals, dtype=np.bool_)
                out.setflags(write=False)
            else:
                out = refusal.make(space.num_intervals)  # type: ignore[operator]
            with pytest.raises(expected_error, match=refusal.pattern) as caught:
                space.get_cardinal_intervals(out=out)
        messages.append(str(caught.value))

    assert messages[0] == messages[1], (
        f"{refusal.label}: the two backends refuse with different text:\n"
        f"  python: {messages[0]}\n  cpp:    {messages[1]}"
    )


@pytest.mark.parametrize("backend", _BACKENDS)
def test_out_is_cleared_before_the_gated_writes(backend: Backend) -> None:
    """An ``out`` arriving full of ``True`` comes back with the False entries cleared.

    Both implementations write only the entries they find cardinal and clear the rest
    up front -- the numba kernel with ``out.fill``, the C++ one with ``std::fill``. A
    caller reusing a buffer across calls, which is what the ``out=`` convention is
    for, is the one who would see that clear go missing, and every other test here
    hands in an array of zeros and so could not tell.
    """
    _demand_the_extension_if_needed(backend)
    with use_backend(backend):
        space = BsplineSpace1D([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0], 2)
        expected = space.get_cardinal_intervals()
        assert not expected.all(), "the fixture must have a False entry or this proves nothing"

        out = np.ones(space.num_intervals, dtype=np.bool_)
        space.get_cardinal_intervals(out=out)

    np.testing.assert_array_equal(out, expected)


def test_a_python_space_is_scanned_by_the_oracle_under_the_cpp_backend(
    cpp_backend: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A space the Python backend built is served by the oracle, not refused.

    The C++ route needs the backend *and* the handle. This pins the second condition,
    which is what lets a caller hold a space across a backend switch -- and what lets
    ``tests/parity/`` read an accessor off two spaces outside a ``use_backend`` block,
    the way every field-by-field comparison in this directory does.

    Nothing is converted: the Python implementation answers about its own knots, and
    the answer is the C++ one. That is asserted here rather than assumed.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (the extension is present here)

    knots = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0]
    with use_backend(Backend.PYTHON):
        python_space = BsplineSpace1D(knots, 2)
    with use_backend(Backend.CPP):
        cpp_space = BsplineSpace1D(knots, 2)

    calls: list[str] = []
    real = _pantr_cpp.bspline_space_cardinal_intervals_1d

    def recording(space: object, *, out: npt.NDArray[np.bool_]) -> None:
        calls.append(type(space).__name__)
        # `object` rather than the two handle classes; see the vacuity guard above.
        real(space, out=out)  # type: ignore[arg-type]

    monkeypatch.setattr(_pantr_cpp, "bspline_space_cardinal_intervals_1d", recording)

    with use_backend(Backend.CPP):
        flags = python_space.get_cardinal_intervals()
        assert calls == [], "a Python-backed space reached the binding"
        np.testing.assert_array_equal(flags, cpp_space.get_cardinal_intervals())
        assert calls == ["BsplineSpace1D64"], "the C++ space did not reach the binding"


@pytest.mark.parametrize("dtype", _DTYPES, ids=["float64", "float32"])
def test_the_length_gate_turns_over_near_the_space_tolerance(
    cpp_backend: None,
    dtype: npt.DTypeLike,
) -> None:
    """A span off by a quarter of the tolerance is the same length; off by four times is not.

    The length gate compares ``abs(span - reference)`` against the space's tolerance,
    and nothing else in this file puts a span anywhere near it: the fixed table is
    built from whole units and the sweep jitters by a third of one, so the gate is
    only ever exercised far from the threshold it is written in terms of.

    The two deltas **bracket** the threshold within a factor of four rather than
    pinning which side of it the exact value falls on. That is deliberate: near the
    last ulp the two backends' arithmetic is a faithfulness question
    ``cpp/include/pantr/bspline/knots.hpp`` discusses at length, and a test there would
    pin which of two defensible answers wins rather than that the two agree.

    The tolerance is read off a space with equal spans rather than written down, so the
    deltas are derived and would move with the storage format and the knot scale.
    """

    def build(delta: float) -> BsplineSpace1D:
        """One space whose third span is longer than the rest by ``delta``."""
        breaks = [0.0, 1.0, 2.0, 3.0 + delta, 4.0 + delta, 5.0 + delta, 6.0 + delta]
        knots = [breaks[0]] * 2 + breaks + [breaks[-1]] * 2
        return BsplineSpace1D(np.asarray(knots, dtype=dtype), 2)

    with use_backend(Backend.CPP):
        tol = build(0.0).tolerance
        inside = build(tol / 4.0)
        outside = build(4.0 * tol)
        inside_flags = inside.get_cardinal_intervals()
        outside_flags = outside.get_cardinal_intervals()

    # Read off the vector: the odd span sits in the window of intervals 1, 2 and 3,
    # interval 4's window is clear of it, and the clamped ends take 0 and 5.
    assert inside_flags.tolist() == [False, True, True, True, True, False]
    assert outside_flags.tolist() == [False, False, False, False, True, False]

    np.testing.assert_array_equal(inside_flags, _oracle(inside))
    np.testing.assert_array_equal(outside_flags, _oracle(outside))


@pytest.mark.parametrize("dtype", _DTYPES, ids=["float64", "float32"])
def test_a_span_off_by_exactly_the_tolerance_is_still_the_same_length(
    cpp_backend: None,
    dtype: npt.DTypeLike,
) -> None:
    """A window difference of exactly ``tol`` does not disqualify an interval.

    The gate is written ``abs(span - reference) > tol`` in C++ against
    ``abs(...) <= tol`` in the oracle, which are exact negations -- so the **tie**
    is the one comparison whose two sides are told apart by the boundary operator
    alone. :func:`test_the_length_gate_turns_over_near_the_space_tolerance`
    deliberately brackets the threshold within a factor of four instead of sitting on
    it, because near the last ulp the two backends' arithmetic is a faithfulness
    question ``cpp/include/pantr/bspline/knots.hpp`` argues at length. That reasoning
    covers *arithmetic noise* and does not cover a **swapped operator**, which is
    deterministic; with ``>`` turned into ``>=`` every case in this file, the sweep
    included, answers exactly as before.

    So the tie is constructed rather than stumbled on, and it is exact rather than
    approximate. The ends are left alone and one interior breakpoint moves, so the
    knot scale -- and therefore ``tol`` -- is the same before and after, which the
    first assertion below checks rather than assumes. With a domain of ``[0, 8]``,
    ``tol`` is ``8 * eps * 8``, a power of two at both storage formats, and
    ``3 + tol`` and the spans ``1 ± tol`` are each exactly representable; both
    differences are exact by Sterbenz, since the operands are within a factor of two
    of one another. Nothing here is build-dependent.

    The expected flags are read off the vector rather than run: the spans are
    ``1, 1, 1+t, 1-t, 1, 1, 1, 1`` and the window at degree 2 is the interval's own
    span with one either side. Intervals 0 and 7 lose to the clamped ends. Interval 2
    sees ``1-t`` against a reference of ``1+t``, and interval 3 sees ``1+t`` against
    ``1-t``: a difference of ``2 * tol``, outside the gate on any reading. Intervals
    1 and 4 each see one neighbour off by exactly ``tol`` and are the two the operator
    decides -- ``True`` here, and ``False`` if the comparison were ``>=``.
    """

    def build(delta: float) -> BsplineSpace1D:
        """A clamped quadratic space over [0, 8] with one interior break moved by ``delta``."""
        breaks = [0.0, 1.0, 2.0, 3.0 + delta, 4.0, 5.0, 6.0, 7.0, 8.0]
        knots = [breaks[0]] * 2 + breaks + [breaks[-1]] * 2
        return BsplineSpace1D(np.asarray(knots, dtype=dtype), 2)

    with use_backend(Backend.CPP):
        tol = build(0.0).tolerance
        space = build(tol)
        flags = space.get_cardinal_intervals()

    assert space.tolerance == tol, "moving an interior break must not move the knot scale"

    spans = np.diff(space.knots)
    gaps = {abs(float(a - b)) for a in spans for b in spans}
    assert tol in gaps, "the construction must put a window difference exactly on the threshold"

    assert flags.tolist() == [False, True, False, False, True, True, True, False]
    np.testing.assert_array_equal(flags, _oracle(space), err_msg=f"exact tie: {_WHY}")


def _draw(rng: np.random.Generator) -> _Case:
    """Draw one non-periodic space whose flags are worth comparing.

    The breakpoints are drawn on a grid and then jittered by less than a third of
    the grid step, so that consecutive knots stay far apart relative to the
    tolerance -- a draw that collapsed two knots would be testing the class scan
    rather than this one -- while the spans still differ enough for the length gate
    to separate them. A share of the draws keeps the grid untouched, because a mesh
    with *equal* spans is the one the gate answers ``True`` for and a purely jittered
    sweep would never produce one.

    Args:
        rng (np.random.Generator): The source of the draw.

    Returns:
        _Case: The drawn case, always non-periodic and always snapped.
    """
    degree = int(rng.integers(0, 6))
    num_breaks = int(rng.integers(2, 12))
    step = 1.0
    breaks = np.arange(num_breaks, dtype=np.float64) * step
    # A share of the draws keeps the grid untouched; see the docstring.
    if rng.random() < 0.7:
        breaks[1:-1] += rng.uniform(-step / 3.0, step / 3.0, size=num_breaks - 2)

    clamped = bool(rng.random() < 0.6)
    knots: list[float] = []
    if clamped:
        knots += [float(breaks[0])] * degree
    else:
        # An unclamped vector needs `degree` knots outside the domain at each end.
        knots += [float(breaks[0]) - step * (degree - k) for k in range(degree)]
    for index, value in enumerate(breaks):
        interior = 0 < index < num_breaks - 1
        multiplicity = int(rng.integers(1, degree + 1)) if interior and degree > 0 else 1
        knots += [float(value)] * multiplicity
    if clamped:
        knots += [float(breaks[-1])] * degree
    else:
        knots += [float(breaks[-1]) + step * (k + 1) for k in range(degree)]

    return _Case("drawn", knots, degree, False, True)


@pytest.mark.slow
@pytest.mark.parametrize("dtype", _DTYPES, ids=["float64", "float32"])
def test_the_scan_agrees_over_a_sweep_ten_times_the_shipped_one(
    cpp_backend: None,
    dtype: npt.DTypeLike,
) -> None:
    """The agreement holds over ten times as many spaces as the table above ships.

    There is no bound to verify -- the quantity is a boolean and the comparison is
    exact -- so what this widens is the input coverage: degrees 0 to 5, clamped and
    unclamped ends, interior multiplicities from 1 to the degree, and meshes that are
    uniform or jittered off it.

    **What it does not draw:** a periodic space, because a valid periodic knot vector
    is not a random draw away from an arbitrary one, and the table above carries two;
    and a mesh fine enough to make the tolerance decide, which would be measuring the
    class scan rather than this one.
    """
    rng = np.random.default_rng(491)
    degrees: set[int] = set()
    cardinal_seen = 0
    for _ in range(10 * len(_CASES)):
        case = _draw(rng)
        with use_backend(Backend.CPP):
            space = _build(case, dtype)
            flags = space.get_cardinal_intervals()
        np.testing.assert_array_equal(
            flags,
            _oracle(space),
            err_msg=(
                f"degree {case.degree} at {np.dtype(dtype).name} over knots {case.knots}: {_WHY}"
            ),
        )
        degrees.add(case.degree)
        cardinal_seen += int(bool(flags.any()))

    # What the docstring above claims the sweep covers, asserted rather than described.
    # A count of the iterations would say nothing -- the loop's own bound already fixes
    # it -- whereas both of these are properties of `_draw` and of the answers, and both
    # would go quiet if the draw narrowed or if every drawn space answered all-False,
    # which would leave this comparing two functions that both return nothing.
    assert degrees == set(range(6)), f"the sweep drew only degrees {sorted(degrees)}"
    assert cardinal_seen > 0, "no drawn space had a cardinal interval; the sweep proves nothing"
