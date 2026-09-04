"""Parity of the `Bspline` value type: the state, the rejections, and the wire format.

What this file compares is a *value*, not a computation. `pantr.bspline.Bspline`
holds a space, a control net and a flag, and answers five questions about them;
between the Python oracle and the C++ port there is no arithmetic on a coefficient
at all, only a copy and one integer subtraction. So every claim here is bitwise or
exact, and a tolerance anywhere in this file would be hiding a transcription error
rather than allowing for rounding. The operations that *do* arithmetic --
evaluation, the degree and knot operations, the product -- are the subject of the
existing `tests/test_bspline_*.py` files, which build their B-splines through this
same type and therefore run under both backends already.

**Each field's claim carries its own argument**, in ``FIELDS`` below, rather than one
decision applied in bulk: the six quantities are exact for six different reasons, and
a shared ``why`` would be a single sentence quoted in six failure messages it does not
fit.

Five things are checked that a field-by-field state comparison does not reach:

- **The rejections.** Two of the four are the wrapper's and are therefore common mode
  by construction, which this file says out loud rather than counting them as
  evidence; the rank refusal is the one a cross-backend comparison actually decides,
  and it is compared at the implementation level where the C++ text is really under
  test.
- **The copy at construction, and the read-only view on the way out.** The C++ value
  does not alias the caller's array at either end, which the oracle does. This is the
  one place the two backends differ on purpose: a write through either end
  desynchronises both of the field's derived memos with nothing raising, which
  `design/bspline_ownership_lifetime.md` records as a defect in today's Python.
- **The reseat contract of the two space-changing in-place methods.**
  ``before = b.space; b.reverse(0, in_place=True)`` must leave ``before`` holding the
  *old* space and ``b.space`` a different object. That is the contract
  `design/bspline_ownership_lifetime.md` reason 3 was measured on, and two of the
  three storage shapes it tabulates break it silently -- one of them by making the
  escaped space start reporting the new value.
- **The derived block, discarded wholesale.** ``design/bspline_derived_caches.md``
  asks this front to turn the oracle's three separate cache-invalidation sites into
  one assignment. The test for that is not "the caches are cold after a mutation" but
  "*both* are, after each of the three mutators", because the defect the block
  removes is one surviving the other.
- **The wire format.** ``__reduce__`` pickles by the constructor's arguments, so a
  pickle written under one backend loads under the other. Without that the backend
  switch would silently become a data-format switch, and a C++ handle is not
  picklable in any case.

## The independent check, and why it is not a mirror

`design/backend_parity.md`'s first rule is that parity says the two backends agree
and not that either is right, so a shared error is invisible to every parity test
here. The independent oracle for a value type is not a rational reimplementation of
an algorithm -- there is no algorithm -- it is a **closed form for the layout**:

- ``test_the_stored_layout_matches_its_closed_form`` builds the control net from an
  explicit function of the multi-index, then reads it back *through the field* at
  each multi-index and compares against that function evaluated again. The expected
  value comes from a formula written in this file; nothing in the comparison consults
  either implementation. A transposition, a wrong stride, a dropped component axis or
  a narrowing cast all move it, and the admissible deviation is zero because the
  formula's values are exactly representable in both storage formats -- dyadic
  fractions times small integers -- so no rounding can enter on either side.
- ``test_the_counts_match_their_closed_forms`` derives the per-direction basis counts
  from the B-spline definition (``len(knots) - degree - 1``, less the periodic
  correction) and the rank from the buffer size, in exact integer arithmetic. Those
  are the numbers the C++ constructor checks the net's extents against, so taking
  them from ``space.num_basis`` instead would have been a mirror. Every case in
  ``CASES`` carries them as literals for the same reason.
"""

from __future__ import annotations

import copy
import gc
import math
import pickle
from collections.abc import Callable
from typing import Any, Final, NamedTuple

import numpy as np
import numpy.typing as npt
import pytest

from pantr._backend import Backend, use_backend
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
from pantr.bspline._bspline import _BsplinePython
from tests._parity_harness import (
    Field,
    assert_accuracy,
    assert_object_parity,
    bitwise_parity,
    derived_accuracy,
    exact_parity,
)

pytestmark = pytest.mark.usefixtures("cpp_backend")

DTYPES: Final = (np.float64, np.float32)
"""Both storage formats. A B-spline stores `float32` too, so the C++ side has two classes."""

FIELDS: Final = (
    Field(
        "control_points",
        bitwise_parity(
            why=(
                "the value type stores the coefficients it is handed and performs no "
                "arithmetic on them, so the two backends can only differ by a transcription "
                "error -- a wrong stride, a transposed axis, a narrowing cast, a byte lost "
                "to the shape. Every one of those is visible bitwise and invisible under any "
                "tolerance wide enough to be called a rounding budget. It is also the only "
                "claim here that would notice a signed zero normalized, a NaN payload "
                "rewritten or a float32 subnormal flushed, none of which a value comparison "
                "can see at all."
            )
        ),
    ),
    Field(
        "is_rational",
        exact_parity(
            why=(
                "the flag is stored and returned; nothing transforms it. It is named as its "
                "own field rather than left implicit because it is what `rank` subtracts, so "
                "a flag lost in the round trip shows up here as well as one field away, and "
                "reading only `rank` could not say which of the two moved."
            )
        ),
    ),
    Field(
        "dim",
        exact_parity(
            why=(
                "the number of parametric directions, forwarded from the space by both "
                "implementations. An integer count, so exactness is the only claim available; "
                "a difference means the field is holding a different space than the one it "
                "was given, which no comparison of the coefficients would reveal on a net "
                "whose flat size happens to match."
            )
        ),
    ),
    Field(
        "degree",
        exact_parity(
            why=(
                "one integer per direction, forwarded from the space in axis order. The order "
                "is the load-bearing part rather than the values: nothing about a degree "
                "reveals a transposition on a field whose directions happen to agree, which "
                "is why every case in `CASES` has directions that differ in it."
            )
        ),
    ),
    Field(
        "rank",
        exact_parity(
            why=(
                "the stored component count less the weight column, one subtraction in exact "
                "integer arithmetic on both sides. A difference is an off-by-one on the "
                "component axis, or the weight counted out of the storage instead of out of "
                "the rank -- and it is the one field that folds the flag against the shape, "
                "so it catches a round trip that preserved the byte count and the "
                "per-direction counts while moving which axis carries the components."
            )
        ),
    ),
    Field(
        "space.num_basis",
        exact_parity(
            why=(
                "the per-direction basis counts the control net is laid out on. Exact integer "
                "counts, and the field's business rather than the space's: this is the tuple "
                "the C++ constructor checks the net's extents against, so a disagreement here "
                "means the two backends laid one buffer out on two different grids. The "
                "space's own parity is asserted in tests/parity/test_bspline_space_nd.py; "
                "what is claimed here is that the field reached the same one."
            )
        ),
    ),
    Field(
        "dtype",
        exact_parity(
            why=(
                "the storage format a caller reads through `Bspline.dtype`, and on the C++ "
                "side it is carried by the class of the handle. A disagreement means "
                "`_impl_class` picked the wrong class, which would silently narrow or widen "
                "the geometry -- and for a coefficient representable in both formats, which "
                "is most of them, the bitwise claim above would not see it."
            )
        ),
    ),
)
"""Every piece of a B-spline's state, one field each, with its own argument.

`degree` and `space.num_basis` are homogeneous tuples of ints, which are one quantity
each and stay one field; the harness refuses a value mixing element kinds, and neither
is one.
"""


class _Case(NamedTuple):
    """One B-spline shape to build under both backends, with its hand-derived counts.

    The counts are **literals**, derived from the B-spline definition rather than read
    off a constructed space, so that the layout and rank checks cannot take their
    expected values from the thing under test. ``test_the_counts_match_their_closed_forms``
    is what holds them to the definition; ``test_the_case_table_earns_its_keep`` is what
    holds them to being asymmetric.

    Attributes:
        knots (tuple[tuple[float, ...], ...]): One knot vector per direction.
        degrees (tuple[int, ...]): One degree per direction.
        periodic (tuple[bool, ...]): Whether each direction wraps.
        num_basis (tuple[int, ...]): The basis count of each direction, hand-derived.
        components (int): The length of the stored component axis, weights included.
        label (str): A short name for the failure message.
    """

    knots: tuple[tuple[float, ...], ...]
    degrees: tuple[int, ...]
    periodic: tuple[bool, ...]
    num_basis: tuple[int, ...]
    components: int
    label: str


_CLAMPED_QUADRATIC: Final = (0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0)
"""Three intervals, degree 2, five basis functions, on ``[0, 3]``."""

_CLAMPED_LINEAR: Final = (10.0, 10.0, 11.0, 12.0, 12.0)
"""Two intervals, degree 1, three basis functions, on ``[10, 12]`` -- a different scale."""

_SINGLE_SPAN: Final = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
"""One interval, degree 2, three basis functions: the Bézier-like case."""

_ASYMMETRIC: Final = (0.0, 0.0, 0.0, 1.0, 3.0, 3.0, 3.0)
"""Two intervals, degree 2, four basis functions, on ``[0, 3]``.

The one knot vector here whose reflection ``a + b - knots[::-1]`` is a *different*
vector: its single interior knot sits at 1 and reflects to 2. Every other vector in
this file is symmetric about its own domain midpoint, which makes a reversal
unobservable in the knots -- ``_CLAMPED_QUADRATIC``'s interior pair ``{1, 2}`` on
``[0, 3]`` reflects onto itself. ``test_a_space_changing_mutator_reseats_the_space``
asserts that asymmetry of this constant before relying on it.
"""

_CLAMPED_LINEAR_4: Final = (10.0, 10.0, 11.0, 12.0, 13.0, 13.0)
"""Three intervals, degree 1, four basis functions, on ``[10, 13]``."""

_CLAMPED_CUBIC: Final = (
    100.0,
    100.0,
    100.0,
    100.0,
    101.0,
    102.0,
    103.0,
    104.0,
    104.0,
    104.0,
    104.0,
)
"""Four intervals, degree 3, seven basis functions, on ``[100, 104]`` -- a third scale."""

_UNIFORM: Final = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
"""A uniform vector, usable clamped or periodic.

Read non-periodically it is degree 2 with ``9 - 2 - 1 = 6`` basis functions; read
periodically the count drops by ``degree - 1 = 1`` more, to 4, because the first
in-domain knot is simple. Both readings appear in ``CASES``, which is what makes the
periodic correction visible rather than assumed.
"""

CASES: Final = (
    _Case((_SINGLE_SPAN,), (2,), (False,), (3,), 1, "a scalar Bezier-like curve"),
    _Case((_CLAMPED_QUADRATIC,), (2,), (False,), (5,), 3, "a space curve"),
    _Case((_CLAMPED_QUADRATIC,), (2,), (False,), (5,), 4, "a rational space curve"),
    _Case(
        (_CLAMPED_QUADRATIC, _CLAMPED_LINEAR),
        (2, 1),
        (False, False),
        (5, 3),
        2,
        "an asymmetric surface",
    ),
    _Case(
        (_CLAMPED_QUADRATIC, _CLAMPED_LINEAR, _CLAMPED_CUBIC),
        (2, 1, 3),
        (False, False, False),
        (5, 3, 7),
        4,
        "an asymmetric volume",
    ),
    _Case((_UNIFORM,), (2,), (True,), (4,), 2, "a periodic curve"),
    _Case(
        (_UNIFORM, _CLAMPED_LINEAR),
        (2, 1),
        (True, False),
        (4, 3),
        2,
        "one periodic direction and one clamped",
    ),
    _Case((_UNIFORM,), (2,), (False,), (6,), 1, "the same vector read non-periodically"),
)
"""The shapes both backends must agree on.

Each multi-direction case differs between its directions in the degree, the basis
count, the domain and the domain's width at once, and its component count is none of
its basis counts, so no permutation of a net's shape is another admissible shape for
its space and a transposition cannot pass. ``test_the_case_table_earns_its_keep``
asserts every one of those rather than leaving this paragraph to be believed.

The periodic pair -- the same knot vector read periodically and not -- is what makes
the periodic basis-count correction visible instead of taken on trust.
"""


def _make_space(case: _Case) -> BsplineSpace:
    """Build ``case``'s space under whichever backend is active.

    Args:
        case (_Case): The shape to build.

    Returns:
        ~pantr.bspline.BsplineSpace: The space, in the active backend's
        implementation.
    """
    return BsplineSpace(
        [
            BsplineSpace1D(np.asarray(knots, dtype=np.float64), degree, periodic=periodic)
            for knots, degree, periodic in zip(case.knots, case.degrees, case.periodic, strict=True)
        ]
    )


def _make_space_at(case: _Case, dtype: npt.DTypeLike) -> BsplineSpace:
    """Build ``case``'s space at a given storage format, under the active backend.

    Args:
        case (_Case): The shape to build.
        dtype (npt.DTypeLike): The knot storage format.

    Returns:
        ~pantr.bspline.BsplineSpace: The space.
    """
    return BsplineSpace(
        [
            BsplineSpace1D(np.asarray(knots, dtype=dtype), degree, periodic=periodic)
            for knots, degree, periodic in zip(case.knots, case.degrees, case.periodic, strict=True)
        ]
    )


def _coefficient(index: tuple[int, ...]) -> float:
    """The coefficient a case's control net holds at one multi-index.

    The closed form the independent check compares against, and the reason the check
    is not a mirror: it is a function of the *index*, written here, and neither
    implementation is consulted to evaluate it.

    Exactly representable in both storage formats at every index a case here reaches,
    which is what lets the accuracy bound be zero rather than a rounding budget: the
    factors are a power of two, a dyadic fraction with denominator 8, and a sign. So
    ``np.asarray(..., dtype=np.float32)`` rounds nothing and the same literal value is
    stored by both backends.

    The exponent moves with the *first* parametric index and the mantissa with the
    rest, so a transposition changes the magnitude by decades rather than by a bit --
    which is what makes a failure legible instead of a last-digit puzzle.

    Args:
        index (tuple[int, ...]): The multi-index, ``(*parametric, component)``.

    Returns:
        float: The coefficient.
    """
    exponent = 3 * index[0] - 12
    mantissa = 1.0
    for position, value in enumerate(index[1:], start=1):
        mantissa += value / float(2**position)
    return (-1.0 if index[-1] % 2 else 1.0) * mantissa * float(2.0**exponent)


def _control_net(case: _Case, dtype: npt.DTypeLike) -> npt.NDArray[np.float32 | np.float64]:
    """Build ``case``'s control net from :func:`_coefficient`, index by index.

    Filled through an explicit loop over ``np.ndindex`` rather than by a vectorized
    expression, so that the array's own layout is numpy's doing and the check reading
    it back through the field is comparing two independent routes to one element.

    Args:
        case (_Case): The shape to build.
        dtype (npt.DTypeLike): The storage format.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The net, C-contiguous, of shape
        ``(*case.num_basis, case.components)``.
    """
    shape = (*case.num_basis, case.components)
    net = np.zeros(shape, dtype=dtype)
    for index in np.ndindex(*shape):
        net[index] = _coefficient(index)
    return net


def _both(
    case: _Case, control_points: npt.NDArray[np.float32 | np.float64], *, is_rational: bool
) -> tuple[Bspline, Bspline]:
    """Build the same B-spline under each backend.

    The space is rebuilt per backend rather than shared, and that is forced rather
    than tidy: each implementation holds the space's own implementation, so a space
    built under one backend is refused by the other.

    Args:
        case (_Case): The shape to build.
        control_points (npt.NDArray[np.float32 | np.float64]): The control net.
        is_rational (bool): Whether the last stored component is a weight.

    Returns:
        tuple[Bspline, Bspline]: ``(py, cpp)``, in the order
        :func:`assert_object_parity` names its arguments.
    """
    dtype = control_points.dtype
    with use_backend(Backend.PYTHON):
        py = Bspline(_make_space_at(case, dtype), control_points, is_rational)
    with use_backend(Backend.CPP):
        cpp = Bspline(_make_space_at(case, dtype), control_points, is_rational)
    return py, cpp


def _cpp_class(dtype: npt.DTypeLike) -> Any:
    """The bound C++ class for one storage format.

    Args:
        dtype (npt.DTypeLike): `float32` or `float64`.

    Returns:
        Any: `pantr._pantr_cpp.Bspline32` or `Bspline64`.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415

    return _pantr_cpp.Bspline32 if np.dtype(dtype) == np.float32 else _pantr_cpp.Bspline64


def _expected_num_basis(case: _Case) -> tuple[int, ...]:
    """``case``'s basis counts from the B-spline definition, not from a space.

    For a non-periodic direction a knot vector of ``m`` entries at degree ``p``
    carries ``m - p - 1`` basis functions. A periodic direction drops the regularity
    at the domain's start as well, which is ``p`` less the multiplicity of the first
    in-domain knot, plus the wrap itself -- so ``m - p - 1 - (p - s) - 1`` where ``s``
    is that multiplicity. Every knot vector in ``CASES`` is strictly increasing where
    it is read periodically, so ``s`` is 1 there.

    Args:
        case (_Case): The shape.

    Returns:
        tuple[int, ...]: One count per direction.
    """
    counts = []
    for knots, degree, periodic in zip(case.knots, case.degrees, case.periodic, strict=True):
        count = len(knots) - degree - 1
        if periodic:
            multiplicity_of_first_in_domain = sum(1 for knot in knots if knot == knots[degree])
            count -= degree - multiplicity_of_first_in_domain + 1
        counts.append(count)
    return tuple(counts)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
@pytest.mark.parametrize("is_rational", [False, True])
def test_the_two_backends_hold_the_same_value(
    case: _Case, dtype: npt.DTypeLike, is_rational: bool
) -> None:
    """Every piece of state agrees, at both storage formats.

    What this catches: a stride or shape mistake in the flat-buffer round trip, a
    narrowing cast in the wrong-dtype class, an off-by-one in `rank`, the weight
    column counted into the rank or out of the storage, and the field reaching a
    different space than the one it was handed.
    """
    if is_rational and case.components < 2:
        pytest.skip("a rational B-spline needs a weight and at least one coordinate")
    control_points = _control_net(case, dtype)
    py, cpp = _both(case, control_points, is_rational=is_rational)
    assert_object_parity(
        py=py,
        cpp=cpp,
        fields=FIELDS,
        context=f"Bspline({case.label}, {np.dtype(dtype).name}, rational={is_rational})",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
def test_the_stored_layout_matches_its_closed_form(case: _Case, dtype: npt.DTypeLike) -> None:
    """Every coefficient reads back, through the field, as the formula that placed it.

    The independent check `design/backend_parity.md` requires, and the reason it is
    not a mirror is in the module docstring: the expected value is
    :func:`_coefficient` evaluated at the multi-index, written in this file, and
    nothing in the comparison consults either implementation.

    The bound is zero, which is a claim about *transport* rather than about rounding:
    :func:`_coefficient` returns a dyadic value exactly representable in both formats
    at every index reached here, so neither the construction of the input nor the
    storage of it can round. A non-zero deviation is therefore a defect and never
    noise, and there is no regime in which the bound would have to be loosened.

    What this catches that the parity comparison cannot: a layout error made *the
    same way* by both backends -- the shared-error case parity is blind to by
    construction. The exponent moves with the first parametric index, so a
    transposition shows as decades rather than as a last digit.
    """
    net = _control_net(case, dtype)
    shape = (*case.num_basis, case.components)
    with use_backend(Backend.CPP):
        field = Bspline(_make_space_at(case, dtype), net)

    stored = np.asarray(field.control_points, dtype=np.float64)
    assert stored.shape == shape, f"{case.label}: the field reshaped the net to {stored.shape}"
    exact = np.zeros(shape, dtype=np.float64)
    for index in np.ndindex(*shape):
        exact[index] = _coefficient(index)

    assert_accuracy(
        stored,
        exact,
        derived_accuracy(
            bound=np.zeros((), dtype=np.float64),
            why=(
                "the type transports its coefficients rather than computing them, and "
                "`_coefficient` returns a dyadic value -- a sign, a mantissa with "
                "denominator a power of two, and a power-of-two scale -- exactly "
                "representable in float32 and float64 alike. So no rounding occurs on "
                "either the way in or the way out, and the admissible deviation is "
                "exactly zero rather than a budget."
            ),
        ),
        context=f"stored layout of {case.label} at {np.dtype(dtype).name}",
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
@pytest.mark.parametrize("is_rational", [False, True])
def test_the_counts_match_their_closed_forms(case: _Case, is_rational: bool) -> None:
    """The basis counts and the rank follow the B-spline definition, on both backends.

    The integer half of the independent check. ``space.num_basis`` is compared against
    ``len(knots) - degree - 1`` -- less the periodic correction -- computed in
    :func:`_expected_num_basis` from the case's own knot vectors, and ``rank`` against
    ``size / prod(num_basis) - is_rational`` in exact integer arithmetic. Neither
    reads a constructed space, which is what stops this being a mirror: those counts
    are exactly what the C++ constructor checks the net's extents against.

    It is also the guard on ``CASES`` itself. Every case states its basis counts as
    literals so that the layout check has an oracle, and a literal nobody checks is a
    comment: if a knot vector is ever edited without its count, this fails and the
    layout check does not.
    """
    if is_rational and case.components < 2:
        pytest.skip("a rational B-spline needs a weight and at least one coordinate")
    expected_basis = _expected_num_basis(case)
    assert case.num_basis == expected_basis, (
        f"{case.label}: the table says {case.num_basis} and the definition gives {expected_basis}"
    )

    net = _control_net(case, np.float64)
    expected_rank = net.size // math.prod(expected_basis) - (1 if is_rational else 0)

    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            field = Bspline(_make_space(case), net, is_rational)
        assert field.space.num_basis == expected_basis, backend.name
        assert field.rank == expected_rank, backend.name
        assert field.dim == len(expected_basis), backend.name


def test_the_case_table_earns_its_keep() -> None:
    """Every multi-direction case really does differ between its directions.

    A test of the case set, not of the library, and the file's own vacuity guard: a
    tensor-product type compared over a table whose directions agree cannot
    distinguish a transposition, an off-by-one in an axis index, or a reduction that
    returns its first argument. So the claim that the table is asymmetric is asserted
    rather than described.

    The two single-direction cases are exempt by construction and are not counted; the
    periodic pair is what covers the axis a single case cannot.

    Gated on the extension by this module's mark even though it needs none, and that
    is consistent rather than an oversight: what it guards -- the layout and count
    checks that take their expected values from this table -- is gated too, so the
    table is never relied on in a configuration where this has not run.
    """
    multi = [case for case in CASES if len(case.degrees) > 1]
    assert len(multi) >= 2, "no multi-direction case is left to compare"
    for case in multi:
        assert len(set(case.degrees)) == len(case.degrees), f"{case.label}: degrees repeat"
        assert len(set(case.num_basis)) == len(case.num_basis), (
            f"{case.label}: basis counts repeat, so a transposed net would pass"
        )
        assert case.components not in case.num_basis, (
            f"{case.label}: the component count equals a basis count, so a permutation "
            f"of the net's shape is another admissible shape for this space"
        )
        # The domains differ in location and in width, which is what makes the
        # per-direction domain and tolerance reductions observable. Derived from the
        # definition -- `(knots[degree], knots[-degree - 1])` -- rather than read off
        # a space, for the reason `_expected_num_basis` is.
        domains = {
            (knots[degree], knots[len(knots) - degree - 1])
            for knots, degree in zip(case.knots, case.degrees, strict=True)
        }
        assert len(domains) == len(case.degrees), f"{case.label}: two directions share a domain"
        widths = {hi - lo for lo, hi in domains}
        assert len(widths) == len(case.degrees), f"{case.label}: two directions share a width"
    # And the periodic axis is exercised in both readings of one vector, which is what
    # makes the basis-count correction observable rather than assumed.
    periodic_labels = {case.label for case in CASES if any(case.periodic)}
    assert periodic_labels, "no periodic direction is exercised"
    same_vector = [case for case in CASES if case.knots == (_UNIFORM,)]
    assert {case.num_basis for case in same_vector} == {(4,), (6,)}, (
        "the periodic and non-periodic readings of one knot vector must give different "
        "counts, or the periodic correction is untested"
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_rational", [False, True])
def test_the_ieee_menagerie_transits_unchanged(dtype: npt.DTypeLike, is_rational: bool) -> None:
    """NaN, both zeros, a subnormal and both infinities survive the copy.

    What this catches: the whole premise of this file is that the two backends only
    copy, and a copy is exactly where these values are cheap to test and expensive to
    lose. A caster round-tripping through a wider intermediate, a narrowing cast, or a
    memcpy replaced by element-wise assignment through a different type all show
    first on a NaN payload, on a ``-0.0`` whose sign an addition drops, or on a
    ``float32`` subnormal flushed to zero -- and every one of those is invisible to a
    value comparison and visible bitwise.

    The finite specials go through the same ``FIELDS``, so ``dtype``, ``degree`` and
    ``rank`` are checked over them too: a B-spline of NaNs is not a special case in
    the type's eyes and must not become one.

    **The two infinities are asserted separately, and not by choice.**
    ``assert_parity`` computes a diagnostic ``|actual - reference|`` before it reports
    a bitwise result, and ``inf - inf`` raises the IEEE invalid flag, which numpy
    reports as ``RuntimeWarning: invalid value encountered in subtract`` and this
    suite turns into an error. So a bitwise claim cannot presently be made about an
    array holding an infinity, and the harness is where that has to be fixed rather
    than here. NaN is unaffected: quiet-NaN propagation raises no flag.
    """
    case = _Case((_SINGLE_SPAN,), (2,), (False,), (3,), 2, "the menagerie")
    finfo = np.finfo(np.dtype(dtype))
    finite_specials = np.asarray(
        [
            [0.0, np.nan],
            [-0.0, float(finfo.smallest_subnormal)],
            [float(finfo.max), float(finfo.tiny)],
        ],
        dtype=dtype,
    )
    py, cpp = _both(case, finite_specials, is_rational=is_rational)
    assert_object_parity(
        py=py,
        cpp=cpp,
        fields=FIELDS,
        context=f"IEEE menagerie ({np.dtype(dtype).name}, rational={is_rational})",
    )
    # Asserted on the bytes rather than through the harness, because that is the
    # specific loss this case exists for: `assert_object_parity` would pass if BOTH
    # backends had flushed the subnormal or normalized the signed zero the same way,
    # and it is the transit that is under test, not the agreement.
    assert cpp.control_points.tobytes() == finite_specials.tobytes()

    with_infinities = np.asarray([[np.inf, -np.inf], [1.0, -0.0], [2.0, 3.0]], dtype=dtype)
    _, cpp_inf = _both(case, with_infinities, is_rational=is_rational)
    assert cpp_inf.control_points.tobytes() == with_infinities.tobytes()


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_non_contiguous_control_net_reaches_both_backends_alike(dtype: npt.DTypeLike) -> None:
    """A Fortran-ordered or strided net builds the same B-spline under either backend.

    What this catches: the ``np.ascontiguousarray`` step in ``_new_impl``. The C++
    binding refuses a non-contiguous array outright rather than copying it, so
    without that step the public constructor would raise a ``TypeError`` about C++
    argument types under one backend and succeed under the other. Nothing else
    exercises the branch: ``test_bspline_binding_contract.py`` tests the raw
    binding's refusals, which is the opposite path.

    A B-spline reshapes its net before storing it, so the interesting input is one
    whose *reshaped* form is still non-contiguous. A Fortran-ordered array of exactly
    the stored shape is that: ``reshape`` returns it unchanged.
    """
    case = _Case((_CLAMPED_QUADRATIC, _CLAMPED_LINEAR), (2, 1), (False, False), (5, 3), 2, "f")
    fortran = np.asfortranarray(np.arange(30.0, dtype=dtype).reshape(5, 3, 2))
    strided = np.arange(60.0, dtype=dtype).reshape(10, 3, 2)[::2]
    for control_points, how in ((fortran, "Fortran-ordered"), (strided, "strided")):
        assert not control_points.flags.c_contiguous, how
        py, cpp = _both(case, control_points, is_rational=False)
        assert_object_parity(
            py=py,
            cpp=cpp,
            fields=FIELDS,
            context=f"{how} control net ({np.dtype(dtype).name})",
        )


def _message_of(build: Any) -> str:
    """The text and type of the exception ``build`` raises.

    Args:
        build (Any): A no-argument call expected to raise.

    Returns:
        str: ``"<ExceptionType>: <message>"``, or a marker saying what happened
        instead, so that the caller's assertion is the one that reports.
    """
    try:
        build()
    except (ValueError, IndexError, TypeError) as error:
        return f"{type(error).__name__}: {error}"
    return "<did not raise>"


_MALFORMED: Final = (
    (
        lambda: Bspline(_make_space(CASES[0]), np.zeros(4, dtype=np.float64)),
        "a coefficient count that is not a multiple of the basis count",
        "wrapper",
    ),
    (
        lambda: Bspline(_make_space(CASES[0]), np.zeros((3, 1), dtype=np.float32)),
        "control points of the wrong dtype",
        "wrapper",
    ),
    (
        lambda: Bspline(_make_space(CASES[0]), np.zeros((3, 1), dtype=np.float64), True),
        "a rational field with a weight and nothing else",
        "implementation",
    ),
    (
        lambda: Bspline(_make_space(CASES[0]), np.zeros(0, dtype=np.float64)),
        "an empty control net over a non-empty space",
        "wrapper",
    ),
    (
        lambda: Bspline(BsplineSpace([]), np.zeros(3, dtype=np.float64)),
        "a space with no directions",
        "wrapper",
    ),
)
"""Every construction both backends must refuse, and which layer owns the refusal.

The third column is the honest part. Only the ``implementation`` entry is decided by
this comparison: the other four are raised by the wrapper, which is one piece of
shared code, so the two backends agree on them by construction and their agreement is
not evidence about the port. They are swept anyway because the *reachability* is what
is claimed -- a C++ constructor that accepted one of them would make the wrapper's
check the only thing standing between a caller and a malformed field, and the count
check is duplicated in the C++ type precisely so that it is not.

``cpp/tests/test_bspline_type.cpp`` is the half that pins the C++ texts against
literals, and it covers three refusals no Python caller can reach at all: a null
space handle, a net whose parametric extents are not the space's basis counts, and
the coefficient-count message of the flat constructor.

**And the ORDER of the wrapper's two refusals is not decided here either**, for the
same reason: every entry violates one rule and so has one possible message. Measured
-- moving the dtype check ahead of the coefficient-count check left this whole file
green. ``tests/test_bspline.py::TestBsplineInit::test_initialization_refusal_order``
is what pins it, with control points that are bad in both ways at once.
"""


@pytest.mark.parametrize(("build", "what", "owner"), _MALFORMED)
def test_refusals_agree_verbatim(build: Any, what: str, owner: str) -> None:
    """Both backends refuse the same constructions with the same exception and text.

    What this catches, for the one entry it can: a reworded rank message on the C++
    side. A caller catching ``pytest.raises(ValueError, match=...)`` must not have to
    know which backend built the object, and the exception type alone does not carry
    that -- both sides raise ``ValueError`` here, so only the text can tell a
    reordered or rewritten check from a faithful one.
    """
    with use_backend(Backend.PYTHON):
        oracle = _message_of(build)
    with use_backend(Backend.CPP):
        ported = _message_of(build)
    assert oracle == ported, f"{what} ({owner}): oracle said {oracle!r}, C++ said {ported!r}"
    assert not oracle.startswith("<"), f"{what}: neither backend refused it"


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_implementations_agree_on_the_rank_refusal(dtype: npt.DTypeLike) -> None:
    """The two implementation classes report a rank-zero field identically.

    The one refusal both implementations own, compared where the C++ text is actually
    under test rather than through the wrapper that would have agreed anyway. Reached
    by handing each class the space's own implementation directly, which is what the
    wrapper does.

    Two ranks are checked and the second is the reason the C++ arithmetic is signed: a
    rational net with no coordinate components at all is rank ``-1``, and an unsigned
    subtraction would refuse it while reporting a number near ``2**64``. That case
    cannot be reached through the public constructor, because numpy's reshape refuses
    an empty buffer over a non-empty space first, so it has no entry in
    ``_MALFORMED``.
    """
    case = CASES[0]
    with use_backend(Backend.PYTHON):
        py_space = _make_space_at(case, dtype)
    with use_backend(Backend.CPP):
        cpp_space = _make_space_at(case, dtype)
    cls = _cpp_class(dtype)

    for components, expected_rank in ((1, 0), (0, -1)):
        net = np.zeros((3, components), dtype=dtype)
        oracle = _message_of(lambda: _BsplinePython(py_space._impl, net, True))  # noqa: B023
        ported = _message_of(lambda: cls(cpp_space._impl, net, True))  # noqa: B023
        assert oracle == ported, f"rank {expected_rank}: {oracle!r} against {ported!r}"
        assert oracle == (
            f"ValueError: The B-spline must have at least rank one. Got rank {expected_rank}"
        ), oracle


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_cpp_value_does_not_alias_the_array_it_was_built_from(dtype: npt.DTypeLike) -> None:
    """Mutating the constructor's argument afterwards does not move the B-spline.

    What this catches: the C++ constructor taking a view of the caller's buffer
    instead of copying, which would let a validated geometry change under its owner's
    feet -- and, worse than for a Bézier, would leave both of the field's derived
    memos describing a geometry it no longer holds.
    """
    case = _Case((_SINGLE_SPAN,), (2,), (False,), (3,), 2, "alias")
    control_points = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=dtype)
    with use_backend(Backend.CPP):
        field = Bspline(_make_space_at(case, dtype), control_points)

    before = field.control_points.copy()
    control_points[0, 0] = 99.0
    assert np.array_equal(field.control_points, before)
    assert not np.shares_memory(field.control_points, control_points)


@pytest.mark.parametrize("dtype", DTYPES)
def test_writing_through_control_points_is_refused(dtype: npt.DTypeLike) -> None:
    """The array handed out is read-only, and the B-spline is unchanged either way.

    What this catches: the way *out* of the same defect. A writeable view would let a
    caller edit a constructed geometry through the property, which is the half a
    criterion about construction alone leaves unpinned.
    """
    case = _Case((_SINGLE_SPAN,), (2,), (False,), (3,), 2, "read-only")
    with use_backend(Backend.CPP):
        field = Bspline(
            _make_space_at(case, dtype),
            np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=dtype),
        )

    handed_out = field.control_points
    assert not handed_out.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        handed_out[0, 0] = 99.0
    assert field.control_points[0, 0] == 0.0


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_view_outlives_the_bspline_it_came_from(dtype: npt.DTypeLike) -> None:
    """The array keeps the C++ storage alive after the handle is dropped.

    What this catches: a view returned without an owner. The values would be read
    from freed memory, which is a use-after-free that usually reads back correct and
    occasionally does not -- so this asserts the values rather than merely that
    nothing crashed.
    """
    case = _Case((_SINGLE_SPAN,), (2,), (False,), (3,), 2, "outlive")
    expected = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=dtype)
    with use_backend(Backend.CPP):
        handed_out = Bspline(_make_space_at(case, dtype), expected.copy()).control_points
    gc.collect()
    assert np.array_equal(handed_out, expected)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_rational", [False, True])
def test_a_bspline_survives_pickling_across_every_backend_pair(
    dtype: npt.DTypeLike, is_rational: bool
) -> None:
    """A pickle written under one backend loads under the other, at both dtypes.

    What this catches: a ``__reduce__`` that reaches the implementation rather than
    the constructor's arguments. The C++ handle is not picklable at all, and a
    payload that carried one would make ``PANTR_BACKEND`` a data-format switch -- a
    pickle written on one machine unreadable on another. ``copy.deepcopy`` goes
    through ``__reduce_ex__`` by the same route, so it is swept over the same pairs.

    The space is part of the payload and is checked as such: it goes out as its
    *wrapper*, so its own ``__reduce__`` runs and the knot vectors survive rather
    than the handle being smuggled through.
    """
    case = CASES[3]
    net = _control_net(case, dtype)
    for writer in (Backend.PYTHON, Backend.CPP):
        with use_backend(writer):
            original = Bspline(_make_space_at(case, dtype), net, is_rational)
            payload = pickle.dumps(original)
        for reader in (Backend.PYTHON, Backend.CPP):
            where = f"{writer.name} -> {reader.name}"
            with use_backend(reader):
                loaded = pickle.loads(payload)
                cloned = copy.deepcopy(original)
            for rebuilt, how in ((loaded, "pickle"), (cloned, "deepcopy")):
                assert rebuilt.control_points.tobytes() == net.tobytes(), f"{where} {how}"
                assert rebuilt.dtype == np.dtype(dtype), f"{where} {how}"
                assert rebuilt.is_rational is is_rational, f"{where} {how}"
                assert rebuilt.degree == original.degree, f"{where} {how}"
                # `rank` folds the flag against the component axis, so it is the one
                # field that would catch a round trip which kept the byte count and
                # the per-direction counts while moving which axis carries the
                # components.
                assert rebuilt.rank == original.rank, f"{where} {how}"
                assert rebuilt.space.num_basis == case.num_basis, f"{where} {how}"
                np.testing.assert_array_equal(
                    np.asarray(rebuilt.space.spaces[0].knots),
                    np.asarray(original.space.spaces[0].knots),
                    err_msg=f"{where} {how}",
                )


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_space_survives_the_round_trip_as_one_object(dtype: npt.DTypeLike) -> None:
    """Pickling a field and its space together restores a pair that still shares it.

    ``pickle`` memoises, so the sharing the identity contract rests on survives one
    ``dumps`` for free -- and only because ``__reduce__`` hands out the space
    *wrapper*. A reduction that rebuilt the space from arrays would restore two
    equal-valued spaces instead, and ``field.space is space`` would stop holding
    across a round trip while every value assertion still passed.

    Two independent ``dumps`` calls do not share, which is also true today and is not
    claimed here.
    """
    case = CASES[3]
    net = _control_net(case, dtype)
    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            space = _make_space_at(case, dtype)
            field = Bspline(space, net)
            restored_field, restored_space = pickle.loads(pickle.dumps((field, space)))
        assert restored_field.space is restored_space, backend.name


@pytest.mark.parametrize("dtype", DTYPES)
def test_an_unpickled_bspline_can_still_be_mutated_in_place(dtype: npt.DTypeLike) -> None:
    """The wire format does not carry one backend's read-only flag to the other.

    What this catches: ``__reduce__`` handing out the C++ backend's read-only view.
    numpy preserves that flag through a pickle, so a payload written under the C++
    backend would rebuild, under the Python backend, a B-spline whose stored array
    cannot be written -- and ``transform(in_place=True)`` would raise on an object the
    caller built by ordinary means.
    """
    from pantr.transform import AffineTransform  # noqa: PLC0415

    case = _Case((_SINGLE_SPAN,), (2,), (False,), (3,), 2, "unpickled")
    net = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=dtype)
    with use_backend(Backend.CPP):
        payload = pickle.dumps(Bspline(_make_space_at(case, dtype), net))
    with use_backend(Backend.PYTHON):
        rebuilt = pickle.loads(payload)
        rebuilt.transform(AffineTransform.translation([1.0, 2.0]), in_place=True)
    np.testing.assert_allclose(rebuilt.control_points, net + np.asarray([1.0, 2.0]))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_rational", [False, True])
def test_the_in_place_mutators_agree_on_the_value_they_leave_behind(
    dtype: npt.DTypeLike, is_rational: bool
) -> None:
    """``reverse``, ``permute_directions`` and ``transform`` agree under both backends.

    What this catches: the C++ path's rebuild-the-implementation strategy losing the
    rationality flag, the reseated space, the dtype or a permuted stride. Only the
    array's *identity* is allowed to differ -- the C++ value owns its storage, so an
    in-place mutation replaces it -- and ``tests/test_transform.py`` pins that
    identity for the Python backend alone.

    All three run in sequence rather than one per test, deliberately: the defect the
    derived block removes is a mutator leaving *part* of the state behind, and a
    single mutation cannot expose a reseat that half happened.
    """
    from pantr.transform import AffineTransform  # noqa: PLC0415

    case = _Case((_CLAMPED_QUADRATIC, _CLAMPED_LINEAR_4), (2, 1), (False, False), (5, 4), 3, "m")
    net = _control_net(case, dtype)
    shift = AffineTransform.translation([1.0, 2.0] if is_rational else [1.0, 2.0, 3.0])

    def mutated(backend: Backend) -> Bspline:
        # A fresh copy per backend, and not a convenience: the Python implementation
        # aliases what it is given, so an in-place mutation under that backend edits
        # `net` itself and the C++ run would then start from an already-reversed net.
        with use_backend(backend):
            field = Bspline(_make_space_at(case, dtype), net.copy(), is_rational)
            field.reverse(1, in_place=True)
            field.permute_directions([1, 0], in_place=True)
            field.transform(shift, in_place=True)
            return field

    assert_object_parity(
        py=mutated(Backend.PYTHON),
        cpp=mutated(Backend.CPP),
        fields=FIELDS,
        context=(
            f"in-place reverse, permute and transform "
            f"({np.dtype(dtype).name}, rational={is_rational})"
        ),
    )


@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.CPP])
def test_a_space_changing_mutator_reseats_the_space_and_leaves_the_old_one_intact(
    backend: Backend,
) -> None:
    """``reverse`` and ``permute_directions`` replace the space; an escaped one keeps its value.

    The contract ``design/bspline_ownership_lifetime.md`` reason 3 was measured on,
    and the reason class H stores a ``shared_ptr<const T>`` and hands out a copy of
    the handle rather than a reference. Two of the three storage shapes that note
    tabulates break this, and one of them breaks it **silently**: with the space
    stored by value and handed out as a reference, the escaped object starts
    reporting the *new* space while ``escaped is field.space`` still reads ``True``.

    So both halves are asserted. Identity alone would pass on that shape, and the
    value alone would pass on a shape that never reseated at all.

    **Two things had to be got right for the second half to say anything**, and both
    were wrong first time. The comparison has to be between two arrays of the same
    shape: ``np.array_equal`` returns ``False`` whenever the shapes differ, before it
    compares a single value, so a knot vector held up against a ``(dim, 2)`` domain
    block reports "different" for a field that was never touched. And the knot vector
    has to be asymmetric about its own domain midpoint, or the reflection is the
    vector it started from -- which ``_CLAMPED_QUADRATIC`` is, so this case uses
    ``_ASYMMETRIC`` and asserts that of it first.

    The reflected vector is stated as a literal rather than recomputed, so the
    expected value does not come from the code under test: reflecting
    ``(0, 0, 0, 1, 3, 3, 3)`` about ``[0, 3]`` gives ``(0, 0, 0, 2, 3, 3, 3)``.
    """
    case = _Case(
        (_ASYMMETRIC, _CLAMPED_LINEAR), (2, 1), (False, False), (4, 3), 2, "reseat"
    )
    reflected = (0.0, 0.0, 0.0, 2.0, 3.0, 3.0, 3.0)
    assert reflected != _ASYMMETRIC, (
        "the case's knot vector reflects onto itself, so a reversal is invisible in "
        "the knots and the value half of this test would pass on a no-op"
    )

    net = _control_net(case, np.float64)
    with use_backend(backend):
        field = Bspline(_make_space_at(case, np.float64), net.copy())
        before = field.space
        before_knots = np.array(before.spaces[0].knots)

        field.reverse(0, in_place=True)
        assert field.space is not before, "the space was not reseated"
        np.testing.assert_array_equal(
            np.array(before.spaces[0].knots),
            before_knots,
            err_msg="the escaped space started reporting the new value",
        )
        # And the new space really is the reflected one, against a hand-derived
        # literal -- so the reseat was not a no-op this test would report as a pass.
        np.testing.assert_array_equal(
            np.array(field.space.spaces[0].knots), np.asarray(reflected)
        )

        after_reverse = field.space
        field.permute_directions([1, 0], in_place=True)
        assert field.space is not after_reverse
        assert field.space.degrees == tuple(reversed(after_reverse.degrees))


@pytest.mark.parametrize("dtype", DTYPES)
def test_reversing_a_periodic_direction_in_place_agrees_with_the_derived_form(
    dtype: npt.DTypeLike,
) -> None:
    """The periodic branch of ``reverse`` gives the same value in place as not.

    The branch this covers is ``_reversed``'s cyclic shift, and it is reached only
    through ``_mutate``: a periodic direction stores fewer coefficients than the
    full sequence expands to, so reversing it needs a plain flip **plus** a shift by
    the ghost count, and the in-place path writes that shift back through
    ``new_cp[:] = np.roll(...)`` while the derived path rebinds a fresh array. This
    split is what the port introduced -- the two used to be one expression -- and
    nothing else in the suite combines a periodic direction with ``in_place=True``.

    Three assertions, and the third is what stops the first two being vacuous:

    - both backends agree on the value the in-place mutation leaves behind;
    - it equals what the derived form returns, whose pointwise correctness
      ``tests/test_review_regressions.py::test_periodic_reverse_is_pointwise_mirror``
      already establishes against an independent mirror -- so this inherits that
      rather than resting on parity alone;
    - it is **not** the plain flip. The shift here is
      ``(n_full - n_stored) % n_stored = (6 - 4) % 4 = 2``, so a flip alone would
      give ``[4, 3, 2, 1]`` and the correct answer is ``[2, 1, 4, 3]``. Without this
      the whole test would pass on an implementation that dropped the shift, since
      both paths would drop it together.

    The expected value is stated as a literal, derived by hand from the ghost count
    rather than taken from either path.
    """
    knots = np.asarray(_UNIFORM, dtype=dtype)
    control_points = np.arange(1.0, 5.0, dtype=dtype).reshape(4, 1)
    expected = np.asarray([[2.0], [1.0], [4.0], [3.0]], dtype=dtype)
    plain_flip = np.asarray([[4.0], [3.0], [2.0], [1.0]], dtype=dtype)

    mutated = {}
    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            space = BsplineSpace([BsplineSpace1D(knots, 2, periodic=True)])
            assert space.num_basis == (4,), "the case no longer has ghost coefficients"
            field = Bspline(space, control_points.copy())
            derived = field.reverse(0)
            field.reverse(0, in_place=True)
            mutated[backend] = field
            np.testing.assert_array_equal(
                np.asarray(field.control_points),
                np.asarray(derived.control_points),
                err_msg=f"{backend.name}: in place and derived disagree",
            )
            np.testing.assert_array_equal(np.asarray(field.control_points), expected)
            assert not np.array_equal(np.asarray(field.control_points), plain_flip), (
                "the reversal is a plain flip, so the cyclic shift the ghost "
                "coefficients need was not applied"
            )

    assert_object_parity(
        py=mutated[Backend.PYTHON],
        cpp=mutated[Backend.CPP],
        fields=FIELDS,
        context=f"periodic reverse in place ({np.dtype(dtype).name})",
    )


@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.CPP])
def test_transform_propagates_the_source_space_rather_than_rebuilding_it(
    backend: Backend,
) -> None:
    """A non-in-place ``transform`` hands back the *source's* space object.

    The one propagation site ``design/bspline_ownership_lifetime.md`` names, pinned by
    ``tests/test_transform.py`` for the default backend and here for both: an affine
    map moves the control points and not the parametrization, so re-wrapping the
    implementation's space would give a different Python object even when the C++
    pointer is identical, and the identity assertion would fail while every value
    still agreed.
    """
    from pantr.transform import AffineTransform  # noqa: PLC0415

    case = CASES[1]
    with use_backend(backend):
        field = Bspline(_make_space_at(case, np.float64), _control_net(case, np.float64))
        moved = field.transform(AffineTransform.translation([1.0, 2.0, 3.0]))
    assert moved.space is field.space
    if backend is Backend.CPP:
        assert moved._impl.space is field._impl.space


def _memos_of(field: Bspline) -> tuple[Any, Any]:
    """Return the field's two derived memos, without narrowing either.

    Read through a function rather than inline, for the reason
    ``tests/test_bspline_locate.py`` gives: an inline ``assert field._derived.locate
    is not None`` narrows the attribute's type for the rest of the test, and mypy
    then calls the later assertions unreachable. A call expression is not narrowed.

    Args:
        field (Bspline): The field to read.

    Returns:
        tuple[Any, Any]: ``(beziers, locate)``, either of which may be ``None``.
    """
    return field._derived.beziers, field._derived.locate


@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.CPP])
@pytest.mark.parametrize("mutator", ["reverse", "permute_directions", "transform"])
def test_every_in_place_mutator_discards_the_whole_derived_block(
    backend: Backend, mutator: str
) -> None:
    """Each of the three mutators leaves *both* memos cold, not one of them.

    ``design/bspline_derived_caches.md`` asks this front to replace the derived block
    wholesale rather than invalidate its parts, and the failure that shape removes is
    one memo surviving the other -- a stale Bézier decomposition of a geometry that
    has been reversed, or a point-inversion hierarchy over control-point boxes that
    have moved. Neither raises; both return wrong answers.

    So the assertion is on both slots after each mutator, and the memos are *filled*
    first: a test that only checked they were ``None`` afterwards would pass on a
    field that never cached anything at all.
    """
    from pantr.transform import AffineTransform  # noqa: PLC0415

    case = _Case((_CLAMPED_QUADRATIC, _CLAMPED_LINEAR), (2, 1), (False, False), (5, 3), 2, "d")
    net = _control_net(case, np.float64)
    with use_backend(backend):
        field = Bspline(_make_space_at(case, np.float64), net.copy())

        field.to_beziers()
        field.locate(np.asarray(field.evaluate(np.asarray([[1.5, 11.0]]))))
        beziers, locate = _memos_of(field)
        assert beziers is not None, "the Bezier memo did not fill"
        assert locate is not None, "the locate memo did not fill"

        if mutator == "reverse":
            field.reverse(0, in_place=True)
        elif mutator == "permute_directions":
            field.permute_directions([1, 0], in_place=True)
        else:
            field.transform(AffineTransform.translation([1.0, 2.0]), in_place=True)

        beziers, locate = _memos_of(field)
        assert beziers is None, f"{mutator} left the Bezier memo behind"
        assert locate is None, f"{mutator} left the locate memo behind"


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_wrapper_holds_the_implementation_its_backend_selects(dtype: npt.DTypeLike) -> None:
    """The dtype picks the class, and the backend picks the family.

    What this catches: ``_impl_class`` ignoring its dtype argument. Every other
    assertion in this file would still pass if it always returned ``Bspline64``,
    because nothing else here can see which class was chosen -- the ``.noconvert()``
    on the binding turns the mistake into a refusal for one direction only.
    """
    case = _Case((_SINGLE_SPAN,), (2,), (False,), (3,), 2, "impl")
    net = np.zeros((3, 2), dtype=dtype)
    with use_backend(Backend.PYTHON):
        assert isinstance(Bspline(_make_space_at(case, dtype), net)._impl, _BsplinePython)
    with use_backend(Backend.CPP):
        assert isinstance(Bspline(_make_space_at(case, dtype), net)._impl, _cpp_class(dtype))


def test_a_space_from_the_other_backend_is_refused_rather_than_mixed() -> None:
    """Building a field over a space of the other backend refuses instead of hybridising.

    The failure this prevents is asymmetric and one half of it is silent. Handing a
    Python oracle space to a C++ field raises a nanobind ``TypeError`` naming C++
    types, which is loud but unreadable. Handing a C++ space handle to the oracle
    *succeeds* and yields a hybrid whose reductions run in Python over C++ values --
    which no parity claim in this file covers and nothing announces.
    ``design/cross_backend_types.md`` forbids exactly that second shape.
    """
    case = CASES[1]
    net = _control_net(case, np.float64)
    with use_backend(Backend.PYTHON):
        py_space = _make_space(case)
    with use_backend(Backend.CPP):
        cpp_space = _make_space(case)

    with use_backend(Backend.CPP), pytest.raises(ValueError, match="active backend"):
        Bspline(py_space, net)
    with use_backend(Backend.PYTHON), pytest.raises(ValueError, match="active backend"):
        Bspline(cpp_space, net)


def test_mutating_under_a_switched_backend_is_refused_rather_than_converted() -> None:
    """An in-place mutator refuses a backend that is not the one this field was built under.

    Rebuilding the implementation reads the *active* backend, so without this check a
    field built under C++ and reversed inside a ``use_backend(Backend.PYTHON)`` block
    would come back as ``_BsplinePython`` -- and the caller's ``control_points`` would
    go from read-only to writeable underneath them, on an array they still held.

    Pins that the refusal leaves the object untouched, and that a mutation under the
    matching backend still works -- a check that merely raised everywhere would pass
    the first half. Each of the three mutators is swept, because the guard lives in
    ``Bspline._mutate`` and a method that reached the value another way would bypass
    it.
    """
    from pantr.transform import AffineTransform  # noqa: PLC0415

    case = _Case((_SINGLE_SPAN,), (2,), (False,), (3,), 2, "switched")
    net = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    shift = AffineTransform.translation([1.0, 2.0])

    mutators: tuple[Callable[[Bspline], None], ...] = (
        lambda field: field.reverse(0, in_place=True),
        lambda field: field.permute_directions([0], in_place=True),
        lambda field: field.transform(shift, in_place=True),
    )
    for mutate in mutators:
        with use_backend(Backend.CPP):
            field = Bspline(_make_space_at(case, np.float64), net.copy())
        assert not field.control_points.flags.writeable

        with use_backend(Backend.PYTHON), pytest.raises(TypeError, match="different backend"):
            mutate(field)

        assert isinstance(field._impl, _cpp_class(np.float64))
        assert not field.control_points.flags.writeable
        np.testing.assert_array_equal(field.control_points, net)

        with use_backend(Backend.CPP):
            mutate(field)
        assert not field.control_points.flags.writeable


@pytest.mark.parametrize("seed", range(24))
def test_a_generated_field_agrees_field_by_field(seed: int) -> None:
    """A random space, net, dtype and rationality agree on every field.

    The shipped sweep. It exists because ``CASES`` is a table someone chose, and a
    table cannot cover the interaction of a degree, a knot multiplicity, a component
    count and a storage format all at once. ``scripts/measure_bspline_type_parity.py``
    runs the same generator over a far larger draw, which is what the claim actually
    rests on: a sweep checked only by its own shipped size has not been checked.
    """
    rng = np.random.default_rng(20260904 + seed)
    dim = int(rng.integers(1, 4))
    dtype = np.float32 if rng.random() < 0.5 else np.float64
    knots = []
    degrees = []
    for _ in range(dim):
        degree = int(rng.integers(0, 4))
        interior = int(rng.integers(0, 4))
        breaks = np.cumsum(rng.uniform(0.5, 2.0, size=interior + 1))
        vector = np.concatenate(
            [np.zeros(degree + 1), np.repeat(breaks[:-1], 1), np.full(degree + 1, breaks[-1])]
        )
        knots.append(np.asarray(vector, dtype=dtype))
        degrees.append(degree)

    components = int(rng.integers(1, 5))
    is_rational = bool(rng.random() < 0.5) and components > 1

    def build() -> Bspline:
        space = BsplineSpace(
            [BsplineSpace1D(vector, degree) for vector, degree in zip(knots, degrees, strict=True)]
        )
        total = space.num_total_basis
        values = rng.standard_normal(total * components) * 2.0 ** rng.integers(
            -20, 20, size=total * components
        )
        return Bspline(space, np.asarray(values, dtype=dtype), is_rational)

    # One draw of the coefficients, so the two backends get the same numbers: the
    # generator is re-seeded rather than shared, because `build` advances `rng`.
    state = rng.bit_generator.state
    with use_backend(Backend.PYTHON):
        py = build()
    rng.bit_generator.state = state
    with use_backend(Backend.CPP):
        cpp = build()

    assert_object_parity(
        py=py,
        cpp=cpp,
        fields=FIELDS,
        context=f"generated field, seed {seed}, {np.dtype(dtype).name}",
    )
