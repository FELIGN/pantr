"""Parity for `pantr.bspline.BsplineSpace`: the C++ tensor product against the oracle.

## What kind of claim this file makes

Every claim here is **exactness**, and none of it is a bound. A tensor-product space
answers counting questions about its directions, so the dimension, the per-direction
degrees, basis counts and interval counts, the two products and the Bézier flag are
integers and booleans reached by the same integer arithmetic on both sides; those are
compared with :func:`exact_parity`, and a tolerance on any of them would be hiding
something.

The two floating-point quantities are compared **bitwise**, and both are claims with
an argument rather than observations that happened to hold:

- ``domain`` is each direction's own ``domain`` copied unchanged, and a direction's
  ``domain`` is two indexed reads of its knot vector. No arithmetic is performed on
  it at either level.
- ``tolerance`` is ``max`` over the directions' tolerances. A **selection**, not a
  combination: the result is one of the inputs, bit for bit, so the only thing that
  could differ is *which* input, which is a verdict rather than a rounding.

Both rest on the directions themselves agreeing bitwise, which
``tests/parity/test_bspline_space_1d.py`` claims and pins independently. That is the
whole of the nD derivation: exact integer arithmetic and exact selection over inputs
that are already equal. ``design/backend_parity.md`` Rule 8 -- a parity claim is only
defined where the quantity has digits -- is satisfied trivially rather than narrowly,
because nothing here loses a digit; and ``CLAUDE.md``'s determinism rule licenses
bit-identity for exactly this case, where finite precision does not bite.

## Why every multi-direction case is asymmetric

This is the file's main structural decision. On a tensor product, two directions that
agree on their degree, their counts and their domain make a whole family of defects
invisible at once: a transposition, an off-by-one in the axis index, a ``min`` written
for a ``max``, and a reduction that quietly returns its first argument all produce the
right answer. Symmetric cases are how a parity file reads as full coverage and
measures nothing.

So no case below repeats a direction except the one that exists to exercise repetition
(``BsplineSpace([s, s])``, which is the common spelling in the suite and which broke a
first draft of the tolerance reduction), and
:func:`test_the_sweep_can_see_each_reduction` asserts of the *case table itself* that
each reduced quantity is distinguished somewhere in it, with the tolerance argmax
appearing both first and last. It is a test of the test set, and it fails if a later
case is added that does not carry its weight.

## The independent accuracy check

`design/backend_parity.md` requires more than agreement with the oracle, and agreement
is all the sweeps establish. The independent check here is a **closed form for a whole
family**: for a tensor product of clamped uniform directions with degrees ``p_d`` over
``n_d`` intervals on ``[a_d, b_d]``, hand arithmetic gives ``num_basis[d] == n_d + p_d``,
``num_intervals[d] == n_d``, the two products, a domain of exactly the requested ends,
and -- because the ends are chosen as small dyadic rationals -- a tolerance of exactly
``max_d 8 * eps * max(b_d - a_d, |a_d|, |b_d|)``.
:func:`test_the_clamped_uniform_closed_form` checks all of them against the formula
rather than against either backend.

Two cross-checks against the *directions* join it, and they are not mirrors of the
implementation: they compare the aggregate against the univariate type, which is
ported and pinned separately, so a reduction that read the wrong axis or dropped a
direction fails them.

## Rule 12

Every test that says something about the *binding* takes the ``cpp_backend`` fixture,
whose ``demand_cpp_backend`` is a skip-or-fail rather than a silent skip. The tests
that state a property of *each* backend take it on the C++ **parameter** instead,
through ``_BACKENDS`` -- because taking it on the test would skip the Python half too,
and the Python half is the only thing here that would catch the oracle regressing.
"""

from __future__ import annotations

import math
import pickle
from typing import TYPE_CHECKING, Any, Final, NamedTuple

import numpy as np
import pytest

from pantr._backend import Backend, use_backend
from pantr.bspline import BsplineSpace, BsplineSpace1D
from tests._parity_harness import (
    Field,
    assert_object_parity,
    bitwise_parity,
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


def _demand_the_extension_if_needed(backend: Backend) -> None:
    """Require the compiled extension, and only for the half that uses it.

    A test parametrized over both backends and *also* taking the ``cpp_backend``
    fixture skips **both** halves when the extension is absent, which silently drops
    the Python half -- and the Python half of the closed-form check and the
    cross-checks is the only thing in this file that would catch the **oracle**
    regressing. ``tests/parity/test_bspline_space_1d.py`` carries the same guard and
    the same reason, including why marking the parameter is not available:
    ``pytest.param`` refuses ``pytest.mark.usefixtures``.

    Args:
        backend (Backend): The backend this parametrization is about.
    """
    if backend is Backend.CPP:
        demand_cpp_backend()


_STATE_WHY: Final = (
    "a tensor-product space counts things in its directions; the dimension, the "
    "per-direction degrees, basis counts and interval counts, the two products and "
    "the Bezier flag are integers and booleans reached by the same integer "
    "arithmetic on both sides, so a difference is a defect and not a rounding"
)

_DOMAIN_WHY: Final = (
    "the domain rows are the directions' own domain ends, copied unchanged; a "
    "direction's domain is two indexed reads of its knot vector, so no arithmetic "
    "is performed on these values at either level and the two backends store the "
    "same bits"
)

_TOLERANCE_WHY: Final = (
    "the tolerance is max over the directions' tolerances -- a selection, not a "
    "combination -- so the result is one of the inputs unmodified. The inputs agree "
    "bitwise by tests/parity/test_bspline_space_1d.py, and the only thing left that "
    "could differ is which input wins, which is a verdict rather than a rounding"
)


class _Direction(NamedTuple):
    """One direction of a tensor-product space.

    Attributes:
        knots (tuple[float, ...]): The knot vector.
        degree (int): The polynomial degree.
        periodic (bool): Whether the direction is periodic.
    """

    knots: tuple[float, ...]
    degree: int
    periodic: bool = False


# The named directions the cases are composed from. Each carries the structural
# feature it contributes and, where it matters, its knot scale -- which is what fixes
# its tolerance and therefore which direction wins the reduction.
_QUAD_3: Final = _Direction((0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0), 2)
"""Clamped quadratic, 3 intervals on [0, 3]: degree 2, 5 basis functions, scale 3."""

_LIN_2: Final = _Direction((10.0, 10.0, 11.0, 12.0, 12.0), 1)
"""Clamped linear, 2 intervals on [10, 12]: degree 1, 3 basis functions, scale 12."""

_CUBIC_1: Final = _Direction((-2.0, -2.0, -2.0, -2.0, -1.0, -1.0, -1.0, -1.0), 3)
"""Clamped cubic single span on [-2, -1]: degree 3, 4 basis functions, scale 2."""

_DEG0_4: Final = _Direction((0.0, 1.0, 2.0, 3.0, 4.0), 0)
"""Degree zero, 4 intervals on [0, 4]: 4 basis functions, scale 4."""

_PERIODIC: Final = _Direction(tuple(float(k) for k in range(10)), 2, True)
"""Periodic uniform, which is the one direction kind that is never Bezier-like."""

_BEZ_2: Final = _Direction((0.0, 0.0, 0.0, 1.0, 1.0, 1.0), 2)
"""Clamped quadratic single span: Bezier-like, 3 basis functions, 1 interval."""

_NOTBEZ_2: Final = _Direction((0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0), 2)
"""The same but with one interior knot: not Bezier-like, 4 basis functions."""

_FAR: Final = _Direction((1e6, 1e6, 1e6, 1e6 + 50.0, 1e6 + 100.0, 1e6 + 100.0, 1e6 + 100.0), 2)
"""A domain far from the origin, where the scale comes from the coordinates."""

_TINY: Final = _Direction((0.0, 0.0, 0.0, 5e-7, 1e-6, 1e-6, 1e-6), 2)
"""A domain smaller than one unit, where the scale comes from the span."""

_STRADDLES_ZERO: Final = _Direction((-1e-7, -1e-7, -1e-7, 0.5, 1.0, 1.0, 1.0), 2)
"""A domain straddling zero, whose ``float32`` tolerance is not a ``float32`` value.

The one direction here whose knot ends are not dyadic, and it is in the table for a
narrow reason. The tolerance is ``8 * eps * scale`` with ``scale = hi - lo``, and for
two ``float32`` operands of opposite sign that difference is a *sum*, which can need
one bit more than ``float32`` has: measured here, the ``float32`` tolerance is
``9.536744117736828e-07`` while the nearest ``float32`` is ``9.5367436e-07``, a
relative gap of ``1.9e-08``.

Why that is worth a case of its own: ``pantr/bspline/knots.hpp`` records that the
tolerance is a ``double`` **at every storage width**, deliberately, and every other
direction in this table has a dyadic scale -- for which ``8 * eps * scale`` is a
``float32`` value exactly, since the factor is a power of two. So on the rest of the
table an implementation that stored the tolerance in ``T`` would agree bit for bit
with one that stored a ``double``, and the claim would pass while measuring nothing.
This is the case where the two spellings differ.
"""


class _Case(NamedTuple):
    """One tensor-product space to build, and what it is in the table for.

    Attributes:
        label (str): What this case is here to exercise.
        directions (tuple[_Direction, ...]): The directions, in axis order.
    """

    label: str
    directions: tuple[_Direction, ...]


_CASES: Final = (
    # The nD case a reduction can get right by accident, kept because it is what the
    # `pantr.cad` layer builds for a curve.
    _Case("one direction", (_QUAD_3,)),
    # Two directions differing in EVERY reduced quantity: degree 2 against 1, 5 basis
    # functions against 3, 3 intervals against 2, and scales 3 against 12 so the
    # tolerances differ too.
    _Case("two directions, all quantities distinct", (_QUAD_3, _LIN_2)),
    # The same two, swapped. Present so that a transposition or an axis-index error
    # cannot pass by agreeing with the other order.
    _Case("the same two, swapped", (_LIN_2, _QUAD_3)),
    # Three distinct directions, so the products are over three unequal factors --
    # which a two-direction case cannot distinguish from a pairwise reduction applied
    # twice in the wrong order.
    _Case("three directions, all distinct", (_QUAD_3, _LIN_2, _DEG0_4)),
    _Case("three directions, another order", (_DEG0_4, _CUBIC_1, _QUAD_3)),
    # One direction repeated, which is how most of the suite builds a square space and
    # which broke a first draft of the tolerance reduction: it asked "is this the
    # first direction?" of the shared handle rather than of the index.
    _Case("one direction repeated", (_QUAD_3, _QUAD_3)),
    # A periodic direction beside a clamped one, in both positions. Periodicity moves
    # the basis count and forces the Bezier flag false whatever the knots say.
    _Case("periodic first", (_PERIODIC, _LIN_2)),
    _Case("periodic last", (_LIN_2, _PERIODIC)),
    # Bezier-likeness needs every direction, so the exception is placed in each
    # position of a three-direction space in turn. A reduction written as `any`, or as
    # "ask the first direction", passes exactly one of these three.
    _Case("all Bezier-like", (_BEZ_2, _BEZ_2, _BEZ_2)),
    _Case("one non-Bezier direction, first", (_NOTBEZ_2, _BEZ_2, _BEZ_2)),
    _Case("one non-Bezier direction, middle", (_BEZ_2, _NOTBEZ_2, _BEZ_2)),
    _Case("one non-Bezier direction, last", (_BEZ_2, _BEZ_2, _NOTBEZ_2)),
    # Scales six orders apart, with the argmax first and then last, so `max` is
    # distinguished from `min`, from `first` and from `last` over one set of values.
    _Case("argmax scale first", (_FAR, _QUAD_3, _TINY)),
    _Case("argmax scale last", (_TINY, _QUAD_3, _FAR)),
    # The case where storing the tolerance in the storage format rather than in a
    # double would show; see `_STRADDLES_ZERO`. It has to be the ARGMAX for that, so
    # it is paired with a direction of scale 1 rather than with `_QUAD_3`, whose
    # scale of 3 would win and whose tolerance is dyadic. A first draft did pair it
    # with `_QUAD_3`, and the narrowing mutation passed the whole suite.
    _Case("a non-dyadic argmax, first", (_STRADDLES_ZERO, _BEZ_2)),
    _Case("a non-dyadic argmax, last", (_BEZ_2, _STRADDLES_ZERO)),
    # The dimensionless space, which the oracle admits and which every reduction has
    # to answer for: the products are 1, `all(())` is true, and only the tolerance
    # has nothing to report.
    _Case("no directions", ()),
)
"""The spaces every field-by-field comparison runs over, and what each is here for."""


def _build(directions: tuple[_Direction, ...], dtype: npt.DTypeLike) -> BsplineSpace:
    """Build one tensor-product space under the active backend.

    Args:
        directions (tuple[_Direction, ...]): The directions, in axis order.
        dtype (npt.DTypeLike): The storage format.

    Returns:
        BsplineSpace: The space.
    """
    return BsplineSpace(
        [
            BsplineSpace1D(
                np.asarray(one_d.knots, dtype=dtype), one_d.degree, periodic=one_d.periodic
            )
            for one_d in directions
        ]
    )


def _both_backends(
    directions: tuple[_Direction, ...],
    dtype: npt.DTypeLike,
) -> tuple[BsplineSpace, BsplineSpace]:
    """Build the same tensor-product space under each backend.

    The directions are built inside each ``use_backend`` block too, and they have to
    be: a space aggregates its directions' *implementations*, and
    :func:`pantr.bspline._bspline_space_nd._new_impl` refuses a direction from the
    other backend rather than reconciling it.

    Args:
        directions (tuple[_Direction, ...]): The directions, in axis order.
        dtype (npt.DTypeLike): The storage format.

    Returns:
        tuple[BsplineSpace, BsplineSpace]: ``(python, cpp)``, in that order, so that a
        call site cannot get :func:`assert_object_parity`'s two keyword arguments the
        wrong way round without saying so.
    """
    with use_backend(Backend.PYTHON):
        python = _build(directions, dtype)
    with use_backend(Backend.CPP):
        cpp = _build(directions, dtype)
    return python, cpp


def _fields(dim: int) -> list[Field]:
    """The state two backends' tensor-product spaces must agree on.

    Args:
        dim (int): The space's dimension, which decides whether ``domain`` exists to
            compare: a dimensionless space raises :class:`IndexError` for it, which
            is the behaviour before the port and is asserted separately.

    Returns:
        list[Field]: One field per piece of state, each with the claim that governs
        it.
    """
    fields = [
        Field("dim", exact_parity(why=_STATE_WHY)),
        Field("degrees", exact_parity(why=_STATE_WHY)),
        Field("num_basis", exact_parity(why=_STATE_WHY)),
        Field("num_total_basis", exact_parity(why=_STATE_WHY)),
        Field("num_intervals", exact_parity(why=_STATE_WHY)),
        Field("num_total_intervals", exact_parity(why=_STATE_WHY)),
        Field(
            "has_Bezier_like_knots",
            exact_parity(why=_STATE_WHY),
            read=lambda s: s.has_Bezier_like_knots(),
        ),
    ]
    if dim > 0:
        fields += [
            Field("tolerance", bitwise_parity(why=_TOLERANCE_WHY)),
            Field("domain", bitwise_parity(why=_DOMAIN_WHY)),
        ]
    return fields


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("case", _CASES, ids=[c.label for c in _CASES])
def test_the_state_agrees_field_by_field(
    cpp_backend: None,
    case: _Case,
    dtype: npt.DTypeLike,
) -> None:
    """Every piece of a tensor-product space's state agrees between the two backends.

    Field by field rather than by a single equality, so that a failure names the
    quantity that moved instead of the object that contains it.
    """
    python, cpp = _both_backends(case.directions, dtype)
    assert_object_parity(
        py=python,
        cpp=cpp,
        fields=_fields(len(case.directions)),
        context=f"BsplineSpace[{case.label}]",
    )


def test_the_sweep_can_see_each_reduction() -> None:
    """Every reduced quantity is distinguished somewhere in the case table.

    A test of the case table rather than of the type, and the file's guard against its
    own worst failure mode: a tensor-product case whose directions agree makes a
    transposition, an axis-index error and a wrong reduction all invisible, so a table
    of such cases reads as coverage and measures nothing. Failing here means a case
    was added or changed in a way that stopped carrying its weight.

    Not gated on the extension and not parametrized: it inspects the table through the
    default backend, and the property it asserts is a property of the table.
    """
    spaces = {
        case.label: _build(case.directions, np.float64)
        for case in _CASES
        if len(case.directions) > 1
    }
    assert spaces, "the table must contain multi-direction cases"

    def somewhere(name: str, read: Any) -> bool:
        return any(len(set(read(space))) > 1 for space in spaces.values())

    assert somewhere("degrees", lambda s: s.degrees), (
        "no case has directions of differing degree, so an axis-index error in "
        "`degrees` would be invisible"
    )
    assert somewhere("num_basis", lambda s: s.num_basis), (
        "no case has directions of differing basis count"
    )
    assert somewhere("num_intervals", lambda s: s.num_intervals), (
        "no case has directions of differing interval count"
    )
    assert somewhere("tolerance", lambda s: [one.tolerance for one in s.spaces]), (
        "no case has directions of differing tolerance, so `max` could be `min`, "
        "`first` or `last` and nothing would notice"
    )
    assert somewhere("bezier", lambda s: [one.has_Bezier_like_knots() for one in s.spaces]), (
        "no case mixes a Bezier-like direction with a non-Bezier one, so `all` could be `any`"
    )
    assert somewhere("domain rows", lambda s: [tuple(row) for row in s.domain]), (
        "no case has directions of differing domain, so a row swap would be invisible"
    )

    # The tolerance argmax must appear both first and last. One position alone leaves
    # `max` indistinguishable from whichever of `first` or `last` matches it.
    argmaxes = set()
    for space in spaces.values():
        tolerances = [one.tolerance for one in space.spaces]
        if len(set(tolerances)) == len(tolerances):
            argmaxes.add(tolerances.index(max(tolerances)) == 0)
            argmaxes.add(tolerances.index(max(tolerances)) == len(tolerances) - 1)
    assert argmaxes == {True, False}, (
        "the tolerance argmax must be the first direction in some case and the last "
        "in another, over cases whose direction tolerances are all distinct"
    )

    # Some case in the table must have a ``float32`` tolerance that is not a
    # ``float32`` value, and the check has to be on the SPACE rather than on a
    # direction. `pantr/bspline/knots.hpp` records that the tolerance is a ``double``
    # at every storage width, deliberately; a space that stored it in ``T`` instead
    # would agree bit for bit everywhere the winning direction's scale is dyadic,
    # which is every other case here. So without this the whole file passes on that
    # mutation -- measured, and it is why the pairing in `_STRADDLES_ZERO`'s two cases
    # is what it is rather than the more natural one.
    non_dyadic = [
        label
        for label, space in {
            case.label: _build(case.directions, np.float32) for case in _CASES if case.directions
        }.items()
        if float(np.float32(space.tolerance)) != space.tolerance
    ]
    assert non_dyadic, (
        "no case's float32 tolerance is outside float32, so storing the tolerance in "
        "the storage format rather than in a double would agree bit for bit"
    )


@pytest.mark.parametrize("backend", _BACKENDS)
def test_the_clamped_uniform_closed_form(backend: Backend) -> None:
    """A clamped uniform tensor product matches hand arithmetic, not the other backend.

    The independent accuracy check `design/backend_parity.md` requires. For degrees
    ``p_d`` over ``n_d`` uniform intervals on ``[a_d, b_d]``, the counts are
    ``n_d + p_d`` and ``n_d``, the totals are their products, the domain is exactly the
    requested ends, and the tolerance is ``8 * eps * max_d max(b_d - a_d, |a_d|, |b_d|)``.

    The ends and the interval counts are dyadic, which is what makes the tolerance a
    closed form rather than an approximation: ``8 * eps`` is a power of two, so the
    product with an exactly representable scale is exact at both storage widths and the
    comparison is ``==``.
    """
    _demand_the_extension_if_needed(backend)
    ends = ((0.0, 4.0), (-2.0, 2.0), (0.0, 8.0))
    with use_backend(backend):
        for dtype in (np.float32, np.float64):
            eps = float(np.finfo(dtype).eps)
            for degrees in ((0, 1), (2, 3), (1, 1), (3, 2, 1)):
                for counts in ((1, 4), (4, 2), (2, 2), (2, 4, 1)):
                    if len(degrees) != len(counts):
                        continue
                    directions = [
                        BsplineSpace1D(
                            np.asarray(
                                _clamped_uniform_knots(*ends[d], counts[d], degrees[d]),
                                dtype=dtype,
                            ),
                            degrees[d],
                        )
                        for d in range(len(degrees))
                    ]
                    space = BsplineSpace(directions)
                    where = f"degrees={degrees} counts={counts} dtype={np.dtype(dtype)}"

                    assert space.dim == len(degrees), where
                    assert space.degrees == tuple(degrees), where
                    assert space.num_basis == tuple(
                        n + p for n, p in zip(counts, degrees, strict=True)
                    ), where
                    assert space.num_intervals == tuple(counts), where
                    assert space.num_total_basis == math.prod(
                        n + p for n, p in zip(counts, degrees, strict=True)
                    ), where
                    assert space.num_total_intervals == math.prod(counts), where
                    np.testing.assert_array_equal(
                        space.domain, np.asarray(ends[: len(degrees)], dtype=dtype), err_msg=where
                    )
                    expected_tolerance = (
                        8.0
                        * eps
                        * max(max(hi - lo, abs(lo), abs(hi)) for lo, hi in ends[: len(degrees)])
                    )
                    assert space.tolerance == expected_tolerance, where
                    assert space.has_Bezier_like_knots() == all(n == 1 for n in counts), where


def _clamped_uniform_knots(lo: float, hi: float, intervals: int, degree: int) -> list[float]:
    """A clamped uniform knot vector, for the closed-form check.

    Args:
        lo (float): The domain start.
        hi (float): The domain end.
        intervals (int): How many equal intervals to divide the domain into.
        degree (int): The polynomial degree, which fixes the end multiplicities.

    Returns:
        list[float]: ``degree + 1`` copies of ``lo``, the interior breakpoints, then
        ``degree + 1`` copies of ``hi``.
    """
    step = (hi - lo) / intervals
    interior = [lo + k * step for k in range(1, intervals)]
    return [lo] * (degree + 1) + interior + [hi] * (degree + 1)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("case", _CASES, ids=[c.label for c in _CASES])
def test_the_aggregate_matches_its_directions(backend: Backend, case: _Case) -> None:
    """Each reduction agrees with the directions it reduces, within one backend.

    Not a mirror of the implementation: it compares the tensor-product type against the
    *univariate* type, which is a separate port with its own parity claim
    (``tests/parity/test_bspline_space_1d.py``). A reduction that read the wrong axis,
    dropped a direction or transposed the domain rows fails this even where both
    backends agree with each other, which agreement alone cannot detect.
    """
    _demand_the_extension_if_needed(backend)
    with use_backend(backend):
        space = _build(case.directions, np.float64)
    directions = space.spaces

    assert space.dim == len(directions)
    assert space.degrees == tuple(one.degree for one in directions)
    assert space.num_basis == tuple(one.num_basis for one in directions)
    assert space.num_intervals == tuple(one.num_intervals for one in directions)
    assert space.num_total_basis == math.prod(one.num_basis for one in directions)
    assert space.num_total_intervals == math.prod(one.num_intervals for one in directions)
    assert space.has_Bezier_like_knots() == all(one.has_Bezier_like_knots() for one in directions)
    if directions:
        # `==` rather than `pytest.approx`: the reduction is a selection, so the
        # result is one of these values unmodified.
        assert space.tolerance == max(one.tolerance for one in directions)
        for d, one in enumerate(directions):
            assert tuple(space.domain[d]) == tuple(one.domain), f"domain row {d}"
    else:
        with pytest.raises(ValueError, match="no directions has no tolerance"):
            _ = space.tolerance
        with pytest.raises(IndexError):
            _ = space.domain


@pytest.mark.parametrize("backend", _BACKENDS)
def test_permuting_the_directions_permutes_the_per_direction_state(backend: Backend) -> None:
    """A permutation of the directions permutes exactly the per-direction quantities.

    The structural check a value comparison cannot make. The per-direction sequences
    and the domain rows must follow the permutation; the two products and the tolerance
    must not move at all, being symmetric functions of the same multiset.
    """
    _demand_the_extension_if_needed(backend)
    directions = (_QUAD_3, _LIN_2, _DEG0_4)
    with use_backend(backend):
        base = _build(directions, np.float64)
        for permutation in ((1, 2, 0), (2, 1, 0), (0, 2, 1)):
            permuted = _build(tuple(directions[i] for i in permutation), np.float64)
            where = f"permutation={permutation}"

            assert permuted.degrees == tuple(base.degrees[i] for i in permutation), where
            assert permuted.num_basis == tuple(base.num_basis[i] for i in permutation), where
            assert permuted.num_intervals == tuple(base.num_intervals[i] for i in permutation), (
                where
            )
            np.testing.assert_array_equal(
                permuted.domain, base.domain[list(permutation)], err_msg=where
            )

            assert permuted.num_total_basis == base.num_total_basis, where
            assert permuted.num_total_intervals == base.num_total_intervals, where
            assert permuted.tolerance == base.tolerance, where
            assert permuted.has_Bezier_like_knots() == base.has_Bezier_like_knots(), where


def test_the_dtype_refusal_agrees_character_for_character(cpp_backend: None) -> None:
    """Mixed-dtype directions are refused identically under both backends.

    The refusal is the wrapper's, not either implementation's -- ``BsplineSpace<T>``
    can hold only ``BsplineSpace1D<T>``, so there is nothing in C++ to check -- which
    is precisely why it is worth asserting under both: a wrapper-level check that
    moved would be a wrapper-level check that stopped firing.
    """
    messages = {}
    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            first = BsplineSpace1D(np.asarray(_QUAD_3.knots, dtype=np.float64), 2)
            second = BsplineSpace1D(np.asarray(_LIN_2.knots, dtype=np.float32), 1)
            with pytest.raises(ValueError) as caught:
                BsplineSpace([first, second])
            messages[backend] = str(caught.value)

    assert messages[Backend.PYTHON] == messages[Backend.CPP]
    assert messages[Backend.PYTHON] == "All B-spline spaces must have the same data type."


def test_the_dimensionless_tolerance_refusal_agrees(cpp_backend: None) -> None:
    """Both backends refuse a dimensionless space's tolerance with the same message.

    The message is stated in both implementations rather than inherited from
    ``max()``: CPython's own text for an empty ``max()`` changed between 3.11 and
    3.12, so leaving it to the builtin would have made the two backends disagree on
    one leg of the test matrix and agree on the others.
    """
    messages = {}
    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            space = BsplineSpace([])
            with pytest.raises(ValueError) as caught:
                _ = space.tolerance
            messages[backend] = str(caught.value)

    assert messages[Backend.PYTHON] == messages[Backend.CPP]
    assert (
        messages[Backend.PYTHON]
        == "tolerance: a B-spline space with no directions has no tolerance"
    )


def test_a_direction_from_the_other_backend_is_refused(cpp_backend: None) -> None:
    """A tensor-product space cannot be built from a direction of the other backend.

    Both directions of the mismatch, and the reverse one is why this test exists: a
    C++ direction handed to the oracle would **succeed**, producing a space whose
    reductions run in Python over C++ values -- a hybrid no parity claim covers and
    which nothing announces. ``design/cross_backend_types.md`` forbids exactly that
    shape.
    """
    with use_backend(Backend.PYTHON):
        python_direction = BsplineSpace1D(np.asarray(_QUAD_3.knots), 2)
    with use_backend(Backend.CPP):
        cpp_direction = BsplineSpace1D(np.asarray(_QUAD_3.knots), 2)

    with use_backend(Backend.CPP), pytest.raises(ValueError, match="from the active backend"):
        BsplineSpace([cpp_direction, python_direction])
    with use_backend(Backend.PYTHON), pytest.raises(ValueError, match="from the active backend"):
        BsplineSpace([python_direction, cpp_direction])


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("case", _CASES, ids=[c.label for c in _CASES])
def test_reduce_round_trips_under_both_backends(
    backend: Backend, case: _Case, dtype: npt.DTypeLike
) -> None:
    """``__reduce__`` restores every piece of state, under each backend on its own."""
    _demand_the_extension_if_needed(backend)
    with use_backend(backend):
        space = _build(case.directions, dtype)
        restored = pickle.loads(pickle.dumps(space))

        assert type(restored) is BsplineSpace
        assert restored.dim == space.dim
        assert restored.degrees == space.degrees
        assert restored.num_basis == space.num_basis
        assert restored.num_total_basis == space.num_total_basis
        assert restored.num_intervals == space.num_intervals
        assert restored.num_total_intervals == space.num_total_intervals
        assert restored.has_Bezier_like_knots() == space.has_Bezier_like_knots()
        if case.directions:
            np.testing.assert_array_equal(restored.domain, space.domain)
            assert restored.dtype == space.dtype
            # `==` and not a bound: `design/bspline_pickle_tolerance.md` derives a
            # drift for a univariate space whose LAST KNOT CLASS has multiplicity
            # above one and whose snapping moved it, and no direction in this table
            # is in that state -- the tolerance is recomputed from the same knots.
            assert restored.tolerance == space.tolerance


@pytest.mark.parametrize("case", _CASES, ids=[c.label for c in _CASES])
def test_a_pickle_crosses_the_backends(cpp_backend: None, case: _Case) -> None:
    """A pickle written under one backend loads under the other, and agrees.

    What ``__reduce__`` reducing to the univariate *wrappers* buys: the wire format is
    the constructor's arguments, so a backend switch cannot become a data-format
    switch.
    """
    for writer, reader in ((Backend.PYTHON, Backend.CPP), (Backend.CPP, Backend.PYTHON)):
        with use_backend(writer):
            original = _build(case.directions, np.float64)
            blob = pickle.dumps(original)
            expected = (
                original.degrees,
                original.num_basis,
                original.num_total_basis,
                original.num_intervals,
                original.num_total_intervals,
                original.has_Bezier_like_knots(),
            )
            domain = None if not case.directions else np.array(original.domain)
            tolerance = None if not case.directions else original.tolerance
        with use_backend(reader):
            restored = pickle.loads(blob)
            where = f"{case.label}: {writer.name} -> {reader.name}"
            assert (
                restored.degrees,
                restored.num_basis,
                restored.num_total_basis,
                restored.num_intervals,
                restored.num_total_intervals,
                restored.has_Bezier_like_knots(),
            ) == expected, where
            if domain is not None:
                np.testing.assert_array_equal(restored.domain, domain, err_msg=where)
                assert restored.tolerance == tolerance, where


@pytest.mark.parametrize("backend", _BACKENDS)
def test_a_joint_pickle_keeps_the_direction_identity(backend: Backend) -> None:
    """Pickling a space together with one of its directions restores the sharing.

    ``pickle`` memoises, so the direction is written once and both references restore
    to one object. That is what makes ``space.spaces[0] is one_d`` survive a round
    trip, and it is the reason ``__reduce__`` reduces to the wrappers rather than to
    their implementations. Sharing does **not** survive two independent ``dumps``
    calls, which is asserted too so the limit is recorded rather than assumed.
    """
    _demand_the_extension_if_needed(backend)
    with use_backend(backend):
        one_d = BsplineSpace1D(np.asarray(_QUAD_3.knots), 2)
        other = BsplineSpace1D(np.asarray(_LIN_2.knots), 1)
        space = BsplineSpace([one_d, other])
        assert space.spaces[0] is one_d

        restored_space, restored_one_d = pickle.loads(pickle.dumps((space, one_d)))
        assert restored_space.spaces[0] is restored_one_d
        assert restored_space.spaces[0] is not one_d

        separate_space = pickle.loads(pickle.dumps(space))
        separate_one_d = pickle.loads(pickle.dumps(one_d))
        assert separate_space.spaces[0] is not separate_one_d


def _sweep_directions() -> list[tuple[_Direction, ...]]:
    """A deterministic grid of direction tuples, for the ten-times sweep.

    Every combination of degree and interval count over three domains, at one, two and
    three directions, with the domains rotated so that no tuple has two directions on
    the same one. Deterministic rather than random, so a failure is reproducible from
    the parametrization alone.

    Returns:
        list[tuple[_Direction, ...]]: The direction tuples to build.
    """
    # Dyadic ends of three different scales, so the tolerance argmax varies over the
    # sweep rather than always being the same direction.
    domains = ((0.0, 4.0), (-2.0, 2.0), (0.0, 1024.0))
    tuples: list[tuple[_Direction, ...]] = []
    for dim in (1, 2, 3):
        for degrees in _tuples_over((0, 1, 2, 3), dim):
            for counts in _tuples_over((1, 2, 3), dim):
                tuples.append(
                    tuple(
                        _Direction(
                            tuple(
                                _clamped_uniform_knots(
                                    *domains[(d + counts[d]) % len(domains)],
                                    counts[d],
                                    degrees[d],
                                )
                            ),
                            degrees[d],
                        )
                        for d in range(dim)
                    )
                )
    return tuples


def _tuples_over(values: tuple[int, ...], length: int) -> list[tuple[int, ...]]:
    """Every tuple of the given length drawn from ``values``.

    Args:
        values (tuple[int, ...]): The values to draw from.
        length (int): The tuple length.

    Returns:
        list[tuple[int, ...]]: All ``len(values) ** length`` tuples, in a fixed order.
    """
    if length == 0:
        return [()]
    return [(value, *rest) for value in values for rest in _tuples_over(values, length - 1)]


def test_the_bitwise_claim_holds_over_a_ten_times_sweep(cpp_backend: None) -> None:
    """The bitwise claim holds over more than ten times the shipped case count.

    The bound is **zero**: every quantity is an exact integer or an unmodified copy or
    selection of a value the two backends already agree on bit for bit, so the sweep
    reports a differing-bit count rather than a margin, and the count must be zero.
    ``design/backend_parity.md`` Rule 8 is what makes that statable -- the claim is
    defined because nothing here loses a digit -- and ``CLAUDE.md``'s determinism rule
    is what licenses bit-identity as the criterion rather than a derived tolerance,
    since finite precision does not bite on a count, an index or a copied value.

    Two vacuity guards, because a sweep that agrees on nothing interesting agrees
    trivially: the quantities compared must actually **vary** across the sweep, and the
    sweep must be at least ten times the shipped parametrization.
    """
    shipped = len(_CASES) * 2  # the two dtypes the field-by-field test runs over
    directions = _sweep_directions()
    swept = len(directions) * 2
    assert swept >= 10 * shipped, (
        f"the sweep is {swept} comparisons against {shipped} shipped, which is under "
        f"the ten-times floor of {10 * shipped}"
    )

    seen_tolerances: set[float] = set()
    seen_totals: set[int] = set()
    differing = 0
    for tuple_of_directions in directions:
        for dtype in (np.float32, np.float64):
            python, cpp = _both_backends(tuple_of_directions, dtype)
            seen_tolerances.add(python.tolerance)
            seen_totals.add(python.num_total_basis)
            if (
                python.dim != cpp.dim
                or python.degrees != cpp.degrees
                or python.num_basis != cpp.num_basis
                or python.num_total_basis != cpp.num_total_basis
                or python.num_intervals != cpp.num_intervals
                or python.num_total_intervals != cpp.num_total_intervals
                or python.has_Bezier_like_knots() != cpp.has_Bezier_like_knots()
                or python.tolerance != cpp.tolerance
                or not np.array_equal(python.domain, cpp.domain)
            ):
                differing += 1

    assert differing == 0, f"{differing} of {swept} comparisons differed"
    assert len(seen_tolerances) > 1, (
        "every space in the sweep had the same tolerance, so the reduction was never asked anything"
    )
    assert len(seen_totals) > 10, (
        f"the sweep saw only {len(seen_totals)} distinct basis totals, so the products "
        f"were barely exercised"
    )
