"""Parity for the Bézier extraction operator builder and its structural identity mask.

The C++ in ``cpp/include/pantr/bspline/extraction.hpp`` is a transliteration of
``pantr.bspline._bspline_extraction_core``: the same identity start, the same
boundary insertions, the same per-element insertion sequence in the same descending
column order, the same column copies between elements, and every step in the knots'
own scalar type. Only the loop bookkeeping differs -- the oracle writes a column at
a time through a NumPy slice where this writes element by element -- and a loop
shape moves no bits.

**So the claim is bitwise, not bounded.** Measured on this machine over every case
below and both dtypes: not one element differs. ``design/backend_parity.md`` says to
make the strongest claim that holds, because a tolerance nothing approaches asserts
nothing.

It is gated, and the gate is the one thing that would break it. Each insertion is
``alpha * C[i][k] + beta * C[i][k-1]``, which a build targeting an ISA with a fused
multiply-add may contract to one instruction with one rounding where the oracle
commits two. :func:`contraction_may_fuse` is that switch, and on such a build the
claim falls back to Rule 10's budget rather than to nothing.
:func:`_fused_claim` builds it, and it was exercised against an extension built at
``-march=native`` rather than reasoned about -- ``design/backend_parity.md`` Rule 10
records what an unevaluated conditional branch is worth.

The two independent accuracy checks
-----------------------------------

Parity says the two backends agree, not that either is right, and a transposed index
would be invisible to it: one side was written from the other, so both would make the
same mistake. Two oracles, neither of them the Python implementation.

**Exact rational values, hand-derived.** For a dyadic uniform open knot vector every
operator entry is a binary rational -- halves at degree 2, quarters at degree 3 --
so it is exactly representable in ``float32`` as well as ``float64`` and the
comparison carries a **zero** bound. :data:`_EXACT_CASES` carries the tables and
``cpp/tests/test_bspline_extraction.cpp`` carries the derivation of the quadratic one
from ``N = C @ B`` directly.

That check is deliberately not a tolerance. Rule 8's concern -- a bound as large as
the values it compares -- cannot arise where the bound is zero and the values are
exact by construction. What has to be watched instead is the opposite, that the
tables are not accidentally trivial, which
:func:`test_the_exact_tables_are_not_trivial` pins.

**The column partition of unity.** Each operator's *columns* sum to one, which
follows from ``sum_i N_i = 1`` and ``sum_j B_j = 1`` plus the linear independence of
the Bernstein basis. Columns and not rows: the quadratic table's rows sum to
``1, 1.5, 0.5``, so this is what catches a transposition, and it holds for every
knot vector rather than for a family. Its bound is derived in
:func:`_column_sum_bound` and
:func:`test_the_partition_of_unity_check_is_not_vacuous` pins that the observed
error reaches it rather than being identically zero.

What the sweep leaves out, and why
----------------------------------

One knot-vector family is **excluded from both accuracy oracles and kept as a parity
case only**: a vector whose first in-domain knot is repeated, of which
``"repeated first in-domain knot"`` is the instance. The shared algorithm's sliding
window starts at ``degree + sum of the in-domain multiplicities``, which is the last
knot index of the element's class only when ``knots[degree]`` is the last knot of
*its* class; where it is not, the window starts inside the repeated knot and the
answer is wrong on both sides -- a division of zero by zero in the worst case. Both
backends reproduce it identically, so parity is exactly what this case is for: it is
the one vector in the table where the boundary multiplicity differs from the first
in-domain **class** multiplicity, so a port that read the class multiplicity instead
would fail here and nowhere else.

Rule 12
-------

Nothing here carries ``pow`` or an integer accumulator, so no gate on the
interpreted oracle is needed: the kernel is built from ``+``, ``-``, ``*`` and
``/``, all of which IEEE 754 pins, and the bitwise claim is as true under
``NUMBA_DISABLE_JIT=1`` as anywhere.

The tests that state a property of *each* backend take the extension requirement on
the C++ **parameter** rather than on the test, because taking it on the test would
skip the Python half too, and the Python half of the accuracy checks is the only
thing here that would catch the **oracle** regressing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, NamedTuple

import numpy as np
import pytest

from pantr._backend import Backend, use_backend
from pantr.bspline import BsplineSpace1D
from pantr.bspline._bspline_extraction import _tabulate_Bspline_Bezier_1D_extraction_impl
from pantr.bspline._extraction_backend import (
    _KERNELS,
    bezier_extraction_kernel,
    bezier_identity_mask_kernel,
)
from pantr.bspline.spanwise_element_extraction import _bezier_structural_identity_mask
from tests._parity_harness import (
    Field,
    Roundings,
    assert_accuracy,
    assert_object_parity,
    assert_parity,
    bitwise_parity,
    bounded_parity,
    contraction_may_fuse,
    demand_cpp_backend,
    derived_accuracy,
    exact_parity,
    unit_roundoff,
)

if TYPE_CHECKING:
    from numpy import typing as npt

DTYPES: Final = [np.float64, np.float32]
"""The two storage formats the builder is instantiated for."""

_BACKENDS: Final = (
    pytest.param(Backend.PYTHON, id="python"),
    pytest.param(Backend.CPP, id="cpp"),
)
"""The two backends, for the tests that state a property of each one separately."""


def _demand_the_extension_if_needed(backend: Backend) -> None:
    """Require the compiled extension, and only for the half that uses it.

    A test parametrized over both backends and *also* taking the ``cpp_backend``
    fixture skips **both** halves when the extension is absent, which silently
    drops the Python half -- and the Python half of the two accuracy checks is the
    only thing here that would catch the oracle regressing. Marking the parameter
    would be neater and pytest forbids it: ``pytest.param`` refuses
    ``pytest.mark.usefixtures``.

    Args:
        backend (Backend): The backend this case runs under.
    """
    if backend is Backend.CPP:
        demand_cpp_backend()


class _Case(NamedTuple):
    """One knot vector to build the operators of, and what it is in the table for.

    A record rather than a positional tuple: two of the four fields are flags and
    ``(..., True, False)`` says nothing at a call site about which is which.

    Attributes:
        label (str): What structural feature this case exercises.
        knots (list[float]): The knot vector.
        degree (int): The polynomial degree.
        accuracy (bool): Whether the analytic invariants may be asserted on it.
            False for the one vector the shared algorithm mishandles; see the module
            docstring.
    """

    label: str
    knots: list[float]
    degree: int
    accuracy: bool


_CASES: Final = (
    _Case("clamped uniform quadratic", [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0], 2, True),
    _Case(
        "clamped uniform cubic",
        [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0],
        3,
        True,
    ),
    _Case(
        "clamped non-uniform quartic",
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0],
        4,
        True,
    ),
    _Case(
        "interior knot of multiplicity two", [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0], 2, True
    ),
    _Case(
        "interior knots at full multiplicity",
        [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
        2,
        True,
    ),
    _Case("unclamped uniform quadratic", [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], 2, True),
    _Case(
        "unclamped non-uniform cubic",
        [0.0, 0.2, 0.45, 0.7, 1.1, 1.6, 2.0, 2.5, 3.0, 3.4],
        3,
        True,
    ),
    _Case("degree zero", [0.0, 1.0], 0, True),
    _Case("degree one", [0.0, 0.0, 1.0, 2.0, 3.0, 3.0], 1, True),
    # A domain based far from the origin, where the knot tolerance is an absolute
    # length of the coordinates' own magnitude rather than of the span. At float32
    # the tolerance is about 0.48 there, so unit-spaced knots still separate.
    _Case(
        "domain at 1e6",
        [1.0e6, 1.0e6, 1.0e6, 1.0e6 + 1.0, 1.0e6 + 2.0, 1.0e6 + 3.0, 1.0e6 + 3.0, 1.0e6 + 3.0],
        2,
        True,
    ),
    # Parity only. See "What the sweep leaves out" in the module docstring: this is
    # the one vector where the boundary multiplicity (1) differs from the first
    # in-domain class multiplicity (2), and the shared algorithm's answer on it is
    # defective.
    _Case("repeated first in-domain knot", [0.0, 0.4, 0.5, 0.5, 1.0, 1.5, 2.0, 2.5], 2, False),
)
"""The shipped parity table: eleven knot vectors, each named for what it exercises."""


class _Exact(NamedTuple):
    """One knot vector whose operator entries are exact binary rationals.

    Attributes:
        label (str): What the case is.
        knots (list[float]): The knot vector.
        degree (int): The polynomial degree.
        operators (list[list[list[float]]]): The exact operators, one per element.
    """

    label: str
    knots: list[float]
    degree: int
    operators: list[list[list[float]]]


_EXACT_CASES: Final = (
    # Derived by hand from `N = C @ B`; the derivation is written out in
    # `cpp/tests/test_bspline_extraction.cpp`. Every entry is a half.
    _Exact(
        "quadratic open, three uniform elements",
        [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0],
        2,
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.5], [0.0, 0.0, 0.5]],
            [[0.5, 0.0, 0.0], [0.5, 1.0, 0.5], [0.0, 0.0, 0.5]],
            [[0.5, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
    ),
    # The interior operator of a uniform quadratic spline does not depend on whether
    # the ends are clamped: a function's restriction to an interior element depends
    # only on the local knot pattern, which is uniform in both. So both operators of
    # the unclamped uniform vector are the middle table above, which was derived
    # somewhere else entirely.
    _Exact(
        "quadratic unclamped, two uniform elements",
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        2,
        [
            [[0.5, 0.0, 0.0], [0.5, 1.0, 0.5], [0.0, 0.0, 0.5]],
            [[0.5, 0.0, 0.0], [0.5, 1.0, 0.5], [0.0, 0.0, 0.5]],
        ],
    ),
    # Degree 1 forces the identity at any knots, because the linear B-spline basis
    # on an element IS the linear Bernstein basis.
    _Exact(
        "linear, three elements",
        [0.0, 0.0, 1.0, 2.0, 3.0, 3.0],
        1,
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
    ),
)
"""Knot vectors with an exactly-representable answer, and that answer."""

_BITWISE_WHY: Final = (
    "the C++ is a transliteration of the oracle's insertion sequence: the same identity "
    "start, the same boundary insertions, the same per-element sequence over the same "
    "descending columns, and the same column copies between elements. Every step runs in "
    "the knots' own scalar type on both sides -- the oracle spells its unit "
    "`knots.dtype.type(1.0)` precisely so that numba does not promote the combination to "
    "float64 -- so the two commit the same roundings on the same operands. Only the loop "
    "bookkeeping differs and a loop shape moves no bits. This holds because the target ISA "
    "has no fused multiply-add; on a build with one, `alpha * x + beta * y` may contract "
    "and the bounded branch takes over"
)
"""Why the two operation sequences agree bit for bit, and what would change it."""


def _insertion_stages(case: _Case, dtype: npt.DTypeLike) -> int:
    """The length of the longest dependency chain of insertions, for this vector.

    Each insertion writes column ``k`` from columns ``k`` and ``k - 1`` of the
    previous outer iteration, so one outer iteration is one stage and the chain
    through an element's own sequence is ``degree - multiplicity`` long. Element
    ``e + 1``'s seeded columns are **copies** of element ``e``'s, so the chain
    accumulates across elements, and the boundary sequence adds
    ``degree - boundary`` on top of element 0. The sum over the whole vector is
    therefore an upper bound on any single entry's chain, and it is taken from this
    vector's own multiplicities rather than from the degree, because an element
    whose right knot is already ``degree``-fold contracts nothing.

    Args:
        case (_Case): The knot vector.
        dtype (npt.DTypeLike): Storage format, which fixes the tolerance and so the
            classes.

    Returns:
        int: The stage count, at least one.
    """
    knots = np.asarray(case.knots, dtype=dtype)
    space = BsplineSpace1D(knots, case.degree, snap_knots=False)
    multiplicity = space.get_unique_knots_and_multiplicity(in_domain=True)[1]
    boundary = int(
        np.count_nonzero(np.abs(knots[: case.degree + 1] - knots[case.degree]) <= space.tolerance)
    )
    stages = max(case.degree - boundary, 0)
    stages += sum(max(case.degree - int(m), 0) for m in multiplicity[1:])
    return max(stages, 1)


def _fused_claim(case: _Case, dtype: npt.DTypeLike) -> Any:
    """Build the Rule 10 claim for a build whose insertions may fuse.

    Args:
        case (_Case): The knot vector.
        dtype (npt.DTypeLike): Storage format.

    Returns:
        Any: The parity claim.
    """
    return bounded_parity(
        roundings=Roundings(
            stages=_insertion_stages(case, dtype), accumulator_per_stage=3, storage_per_stage=0
        ),
        accumulator=dtype,
        storage=dtype,
        amplification=np.array(1.0),
        why=(
            "this build's target ISA has a fused multiply-add, so at each insertion the C++ "
            "may compute `fl(alpha*x + beta*y)` with one of the two products contracted "
            "where the oracle computes both products first. design/backend_parity.md Rule 10 "
            "budgets that at three accumulator roundings per fused site. The insertion "
            "weight and its complement are common mode -- a subtraction and a division "
            "carry no fusible site, so both backends hold the same alpha and the same beta "
            "bit for bit -- and the accumulator is the storage format on both sides, so no "
            "store narrows and storage_per_stage is zero. The amplification is one because "
            "every intermediate lies in [0, 1]: the operators start as the identity, whose "
            "entries are 0 and 1, and each insertion replaces a column by a convex "
            "combination of two columns, so the largest entry of a row cannot grow. One is "
            "therefore the exact bound on the intermediates, and it cannot be tightened "
            "using the final values, which are smaller than the intermediates by exactly "
            "what the recurrence shrank them"
        ),
    )


def _column_sum_bound(degree: int, dtype: npt.DTypeLike) -> float:
    """The absolute error a column sum of one may carry.

    Each entry is a chain of at most ``degree`` insertions, each committing four
    roundings -- the complement ``1 - alpha``, the two products, and their sum --
    against an exact convex combination of values in ``[0, 1]``, so an entry carries
    at most ``gamma_{4 degree}`` of absolute error. Summing ``degree + 1`` of them
    against an exact total of one adds ``gamma_{degree}``. First order in ``u``,
    that is ``(4 degree (degree + 1) + degree) u``.

    Args:
        degree (int): The polynomial degree.
        dtype (npt.DTypeLike): Storage format.

    Returns:
        float: The bound, zero at degree 0 where nothing is computed.
    """
    u = unit_roundoff(dtype)
    return (4.0 * degree * (degree + 1.0) + degree) * u


def _build(case: _Case, dtype: npt.DTypeLike, backend: Backend) -> npt.NDArray[Any]:
    """Build one case's operators under one backend.

    Args:
        case (_Case): The knot vector.
        dtype (npt.DTypeLike): Storage format.
        backend (Backend): Which implementation to run.

    Returns:
        npt.NDArray[Any]: The ``(n_intervals, degree+1, degree+1)`` operators.
    """
    knots = np.asarray(case.knots, dtype=dtype)
    space = BsplineSpace1D(knots, case.degree, snap_knots=False)
    with use_backend(backend):
        return np.asarray(
            _tabulate_Bspline_Bezier_1D_extraction_impl(space.knots, case.degree, space.tolerance)
        )


def _claim(case: _Case, dtype: npt.DTypeLike) -> Any:
    """The parity claim for one case, on whichever build is running.

    Args:
        case (_Case): The knot vector.
        dtype (npt.DTypeLike): Storage format.

    Returns:
        Any: A bitwise claim, or Rule 10's bounded one where contraction is live.
    """
    if contraction_may_fuse():
        return _fused_claim(case, dtype)
    return bitwise_parity(why=_BITWISE_WHY)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", _CASES, ids=[entry.label for entry in _CASES])
def test_the_operators_agree(cpp_backend: None, case: _Case, dtype: npt.DTypeLike) -> None:
    """The two backends build the same operators.

    Args:
        cpp_backend (None): Requires the compiled extension.
        case (_Case): The knot vector.
        dtype (npt.DTypeLike): Storage format.
    """
    reference = _build(case, dtype, Backend.PYTHON)
    actual = _build(case, dtype, Backend.CPP)
    assert_parity(
        actual, reference, _claim(case, dtype), context=f"{case.label} in {np.dtype(dtype).name}"
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", _CASES, ids=[entry.label for entry in _CASES])
def test_the_identity_mask_agrees(cpp_backend: None, case: _Case, dtype: npt.DTypeLike) -> None:
    """The two backends mark the same elements as already-Bézier.

    A boolean verdict per element, so the claim is exactness rather than a
    tolerance: ``design/backend_parity.md`` Rule 11's distinction, and the mask is
    reached by the same integer comparisons on both sides.

    Args:
        cpp_backend (None): Requires the compiled extension.
        case (_Case): The knot vector.
        dtype (npt.DTypeLike): Storage format.
    """
    space = BsplineSpace1D(np.asarray(case.knots, dtype=dtype), case.degree, snap_knots=False)
    with use_backend(Backend.PYTHON):
        reference = _bezier_structural_identity_mask(space)
    with use_backend(Backend.CPP):
        actual = _bezier_structural_identity_mask(space)
    assert_object_parity(
        py=reference,
        cpp=actual,
        fields=[
            Field(
                name="identity mask",
                claim=exact_parity(
                    why=(
                        "the mask is two integer comparisons of a knot multiplicity against "
                        "degree + 1, so both sides reach the same boolean by the same exact "
                        "arithmetic and a difference is a defect rather than a rounding"
                    )
                ),
                read=lambda mask: mask,
            )
        ],
        context=f"{case.label} in {np.dtype(dtype).name}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("case", _EXACT_CASES, ids=[entry.label for entry in _EXACT_CASES])
def test_matches_the_exact_rational_operators(
    case: _Exact, backend: Backend, dtype: npt.DTypeLike
) -> None:
    """Each backend reproduces the hand-derived exact table, with a zero bound.

    Args:
        case (_Exact): The knot vector and its exact operators.
        backend (Backend): Which implementation to run.
        dtype (npt.DTypeLike): Storage format.
    """
    _demand_the_extension_if_needed(backend)
    knots = np.asarray(case.knots, dtype=dtype)
    space = BsplineSpace1D(knots, case.degree, snap_knots=False)
    with use_backend(backend):
        computed = np.asarray(
            _tabulate_Bspline_Bezier_1D_extraction_impl(space.knots, case.degree, space.tolerance)
        )
    exact = np.asarray(case.operators, dtype=dtype)
    assert computed.shape == exact.shape, f"{case.label}: shape {computed.shape} vs {exact.shape}"
    assert_accuracy(
        computed,
        exact,
        derived_accuracy(
            bound=np.zeros_like(exact, dtype=np.float64),
            why=(
                "every entry of this vector's operators is a binary rational -- a half at "
                "degree 2, a quarter at degree 3 -- so it is exactly representable in both "
                "storage formats, and every insertion weight along the way is a half too. "
                "No rounding occurs, so the bound is zero rather than derived from one"
            ),
        ),
        context=f"{case.label} in {np.dtype(dtype).name} on {backend.name}",
    )


def test_the_exact_tables_are_not_trivial() -> None:
    """A zero bound says something only if the values it guards are not all forced.

    The identity is what the builder starts from, so a table that is the identity
    everywhere would pass against a builder that did nothing at all. This pins that
    the tables carry entries strictly between zero and one, that they are not all
    the identity, and that transposing one changes it -- which is what makes the
    comparison able to see an index-order error.
    """
    interesting = [
        np.asarray(case.operators, dtype=np.float64) for case in _EXACT_CASES if case.degree >= 2
    ]
    assert interesting, "no exact case has an answer that is not forced"
    for table in interesting:
        strictly_inside = (table > 0.0) & (table < 1.0)
        assert strictly_inside.any(), "an exact table is entirely zeros and ones"
        identity = np.broadcast_to(np.eye(table.shape[1]), table.shape)
        assert not np.array_equal(table, identity), "an exact table is the identity everywhere"
        assert not np.array_equal(table, np.swapaxes(table, 1, 2)), (
            "an exact table is symmetric, so comparing against it could not tell a "
            "transposed operator from the right one"
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize(
    "case",
    [entry for entry in _CASES if entry.accuracy],
    ids=[e.label for e in _CASES if e.accuracy],
)
def test_the_columns_are_a_partition_of_unity(
    case: _Case, backend: Backend, dtype: npt.DTypeLike
) -> None:
    """Every operator's columns sum to one, in either backend.

    The analytic oracle: ``sum_i N_i = 1`` on the element and ``sum_j B_j = 1`` on
    the reference interval, so ``sum_i sum_j C_ij B_j = sum_j B_j`` and the
    Bernstein basis being independent forces ``sum_i C_ij = 1`` for every column
    ``j``. Rows are not what sums to one, which is why this catches a transposition.

    Args:
        case (_Case): The knot vector.
        backend (Backend): Which implementation to run.
        dtype (npt.DTypeLike): Storage format.
    """
    _demand_the_extension_if_needed(backend)
    operators = _build(case, dtype, backend)
    column_sums = operators.sum(axis=1)
    assert_accuracy(
        column_sums,
        np.ones_like(column_sums),
        derived_accuracy(
            bound=np.full(column_sums.shape, _column_sum_bound(case.degree, dtype)),
            why=(
                "an entry is a chain of at most `degree` insertions, each committing four "
                "roundings -- the complement, the two products and their sum -- against an "
                "exact convex combination of values in [0, 1], so it carries at most "
                "gamma_{4 degree}; summing degree + 1 of them against an exact total of one "
                "adds gamma_{degree}. First order in u that is (4 degree (degree + 1) + "
                "degree) u"
            ),
        ),
        context=f"{case.label} in {np.dtype(dtype).name} on {backend.name}",
    )
    assert not np.any(operators < 0.0), (
        "an entry is a product of convex combinations and cannot be negative"
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_partition_of_unity_check_is_not_vacuous(dtype: npt.DTypeLike) -> None:
    """The column-sum bound is compared against a nonzero error somewhere.

    ``design/backend_parity.md``'s rule the hard way: a bound compared only against
    zero has not been checked. A dyadic knot vector rounds nothing, so its column
    sums are exactly one; a non-dyadic one does not, and this pins that at least one
    case in the table reaches a nonzero deviation and that the bound is above it.

    Args:
        dtype (npt.DTypeLike): Storage format.
    """
    worst = 0.0
    for case in _CASES:
        if not case.accuracy or case.degree < 1:
            continue
        operators = _build(case, dtype, Backend.PYTHON)
        deviation = np.abs(operators.sum(axis=1).astype(np.float64) - 1.0)
        worst = max(worst, float(deviation.max(initial=0.0)))
        assert worst <= max(_column_sum_bound(c.degree, dtype) for c in _CASES if c.accuracy)
    assert worst > 0.0, (
        "every case in the table has exactly-summing columns, so the derived bound is "
        "only ever compared against zero and asserts nothing"
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("backend", _BACKENDS)
def test_a_strided_out_is_filled(backend: Backend, dtype: npt.DTypeLike) -> None:
    """A caller's non-contiguous ``out`` comes back filled, under either backend.

    The binding declares ``out`` C-contiguous, so the adapter in
    :mod:`pantr.bspline._extraction_backend` computes into a fresh buffer and copies
    back. Nothing else in this file reaches that path, and a silently unfilled
    ``out`` would look like a wrong answer far from its cause.

    Args:
        backend (Backend): Which implementation to run.
        dtype (npt.DTypeLike): Storage format.
    """
    _demand_the_extension_if_needed(backend)
    case = _CASES[0]
    expected = _build(case, dtype, backend)
    # Every other slice of a doubled leading axis: C-contiguous in the last two
    # axes and strided in the first, which is what the adapter has to absorb.
    canvas = np.full((2 * expected.shape[0], *expected.shape[1:]), np.nan, dtype=dtype)
    strided = canvas[::2]
    assert not strided.flags["C_CONTIGUOUS"]
    knots = np.asarray(case.knots, dtype=dtype)
    space = BsplineSpace1D(knots, case.degree, snap_knots=False)
    with use_backend(backend):
        returned = _tabulate_Bspline_Bezier_1D_extraction_impl(
            space.knots, case.degree, space.tolerance, out=strided
        )
    assert returned is strided
    assert np.array_equal(strided, expected), "the strided out was not filled with the answer"


def test_the_claim_is_not_vacuous(cpp_backend: None) -> None:
    """A bitwise claim asserts nothing unless the two paths could have differed.

    The specific risk: if the catalogue handed back the Numba kernel for both
    backends, every assertion above would pass for the wrong reason. Identity is
    taken against the catalogue's own table rather than against a kernel's ``repr``
    or ``py_func``, neither of which exists under ``NUMBA_DISABLE_JIT=1`` -- Rule
    12's shape, and what a first version of the sibling assertion got wrong.
    """
    python_builder = bezier_extraction_kernel(Backend.PYTHON)
    cpp_builder = bezier_extraction_kernel(Backend.CPP)
    assert python_builder is not cpp_builder
    python_mask = bezier_identity_mask_kernel(Backend.PYTHON)
    cpp_mask = bezier_identity_mask_kernel(Backend.CPP)
    assert python_mask is not cpp_mask

    # The apply catalogue is the sibling table; the builders must not have landed in
    # it, since the two are separate C++ registrations with separate claims.
    assert python_builder not in _KERNELS.values()
    assert cpp_builder not in _KERNELS.values()


def test_the_repeated_first_knot_case_separates_the_two_multiplicities(
    cpp_backend: None,
) -> None:
    """The parity-only case really is the one that discriminates the boundary count.

    ``design/extraction_port.md``'s 2026-09-01 amendment says the boundary
    multiplicity the builder opens with is the front entry of
    ``multiplicity_in_domain()``. It is not: they are different computations over
    different index ranges. This pins that the table holds a vector where they
    differ, so a port that read the class multiplicity would fail
    :func:`test_the_operators_agree` on it -- and that a clamped vector is where the
    two agree, which is why the confusion survived.
    """
    repeated = next(entry for entry in _CASES if entry.label == "repeated first in-domain knot")
    knots = np.asarray(repeated.knots, dtype=np.float64)
    space = BsplineSpace1D(knots, repeated.degree, snap_knots=False)
    multiplicity = space.get_unique_knots_and_multiplicity(in_domain=True)[1]
    boundary = int(
        np.count_nonzero(
            np.abs(knots[: repeated.degree + 1] - knots[repeated.degree]) <= space.tolerance
        )
    )
    assert boundary != int(multiplicity[0]), (
        "the parity table no longer holds a vector separating the boundary "
        "multiplicity from the first in-domain class multiplicity, so nothing here "
        "would notice a port that confused the two"
    )

    clamped = next(entry for entry in _CASES if entry.label == "clamped uniform quadratic")
    clamped_knots = np.asarray(clamped.knots, dtype=np.float64)
    clamped_space = BsplineSpace1D(clamped_knots, clamped.degree, snap_knots=False)
    clamped_multiplicity = clamped_space.get_unique_knots_and_multiplicity(in_domain=True)[1]
    clamped_boundary = int(
        np.count_nonzero(
            np.abs(clamped_knots[: clamped.degree + 1] - clamped_knots[clamped.degree])
            <= clamped_space.tolerance
        )
    )
    assert clamped_boundary == int(clamped_multiplicity[0])


def _draw(rng: np.random.Generator, dtype: npt.DTypeLike) -> _Case:
    """Draw one random knot vector the shared algorithm handles.

    The draw stays inside the family where the sliding window is aligned: either the
    left end is clamped with multiplicity exactly ``degree + 1``, or the leading
    knots are strictly increasing. Outside it the algorithm mishandles the vector on
    both sides, which is a defect rather than a parity question, and the module
    docstring says so.

    Args:
        rng (np.random.Generator): The source of randomness.
        dtype (npt.DTypeLike): Storage format.

    Returns:
        _Case: The drawn vector, marked as safe for the accuracy oracles.
    """
    degree = int(rng.integers(0, 5))
    n_intervals = int(rng.integers(1, 6))
    # Strictly increasing breakpoints, well separated against the knot tolerance.
    gaps = rng.uniform(0.4, 2.0, size=n_intervals)
    breaks = np.concatenate(([0.0], np.cumsum(gaps)))
    clamped = bool(rng.integers(0, 2))

    interior: list[float] = []
    for value in breaks[1:-1]:
        multiplicity = int(rng.integers(1, degree + 1)) if degree >= 1 else 1
        interior.extend([float(value)] * multiplicity)

    if clamped:
        clamp = degree + 1
        knots = [
            *[float(breaks[0])] * clamp,
            *interior,
            *[float(breaks[-1])] * clamp,
        ]
    else:
        # Strictly increasing outside the domain, so `knots[degree]` is the last
        # knot of its class and the window stays aligned.
        step = float(gaps[0])
        left = [float(breaks[0]) - step * (degree - i) for i in range(degree)]
        right = [float(breaks[-1]) + step * (i + 1) for i in range(degree)]
        knots = [*left, float(breaks[0]), *interior, float(breaks[-1]), *right]

    label = f"drawn degree {degree}, {n_intervals} intervals, {'clamped' if clamped else 'open'}"
    case = _Case(label, knots, degree, True)
    # A draw that the space itself refuses is not a case; redraw deterministically by
    # falling back to the clamped uniform form, which is always legal.
    try:
        BsplineSpace1D(np.asarray(knots, dtype=dtype), degree, snap_knots=False)
    except ValueError:  # pragma: no cover  (the construction above is legal by design)
        uniform = [0.0] * (degree + 1) + [float(i + 1) for i in range(n_intervals - 1)]
        uniform += [float(n_intervals)] * (degree + 1)
        case = _Case(label + ", replaced", uniform, degree, True)
    return case


@pytest.mark.slow
@pytest.mark.parametrize("dtype", DTYPES)
def test_the_claim_holds_over_a_sweep_ten_times_the_shipped_one(
    cpp_backend: None, dtype: npt.DTypeLike
) -> None:
    """A bound checked only by the sweep that ships with it has not been checked.

    The shipped parametrization is eleven knot vectors per dtype. This draws 110
    random ones per dtype, from the family the algorithm handles, and asserts the
    parity claim, the identity mask and the partition of unity on each.

    Args:
        cpp_backend (None): Requires the compiled extension.
        dtype (npt.DTypeLike): Storage format.
    """
    draws = 10 * len(_CASES)
    checked = 0
    worst_column_sum = 0.0
    for draw in range(draws):
        rng = np.random.default_rng(700_000 + 13 * draw)
        case = _draw(rng, dtype)
        reference = _build(case, dtype, Backend.PYTHON)
        actual = _build(case, dtype, Backend.CPP)
        context = f"{case.label} in {np.dtype(dtype).name} (draw {draw})"
        assert_parity(actual, reference, _claim(case, dtype), context=context)

        space = BsplineSpace1D(np.asarray(case.knots, dtype=dtype), case.degree, snap_knots=False)
        with use_backend(Backend.PYTHON):
            mask_reference = _bezier_structural_identity_mask(space)
        with use_backend(Backend.CPP):
            mask_actual = _bezier_structural_identity_mask(space)
        assert np.array_equal(mask_actual, mask_reference), f"{context}: the masks differ"

        column_sums = actual.sum(axis=1).astype(np.float64)
        deviation = float(np.abs(column_sums - 1.0).max(initial=0.0))
        assert deviation <= _column_sum_bound(case.degree, dtype), (
            f"{context}: a column sums to one only within {deviation}"
        )
        worst_column_sum = max(worst_column_sum, deviation)
        checked += 1

    assert checked == draws, f"the sweep ran {checked} cases, expected {draws}"
    assert worst_column_sum > 0.0, (
        "no drawn case rounded at all, so the partition-of-unity bound was compared "
        "against zero throughout the sweep"
    )
