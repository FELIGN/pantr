"""Parity for `Bspline.reverse`, `.permute_directions` and `.transform` across backends.

The three operations split into **two regimes, and the split is not by method** --
``reverse`` spans both.

Exact, asserted bitwise
-----------------------

The **control points** of all three. They are permuted or copied and never
arithmetically combined: the oracle spells them ``np.flip``, ``np.roll``,
``np.transpose``, and for ``transform``'s weight column a plain assignment. A tolerance
on these would hide exactly the defect the test exists to catch -- a wrong stride, a
transposed axis, a lost roll -- every one of which is visible bitwise and invisible
under any bound wide enough to be called a rounding budget.

``permute_directions``' **knot vectors** too, which are only reordered.

And **``reverse``'s reflected knot vector**, which is where this file departs from the
ticket that asked for it.
FELIGN/pantr#495's AC3 classifies that vector under the derived-bound regime, on the
correct observation that ``new_knots = (a + b) - knots[::-1]`` is real arithmetic near
a cancelling difference. It is real arithmetic, and the *accuracy* consequence the
ticket draws is right: the reflected value's error against the exact reflection is
bounded relative to the domain magnitude ``|a| + |b|`` and not relative to the
possibly-tiny result. But **accuracy is not what a parity claim measures.** The two
backends evaluate the same expression on the same operands in the same order:

- ``a`` and ``b`` are read from the same knot vector, which the two backends already
  agree on bitwise -- ``tests/parity/test_bspline_space_1d.py`` is what asserts that.
- ``fl(a + b)`` is formed once, in the storage format, on both sides.
- ``fl(sum - k)`` is then one correctly-rounded subtraction per knot.

IEEE-754 addition and subtraction are correctly rounded, so each step has exactly one
right answer. Nothing in the expression is a multiplication, so **no fused multiply-add
can absorb a rounding on one side and not the other** -- which is the usual reason a
bitwise arm has to be conditioned on ``contraction_may_fuse()``, and the reason this one
does not. What would break it is a build that *reassociates*, which needs fast-math,
which ``cpp/include/pantr/bspline/shape.hpp`` and the project's flags both exclude.

So both backends commit the *same* error, which makes the parity claim exact while the
accuracy claim is not. Asserting a bound here instead would assert something weaker than
what holds and would never be approached, which is the vacuity
``design/backend_parity.md`` Rule 3 and AC5's second guard both exist to refuse.
The claim is not left to a measurement taken once and written down here, which nothing
would re-run. :func:`test_the_claims_hold_over_a_ten_times_sweep` asserts
``reverse_knot_diffs == 0`` over the whole sweep on every run, and its message says
plainly that a failure there refutes the argument above rather than asking for a looser
bound.

Derived bound
-------------

**``transform``'s control points**, and only those. The oracle is
``cp @ A.T + b`` (``pantr._transform_control_points._apply_affine_to_control_points``,
shared with ``Bezier.transform``), a matrix product that reaches BLAS, whose summation
order is not reproducible. The budget and its amplification are
``tests/parity/test_bezier_shape.py``'s, restated here rather than imported because the
two files derive it over their own operand -- the same expression, the same helper, the
same derivation.

What is **not** checked here
----------------------------

``in_place=True`` does not dispatch; FELIGN/pantr#495's AC1 scopes the port to the
value-returning form and its invariants keep ``Bspline._mutate`` the single writer. So
under the C++ backend the mutating form of ``transform`` runs the oracle's matrix
product while the value-returning form runs the C++ loop, and the two may differ in the
last bits. :func:`test_the_in_place_form_agrees_with_the_value_form` is where that shows
and it carries the same bound for that reason; the two rearrangements cannot differ at
all and are asserted bitwise there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt
import pytest

from pantr._backend import Backend, use_backend
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
from pantr.bspline._shape_backend import (
    permute_field_directions,
    reverse_field,
    transform_field,
)
from pantr.transform import AffineTransform
from tests._parity_harness import (
    Field,
    Roundings,
    assert_accuracy,
    assert_object_parity,
    assert_parity,
    bitwise_parity,
    bounded_parity,
    derived_accuracy,
    exact_parity,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    _Act: TypeAlias = "Callable[[Bspline], Any]"
    """What a test does to a field it has just built under one backend."""

pytestmark = pytest.mark.usefixtures("cpp_backend")

DTYPES: Final = (np.float64, np.float32)
"""The two storage formats. Both are exercised because the transform's budget is a
function of the format and because the oracle casts the affine map into it."""


class _Vector(NamedTuple):
    """One direction's knot vector, degree and periodicity.

    Attributes:
        label (str): What to call it in a test id.
        degree (int): The polynomial degree.
        knots (tuple[float, ...]): The knot vector, in ``double``.
        periodic (bool): Whether the entries describe a periodic space.
    """

    label: str
    degree: int
    knots: tuple[float, ...]
    periodic: bool


class _Case(NamedTuple):
    """One field to build and act on.

    Attributes:
        label (str): What to call it in a test id.
        vectors (tuple[_Vector, ...]): One per parametric direction.
        rank (int): The number of value components, weight column excluded.
        is_rational (bool): Whether a homogeneous weight column is stored.
    """

    label: str
    vectors: tuple[_Vector, ...]
    rank: int
    is_rational: bool


_UNIT = _Vector("unit", 2, (0.0, 0.0, 0.0, 0.3, 1.0, 1.0, 1.0), False)
"""A clamped cubic-ish vector on the unit interval, with one interior knot."""

_OFFSET = _Vector("offset", 2, (0.1, 0.1, 0.1, 0.3, 0.4, 0.9, 0.9, 0.9), False)
"""Clamped, and deliberately **not symmetric about its own domain midpoint**: a
symmetric vector is its own reflection, so it cannot tell a `reverse` that reflects from
one that leaves the knots alone."""

_LINEAR = _Vector("linear", 1, (0.0, 0.0, 0.25, 0.5, 1.0, 1.0), False)
"""A degree-1 vector, so a direction whose reversal moves few coefficients."""

_PERIODIC = _Vector("periodic", 1, (-0.25, 0.0, 0.125, 0.5, 0.75, 1.0, 1.125), True)
"""A periodic direction, the only input for which `reverse` rolls as well as flips.

**Deliberately not uniform.** A uniform periodic vector over a domain like ``[0, 1]`` is
symmetric about its own domain midpoint, so ``(a + b) - knots[::-1]`` reproduces it
exactly and a backend that skipped the reflection entirely would agree with one that
performed it. This vector's interior spans are ``(0.125, 0.375, 0.25, 0.25)``, which
keeps the cyclic ghost structure a periodic space needs while making the reflection
move four of the seven knots. Every entry stays dyadic, so the reflection is exact in
both storage formats and the bitwise claim is still a claim about transcription rather
than about rounding.
"""

_LARGE = _Vector("large", 2, (1e5, 1e5, 1e5, 1.00003e5, 1.0001e5, 1.0001e5, 1.0001e5), False)
"""A domain far from the origin, where `(a + b) - k` cancels hardest and an absolute
tolerance derived at unit scale would be meaningless."""

_NEGATIVE = _Vector("negative", 2, (-7.3, -7.3, -7.3, -2.1, 0.7, 0.7, 0.7), False)
"""A domain straddling zero, where `a + b` is itself a cancelling sum."""

VECTORS: Final = (_UNIT, _OFFSET, _LINEAR, _PERIODIC, _LARGE, _NEGATIVE)
"""Every direction a case is built from."""

CASES: Final = (
    _Case("curve", (_UNIT,), 2, False),
    _Case("offset-curve", (_OFFSET,), 3, False),
    _Case("periodic-curve", (_PERIODIC,), 2, False),
    _Case("large-curve", (_LARGE,), 2, False),
    _Case("negative-curve", (_NEGATIVE,), 2, False),
    _Case("rational-curve", (_UNIT,), 2, True),
    _Case("surface", (_UNIT, _LINEAR), 2, False),
    _Case("rational-surface", (_OFFSET, _LINEAR), 3, True),
    _Case("mixed-volume", (_UNIT, _LINEAR, _OFFSET), 3, False),
    _Case("scalar-curve", (_OFFSET,), 1, False),
    _Case("scalar-surface", (_UNIT, _LINEAR), 1, False),
)
"""The shipped comparison set. `mixed-volume`'s three directions have different basis
counts and different degrees, which is what makes a transposed stride visible."""


def _build(case: _Case, dtype: npt.DTypeLike) -> Bspline:
    """Build ``case``'s field under whichever backend is active.

    The control net is a fixed pseudo-random draw rather than a ramp: a ramp is a
    separable, monotone net, and a permutation of one can agree with the wrong
    permutation on a symmetric shape. The weight column is pushed positive because a
    rational field's weights must be.

    Args:
        case (_Case): The field to build.
        dtype (npt.DTypeLike): The storage format.

    Returns:
        ~pantr.bspline.Bspline: The field.
    """
    spaces = [
        BsplineSpace1D(np.asarray(v.knots, dtype=dtype), v.degree, periodic=v.periodic)
        for v in case.vectors
    ]
    space = BsplineSpace(spaces)
    components = case.rank + (1 if case.is_rational else 0)
    rng = np.random.default_rng(20260917)
    net = rng.standard_normal((*space.num_basis, components)).astype(dtype)
    if case.is_rational:
        # A Python float is a weak scalar under NEP 50, so this addition stays in the
        # net's own format and the weight column keeps the storage width it was built
        # with. Writing ``dtype(0.5)`` would mean calling a ``DTypeLike``, which is not
        # a constructor.
        net[..., -1] = np.abs(net[..., -1]) + 0.5
    return Bspline(space, net, is_rational=case.is_rational)


def _both(case: _Case, dtype: npt.DTypeLike, act: _Act) -> tuple[Any, Any]:
    """Build and act on ``case``'s field once under each backend.

    Everything happens inside the ``use_backend`` block, the operation included: the
    branch :mod:`pantr.bspline._shape_backend` takes is decided by the *active* backend,
    so acting outside the block would measure the wrong path.

    Args:
        case (_Case): The field to build.
        dtype (npt.DTypeLike): The storage format.
        act (_Act): What to do to the field.

    Returns:
        tuple[Any, Any]: ``(py, cpp)``, in the order
        :func:`~tests._parity_harness.assert_object_parity` names its arguments.
    """
    with use_backend(Backend.PYTHON):
        py = act(_build(case, dtype))
    with use_backend(Backend.CPP):
        cpp = act(_build(case, dtype))
    return py, cpp


_AFFINE_2D: Final = AffineTransform(np.array([[0.3, -1.7], [2.9, 0.11]]), np.array([5.5, -0.25]))
"""A general, non-symmetric, non-dyadic map, so the product actually rounds."""

_AFFINE_1D: Final = AffineTransform(np.array([[-0.7]]), np.array([2.25]))
"""The rank-1 twin, for a scalar-valued field. Scalar B-splines are the common case in
this library and `_affine_for` could not reach one before."""

_AFFINE_3D: Final = AffineTransform(
    np.array([[0.3, -1.7, 0.02], [2.9, 0.11, -4.0], [1.25, 0.9, 0.31]]),
    np.array([5.5, -0.25, 11.0]),
)
"""The rank-3 twin of :data:`_AFFINE_2D`."""


def _affine_for(rank: int) -> AffineTransform:
    """The general map of the right rank.

    Args:
        rank (int): The field's geometric rank, 1, 2 or 3.

    Returns:
        ~pantr.transform.AffineTransform: The map.
    """
    return {1: _AFFINE_1D, 2: _AFFINE_2D}.get(rank, _AFFINE_3D)


# The three operations as factories rather than as lambdas written at each call site.
# A lambda closing over a loop variable is the shape that reads correctly today and
# silently reads the last iteration's value the day a call is deferred, which is what
# ruff's B023 is about; a factory binds its argument at definition and cannot.


def _reversing(direction: int, *, in_place: bool = False) -> _Act:
    """An action that reverses one direction.

    Args:
        direction (int): The direction to reverse.
        in_place (bool): Whether to mutate the receiver instead of deriving.

    Returns:
        _Act: The action.
    """

    def act(field: Bspline) -> Any:
        return field.reverse(direction, in_place=in_place)  # type: ignore[call-overload]

    return act


def _permuting(permutation: Sequence[int], *, in_place: bool = False) -> _Act:
    """An action that reorders the directions.

    Args:
        permutation (Sequence[int]): A permutation of ``range(dim)``.
        in_place (bool): Whether to mutate the receiver instead of deriving.

    Returns:
        _Act: The action.
    """

    def act(field: Bspline) -> Any:
        return field.permute_directions(permutation, in_place=in_place)  # type: ignore[call-overload]

    return act


def _transforming(affine: AffineTransform, *, in_place: bool = False) -> _Act:
    """An action that applies an affine map.

    Args:
        affine (~pantr.transform.AffineTransform): The map to apply.
        in_place (bool): Whether to mutate the receiver instead of deriving.

    Returns:
        _Act: The action.
    """

    def act(field: Bspline) -> Any:
        return field.transform(affine, in_place=in_place)  # type: ignore[call-overload]

    return act


# ---------------------------------------------------------------------------
# The claims
# ---------------------------------------------------------------------------

_REARRANGEMENT_WHY: Final = (
    "the control points of all three shape operations are moved or copied and never "
    "arithmetically combined: the oracle spells them np.flip, np.roll, np.transpose, "
    "and transform's weight column a plain assignment. There is no arithmetic for a "
    "fused multiply-add to change, so this holds on any build and at any storage "
    "format by construction rather than by measurement. A difference could only be a "
    "wrong stride, a transposed axis, a lost roll or a narrowing cast -- every one of "
    "them visible bitwise and invisible under any bound wide enough to be called a "
    "rounding budget"
)

_REFLECTED_KNOTS_WHY: Final = (
    "reverse's reflected knot vector, which IS arithmetic -- the oracle writes it "
    "(a + b) - knots[::-1] -- and is still exact ACROSS the two backends. Both read a "
    "and b from a knot vector the two already agree on bitwise, both form fl(a + b) "
    "once in the storage format, and both then commit one correctly-rounded "
    "subtraction per knot. IEEE-754 addition and subtraction each have exactly one "
    "right answer, and the expression contains no multiplication, so no fused "
    "multiply-add can absorb a rounding on one side and not the other -- which is why "
    "this bitwise arm needs no contraction gate. Only reassociation could break it, "
    "and that needs fast-math, which the project's flags exclude. This says nothing "
    "about ACCURACY: (a + b) - k cancels near either domain end, so the error against "
    "the exact reflection is bounded relative to |a| + |b| and not relative to the "
    "result. Both backends make that same error, which is what makes parity exact "
    "while accuracy is not"
)

_PERMUTED_KNOTS_WHY: Final = (
    "permute_directions reorders the directions and computes nothing at all, so each "
    "resulting knot vector is a direction's own vector carried across unchanged. A "
    "difference could only be a lost entry or a direction read from the wrong slot, "
    "and the second is exactly what a bound would hide: two directions' vectors can "
    "be close enough to pass a tolerance and still be the wrong one"
)

_TRANSFORM_WHY: Final = (
    "the oracle is cp @ A.T + b, a matrix product that reaches BLAS, so its summation "
    "order is not reproducible and there is no bitwise arm to condition on. The budget "
    "is Higham's dot-product result: n roundings for a length-n inner product, plus "
    "one for the translation, so gamma_(n+1) -- at the STORAGE width, because the "
    "oracle casts the matrix to the control points' dtype before multiplying rather "
    "than after, which is what pantr/bspline/shape.hpp transcribes. The amplification "
    "is the row action |cp| @ |A.T| + |b|, which is the reachable magnitude because it "
    "is the same expression on absolute values. A fused multiply-add on either side "
    "only removes a rounding, so the bound holds whether or not the build contracts "
    "and no contraction gate is needed. This is tests/parity/test_bezier_shape.py's "
    "derivation over the same helper and the same expression"
)


def _permutations_of(dim: int) -> tuple[list[int], ...]:
    """The permutations a test should try on a field of ``dim`` directions.

    The cycle alone is not enough. On two directions it *is* a transposition and is its
    own inverse, so it cannot tell a permutation from its inverse; on three it has no
    fixed point at all, so nothing ever exercises "one direction stays put while two
    move" -- which is the arrangement a catalogue that reused the prior wrapper list
    positionally would get wrong in the least visible way. The identity is included for
    the same reason in reverse: it must be a complete no-op and nothing else tests that.

    Args:
        dim (int): The number of parametric directions.

    Returns:
        tuple[list[int], ...]: The identity, the forward cycle, and -- from three
        directions up -- a transposition that fixes the last direction.
    """
    identity = list(range(dim))
    if dim == 1:
        return (identity,)
    cycle = [*range(1, dim), 0]
    if dim == 2:
        return (identity, cycle)
    transposition = [1, 0, *range(2, dim)]
    return (identity, cycle, transposition)


def _transform_roundings(rank: int) -> Roundings:
    """The budget one affine map of a rank-``n`` point commits.

    Args:
        rank (int): The geometric rank ``n``.

    Returns:
        Roundings: ``n`` roundings for the inner product and one for the translation.
    """
    return Roundings(stages=rank + 1, accumulator_per_stage=1, storage_per_stage=0)


def _transform_amplification(
    net: npt.NDArray[np.floating[Any]], affine: AffineTransform, *, is_rational: bool
) -> npt.NDArray[np.float64]:
    """The reachable magnitude of ``cp @ A.T + b``, elementwise.

    The same expression as the oracle with every operand replaced by its absolute
    value, which is what makes it a bound on the result's magnitude rather than an
    estimate of it. For a rational field the translation is scaled by the stored
    weight, because the oracle's homogeneous form is ``A (w x) + w b``.

    Args:
        net (npt.NDArray[np.floating[Any]]): The source control points.
        affine (~pantr.transform.AffineTransform): The map.
        is_rational (bool): Whether the last component is a weight.

    Returns:
        npt.NDArray[np.float64]: One amplification per transformed coordinate.
    """
    coefficients = np.asarray(net, dtype=np.float64)
    rank = affine.matrix.shape[0]
    coordinates = coefficients[..., :rank]
    weight = np.abs(coefficients[..., rank : rank + 1]) if is_rational else 1.0
    linear = np.asarray(np.abs(coordinates) @ np.abs(affine.matrix).T, dtype=np.float64)
    return np.asarray(linear + (weight * np.abs(affine.offset)), dtype=np.float64)


def _shared_fields(dim: int) -> tuple[Field, ...]:
    """The pieces of a result every one of the three operations must agree on.

    Args:
        dim (int): The result's number of parametric directions.

    Returns:
        tuple[Field, ...]: The field list, knot vectors excluded -- those differ per
        operation and each test adds its own.
    """
    per_direction: list[Field] = []
    for direction in range(dim):
        per_direction.append(
            Field(
                f"space.spaces[{direction}].tolerance",
                bitwise_parity(
                    why=(
                        "each resulting space derives its own tolerance from its own "
                        "knot vector, so it is a new quantity of the result rather "
                        "than one carried over, and it is what every later comparison "
                        "on that space will use. The knots agree bit for bit and "
                        "knot_tolerance is the same reduction over them on both sides. "
                        "A difference would mean the two backends disagree about which "
                        "knots are the same knot, which no comparison of the knots "
                        "themselves would reveal"
                    )
                ),
                read=lambda field, d=direction: field.space.spaces[d].tolerance,  # type: ignore[misc]
            )
        )
        per_direction.append(
            Field(
                f"space.spaces[{direction}].periodic",
                exact_parity(
                    why=(
                        "periodicity is a flag and all three operations carry it "
                        "structurally: reverse keeps the reversed direction's, "
                        "permute_directions moves each direction's with it, and "
                        "transform touches no space at all. A difference would be a "
                        "different KIND of space, which no tolerance covers"
                    )
                ),
                read=lambda field, d=direction: field.space.spaces[d].periodic,  # type: ignore[misc]
            )
        )
    return (
        Field(
            "dim",
            exact_parity(
                why=(
                    "the parametric dimension is a count, and none of the three may "
                    "change it. It must be asserted separately from the coefficients, "
                    "which compare elementwise and cannot see two results of different "
                    "shape"
                )
            ),
        ),
        Field(
            "space.num_basis",
            exact_parity(
                why=(
                    "the basis counts are integers derived from the resulting knot "
                    "vectors and degrees. They are what a permutation applied in the "
                    "opposite sense moves first, and design/backend_parity.md Rule 11 "
                    "is why a count is asserted on its own rather than folded into a "
                    "bounded comparison"
                )
            ),
        ),
        Field(
            "degree",
            exact_parity(
                why=(
                    "the degree of each direction is carried, never recomputed, and "
                    "permute_directions is the one operation here that reorders the "
                    "tuple -- so this is what catches a permutation applied to the net "
                    "but not to the space"
                )
            ),
        ),
        Field(
            "rank",
            exact_parity(
                why=(
                    "the rank is the component count less the weight column. None of "
                    "the three touches the component axis, so a difference would mean "
                    "one backend dropped or projected a component"
                )
            ),
        ),
        Field(
            "is_rational",
            exact_parity(
                why=(
                    "the rationality flag is the invariant an unpack-and-reassemble "
                    "across the seam would lose, which is why the field crosses rather "
                    "than its arrays. design/cross_backend_types.md names it"
                )
            ),
        ),
        *per_direction,
    )


def _knot_fields(dim: int, why: str) -> tuple[Field, ...]:
    """One bitwise knot-vector field per direction.

    Args:
        dim (int): The result's number of parametric directions.
        why (str): The derivation for this operation's knot vectors.

    Returns:
        tuple[Field, ...]: The fields.
    """
    return tuple(
        Field(
            f"space.spaces[{d}].knots",
            bitwise_parity(why=why),
            read=lambda field, direction=d: field.space.spaces[direction].knots,  # type: ignore[misc]
        )
        for d in range(dim)
    )


# ---------------------------------------------------------------------------
# AC1 -- the three methods dispatch
# ---------------------------------------------------------------------------

_DISPATCH_CASE: Final = _Case("dispatch", (_UNIT, _LINEAR), 2, False)
"""A two-direction field, so a permutation has something to permute."""


@pytest.mark.parametrize(
    ("binding", "call"),
    [
        ("reverse_bspline", lambda f: f.reverse(0)),
        ("permute_bspline_directions", lambda f: f.permute_directions([1, 0])),
        (
            "transform_bspline",
            lambda f: f.transform(AffineTransform.translation([1.0, 2.0])),
        ),
    ],
    ids=["reverse", "permute_directions", "transform"],
)
def test_the_method_reaches_its_own_binding(
    monkeypatch: pytest.MonkeyPatch, binding: str, call: Callable[[Bspline], Bspline]
) -> None:
    """Each method's value-returning form goes through its catalogue entry.

    The binding is replaced by one that raises a sentinel, and the method is required to
    raise it. **That is what makes this test fail if the catalogue entry is removed**: a
    method that fell back to the oracle would return a perfectly good field and every
    value assertion in this file would still pass. Asserting the result cannot see the
    difference; asserting the route can.

    Args:
        monkeypatch (pytest.MonkeyPatch): Used to replace the binding.
        binding (str): The ``pantr._pantr_cpp`` name the method must reach.
        call (Callable): The method call under test.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    sentinel = RuntimeError(f"{binding} was reached")

    def _refuse(*args: object, **kwargs: object) -> object:
        raise sentinel

    monkeypatch.setattr(_pantr_cpp, binding, _refuse)
    with use_backend(Backend.CPP):
        field = _build(_DISPATCH_CASE, np.float64)
        with pytest.raises(RuntimeError) as raised:
            call(field)
    assert raised.value is sentinel


@pytest.mark.parametrize(
    ("entry", "call"),
    [
        (reverse_field, lambda f: reverse_field(f, 0)),
        (permute_field_directions, lambda f: permute_field_directions(f, [1, 0])),
        (
            transform_field,
            lambda f: transform_field(f, AffineTransform.translation([1.0, 2.0])),
        ),
    ],
    ids=["reverse", "permute_directions", "transform"],
)
def test_the_catalogue_entry_stays_on_the_oracle_under_the_python_backend(
    entry: object, call: Callable[[Bspline], Bspline]
) -> None:
    """The same entry point runs the oracle when the Python backend is active.

    The other half of the dispatch claim, and the one that catches a catalogue that
    routes to C++ unconditionally: ``design/cross_backend_types.md``'s never-fall-back
    rule runs in both directions.

    Args:
        entry (object): The catalogue entry, named only so the id reads.
        call (Callable): The call under test.
    """
    del entry
    with use_backend(Backend.PYTHON):
        field = _build(_DISPATCH_CASE, np.float64)
        result = call(field)
    assert type(result._impl) is type(field._impl)


# ---------------------------------------------------------------------------
# AC2 -- in_place under both backends
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES, ids=["float64", "float32"])
def test_the_in_place_form_mutates_the_receiver_under_both_backends(
    dtype: npt.DTypeLike,
) -> None:
    """Each ``in_place=True`` call returns ``None`` and leaves the receiver changed.

    The receiver's own identity is asserted too, which is the part
    ``Bspline._mutate``'s C++ branch could break: it replaces the *implementation*
    wholesale, and a wrapper rebuilt rather than reseated would be a different object
    while every value agreed. Where the array itself landed is deliberately **not**
    asserted -- ``tests/test_transform.py`` records that as backend-dependent and not a
    contract.

    Args:
        dtype (npt.DTypeLike): The storage format.
    """
    case = _Case("in-place", (_UNIT, _LINEAR), 2, False)
    mutations: tuple[_Act, ...] = (
        _reversing(0, in_place=True),
        _permuting([1, 0], in_place=True),
        _transforming(_AFFINE_2D, in_place=True),
    )
    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            for mutate in mutations:
                receiver = _build(case, dtype)
                identity = id(receiver)
                before = np.array(receiver.control_points)
                assert mutate(receiver) is None, (
                    f"an in_place=True call returned something under {backend.name}; "
                    f"the overloads say it returns None and a caller assigning the "
                    f"result would silently get it"
                )
                after = np.array(receiver.control_points)
                assert id(receiver) == identity, (
                    f"an in_place=True call replaced the receiver under "
                    f"{backend.name} instead of mutating it"
                )
                assert before.shape != after.shape or not np.array_equal(before, after), (
                    f"an in_place=True call left the receiver unchanged under "
                    f"{backend.name}, so it returned None and did nothing"
                )


@pytest.mark.parametrize("case", CASES, ids=[c.label for c in CASES])
@pytest.mark.parametrize("dtype", DTYPES, ids=["float64", "float32"])
def test_the_in_place_form_agrees_with_the_value_form(case: _Case, dtype: npt.DTypeLike) -> None:
    """Mutating and deriving leave the same value, on each backend separately.

    This is a self-comparison rather than a parity check, and it is here because the
    mutating form deliberately does **not** dispatch: under the C++ backend it runs the
    oracle's arrays while the value form runs the C++ loop. For the two rearrangements
    that is bitwise, because neither computes anything. For ``transform`` it is the
    derived bound, because one side is a BLAS product and the other is a loop.

    Args:
        case (_Case): The field to build.
        dtype (npt.DTypeLike): The storage format.
    """
    dim = len(case.vectors)
    permutation = [*range(1, dim), 0]
    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            table: tuple[tuple[str, _Act, _Act], ...] = (
                ("reverse", _reversing(0), _reversing(0, in_place=True)),
                (
                    "permute_directions",
                    _permuting(permutation),
                    _permuting(permutation, in_place=True),
                ),
                (
                    "transform",
                    _transforming(_affine_for(case.rank)),
                    _transforming(_affine_for(case.rank), in_place=True),
                ),
            )
            for label, derive, mutate in table:
                derived = derive(_build(case, dtype))
                receiver = _build(case, dtype)
                mutate(receiver)
                context = f"{label} in place against derived, {case.label} {backend.name}"
                if label == "transform":
                    # The coordinate block only: the weight column is copied on both
                    # paths, so it is asserted exactly rather than inside a budget
                    # derived for a matrix product it never went through.
                    rank = case.rank
                    assert_parity(
                        np.asarray(receiver.control_points)[..., :rank],
                        np.asarray(derived.control_points)[..., :rank],
                        bounded_parity(
                            roundings=_transform_roundings(rank),
                            accumulator=dtype,
                            storage=dtype,
                            amplification=_transform_amplification(
                                _build(case, dtype).control_points,
                                _affine_for(rank),
                                is_rational=case.is_rational,
                            ),
                            why=_TRANSFORM_WHY,
                        ),
                        context=context,
                    )
                    if case.is_rational:
                        np.testing.assert_array_equal(
                            np.asarray(receiver.control_points)[..., rank],
                            np.asarray(derived.control_points)[..., rank],
                            err_msg=f"{context}: the weight column is copied, not computed",
                        )
                else:
                    np.testing.assert_array_equal(
                        np.asarray(receiver.control_points),
                        np.asarray(derived.control_points),
                        err_msg=context,
                    )


# ---------------------------------------------------------------------------
# AC3 -- parity per method
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=[c.label for c in CASES])
@pytest.mark.parametrize("dtype", DTYPES, ids=["float64", "float32"])
def test_reverse_agrees_across_the_backends(case: _Case, dtype: npt.DTypeLike) -> None:
    """``reverse``'s control points and reflected knot vector are both bitwise.

    See the module docstring for why the reflected vector is claimed bitwise where
    FELIGN/pantr#495 asked for a bound, and what that claim does and does not say.

    Args:
        case (_Case): The field to build.
        dtype (npt.DTypeLike): The storage format.
    """
    dim = len(case.vectors)
    for direction in range(dim):
        py, cpp = _both(case, dtype, _reversing(direction))
        fields = (
            Field("control_points", bitwise_parity(why=_REARRANGEMENT_WHY)),
            Field(
                f"space.spaces[{direction}].knots",
                bitwise_parity(why=_REFLECTED_KNOTS_WHY),
                read=lambda field, d=direction: field.space.spaces[d].knots,  # type: ignore[misc]
            ),
            *_knot_fields(dim, _PERMUTED_KNOTS_WHY)[direction + 1 :],
            *_shared_fields(dim),
        )
        assert_object_parity(
            py=py,
            cpp=cpp,
            fields=fields,
            context=f"reverse({direction}) of {case.label} at {np.dtype(dtype).name}",
        )


@pytest.mark.parametrize("case", CASES, ids=[c.label for c in CASES])
@pytest.mark.parametrize("dtype", DTYPES, ids=["float64", "float32"])
def test_permute_directions_agrees_across_the_backends(case: _Case, dtype: npt.DTypeLike) -> None:
    """``permute_directions`` is bitwise in every quantity it produces.

    A cyclic permutation is used rather than a reversal of the axis order, because on a
    two-direction field the two coincide and on a three-direction one they do not: the
    cycle is what distinguishes a permutation applied in the opposite sense.

    Args:
        case (_Case): The field to build.
        dtype (npt.DTypeLike): The storage format.
    """
    dim = len(case.vectors)
    permutation = [*range(1, dim), 0]
    py, cpp = _both(case, dtype, _permuting(permutation))
    assert_object_parity(
        py=py,
        cpp=cpp,
        fields=(
            Field("control_points", bitwise_parity(why=_REARRANGEMENT_WHY)),
            *_knot_fields(dim, _PERMUTED_KNOTS_WHY),
            *_shared_fields(dim),
        ),
        context=(f"permute_directions({permutation}) of {case.label} at {np.dtype(dtype).name}"),
    )


@pytest.mark.parametrize("case", CASES, ids=[c.label for c in CASES])
@pytest.mark.parametrize("dtype", DTYPES, ids=["float64", "float32"])
def test_transform_agrees_across_the_backends(case: _Case, dtype: npt.DTypeLike) -> None:
    """``transform``'s coordinates carry the bound; its weights and space do not.

    The weight column is asserted **bitwise and separately**. It is copied rather than
    computed, so folding it into the bounded comparison would let a transform that
    quietly scaled a weight pass inside the coordinates' own budget.

    Args:
        case (_Case): The field to build.
        dtype (npt.DTypeLike): The storage format.
    """
    dim = len(case.vectors)
    affine = _affine_for(case.rank)
    py, cpp = _both(case, dtype, _transforming(affine))
    source = _build(case, dtype).control_points

    rank = case.rank
    assert_parity(
        np.asarray(cpp.control_points)[..., :rank],
        np.asarray(py.control_points)[..., :rank],
        bounded_parity(
            roundings=_transform_roundings(rank),
            accumulator=dtype,
            storage=dtype,
            amplification=_transform_amplification(source, affine, is_rational=case.is_rational),
            why=_TRANSFORM_WHY,
        ),
        context=f"transform of {case.label} at {np.dtype(dtype).name}",
    )
    if case.is_rational:
        np.testing.assert_array_equal(
            np.asarray(cpp.control_points)[..., rank],
            np.asarray(py.control_points)[..., rank],
            err_msg=(
                f"the weight column of {case.label} is copied rather than computed, so "
                f"the two backends must agree on it bitwise"
            ),
        )
    assert_object_parity(
        py=py,
        cpp=cpp,
        fields=(*_knot_fields(dim, _PERMUTED_KNOTS_WHY), *_shared_fields(dim)),
        context=f"transform of {case.label} at {np.dtype(dtype).name}",
    )


@pytest.mark.parametrize("case", CASES, ids=[c.label for c in CASES])
def test_transform_hands_back_the_source_space_object_under_both_backends(
    case: _Case,
) -> None:
    """``s.transform(t).space is s.space``, which no value comparison can see.

    ``design/bspline_ownership_lifetime.md``'s **F6** names this as the stronger of its
    two identity assertions, because a space wrapper rebuilt over the same C++ handle is
    a different Python object while every value agrees.

    Args:
        case (_Case): The field to build.
    """
    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            field = _build(case, np.float64)
            assert field.transform(_affine_for(case.rank)).space is field.space, (
                f"transform rebuilt the space wrapper under {backend.name}"
            )


@pytest.mark.parametrize("case", CASES, ids=[c.label for c in CASES])
def test_the_rearrangements_reuse_every_untouched_direction_wrapper(
    case: _Case,
) -> None:
    """A direction an operation left alone comes back as the object that went in.

    ``reverse`` rebuilds one direction and must carry the rest; ``permute_directions``
    rebuilds none and must carry all of them, **to their new positions**. The second is
    what catches a catalogue that handed ``_wrap_over`` the unpermuted prior list: every
    value would agree and direction ``k`` would be wearing direction ``k``'s old
    wrapper, which is a different direction's.

    Args:
        case (_Case): The field to build.
    """
    dim = len(case.vectors)
    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            field = _build(case, np.float64)
            # Every direction, not just direction 0: a `reverse` that rebuilt the
            # *wrong* direction's space would carry direction 0's handle through
            # correctly and fail only on a later axis.
            for reversed_axis in range(dim):
                result = field.reverse(reversed_axis)
                for d in range(dim):
                    if d == reversed_axis:
                        continue
                    assert result.space.spaces[d] is field.space.spaces[d], (
                        f"reverse({reversed_axis}) rebuilt untouched direction {d} "
                        f"under {backend.name}"
                    )
            for permutation in _permutations_of(dim):
                permuted = field.permute_directions(permutation)
                for k, source in enumerate(permutation):
                    assert permuted.space.spaces[k] is field.space.spaces[source], (
                        f"permute_directions({permutation}) gave new direction {k} the "
                        f"wrapper of the wrong source direction under {backend.name}"
                    )


# ---------------------------------------------------------------------------
# AC4 -- the independent check
# ---------------------------------------------------------------------------

_MAPS_BY_RANK: Final = {
    1: (
        ("negate", np.array([[-1.0]]), np.array([0.0])),
        ("scale-shift", np.array([[0.5]]), np.array([6.0])),
    ),
    2: (
        ("permutation", np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([0.0, 0.0])),
        ("power-of-two", np.array([[4.0, 0.0], [0.0, 0.25]]), np.array([0.0, 0.0])),
        ("integer-shift", np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([3.0, -7.0])),
        ("shear", np.array([[2.0, 0.5], [0.0, 4.0]]), np.array([-3.0, 8.0])),
    ),
    3: (
        (
            "cyclic-permutation",
            np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
            np.array([0.0, 0.0, 0.0]),
        ),
        (
            "lower-triangular",
            np.array([[2.0, 0.0, 0.0], [0.5, 4.0, 0.0], [-1.0, 0.25, 8.0]]),
            np.array([3.0, -7.0, 16.0]),
        ),
    ),
}
"""Exactly-representable maps per geometric rank. Rank 3 was absent entirely until a
test audit found it: a loop bound wrong only at `n = 3` would have been invisible to
this check, and a cross-backend comparison cannot see a fault both backends share.

Each rank carries at least one **non-symmetric** matrix, because a symmetric one is its
own transpose and cannot tell `A` from `A` transposed."""

_EXACT_MAPS: Final = (
    ("permutation", np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([0.0, 0.0])),
    ("power-of-two", np.array([[4.0, 0.0], [0.0, 0.25]]), np.array([0.0, 0.0])),
    ("integer-shift", np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([3.0, -7.0])),
    ("shear", np.array([[2.0, 0.5], [0.0, 4.0]]), np.array([-3.0, 8.0])),
)
"""Maps whose every entry is a power of two, zero or a small integer, so the closed form
below is exact in both storage formats. The shear is non-symmetric on purpose: a
diagonal matrix is its own transpose and cannot tell ``A`` from ``A.T``."""


class _ExactMap(NamedTuple):
    """One exactly-representable affine map and the field shape to apply it to.

    A record rather than a five-tuple, which is this file's own convention for
    :class:`_Vector` and :class:`_Case` and what keeps the test's signature readable.

    Attributes:
        rank (int): The field's geometric rank.
        label (str): What to call it in a test id.
        matrix (npt.NDArray[np.float64]): The ``(rank, rank)`` linear part.
        offset (npt.NDArray[np.float64]): The translation.
        rational (bool): Whether to store a homogeneous weight column.
    """

    rank: int
    label: str
    matrix: npt.NDArray[np.float64]
    offset: npt.NDArray[np.float64]
    rational: bool


_EXACT_CASES: Final = tuple(
    _ExactMap(rank, label, matrix, offset, rational)
    for rank, maps in _MAPS_BY_RANK.items()
    for label, matrix, offset in maps
    for rational in (False, True)
)
"""Every exact map, at every rank, non-rational and rational. The rational arm was
absent until a test audit found it: the weight-scaled translation `w b` is arithmetic no
non-rational case reaches, and it was checked only by the cross-backend comparison,
which cannot see a fault both backends share."""


@pytest.mark.parametrize(
    "exact",
    _EXACT_CASES,
    ids=[f"r{c.rank}-{c.label}{'-w' if c.rational else ''}" for c in _EXACT_CASES],
)
@pytest.mark.parametrize("dtype", DTYPES, ids=["float64", "float32"])
def test_transform_matches_its_closed_form_on_an_exact_map(
    exact: _ExactMap, dtype: npt.DTypeLike
) -> None:
    """An exactly-representable map has a closed form, asserted against it.

    ``design/backend_parity.md`` asks for a check that is independent of the oracle
    rather than a comparison against it, because an error present in both backends is
    invisible to a parity comparison however tight. Every entry of these maps is a power
    of two, zero or a small integer, so ``A x + b`` commits no rounding at all on a net
    of small integers and the admissible deviation is exactly zero rather than a budget.

    For a rational field the closed form is the homogeneous one,
    ``w (A x + b) = A (w x) + w b``, with the weight column copied through. The weights
    are powers of two so the scaled translation stays exact too.

    Args:
        exact (_ExactMap): The map and the field shape to apply it to.
        dtype (npt.DTypeLike): The storage format.
    """
    rank, matrix, offset, rational = exact.rank, exact.matrix, exact.offset, exact.rational
    # Small integers, so every product with a power of two and every sum stays exact.
    count = len(_UNIT.knots) - _UNIT.degree - 1
    components = rank + (1 if rational else 0)
    net = np.arange(1.0, 1.0 + (count * components)).reshape(-1, components).astype(dtype)
    if rational:
        # Powers of two, so `w b` and the weighted coordinates stay exact.
        net[:, rank] = np.asarray([2.0, 0.5, 4.0, 1.0][:count], dtype=dtype)
    affine = AffineTransform(matrix, offset)

    expected = np.empty_like(net, dtype=np.float64)
    for k in range(net.shape[0]):
        weight = float(net[k, rank]) if rational else 1.0
        for i in range(rank):
            expected[k, i] = sum(float(net[k, j]) * float(matrix[i, j]) for j in range(rank)) + (
                weight * float(offset[i])
            )
        if rational:
            expected[k, rank] = weight

    for backend in (Backend.PYTHON, Backend.CPP):
        # The space is built inside the block: it holds the active backend's own
        # implementation, and a field refuses a space built under the other one.
        with use_backend(backend):
            space = BsplineSpace(
                [BsplineSpace1D(np.asarray(_UNIT.knots, dtype=dtype), _UNIT.degree)]
            )
            result = Bspline(space, net, is_rational=rational).transform(affine)
            assert_accuracy(
                np.asarray(result.control_points, dtype=np.float64),
                expected,
                derived_accuracy(
                    bound=np.zeros((), dtype=np.float64),
                    why=(
                        "every entry of this map is a power of two, zero or a small "
                        "integer, and the control points are small integers, so each "
                        "product and each sum is exactly representable in float32 and "
                        "float64 alike. No rounding occurs anywhere in A x + b, and the "
                        "admissible deviation is exactly zero rather than a budget"
                    ),
                ),
                context=(
                    f"rank-{rank} {exact.label} map, rational={rational}, on "
                    f"{np.dtype(dtype).name} under {backend.name}"
                ),
            )


# ---------------------------------------------------------------------------
# AC5 -- the ten-times sweep, with both guards
# ---------------------------------------------------------------------------


def _sweep_vectors() -> tuple[_Vector, ...]:
    """Every direction the sweep draws from, including scaled copies of the shipped set.

    Scaling a knot vector is what makes the sweep reach magnitudes the shipped cases do
    not, which is where an absolute tolerance derived at unit scale would fail.

    Returns:
        tuple[_Vector, ...]: The vectors.
    """
    scaled: list[_Vector] = list(VECTORS)
    for factor in (1e-3, 1e3):
        for vector in (_UNIT, _OFFSET, _LINEAR):
            scaled.append(
                _Vector(
                    f"{vector.label}x{factor:g}",
                    vector.degree,
                    tuple(k * factor for k in vector.knots),
                    vector.periodic,
                )
            )
    return tuple(scaled)


def _sweep_cases() -> tuple[_Case, ...]:
    """The sweep's field set: every vector as a curve, and every pair as a surface.

    Returns:
        tuple[_Case, ...]: The cases.
    """
    vectors = _sweep_vectors()
    cases: list[_Case] = []
    for vector in vectors:
        # Rank 1 is in the sweep as well as in the shipped set: a scalar field is the
        # common case in this library and the transform's inner product degenerates to
        # a single term there, which is its own arm.
        for rank, rational in ((1, False), (2, False), (3, False), (2, True)):
            cases.append(
                _Case(f"{vector.label}-r{rank}{'w' if rational else ''}", (vector,), rank, rational)
            )
    # Every ordered pair from the first eight, so a surface's two directions differ in
    # degree, in basis count and in magnitude, and each appears in both slots -- which
    # is what a transposed stride needs in order to be visible.
    for first in vectors[:8]:
        for second in vectors[:8]:
            cases.append(_Case(f"{first.label}+{second.label}", (first, second), 2, False))
    return tuple(cases)


@pytest.mark.slow
def test_the_claims_hold_over_a_ten_times_sweep() -> None:
    """Every claim holds over more than ten times the shipped parametrization.

    FELIGN/pantr#495's AC5 asks that the bounds be verified over a sweep at least ten
    times the one that ships, on the reasoning that a bound checked only by its own
    shipped sweep has not been checked. **Two guards keep the sweep from passing
    trivially**: it must be ten times the shipped comparison count, *and* the bound must
    actually be **approached** somewhere -- a sweep on which every coefficient is exact
    says nothing about a bound. ``tests/parity/test_bspline_refinement.py`` states the
    second guard and is the pattern copied here.

    The bitwise families carry the count floor only. There is nothing for them to
    approach: a bit pattern either matches or it does not, and the guard that keeps
    *them* honest is that the nets are pseudo-random rather than symmetric, so a wrong
    stride cannot coincide with a right one.
    """
    shipped = len(CASES) * len(DTYPES) * 3  # the three operations
    cases = _sweep_cases()
    swept = len(cases) * len(DTYPES) * 3
    assert swept >= 10 * shipped, (
        f"the sweep is {swept} comparisons against {shipped} shipped, which is under "
        f"the ten-times floor of {10 * shipped}"
    )

    reverse_knot_diffs = 0
    rearrangement_diffs = 0
    worst_ratio = 0.0
    approached = 0
    transform_comparisons = 0

    for case in cases:
        dim = len(case.vectors)
        permutation = [*range(1, dim), 0]
        affine = _affine_for(case.rank)
        for dtype in DTYPES:
            source = _build(case, dtype).control_points

            py, cpp = _both(case, dtype, _reversing(0))
            if not np.array_equal(np.asarray(py.control_points), np.asarray(cpp.control_points)):
                rearrangement_diffs += 1
            if not np.array_equal(
                np.asarray(py.space.spaces[0].knots), np.asarray(cpp.space.spaces[0].knots)
            ):
                reverse_knot_diffs += 1

            py, cpp = _both(case, dtype, _permuting(permutation))
            if not np.array_equal(np.asarray(py.control_points), np.asarray(cpp.control_points)):
                rearrangement_diffs += 1

            py, cpp = _both(case, dtype, _transforming(affine))
            deviation = assert_parity(
                np.asarray(cpp.control_points)[..., : case.rank],
                np.asarray(py.control_points)[..., : case.rank],
                bounded_parity(
                    roundings=_transform_roundings(case.rank),
                    accumulator=dtype,
                    storage=dtype,
                    amplification=_transform_amplification(
                        source, affine, is_rational=case.is_rational
                    ),
                    why=_TRANSFORM_WHY,
                ),
                context=f"sweep: transform of {case.label} at {np.dtype(dtype).name}",
            )
            transform_comparisons += 1
            worst_ratio = max(worst_ratio, deviation.max_ratio_to_bound)
            approached += int(deviation.num_differing > 0)

    assert rearrangement_diffs == 0, (
        f"{rearrangement_diffs} rearrangement comparisons differed bitwise, and a "
        f"rearrangement computes nothing that could round"
    )
    assert reverse_knot_diffs == 0, (
        f"{reverse_knot_diffs} reflected knot vectors differed bitwise. The module "
        f"docstring's argument is that they cannot: same operands, same order, no "
        f"multiplication to contract. A failure here refutes that argument rather than "
        f"asking for a looser bound"
    )
    assert worst_ratio > 0.0, (
        "no transformed coefficient in the whole sweep differed between the backends, "
        "so the derived bound was never asked a question. Add a non-dyadic matrix or a "
        "wider net"
    )
    assert approached > transform_comparisons // 10, (
        f"only {approached} of {transform_comparisons} transform comparisons deviated "
        f"at all, so the sweep is mostly exact arithmetic and is not exercising the "
        f"bound"
    )


# ---------------------------------------------------------------------------
# AC6 -- the refusals
# ---------------------------------------------------------------------------


def _refusal(field: Bspline, call: Callable[[Bspline], object]) -> str:
    """The ``ValueError`` text a call on ``field`` raised.

    The field is a parameter rather than something ``call`` closes over, so no caller
    has to bind a loop variable by hand to keep the closure honest.

    Args:
        field (~pantr.bspline.Bspline): The receiver.
        call (Callable[[Bspline], object]): What to do to it.

    Returns:
        str: The message.
    """
    with pytest.raises(ValueError) as raised:
        call(field)
    return str(raised.value)


@pytest.mark.parametrize("dtype", DTYPES, ids=["float64", "float32"])
def test_the_refusals_read_the_same_under_both_backends(dtype: npt.DTypeLike) -> None:
    """Every Layer 1 refusal of the three methods carries the oracle's own text.

    Compared as **text**, not just as an exception type: each of these checks runs in
    the wrapper above the branch, so a backend that moved one into its own half would
    still raise ``ValueError`` and would say something else. The expected strings are
    written out rather than derived, so that a change to either half has to be made
    here too.

    Args:
        dtype (npt.DTypeLike): The storage format.
    """
    case = _Case("refusals", (_UNIT, _LINEAR), 2, False)
    expected = {
        "reverse-high": "direction must be in [0, 2), got 2.",
        "reverse-low": "direction must be in [0, 2), got -1.",
        "permute": "permutation must be a permutation of range(2), got [0, 0].",
        "transform": (
            "Transform dimension (3) does not match the geometric rank (2) of the control points."
        ),
    }
    seen: dict[str, dict[Backend, str]] = {name: {} for name in expected}
    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            field = _build(case, dtype)
            seen["reverse-high"][backend] = _refusal(field, lambda f: f.reverse(2))
            seen["reverse-low"][backend] = _refusal(field, lambda f: f.reverse(-1))
            seen["permute"][backend] = _refusal(field, lambda f: f.permute_directions([0, 0]))
            seen["transform"][backend] = _refusal(
                field, lambda f: f.transform(AffineTransform.identity(3))
            )

    for name, text in expected.items():
        assert seen[name][Backend.PYTHON] == text, (
            f"{name} under the Python backend reads {seen[name][Backend.PYTHON]!r}"
        )
        assert seen[name][Backend.CPP] == text, (
            f"{name} under the C++ backend reads {seen[name][Backend.CPP]!r}"
        )


def test_a_cross_backend_field_is_refused_rather_than_converted() -> None:
    """A field built under one backend is refused by the other's catalogue.

    ``design/cross_backend_types.md`` forbids converting one implementation into the
    other, so the catalogue raises rather than reaching for a conversion. The refusal is
    a property of taking the C++ route, which is why it fires one way only.
    """
    with use_backend(Backend.PYTHON):
        python_field = _build(_DISPATCH_CASE, np.float64)
    with use_backend(Backend.CPP), pytest.raises(TypeError, match="different backend"):
        reverse_field(python_field, 0)
