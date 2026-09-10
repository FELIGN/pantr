"""Parity of the four structural field operations on a `Bspline`.

`to_open_bspline`, `split`, `slice` and `boundary`.

Like ``tests/parity/test_bspline_refinement.py`` this compares a **computation** rather
than a value the two backends only copy, so the criterion is argued per quantity rather
than assumed once. Unlike that file, two of the three computations carry no new
arithmetic at all, and the third is the first *evaluator* in the port.

## The criterion, quantity by quantity

**Bitwise, for every coefficient and every knot**, and the reason splits three ways.

*The open conversion and the split* reach `inserted_knot_vector`, `oslo_bands_1d` and
`refine_along_axis`, whose bitwise claim ``test_bspline_refinement.py`` already derives
and ``cpp/include/pantr/bspline/refinement.hpp`` carries beside the code. Nothing in
``cpp/include/pantr/bspline/structural.hpp`` re-rounds their output: the trim and the
partition are row selections, and a row selection performs no arithmetic.

*The slice* is the de Boor corner cut, whose statement is
``alpha * R[i + 1] + (1.0 - alpha) * R[i]``. Three arithmetic-width facts decide whether
the two sides can match at all, and each was settled by executing the oracle:

 - ``alpha`` and its knot differences are Python floats, so the weights are computed in
   ``double`` whatever the knots are stored in;
 - the update is a numpy array expression with a *weak* Python float scalar, so under
   NEP 50 both weights are rounded to the storage format and the update runs there. At
   ``float32`` this differs in the last bit from computing the update in ``double``;
 - the span search and the multiplicity count read every knot through ``float(...)``,
   and ``np.searchsorted`` promotes its needle rather than weakening it, so both are
   ``double``.

*The end multiplicities and the split's multiplicity count* go the other way: they are
numpy array expressions, so the ``double`` tolerance is **weakened to the storage
format** before the comparison. That one is a discrete verdict rather than a
displacement -- it decides how many knots are inserted -- so
``design/backend_parity.md`` Rule 11 applies and there is no bound that could absorb a
disagreement. It is not directly asserted here, because a divergence would surface as a
different knot count, which the knot-vector comparison reports as a shape mismatch
rather than as a tolerance failure. **The band where the two spellings disagree is
ulp-wide and no case below constructs one**, which is recorded as a limit of this file
rather than left to be assumed away.

Bitwise is a claim about **this build**, per ``design/backend_parity.md`` Rule 7: the
corner cut's two products and their sum would contract on a target with a fused
multiply-add, and the baseline x86-64 this project compiles for has none.
:func:`~tests._parity_harness.contraction_may_fuse` is the gate, and a fusing build
skips the claim rather than weakening it.

**Exact, for the counts, the degrees, the flags and the dimensions.** A slice's
dimension and a half's basis count are integers; Rule 11 is why they are not folded in
with the coefficients, which compare elementwise and cannot see two results of different
length.

## What this file can check that the refinement file could not

``test_bspline_refinement.py`` records that the natural invariant for knot insertion --
that it must not move the curve -- was unavailable to it, because evaluating a field
needs ``BsplineSpace1D.tabulate_basis`` and that is a separate port.
**`slice_point` is an evaluator that needs no basis function**, so the invariant is
available now, and :func:`test_the_open_conversion_does_not_move_the_curve` and
:func:`test_the_split_does_not_move_the_curve` state it. They are not parity checks: they
compare one backend against *itself* across a representation change, so they would catch
a shared error that every comparison above is blind to, which is the gap
``design/backend_parity.md`` opens by naming.

The closed-form checks that pin the evaluator itself -- partition of unity and linear
precision over the Greville abscissae -- live in ``cpp/tests/test_bspline_structural.cpp``
rather than here, because they are properties of the C++ kernel and need no interpreter.

## The divergence this file pins rather than removes

:meth:`pantr.bspline.Bspline.slice`'s out-of-domain message interpolates a numpy scalar
of the storage format, and the C++ half renders the widened value. The wrapper makes
that check before dispatching, so **the divergence is unreachable** and what is asserted
here is that the *wrapper's* text is identical under both backends --
:func:`test_the_wrapper_refusals_are_identical`. ``pantr.bspline._structural_backend``
and ``cpp/include/pantr/bspline/structural.hpp`` both carry why the repair belongs to
``pantr/core/format.hpp``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt
import pytest

from pantr._backend import Backend, use_backend
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
from pantr.bspline._structural_backend import slice_field, split_field, to_open_field
from tests._parity_harness import (
    Field,
    assert_object_parity,
    assert_parity,
    bitwise_parity,
    contraction_may_fuse,
    exact_parity,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    _Act: TypeAlias = "Callable[[Bspline], Any]"
    """What a test does to a field it has just built under one backend."""

pytestmark = pytest.mark.usefixtures("cpp_backend")

DTYPES: Final = (np.float64, np.float32)
"""The two storage formats, both of which the three width facts above depend on."""

_FUSING: Final = (
    "a build whose contraction may fuse: the corner cut's `wa * next + wb * row` is "
    "exactly the site an FMA would contract, so the bitwise claim is a property of "
    "this build and `design/backend_parity.md` Rule 7 says to skip rather than weaken "
    "it"
)


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
    periodic: bool = False


P1: Final = _Vector("p1", 1, (0.0, 0.0, 0.3, 0.7, 1.0, 1.0))
P2: Final = _Vector("p2", 2, (0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0))
P3: Final = _Vector("p3", 3, (0.0, 0.0, 0.0, 0.0, 0.4, 1.0, 1.0, 1.0, 1.0))
P2_C0: Final = _Vector("p2c0", 2, (0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0))
P0: Final = _Vector("p0", 0, (0.0, 0.4, 1.0))
UNCLAMPED: Final = _Vector("uncl", 2, (-0.2, -0.1, 0.0, 0.25, 0.5, 0.75, 1.0, 1.1, 1.2))
UNCLAMPED_LEFT: Final = _Vector("unclleft", 2, (-0.2, -0.1, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0))
"""Unclamped on the left and clamped on the right, and named for the end it is *not*
clamped at -- which is the reading ``to_open`` and ``has_open_knots()`` use, where
"open" means clamped."""
PERIODIC: Final = _Vector("per", 2, (-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5), True)
SHIFTED: Final = _Vector("shift", 2, (2.0, 2.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.0, 4.0))
"""A vector based away from the origin, where an absolute tolerance and a relative one
part company; ``design/backend_parity.md`` Rule 2 is why one of these is always in a
case list."""


class _Case(NamedTuple):
    """A field to build, and what to do to it.

    Attributes:
        label (str): The test id.
        vectors (tuple[_Vector, ...]): One per parametric direction.
        rank (int): How many value components the field carries, weight column
            excluded.
        rational (bool): Whether the last stored component is a homogeneous weight.
    """

    label: str
    vectors: tuple[_Vector, ...]
    rank: int
    rational: bool = False


def _build(case: _Case, dtype: npt.DTypeLike) -> Bspline:
    """Build ``case``'s field under the active backend.

    The coefficients are an affine ramp over the flat index rather than random: it is
    reproducible without a seed, it makes every coefficient distinct so a transposed or
    mis-strided sweep cannot pass, and it is exactly representable at both formats so
    the field itself introduces no rounding.

    Args:
        case (_Case): The field to build.
        dtype (npt.DTypeLike): The storage format.

    Returns:
        ~pantr.bspline.Bspline: The field.
    """
    spaces = [
        BsplineSpace1D(
            np.asarray(vector.knots, dtype=dtype), vector.degree, periodic=vector.periodic
        )
        for vector in case.vectors
    ]
    space = BsplineSpace(spaces)
    counts = [one_d.num_basis for one_d in spaces]
    components = case.rank + (1 if case.rational else 0)
    total = int(np.prod(counts)) * components
    values = (np.arange(total, dtype=dtype) + np.asarray(1.0, dtype=dtype)) * np.asarray(
        0.25, dtype=dtype
    )
    return Bspline(space, values.reshape(*counts, components), is_rational=case.rational)


def _both(case: _Case, dtype: npt.DTypeLike, act: _Act) -> tuple[Any, Any]:
    """Build and act on ``case``'s field once under each backend.

    Everything happens inside the ``use_backend`` block, the operation included: the
    branch :mod:`pantr.bspline._structural_backend` takes is decided by the *active*
    backend, so acting outside the block would measure the wrong path.

    Args:
        case (_Case): The field to build.
        dtype (npt.DTypeLike): The storage format.
        act (_Act): What to do to the field; takes it and returns anything.

    Returns:
        tuple[Any, Any]: ``(py, cpp)``, in the order
        :func:`~tests._parity_harness.assert_object_parity` names its arguments.
    """
    with use_backend(Backend.PYTHON):
        py = act(_build(case, dtype))
    with use_backend(Backend.CPP):
        cpp = act(_build(case, dtype))
    return py, cpp


def _as_curve(field: Bspline, keep: int) -> tuple[Bspline, int]:
    """Reduce a field to a curve by fixing every direction but ``keep``.

    Each removed direction is fixed at its own domain midpoint. Removing them from the
    last one backwards keeps the surviving index computable: a direction after ``keep``
    does not move it and one before it shifts it down by one.

    This composes the operation under test with itself, which is fine for an
    *invariant* check and would not be for a parity one: what
    :func:`test_the_open_conversion_does_not_move_the_curve` compares is two
    representations of one map, both reduced the same way, so a slice that is wrong in
    the same way on both sides cancels and a wrong *trim* or *partition* does not. The
    closed-form checks that pin the slice itself are in
    ``cpp/tests/test_bspline_structural.cpp``.

    Args:
        field (~pantr.bspline.Bspline): The field to reduce.
        keep (int): The direction to keep.

    Returns:
        tuple[~pantr.bspline.Bspline, int]: The curve, and ``keep``'s index in it --
        which is always 0, returned so that a caller does not have to re-derive it.
    """
    surviving = keep
    for axis in reversed(range(field.dim)):
        if axis == keep:
            continue
        lo, hi = (float(edge) for edge in field.space.spaces[axis].domain)
        reduced = field.slice(axis, 0.5 * (lo + hi))
        assert isinstance(reduced, Bspline), "a field of dimension >= 2 slices to a field"
        field = reduced
        if axis < surviving:
            surviving -= 1
    return field, surviving


def _insertions_to_open(field: Bspline) -> int:
    """How many knot insertions :meth:`~pantr.bspline.Bspline.to_open_bspline` performs.

    One per direction it actually converts, which is one per direction that is periodic
    or not clamped at both ends -- the predicate
    :func:`~pantr.bspline._bspline_knot_insertion._to_open_bspline_impl` skips on.

    Args:
        field (~pantr.bspline.Bspline): The field about to be converted.

    Returns:
        int: The count, at least 1 for any field the conversion does not refuse.
    """
    return sum(1 for one_d in field.space.spaces if one_d.periodic or not one_d.has_open_knots())


def _invariance_bound(field: Bspline, degree: int, dtype: npt.DTypeLike, insertions: int) -> float:
    """The absolute bound on how far a representation change may move the map.

    `gamma_K` times the coefficients' magnitude, with `K` assembled rather than fitted:

     - `insertions * (2p + 3 + (7p + 2))` for the knot insertions the changed
       representation went through, which is
       ``tests/parity/test_bspline_refinement.py``'s derivation at one refined direction,
       charged once per insertion because sequential transforms add their counts. **The
       factor is not 1**: `to_open_bspline` inserts once per converted direction, and a
       *periodic* split inserts twice -- once converting to the open form and again
       raising the split value's multiplicity
       (``pantr/bspline/_bspline_split.py``'s two steps).
     - `5 * p_d` per corner cut, being 3 roundings for the two products and the sum and
       2 for the weight pair's departure from summing to one. Reducing to a curve and
       sampling it is one cut per direction, on each of the two sides, so `10 * sum p_d`.

    The weights are in `[0, 1]` for a parameter inside its span, so no cut amplifies the
    magnitude and the counts compose additively.

    Args:
        field (~pantr.bspline.Bspline): The field whose coefficients set the magnitude.
        degree (int): The degree of the direction the operation changed; the largest of
            them where it changed more than one.
        dtype (npt.DTypeLike): The storage format the roundings are charged in.
        insertions (int): How many knot insertions the changed representation went
            through.

    Returns:
        float: The bound, in the units of the coefficients.
    """
    roundings = insertions * (2 * degree + 3 + 7 * degree + 2) + 10 * sum(field.degree)
    unit = 0.5 * float(np.finfo(np.dtype(dtype)).eps)
    magnitude = float(np.max(np.abs(np.asarray(field.control_points, dtype=np.float64))))
    return roundings * unit / (1.0 - roundings * unit) * magnitude


_COEFFICIENTS_WHY: Final = (
    "the coefficients. On the open-conversion and split paths every arithmetic "
    "operation is `refine_along_axis`'s, whose bitwise claim "
    "`tests/parity/test_bspline_refinement.py` derives: the sweep accumulates in the "
    "storage format, in ascending band order, from an exact zero. The trim and the "
    "partition around it are row selections and perform no arithmetic. On the slice "
    "path every operation is the corner cut's `wa * next + wb * row`, whose two "
    "weights are `double` expressions rounded once to the storage format -- measured, "
    "not inferred -- and whose update then runs in that format, which is what numpy's "
    "weak-scalar promotion makes the oracle do. Same operations, same order, same "
    "operands."
)

_KNOTS_WHY: Final = (
    "a resulting knot vector. Both operations build theirs by selection and "
    "concatenation of values that already exist: `inserted_knot_vector` stable-sorts "
    "the old vector against the boundary or split value it was handed, and the trim "
    "and the partition take subranges of that. No arithmetic touches a knot, so a "
    "difference could only be a lost entry, a reordering, a narrowing cast, or a "
    "multiplicity count that decided to insert a different number of knots -- which "
    "would show here as a length mismatch rather than as a displaced value."
)

_TOLERANCE_WHY: Final = (
    "the resulting space derives its own tolerance from its own knot vector, so it is "
    "a *new* quantity of the result and not one carried over, and it is what every "
    "later comparison on that space will use. The knots agree bit for bit and "
    "`knot_tolerance` is the same reduction over them on both sides. A difference "
    "would mean the two backends disagree about which knots are the same knot, which "
    "no comparison of the knots themselves would reveal."
)


def _fields(dim: int) -> tuple[Field, ...]:
    """Every piece of a resulting field's state, one field each with its own reason.

    Args:
        dim (int): The result's number of parametric directions.

    Returns:
        tuple[Field, ...]: The field list for
        :func:`~tests._parity_harness.assert_object_parity`.
    """
    per_direction: list[Field] = []
    for direction in range(dim):
        per_direction.append(
            Field(
                f"space.spaces[{direction}].knots",
                bitwise_parity(why=_KNOTS_WHY),
                read=lambda field, d=direction: field.space.spaces[d].knots,  # type: ignore[misc]
            )
        )
        per_direction.append(
            Field(
                f"space.spaces[{direction}].tolerance",
                bitwise_parity(why=_TOLERANCE_WHY),
                read=lambda field, d=direction: field.space.spaces[d].tolerance,  # type: ignore[misc]
            )
        )
        per_direction.append(
            Field(
                f"space.spaces[{direction}].periodic",
                exact_parity(
                    why=(
                        "periodicity is a flag, and both operations decide it "
                        "structurally rather than numerically: the open conversion "
                        "clears it and the split clears it in the cut direction, "
                        "whatever the input was. A slice carries a surviving "
                        "direction's flag through untouched. A difference would be a "
                        "different *kind* of space, which no tolerance covers."
                    )
                ),
                read=lambda field, d=direction: field.space.spaces[d].periodic,  # type: ignore[misc]
            )
        )
    return (
        Field("control_points", bitwise_parity(why=_COEFFICIENTS_WHY)),
        Field(
            "dim",
            exact_parity(
                why=(
                    "the parametric dimension is a count. It is the field that would "
                    "catch a slice that failed to drop its axis, and it must be "
                    "asserted separately from the coefficients, which compare "
                    "elementwise and cannot see two results of different shape."
                )
            ),
        ),
        Field(
            "space.num_basis",
            exact_parity(
                why=(
                    "the basis counts are integers derived from the resulting knot "
                    "vectors and degrees, so they are exact whenever those are. They "
                    "are what a wrong trim or a wrong partition index moves first, and "
                    "`design/backend_parity.md` Rule 11 is why a count is asserted on "
                    "its own rather than folded into a bounded comparison."
                )
            ),
        ),
        Field(
            "degree",
            exact_parity(why="the degree of each direction is carried, never recomputed."),
        ),
        Field(
            "rank",
            exact_parity(
                why=(
                    "the rank is the component count less the weight column. None of "
                    "these operations touches the component axis, so a difference "
                    "would mean one backend dropped or projected a component."
                )
            ),
        ),
        Field(
            "is_rational",
            exact_parity(
                why=(
                    "the rationality flag is the invariant an unpack-and-reassemble "
                    "across the seam would lose, which is why the field crosses rather "
                    "than its arrays. `design/cross_backend_types.md` names it."
                )
            ),
        ),
        *per_direction,
    )


# ---------------------------------------------------------------------------
# The open conversion
# ---------------------------------------------------------------------------

OPEN_CASES: Final = (
    _Case("uncl-1d", (UNCLAMPED,), 3),
    _Case("unclamped-left-1d", (UNCLAMPED_LEFT,), 2),
    _Case("per-1d", (PERIODIC,), 3),
    _Case("per-1d-rational", (PERIODIC,), 2, rational=True),
    _Case("per-uncl-2d", (PERIODIC, UNCLAMPED), 2),
    _Case("open-uncl-2d", (P2, UNCLAMPED), 2),
    _Case("uncl-open-2d", (UNCLAMPED, P1), 3),
    _Case("per-open-uncl-3d", (PERIODIC, P1, UNCLAMPED), 1),
    _Case("shifted-uncl-2d", (SHIFTED, UNCLAMPED), 2),
)
"""Fields with something to convert. Every combination of "already open", "unclamped"
and "periodic" appears in at least one direction of at least one case, and the mixed
cases are what catch a conversion that reads the wrong axis's extents."""


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", OPEN_CASES, ids=lambda case: case.label)
def test_the_open_conversion_agrees_field_by_field(case: _Case, dtype: npt.DTypeLike) -> None:
    """Every piece of the open field's state agrees, under its own claim.

    Args:
        case (_Case): The field to build.
        dtype (npt.DTypeLike): The storage format.
    """
    if contraction_may_fuse():
        pytest.skip(_FUSING)
    py, cpp = _both(case, dtype, lambda field: field.to_open_bspline())
    assert_object_parity(
        py=py,
        cpp=cpp,
        fields=_fields(len(case.vectors)),
        context=f"to_open_bspline on {case.label} at {np.dtype(dtype).name}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", OPEN_CASES, ids=lambda case: case.label)
def test_the_open_conversion_does_not_move_the_curve(case: _Case, dtype: npt.DTypeLike) -> None:
    """The open form evaluates to the same map as the field it came from.

    Not a parity check: it compares the C++ backend against *itself* across a
    representation change, so unlike everything above it would catch an error the two
    backends share. The evaluator is `slice_point`, which needs no basis function and so
    is available in this cut -- ``tests/parity/test_bspline_refinement.py`` records that
    it was not available to the refinement port.

    A multi-dimensional field is reduced to a curve first, by :func:`_as_curve`, so
    every case is sampled rather than only the univariate ones -- the strided sweep is
    exactly what a multi-dimensional case exercises and a univariate one cannot.

    Args:
        case (_Case): The field to build.
        dtype (npt.DTypeLike): The storage format.
    """
    with use_backend(Backend.CPP):
        field = _build(case, dtype)
        opened = field.to_open_bspline()
        bound = _invariance_bound(
            field,
            max(vector.degree for vector in case.vectors),
            dtype,
            _insertions_to_open(field),
        )
        for direction in range(field.dim):
            before, keep = _as_curve(field, direction)
            after, _ = _as_curve(opened, direction)
            lo, hi = (float(edge) for edge in before.space.spaces[keep].domain)
            for step in range(9):
                at = lo + (hi - lo) * step / 8.0
                was = np.atleast_1d(np.asarray(before.slice(keep, at), dtype=np.float64))
                now = np.atleast_1d(np.asarray(after.slice(keep, at), dtype=np.float64))
                assert np.all(np.abs(was - now) <= bound), (
                    f"to_open_bspline moved the curve of {case.label} at "
                    f"{np.dtype(dtype).name}, direction {direction}, u={at}: "
                    f"{was} against {now}, bound {bound}"
                )


def test_the_open_conversion_refuses_an_open_field_the_same_way() -> None:
    """A field that is already open in every direction is refused with one text.

    The only refusal of the four operations that no wrapper check makes first, so it is
    the only one raised by whichever implementation runs and the only one whose text
    could differ between the backends without anything else noticing.
    """
    case = _Case("already-open", (P2, P1), 2)
    messages: list[str] = []
    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            field = _build(case, np.float64)
            with pytest.raises(ValueError) as raised:
                field.to_open_bspline()
            messages.append(str(raised.value))
    assert messages[0] == "B-spline is already open in every direction.", messages[0]
    assert messages[0] == messages[1], messages


def test_the_open_conversion_shares_an_untouched_direction() -> None:
    """An already-open direction keeps the wrapper it had, under both backends.

    The identity contract ``design/bspline_ownership_lifetime.md`` states and the reason
    the C++ half carries a space handle rather than rebuilding an equal-valued space. No
    value comparison could report a difference here.
    """
    case = _Case("open-uncl-2d", (P2, UNCLAMPED), 2)
    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            field = _build(case, np.float64)
            opened = field.to_open_bspline()
            assert opened.space.spaces[0] is field.space.spaces[0], backend
            assert opened.space.spaces[1] is not field.space.spaces[1], backend


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------


class _SplitCase(NamedTuple):
    """A field to split, the direction and the parameter.

    Attributes:
        label (str): The test id.
        case (_Case): The field.
        direction (int): The direction to cut.
        value (float): Where to cut it.
    """

    label: str
    case: _Case
    direction: int
    value: float


SPLIT_CASES: Final = (
    _SplitCase("p1-mid", _Case("p1", (P1,), 2), 0, 0.5),
    _SplitCase("p2-off-knot", _Case("p2", (P2,), 3), 0, 0.4),
    _SplitCase("p2-on-knot", _Case("p2", (P2,), 3), 0, 0.5),
    _SplitCase("p3-off-knot", _Case("p3", (P3,), 1), 0, 0.6),
    _SplitCase("p2-at-c0-knot", _Case("p2c0", (P2_C0,), 2), 0, 0.5),
    _SplitCase("p0", _Case("p0", (P0,), 2), 0, 0.5),
    _SplitCase("2d-first", _Case("2d", (P2, P1), 3), 0, 0.3),
    _SplitCase("2d-second", _Case("2d", (P2, P1), 3), 1, 0.55),
    _SplitCase("3d-middle", _Case("3d", (P2, P1, P3), 2), 1, 0.42),
    _SplitCase("rational", _Case("rat", (P2, P1), 3, rational=True), 0, 0.6),
    _SplitCase("unclamped", _Case("uncl", (UNCLAMPED,), 2), 0, 0.5),
    _SplitCase("periodic", _Case("per", (PERIODIC,), 3), 0, 0.4),
    _SplitCase("periodic-2d", _Case("per2d", (PERIODIC, P1), 2), 0, 0.6),
    _SplitCase("shifted", _Case("shift", (SHIFTED,), 2), 0, 2.75),
)
"""Cuts off a knot, on a knot of multiplicity 1, on a knot already at C0, at degree 0,
on each axis of a two- and a three-direction field, on a rational field, and on
unclamped and periodic directions -- the two that take a different path into the cut."""


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("split", SPLIT_CASES, ids=lambda split: split.label)
def test_the_split_agrees_half_by_half(split: _SplitCase, dtype: npt.DTypeLike) -> None:
    """Every piece of both halves' state agrees, under its own claim.

    Args:
        split (_SplitCase): The field, the direction and the parameter.
        dtype (npt.DTypeLike): The storage format.
    """
    if contraction_may_fuse():
        pytest.skip(_FUSING)
    py, cpp = _both(split.case, dtype, lambda field: field.split(split.direction, split.value))
    fields = _fields(len(split.case.vectors))
    for index, side in enumerate(("left", "right")):
        assert_object_parity(
            py=py[index],
            cpp=cpp[index],
            fields=fields,
            context=(
                f"split({split.direction}, {split.value}) {side} half on "
                f"{split.label} at {np.dtype(dtype).name}"
            ),
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("split", SPLIT_CASES, ids=lambda split: split.label)
def test_the_split_does_not_move_the_curve(split: _SplitCase, dtype: npt.DTypeLike) -> None:
    """Each half evaluates to the same map as the field it came from, on its own domain.

    The other check that compares a backend against itself rather than against the other
    one; see :func:`test_the_open_conversion_does_not_move_the_curve`. It is what a wrong
    partition index moves, and a wrong index that shifts both the knot vector and the
    coefficients consistently would pass every comparison above.

    A multi-dimensional field is reduced to a curve along the cut direction first, by
    :func:`_as_curve`, so every case is sampled and the strided partition is covered.

    Args:
        split (_SplitCase): The field, the direction and the parameter.
        dtype (npt.DTypeLike): The storage format.
    """
    with use_backend(Backend.CPP):
        field = _build(split.case, dtype)
        halves = field.split(split.direction, split.value)
        degree = split.case.vectors[split.direction].degree
        # A periodic direction is converted to the open form first and *then* has the
        # split value's multiplicity raised: two insertions, not one.
        insertions = 2 if split.case.vectors[split.direction].periodic else 1
        bound = _invariance_bound(field, degree, dtype, insertions)
        whole, keep = _as_curve(field, split.direction)
        for side, half in zip(("left", "right"), halves, strict=True):
            reduced, _ = _as_curve(half, split.direction)
            lo, hi = (float(edge) for edge in reduced.space.spaces[keep].domain)
            for step in range(5):
                at = lo + (hi - lo) * step / 4.0
                was = np.atleast_1d(np.asarray(whole.slice(keep, at), dtype=np.float64))
                now = np.atleast_1d(np.asarray(reduced.slice(keep, at), dtype=np.float64))
                assert np.all(np.abs(was - now) <= bound), (
                    f"split moved the {side} half of {split.label} at "
                    f"{np.dtype(dtype).name}, u={at}: {was} against {now}, bound {bound}"
                )


def test_the_split_halves_share_the_untouched_directions() -> None:
    """Every direction but the cut one keeps its wrapper, in both halves and both backends.

    The two halves are wrapped over the *same* prior direction list, so an untouched
    direction's wrapper is one object shared three ways.
    """
    case = _Case("3d", (P2, P1, P3), 2)
    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            field = _build(case, np.float64)
            left, right = field.split(1, 0.5)
            for half in (left, right):
                assert half.space.spaces[0] is field.space.spaces[0], backend
                assert half.space.spaces[2] is field.space.spaces[2], backend
                assert half.space.spaces[1] is not field.space.spaces[1], backend


# ---------------------------------------------------------------------------
# The slice and the boundary
# ---------------------------------------------------------------------------


class _SliceCase(NamedTuple):
    """A field to slice, the axis and the parameter.

    Attributes:
        label (str): The test id.
        case (_Case): The field.
        axis (int): The direction to fix.
        value (float): Where to fix it.
    """

    label: str
    case: _Case
    axis: int
    value: float


SLICE_CASES: Final = (
    _SliceCase("1d-off-knot", _Case("p2", (P2,), 3), 0, 0.4),
    _SliceCase("1d-on-knot", _Case("p2", (P2,), 3), 0, 0.5),
    _SliceCase("1d-at-start", _Case("p2", (P2,), 3), 0, 0.0),
    _SliceCase("1d-at-end", _Case("p2", (P2,), 3), 0, 1.0),
    _SliceCase("1d-at-c0-knot", _Case("p2c0", (P2_C0,), 2), 0, 0.5),
    _SliceCase("1d-degree-0", _Case("p0", (P0,), 2), 0, 0.6),
    _SliceCase("1d-p1", _Case("p1", (P1,), 2), 0, 0.45),
    _SliceCase("1d-p3", _Case("p3", (P3,), 1), 0, 0.15),
    _SliceCase("1d-rational", _Case("rat", (P2,), 3, rational=True), 0, 0.4),
    _SliceCase("1d-periodic", _Case("per", (PERIODIC,), 3), 0, 0.4),
    _SliceCase("1d-unclamped", _Case("uncl", (UNCLAMPED,), 2), 0, 0.4),
    _SliceCase("1d-shifted", _Case("shift", (SHIFTED,), 2), 0, 2.75),
    _SliceCase("2d-first", _Case("2d", (P2, P1), 3), 0, 0.3),
    _SliceCase("2d-second", _Case("2d", (P2, P1), 3), 1, 0.55),
    _SliceCase("2d-rational", _Case("rat2d", (P2, P1), 3, rational=True), 1, 0.55),
    _SliceCase("2d-periodic-first", _Case("per2d", (PERIODIC, P1), 2), 0, 0.6),
    _SliceCase("2d-periodic-second", _Case("2dper", (P1, PERIODIC), 2), 1, 0.6),
    _SliceCase("3d-middle", _Case("3d", (P2, P1, P3), 2), 1, 0.42),
    _SliceCase("3d-last", _Case("3d", (P2, P1, P3), 2), 2, 0.42),
    _SliceCase("3d-first", _Case("3d", (P2, P1, P3), 2), 0, 0.42),
)
"""Both endpoints and an interior knot -- the three places the span search branches --
plus each axis of a two- and a three-direction field, which is what catches a sweep that
strides the wrong axis, and the periodic and rational paths."""

_POINT_WHY: Final = (
    "the point a one-dimensional slice returns. Every operation is the corner cut's, "
    "operand for operand: the weights are `double` expressions rounded once to the "
    "storage format and the update runs in that format. For a rational field the "
    "division by the weight is one numpy expression on both paths, because the C++ "
    "binding hands back the homogeneous components and "
    "`pantr.bspline._structural_backend` divides."
)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", SLICE_CASES, ids=lambda case: case.label)
def test_the_slice_agrees(case: _SliceCase, dtype: npt.DTypeLike) -> None:
    """The slice agrees, field by field for a surface and pointwise for a curve.

    Args:
        case (_SliceCase): The field, the axis and the parameter.
        dtype (npt.DTypeLike): The storage format.
    """
    if contraction_may_fuse():
        pytest.skip(_FUSING)
    py, cpp = _both(case.case, dtype, lambda field: field.slice(case.axis, case.value))
    context = f"slice({case.axis}, {case.value}) on {case.label} at {np.dtype(dtype).name}"
    if len(case.case.vectors) == 1:
        assert isinstance(py, np.ndarray) and isinstance(cpp, np.ndarray), context
        assert py.shape == (case.case.rank,), f"{context}: oracle shape {py.shape}"
        assert_parity(cpp, py, claim=bitwise_parity(why=_POINT_WHY), context=context)
        return
    assert_object_parity(
        py=py,
        cpp=cpp,
        fields=_fields(len(case.case.vectors) - 1),
        context=context,
    )


def test_the_slice_drops_the_axis_and_shares_the_survivors() -> None:
    """A slice keeps the surviving directions' wrappers, in order, under both backends.

    The property that forced :meth:`pantr.bspline.Bspline._wrap_over` to take a
    per-direction list rather than the whole prior space: the reuse is positional
    against the *result's* directions, so the sliced axis has to be dropped from the
    prior list too. Reusing the unreduced list would hand back a wrapper for a different
    direction while every value comparison agreed, which is exactly what this asserts
    did not happen.
    """
    case = _Case("3d", (P2, P1, P3), 2)
    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            field = _build(case, np.float64)
            sliced = field.slice(1, 0.5)
            assert isinstance(sliced, Bspline), backend
            assert sliced.dim == 2, backend
            assert sliced.space.spaces[0] is field.space.spaces[0], backend
            assert sliced.space.spaces[1] is field.space.spaces[2], backend


def test_a_sliced_field_holds_the_space_its_implementation_holds() -> None:
    """The wrapper's space is the one its implementation holds, not an equal rebuild.

    ``design/bspline_derived_caches.md``'s reason for ``_wrap_over`` at all: two
    computations of one answer is a second truth about a value. Only meaningful under
    the C++ backend, where the two objects could differ.
    """
    with use_backend(Backend.CPP):
        field = _build(_Case("2d", (P2, P1), 3), np.float64)
        sliced = field.slice(0, 0.3)
        assert isinstance(sliced, Bspline)
        assert sliced.space._impl is sliced._impl.space
        left, right = field.split(0, 0.3)
        assert left.space._impl is left._impl.space
        assert right.space._impl is right._impl.space
        opened = _build(_Case("uncl", (UNCLAMPED,), 2), np.float64).to_open_bspline()
        assert opened.space._impl is opened._impl.space


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("side", (0, 1))
def test_the_boundary_agrees(side: int, dtype: npt.DTypeLike) -> None:
    """The boundary agrees, and is the slice at the endpoint it is defined as.

    ``boundary`` has no entry point in :mod:`pantr.bspline._structural_backend` and no
    binding of its own, so what this checks is that routing it through
    :meth:`~pantr.bspline.Bspline.slice` really does reach C++ and really does agree.

    Args:
        side (int): Which end of the domain.
        dtype (npt.DTypeLike): The storage format.
    """
    if contraction_may_fuse():
        pytest.skip(_FUSING)
    case = _Case("2d", (SHIFTED, P1), 3)
    py, cpp = _both(case, dtype, lambda field: field.boundary(0, side))
    assert_object_parity(
        py=py,
        cpp=cpp,
        fields=_fields(1),
        context=f"boundary(0, {side}) at {np.dtype(dtype).name}",
    )
    with use_backend(Backend.CPP):
        field = _build(case, dtype)
        face = field.boundary(0, side)
        edge = float(field.space.spaces[0].domain[side])
        assert isinstance(face, Bspline)
        direct = field.slice(0, edge)
        assert isinstance(direct, Bspline)
        assert np.array_equal(face.control_points, direct.control_points), side


# ---------------------------------------------------------------------------
# The refusals, and the seam
# ---------------------------------------------------------------------------


def _refusal(backend: Backend, act: _Act) -> str:
    """The text of the ``ValueError`` ``act`` raises under ``backend``.

    Args:
        backend (Backend): The backend to run under.
        act (_Act): Takes a field and does something that must refuse.

    Returns:
        str: The message.
    """
    with use_backend(backend):
        field = _build(_Case("2d", (P2, P1), 3), np.float64)
        with pytest.raises(ValueError) as raised:
            act(field)
        return str(raised.value)


@pytest.mark.parametrize(
    ("label", "act", "expected"),
    [
        (
            "split-direction",
            lambda field: field.split(2, 0.5),
            "direction must be in [0, 2), got 2.",
        ),
        (
            "split-at-start",
            lambda field: field.split(0, 0.0),
            "value must be strictly inside the domain (0.0, 1.0), got 0.0.",
        ),
        (
            "split-outside",
            lambda field: field.split(0, 1.7),
            "value must be strictly inside the domain (0.0, 1.0), got 1.7.",
        ),
        ("slice-axis", lambda field: field.slice(2, 0.5), "axis must be in [0, 2), got 2."),
        (
            "slice-outside",
            lambda field: field.slice(1, 1.7),
            "value 1.7 is outside the domain [0.0, 1.0] of direction 1.",
        ),
        ("boundary-side", lambda field: field.boundary(0, 2), "side must be 0 or 1, got 2."),
        ("boundary-axis", lambda field: field.boundary(2, 0), "axis must be in [0, 2), got 2."),
        # Bad in both ways. The wrapper checks the side first, so that is the message,
        # and the C++ `boundary` restates the same order; the claim had no test before.
        (
            "boundary-side-and-axis",
            lambda field: field.boundary(7, 2),
            "side must be 0 or 1, got 2.",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_the_wrapper_refusals_are_identical(label: str, act: _Act, expected: str) -> None:
    """Every Layer 1 refusal reads the same under both backends, and is the oracle's.

    All seven are made by the wrapper above the branch, which is what makes them
    common mode. Asserting the literal as well as the equality is what would catch both
    backends drifting together -- the failure a parity comparison cannot see.

    The out-of-domain slice is the one whose C++ text differs at ``float32``, and this
    is the test that pins the divergence as unreachable: the wrapper refuses first, so
    what a caller reads is this text on both backends. See the module docstring.

    Args:
        label (str): The case's name.
        act (_Act): What to do to the field.
        expected (str): The oracle's message.
    """
    texts = [_refusal(backend, act) for backend in (Backend.PYTHON, Backend.CPP)]
    assert texts[0] == expected, f"{label}: the oracle's text is {texts[0]!r}"
    assert texts[0] == texts[1], f"{label}: {texts}"


def test_a_cross_backend_field_is_refused_on_both_routes() -> None:
    """Neither route converts a field the other backend built, and the two differ in how.

    ``design/cross_backend_types.md`` forbids the conversion. **Both directions are
    refused, by different mechanisms and with different messages**, and the asymmetry is
    measured rather than assumed:

     - the C++ route refuses it up front, in :func:`_cpp_handle`, with a ``TypeError``
       naming the backend;
     - the *oracle* route refuses it late and incidentally, in
       :class:`~pantr.bspline.BsplineSpace`'s own constructor, with a ``ValueError``:
       both operations carry at least one untouched direction's wrapper into the result,
       and a space cannot aggregate a direction from the other backend.

    That second half is why the claim in
    :mod:`pantr.bspline._refinement_backend`'s docstring -- that the oracle runs happily
    over a C++ field -- needed the qualification it now carries: it holds only where
    *every* direction is rebuilt, which for these operations is never.
    """
    with use_backend(Backend.PYTHON):
        python_field = _build(_Case("2d", (P2, P1), 3), np.float64)
    with use_backend(Backend.CPP):
        native = _build(_Case("2d", (P2, P1), 3), np.float64)
        for act in (
            lambda: to_open_field(python_field),
            lambda: split_field(python_field, 0, 0.5),
            lambda: slice_field(python_field, 0, 0.5),
        ):
            with pytest.raises(TypeError, match="different backend"):
                act()
    with use_backend(Backend.PYTHON):
        for act in (
            lambda: split_field(native, 0, 0.5),
            lambda: slice_field(native, 0, 0.5),
        ):
            with pytest.raises(ValueError, match="must come from the active backend"):
                act()


def test_the_cpp_route_is_the_one_that_ran() -> None:
    """Under the C++ backend the results hold C++ implementations, not Python ones.

    The vacuity guard for every parity test above: a dispatcher that silently fell back
    to the oracle would make all of them pass while measuring nothing, and no comparison
    of values could tell.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415

    cpp_types = (_pantr_cpp.Bspline32, _pantr_cpp.Bspline64)
    with use_backend(Backend.CPP):
        field = _build(_Case("2d", (P2, UNCLAMPED), 3), np.float64)
        assert isinstance(field.to_open_bspline()._impl, cpp_types)
        left, right = field.split(0, 0.3)
        assert isinstance(left._impl, cpp_types)
        assert isinstance(right._impl, cpp_types)
        sliced = field.slice(0, 0.3)
        assert isinstance(sliced, Bspline)
        assert isinstance(sliced._impl, cpp_types)


def test_wrap_over_refuses_the_unreduced_prior_list() -> None:
    """A dimension-reducing result cannot be wrapped over the field's own direction list.

    The precondition :meth:`pantr.bspline.Bspline._wrap_over` documents and the length
    check one level down enforces. It is what stops a slice from reusing direction 1's
    wrapper for what is now direction 1 of a smaller space, which would be a wrapper for
    a different direction with every value still agreeing.
    """
    with use_backend(Backend.CPP):
        field = _build(_Case("3d", (P2, P1, P3), 2), np.float64)
        sliced = field.slice(1, 0.5)
        assert isinstance(sliced, Bspline)
        assert len(field.space.spaces) == 3, "the guard needs a prior list that is too long"
        with pytest.raises(ValueError, match="one prior wrapper per direction"):
            Bspline._wrap_over(sliced._impl, field.space.spaces)


def test_the_results_are_still_immutable() -> None:
    """A structural result is as frozen as any other field wrapper.

    ``__slots__``-only and no ``__dict__``: the three slots refuse assignment, so an
    operation that handed back a half-built wrapper would be caught here rather than by
    whatever read it next.
    """
    with use_backend(Backend.CPP):
        field = _build(_Case("2d", (P2, UNCLAMPED), 3), np.float64)
        results: Sequence[Bspline] = (field.to_open_bspline(), *field.split(0, 0.3))
        for result in results:
            with pytest.raises(AttributeError):
                result._impl = field._impl
            with pytest.raises(AttributeError):
                del result._space
