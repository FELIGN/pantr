"""Parity of the two field refinements: `Bspline.insert_knots` and `.subdivide`.

What this file compares is a **computation**, which is what separates it from
``tests/parity/test_bspline_type.py``: that one compares a value the two backends only
copy, so a tolerance there would be hiding a transcription error. Here the control net
is pushed through a two-scale matrix, so there is real arithmetic on every coefficient
and the criterion has to be argued rather than assumed.

It is argued **per quantity** in :func:`_fields`, not once in bulk: the coefficients,
the merged knot vectors, the tolerance the refined space derives from them, the counts,
the degrees and the flags are exact for six different reasons, and a shared ``why``
would be one sentence quoted in six failure messages it does not fit.

## The criterion, and the host it depends on

**Bitwise, for the coefficients and the knots.** The C++ sweep accumulates in the
storage format, in ascending band order, from an exact zero, skipping the off-band
columns rather than adding their exact zeros -- which is
``pantr/bspline/_bspline_knot_insertion_core.py``'s ``_insert_knots_1d_core`` operation
for operation. ``design/backend_parity.md`` Rule 9 is why that had to be read off the
kernel rather than inferred: the oracle allocates its accumulator in ``ctrl.dtype`` and
never widens it, so a C++ side accumulating in ``double`` would be exact at ``float64``
and quietly wrong at ``float32``.

The band recurrence *does* widen three variables under numba -- ``saved``, ``to_left``
and ``to_right`` are each seeded with the literal ``0.0`` and so unify to ``float64`` --
and it changes nothing, because each then holds the value of a ``float32`` expression
and the exact sum of two ``float32`` numbers is representable in ``float64``. There is
no double rounding to inherit, which is why ``oslo_bands_1d`` computing entirely in
``T`` matches. ``cpp/include/pantr/bspline/refinement.hpp`` carries the same argument
beside the code.

Bitwise is a claim about **this build**, per ``design/backend_parity.md`` Rule 7. The
sweep's inner statement is ``row[k] + weight * source[k]``, which a target with a fused
multiply-add would contract; on the baseline x86-64 this project compiles for there is
no FMA to contract into. :func:`~tests._parity_harness.contraction_may_fuse` is the
gate, and where it reports a fusing build the bitwise claim is skipped rather than
weakened -- Rule 10's budget would apply and no host in this project can execute it to
check.

**The interpreted oracle makes no difference here, and that is a guarantee rather than
an observation.** ``design/backend_parity.md`` Rule 12 names the three things that
differ under ``NUMBA_DISABLE_JIT=1`` -- a widened storage width, ``pow``, and an integer
accumulator that wraps when compiled -- and its complementary half says a kernel built
from ``+``, ``-``, ``*``, ``/`` and ``sqrt`` reproduces its own bits by IEEE 754 rather
than by luck. Both Oslo kernels are of that kind: no ``pow``, and the only integer they
carry is a column index bounded by an array length. The one width that does move is the
``saved`` unification named above, in the other direction -- interpreted, NEP 50 keeps it
``float32`` -- and it is exact either way for the same reason. So no gate is needed, and
running this file under that configuration confirms it at both storage formats.

## The independent check, and why the obvious one was not available

``design/backend_parity.md`` opens with the fact everything else there follows from:
parity says the two backends agree and not that either is right, so a shared error is
invisible to every parity test in this file.

The natural invariant for knot insertion is that **it must not move the curve**, and
the natural way to check that is to evaluate before and after. That needs
``BsplineSpace1D.tabulate_basis``, which ``cpp/include/pantr/bspline/space_1d.hpp``
keeps off this milestone, so it is not available. The invariant is, though, written in
**coefficients** instead of in values.

**Marsden's identity.** For a knot vector ``t`` and degree ``p``,

    (u - y)^p = sum_i prod_{k=1}^{p} (t_{i+k} - y) N_{i,p}(u),

and matching the coefficient of ``(-y)^{p-r}`` on both sides gives, for every ``r`` in
``[0, p]``,

    u^r = sum_i [ e_r(t_{i+1}, ..., t_{i+p}) / C(p, r) ] N_{i,p}(u),

with ``e_r`` the elementary symmetric polynomial. So the B-spline coefficients of the
monomial ``u^r`` are a closed form in the knots alone. A field whose control points are
those numbers *is* the map ``u -> u^r``; refinement must not move it; therefore the
refined coefficients must be the same closed form on the **refined** knot vector.

Three things make it a real oracle rather than a restatement:

- **It is complete.** ``r = 1`` alone is the Greville abscissa and says only that the
  refinement preserves affine maps; ``r = 0`` alone is the partition of unity. Either
  leaves a band of ``p + 1`` entries pinned by one linear functional. All of
  ``r`` in ``[0, p]`` pins ``p + 1`` independent functionals per band, which determines
  it.
- **It is not a mirror.** Elementary symmetric polynomials of knots and a binomial
  coefficient, computed in :func:`_marsden_column` in this file. Nothing in it consults
  the Oslo recurrence, either backend, or the refinement matrix.
- **It survives several directions.** The refinement is a tensor product of the
  per-direction matrices, so the field ``prod_d u_d^{r_d}`` has control points
  ``prod_d A^d_{i_d, r_d}``, and each component of the net below carries one power
  multi-index. Refining direction ``d`` must move that direction's factor and leave the
  others alone, which is what catches a sweep applied to the wrong axis.

``cpp/tests/test_bspline_refinement.cpp`` runs the same oracle against the C++ side
alone, so the two halves of the check are independent of the Python seam.

## The accuracy bound

Every quantity compared is relative on non-negative data, so the bound is
``gamma_K * magnitude`` with ``K`` counted rather than fitted:

- **the oracle's own formation**, ``e_r`` by ``E_j <- E_j + x E_{j-1}``: two roundings
  per knot along the dominant chain, plus the division by ``C(p, r)``. ``2p + 1``;
- **the cast into the field's storage**: :func:`_make_field` forms the closed form in
  ``float64`` and stores it in the field's dtype, which at ``float32`` is a real
  rounding on the *input* the sweep then propagates. It costs one, because a convex
  combination does not amplify a relative perturbation. ``1``;
- **one direction's band recurrence**: ``p`` levels, each committing two subtractions,
  a division, a multiplication and an addition on its dominant path. ``5p``;
- **one direction's control-point sweep**: ``p + 1`` terms, each a multiplication and an
  addition. ``2p + 2``;
- relative errors compose sub-additively and ``gamma_a + gamma_b <= gamma_{a+b}``, so
  ``D`` refined directions cost ``D (7p + 2)``.

``K = 2p + 3 + D (7p + 2)``, the last 1 being the store into the result. The magnitude
is ``prod_d max_i A^d_{i, r_d}`` per component: the discrete B-splines of a refinement
are non-negative, so each output coefficient is a convex combination of coarse ones and
no partial sum of it exceeds the largest. That non-negativity is a classical property
(Cohen, Lyche & Riesenfeld 1980) and ``cpp/tests/test_bspline_refinement.cpp`` asserts
it rather than assuming it, because it is the premise this bound rests on.

The relative term is not the whole bound. A coefficient of an exact zero -- ``e_p`` of a
window containing the knot 0 -- has a relative bound of zero, and floating point does
not: :func:`~tests._parity_harness.underflow_floor` is added ``K`` times, which is the
absolute half of Higham's model the harness's own docstring says every tolerance here
carries.

The counts are charged at ``unit_roundoff(dtype)`` throughout, the oracle's included,
even though the oracle is formed in ``float64`` whatever the field stores. That
over-states the ``float32`` case and keeps one derivation for both.

## The two places the backends do not meet

Both are recorded in :mod:`pantr.bspline._refinement_backend` and both are pinned here
rather than left to be met.

- **A periodic direction that receives knots runs the oracle**, because refining one
  needs the open-to-periodic conversions, which are their own port. The C++ half refuses
  it outright and
  :func:`test_a_periodic_direction_is_refused_by_the_binding_and_served_by_the_oracle`
  pins both halves of that. Parity on such a case is **common mode** -- one
  implementation ran twice -- which this file says out loud rather than counting as
  evidence.
- **One shape of bad input is refused in a different order.** The oracle checks an
  insertion array's rank inside its per-direction loop; a ``std::span`` has no rank, so
  the C++ path checks every rank before the call.
  :func:`test_the_refusal_order_divergence_is_exactly_one_input` pins both orders.
"""

from __future__ import annotations

import math
import pickle
from typing import TYPE_CHECKING, Any, Final, NamedTuple

import numpy as np
import numpy.typing as npt
import pytest

from pantr._backend import Backend, use_backend
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
from pantr.bspline._refinement_backend import insert_knots_into_field, subdivide_field
from tests._parity_harness import (
    AccuracyClaim,
    Field,
    assert_accuracy,
    assert_object_parity,
    bitwise_parity,
    contraction_may_fuse,
    derived_accuracy,
    exact_parity,
    underflow_floor,
    unit_roundoff,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

pytestmark = pytest.mark.usefixtures("cpp_backend")

DTYPES: Final = (np.float64, np.float32)
"""Both storage formats: a field stores `float32` too, so the C++ side has two classes."""

_FUSING: Final = (
    "this build's target ISA has a fused multiply-add, so the sweep's "
    "`row[k] + weight * source[k]` may contract and the bitwise claim does not hold. "
    "design/backend_parity.md Rule 10's budget would apply; no host in this project "
    "can execute that branch to check it, so it is skipped rather than written blind."
)
"""Skip reason for the bitwise claims on a build that can fuse."""


# ---------------------------------------------------------------------------
# The independent oracle: Marsden's identity
# ---------------------------------------------------------------------------


def _marsden_column(
    knots: npt.NDArray[np.float32 | np.float64], degree: int, power: int
) -> npt.NDArray[np.float64]:
    """The B-spline coefficients of ``u ** power`` over one knot vector.

    ``A_i = e_power(knots[i+1], ..., knots[i+degree]) / C(degree, power)``; see the
    module docstring for the identity. Formed in ``float64`` whatever the knots are
    stored in, by the elementary-symmetric recurrence, so nothing here re-runs either
    backend.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): The knot vector.
        degree (int): The polynomial degree.
        power (int): The monomial power, in ``[0, degree]``.

    Returns:
        npt.NDArray[np.float64]: One coefficient per basis function,
        ``len(knots) - degree - 1`` of them.
    """
    count = len(knots) - degree - 1
    out = np.zeros(count, dtype=np.float64)
    for i in range(count):
        symmetric = np.zeros(degree + 1, dtype=np.float64)
        symmetric[0] = 1.0
        for k in range(1, degree + 1):
            knot = float(knots[i + k])
            for j in range(k, 0, -1):
                symmetric[j] += knot * symmetric[j - 1]
        out[i] = symmetric[power] / float(math.comb(degree, power))
    return out


class _Marsden(NamedTuple):
    """The Marsden net over a set of directions, and the magnitude of each entry.

    Attributes:
        values (npt.NDArray[np.float64]): The coefficients, shape
            ``(*num_basis, num_components)``. Component ``c`` decodes row-major into a
            power multi-index, so one net carries every monomial the degrees admit.
        magnitude (npt.NDArray[np.float64]): ``prod_d max_i A^d_{i, r_d}`` per entry,
            the factor the relative bound multiplies. See the module docstring.
    """

    values: npt.NDArray[np.float64]
    magnitude: npt.NDArray[np.float64]


def _marsden_net(spaces: Sequence[BsplineSpace1D]) -> _Marsden:
    """The Marsden net over the given directions, with its per-entry magnitude.

    Args:
        spaces (Sequence[BsplineSpace1D]): One univariate space per direction, in axis
            order.

    Returns:
        _Marsden: The coefficients and their magnitudes.
    """
    tables = [
        [_marsden_column(space.knots, space.degree, power) for power in range(space.degree + 1)]
        for space in spaces
    ]
    largest = [
        [float(np.abs(column).max()) for column in per_direction] for per_direction in tables
    ]
    counts = tuple(space.num_basis for space in spaces)
    powers = tuple(space.degree + 1 for space in spaces)
    num_components = math.prod(powers)

    values = np.zeros((*counts, num_components), dtype=np.float64)
    magnitude = np.zeros((*counts, num_components), dtype=np.float64)
    for index in np.ndindex(*counts):
        for component in range(num_components):
            rest = component
            value = 1.0
            bound = 1.0
            for direction in reversed(range(len(spaces))):
                power = rest % powers[direction]
                rest //= powers[direction]
                value *= float(tables[direction][power][index[direction]])
                bound *= largest[direction][power]
            values[(*index, component)] = value
            magnitude[(*index, component)] = bound
    return _Marsden(values, magnitude)


def _accuracy_claim(
    refined: Bspline, magnitude: npt.NDArray[np.float64], num_refined: int
) -> AccuracyClaim:
    """The elementwise bound between a refined net and Marsden's closed form.

    Args:
        refined (~pantr.bspline.Bspline): The refined field, for its degrees and dtype.
        magnitude (npt.NDArray[np.float64]): The per-entry magnitude from
            :func:`_marsden_net`.
        num_refined (int): How many directions received knots, the ``D`` of the
            derivation.

    Returns:
        AccuracyClaim: The bound and its derivation.
    """
    degree = max(refined.degree)
    roundings = 2 * degree + 3 + num_refined * (7 * degree + 2)
    unit = unit_roundoff(refined.dtype)
    relative = roundings * unit / (1.0 - roundings * unit)
    return derived_accuracy(
        bound=relative * magnitude + roundings * underflow_floor(refined.dtype),
        why=(
            f"Marsden's identity gives the coefficients of u**r in closed form, so a "
            f"refinement that does not move the field must reproduce it on the refined "
            f"knots. gamma_{roundings} at unit roundoff {unit:.3e}, times "
            f"prod_d max_i A^d_(i, r_d) per component: {2 * degree + 1} roundings form "
            f"the closed form itself, one is the cast of that closed form into the "
            f"field's storage, {num_refined} refined direction(s) cost "
            f"{7 * degree + 2} each -- {5 * degree} in the band recurrence and "
            f"{2 * degree + 2} in the control-point sweep -- and one is the store. The "
            f"magnitude is a bound because the discrete B-splines of a refinement are "
            f"non-negative, so no partial sum exceeds the largest coarse coefficient. "
            f"{roundings} underflow floors are added for the components whose closed "
            f"form is an exact zero, where a relative bound asserts bit-identity it has "
            f"no grounds for."
        ),
    )


# ---------------------------------------------------------------------------
# The cases
# ---------------------------------------------------------------------------


class _Case(NamedTuple):
    """One field to refine under both backends, and how.

    The control net is always the Marsden net over the case's own directions, so one
    case serves the field-by-field parity comparison and the independent accuracy check
    without two nets to keep in step.

    Attributes:
        knots (tuple[tuple[float, ...], ...]): One knot vector per direction.
        degrees (tuple[int, ...]): One degree per direction.
        insertions (tuple[tuple[float, ...] | None, ...]): Knots to insert per
            direction; ``None`` skips it.
        counts (tuple[int | None, ...]): Subdivision factors per direction; ``None`` and
            1 both skip it.
        regularity (int | None): The continuity at every inserted knot, or ``None`` for
            ``degree - 1`` per direction.
        is_rational (bool): Whether the last stored component is declared a homogeneous
            weight. It changes the rank and nothing else here: refinement is linear and
            component-blind, and nothing in this file divides by a weight, so a
            "weight" that is a Marsden coefficient is a legitimate net for it even
            where it would be a degenerate NURBS.
        label (str): A short name for the failure message.
    """

    knots: tuple[tuple[float, ...], ...]
    degrees: tuple[int, ...]
    insertions: tuple[tuple[float, ...] | None, ...]
    counts: tuple[int | None, ...]
    regularity: int | None
    is_rational: bool
    label: str


_THIRD: Final = 1.0 / 3.0
"""A knot no binary format represents, so a refinement through it cannot round nothing."""

_CONSTANT: Final = (0.0, 0.5, 1.0)
"""Degree 0, two intervals, three basis functions: the piecewise-constant case."""

_LINEAR: Final = (0.0, 0.0, 0.5, 1.0, 1.0)
"""Degree 1, two intervals, three basis functions."""

_QUADRATIC: Final = (0.0, 0.0, 0.0, 0.25, 0.75, 1.0, 1.0, 1.0)
"""Degree 2, three intervals, five basis functions, dyadic throughout."""

_QUADRATIC_THIRDS: Final = (0.0, 0.0, 0.0, _THIRD, 1.0, 1.0, 1.0)
"""Degree 2, two intervals, four basis functions, and one knot that is not dyadic."""

_CUBIC: Final = (0.0, 0.0, 0.0, 0.0, 0.25, 3.0 / 7.0, 1.0, 1.0, 1.0, 1.0)
"""Degree 3, three intervals, six basis functions, two of its knots non-dyadic."""

_QUARTIC: Final = (0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.55, 1.0, 1.0, 1.0, 1.0, 1.0)
"""Degree 4, three intervals, seven basis functions. Sweep only: the highest degree here,
where the band is widest and the rounding budget largest."""

_SHIFTED_LINEAR: Final = (10.0, 10.0, 11.0, 12.0, 12.0)
"""Degree 1 on ``[10, 12]``: a different scale, so a scale-dependent slip shows."""

_UNCLAMPED: Final = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
"""Strictly increasing and clamped at neither end: legal, and the case where a band is
truncated because it addresses columns the coarse space does not have."""

_PERIODIC_UNIFORM: Final = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
"""The same vector read periodically, for the direction the C++ half refuses."""

CASES: Final = (
    _Case((_CONSTANT,), (0,), ((0.25,),), (3,), None, False, "a piecewise-constant curve"),
    _Case((_LINEAR,), (1,), ((0.25, _THIRD),), (4,), None, False, "a linear curve"),
    _Case((_QUADRATIC,), (2,), ((0.5,),), (2,), None, False, "a dyadic quadratic curve"),
    _Case(
        (_QUADRATIC_THIRDS,),
        (2,),
        ((_THIRD, _THIRD, 0.9),),
        (3,),
        0,
        False,
        "a quadratic curve refined through a repeat",
    ),
    _Case((_CUBIC,), (3,), ((0.1, 0.6),), (3,), 1, False, "a cubic curve"),
    _Case((_CUBIC,), (3,), ((0.1,),), (2,), -1, False, "a cubic curve cut to C^-1"),
    _Case(
        (_QUADRATIC_THIRDS, _SHIFTED_LINEAR),
        (2, 1),
        (None, (11.5,)),
        (1, 3),
        None,
        False,
        "a surface with one direction refined",
    ),
    _Case(
        (_QUADRATIC_THIRDS, _SHIFTED_LINEAR),
        (2, 1),
        ((0.5,), (11.5, 11.5)),
        (2, 3),
        0,
        False,
        "a surface with both directions refined",
    ),
    _Case(
        (_QUADRATIC_THIRDS, _SHIFTED_LINEAR, _CUBIC),
        (2, 1, 3),
        ((0.5,), None, (0.6,)),
        (2, 1, 3),
        None,
        True,
        "a rational volume with two directions refined",
    ),
    _Case((_UNCLAMPED,), (2,), ((0.5,),), (3,), None, False, "an unclamped curve"),
)
"""The shapes both backends must agree on, for both operations.

Each multi-direction case differs between its directions in the degree, the basis
count, the domain and the domain's width at once, so no permutation of the net's shape
is another admissible shape for its space and a transposed sweep cannot pass.
:func:`test_the_case_table_earns_its_keep` asserts every one of those rather than
leaving this paragraph to be believed.
"""


def _make_space(case: _Case, dtype: npt.DTypeLike) -> BsplineSpace:
    """Build ``case``'s space under whichever backend is active.

    Args:
        case (_Case): The shape to build.
        dtype (npt.DTypeLike): The knot storage format.

    Returns:
        ~pantr.bspline.BsplineSpace: The space, in the active backend's implementation.
    """
    return BsplineSpace(
        [
            BsplineSpace1D(np.asarray(knots, dtype=dtype), degree)
            for knots, degree in zip(case.knots, case.degrees, strict=True)
        ]
    )


def _make_field(case: _Case, dtype: npt.DTypeLike) -> tuple[Bspline, _Marsden]:
    """Build ``case``'s Marsden field under whichever backend is active.

    Args:
        case (_Case): The shape to build.
        dtype (npt.DTypeLike): The storage format.

    Returns:
        tuple[~pantr.bspline.Bspline, _Marsden]: The field, and the closed form it was
        built from.
    """
    space = _make_space(case, dtype)
    marsden = _marsden_net(space.spaces)
    return Bspline(space, marsden.values.astype(dtype), case.is_rational), marsden


def _insert(case: _Case, dtype: npt.DTypeLike) -> Bspline:
    """Refine ``case`` by :meth:`~pantr.bspline.Bspline.insert_knots`, active backend.

    Args:
        case (_Case): The shape and the knots to insert.
        dtype (npt.DTypeLike): The storage format.

    Returns:
        ~pantr.bspline.Bspline: The refined field.
    """
    field, _ = _make_field(case, dtype)
    if len(case.degrees) == 1:
        # A one-direction field takes the flat array, not a sequence of one: the oracle
        # runs `np.asarray(new_knots, dtype)` on it, which would read `[[0.25]]` as a
        # `(1, 1)` array and refuse it. `tests/_patches.py` in the downstream consumer
        # records the same asymmetry.
        return field.insert_knots(np.asarray(case.insertions[0], dtype=dtype))
    return field.insert_knots(
        [None if values is None else np.asarray(values, dtype=dtype) for values in case.insertions]
    )


def _subdivide(case: _Case, dtype: npt.DTypeLike) -> Bspline:
    """Refine ``case`` by :meth:`~pantr.bspline.Bspline.subdivide`, active backend.

    Args:
        case (_Case): The shape and the factors.
        dtype (npt.DTypeLike): The storage format.

    Returns:
        ~pantr.bspline.Bspline: The refined field.
    """
    field, _ = _make_field(case, dtype)
    return field.subdivide(list(case.counts), case.regularity)


def _both(case: _Case, dtype: npt.DTypeLike, *, subdivide: bool) -> tuple[Bspline, Bspline]:
    """Refine the same field under each backend.

    Everything happens inside the ``use_backend`` block, the refinement included: the
    branch :mod:`pantr.bspline._refinement_backend` takes is decided by the *active*
    backend, so refining outside the block would measure the wrong path.

    Args:
        case (_Case): The shape to build and how to refine it.
        dtype (npt.DTypeLike): The storage format.
        subdivide (bool): Whether to subdivide rather than insert explicit knots.

    Returns:
        tuple[~pantr.bspline.Bspline, ~pantr.bspline.Bspline]: ``(py, cpp)``, in the
        order :func:`~tests._parity_harness.assert_object_parity` names its arguments.
    """
    refine = _subdivide if subdivide else _insert
    with use_backend(Backend.PYTHON):
        py = refine(case, dtype)
    with use_backend(Backend.CPP):
        cpp = refine(case, dtype)
    return py, cpp


def _fields(dim: int) -> tuple[Field, ...]:
    """Every piece of a refined field's state, one field each, with its own argument.

    Args:
        dim (int): The field's number of parametric directions, which decides how many
            per-direction knot vectors there are to compare.

    Returns:
        tuple[Field, ...]: The field list for
        :func:`~tests._parity_harness.assert_object_parity`.
    """
    per_direction: list[Field] = []
    for direction in range(dim):
        per_direction.append(
            Field(
                f"space.spaces[{direction}].knots",
                bitwise_parity(
                    why=(
                        "the merged knot vector. `inserted_knot_vector` concatenates the old "
                        "vector with the insertions and stable-sorts it, so no arithmetic "
                        "touches a knot at all on the insertion path and a difference could "
                        "only be a lost value, a reordering or a narrowing cast. On the "
                        "subdivision path each new knot is `j * ((hi - lo) / n) + lo`, which "
                        "is numpy's own linspace expression reproduced rather than "
                        "simplified and computed in `double` whatever the storage is, so it "
                        "is the same three operations in the same order on both sides."
                    )
                ),
                read=lambda field, d=direction: field.space.spaces[d].knots,  # type: ignore[misc]
            )
        )
        per_direction.append(
            Field(
                f"space.spaces[{direction}].tolerance",
                bitwise_parity(
                    why=(
                        "the refined space derives its own tolerance from the merged vector, "
                        "before snapping, so it is a *new* quantity of the result rather than "
                        "one carried over -- and it is what every later domain and "
                        "multiplicity comparison on that space will use. The knots above "
                        "agree bit for bit and `knot_tolerance` is the same reduction over "
                        "them on both sides, so this does too; a difference would mean the "
                        "two backends disagree about which knots are the same knot, which no "
                        "comparison of the knots themselves would reveal."
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
                        "a flag, and the one the C++ half is allowed to be wrong about in "
                        "principle: it refuses to refine a periodic direction and carries an "
                        "unrefined one over whole, so a direction that came back "
                        "non-periodic would mean it converted a representation instead of "
                        "carrying it. Nothing in the coefficients or the knots would say so."
                    )
                ),
                read=lambda field, d=direction: field.space.spaces[d].periodic,  # type: ignore[misc]
            )
        )
    return (
        Field(
            "control_points",
            bitwise_parity(
                why=(
                    "the refined coefficients, and the only quantity here that arithmetic "
                    "touches. The two backends run the same IEEE-754 operations in the same "
                    "order: the band recurrence forms its quotient before multiplying by the "
                    "carried value, the sweep accumulates in the storage format in ascending "
                    "band order from an exact zero, and an off-band column is skipped rather "
                    "than added as an exact zero. See the module docstring for why the "
                    "oracle's `float64` unification of `saved` does not break that, and for "
                    "the build this claim depends on."
                )
            ),
        ),
        *per_direction,
        Field(
            "space.num_basis",
            exact_parity(
                why=(
                    "the per-direction basis counts the refined net is laid out on. Exact "
                    "integer counts, and the field's business rather than the space's: this "
                    "is the tuple the C++ constructor checks the refined net's extents "
                    "against, so a disagreement means the two backends laid one refinement "
                    "out on two different grids. A count that moved with the knots agreeing "
                    "would mean the sweep produced the wrong number of rows."
                )
            ),
        ),
        Field(
            "degree",
            exact_parity(
                why=(
                    "one integer per direction, in axis order, and refinement must not "
                    "change any of them -- that is the whole point of inserting a knot "
                    "rather than elevating. The order is the load-bearing part: nothing "
                    "about a degree reveals a transposition on a field whose directions "
                    "happen to agree, which is why every multi-direction case here differs "
                    "in it."
                )
            ),
        ),
        Field(
            "rank",
            exact_parity(
                why=(
                    "the stored component count less the weight column. Refinement touches "
                    "no component axis, so a rank that moved means the sweep consumed or "
                    "produced an axis -- and it is the one field that folds the rationality "
                    "flag against the shape, so it catches a refinement that preserved the "
                    "coefficient count while moving which axis carries the components."
                )
            ),
        ),
        Field(
            "is_rational",
            exact_parity(
                why=(
                    "the flag is carried through, and nothing transforms it. It is named as "
                    "its own field rather than left implicit because it is what `rank` "
                    "subtracts, so a flag dropped in the refinement shows up here as well as "
                    "one field away, and reading only `rank` could not say which moved."
                )
            ),
        ),
        Field(
            "dtype",
            exact_parity(
                why=(
                    "the storage format the refined field reports, carried by the class of "
                    "the C++ handle. A disagreement means the refinement changed precision, "
                    "which for a coefficient representable in both formats -- most of them -- "
                    "the bitwise claim above would not see."
                )
            ),
        ),
    )


# ---------------------------------------------------------------------------
# The field-by-field comparison
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
@pytest.mark.parametrize("subdivide", [False, True], ids=["insert_knots", "subdivide"])
def test_the_refinement_agrees_field_by_field(
    case: _Case, dtype: npt.DTypeLike, subdivide: bool
) -> None:
    """Every piece of the refined field's state agrees, under its own claim.

    Args:
        case (_Case): The shape to build and how to refine it.
        dtype (npt.DTypeLike): The storage format.
        subdivide (bool): Whether to subdivide rather than insert explicit knots.
    """
    if contraction_may_fuse():
        pytest.skip(_FUSING)
    py, cpp = _both(case, dtype, subdivide=subdivide)
    assert_object_parity(
        py=py,
        cpp=cpp,
        fields=_fields(len(case.degrees)),
        context=f"{'subdivide' if subdivide else 'insert_knots'} on {case.label} at {dtype}",
    )


# ---------------------------------------------------------------------------
# The independent accuracy check
# ---------------------------------------------------------------------------


def _clamped(case: _Case) -> bool:
    """Report whether every direction of ``case`` is clamped at both ends.

    Marsden's identity describes every coefficient only where it is. On an unclamped
    vector the leading and trailing bands address columns the coarse space does not
    have, they are truncated, and their coefficients are not the closed form; see
    ``cpp/tests/test_bspline_refinement.cpp``, which checks the interior rows of such a
    vector with a vacuity guard on how many it found.

    Args:
        case (_Case): The shape.

    Returns:
        bool: True when the identity covers the whole net.
    """
    return all(
        knots[0] == knots[degree] and knots[-1] == knots[-degree - 1]
        for knots, degree in zip(case.knots, case.degrees, strict=True)
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
@pytest.mark.parametrize("subdivide", [False, True], ids=["insert_knots", "subdivide"])
def test_marsden_s_identity_survives_the_refinement(
    case: _Case, dtype: npt.DTypeLike, subdivide: bool
) -> None:
    """The refined coefficients are the closed form again, on the refined knots.

    The independent check. It says the refinement did not move the field, which no
    comparison between the two backends can say; see the module docstring for the
    identity, for why it is complete rather than partial, and for the bound.

    Run on the C++ result, which is the backend under test. The oracle is covered by
    the bitwise claim above carrying the same numbers to it, and by
    ``tests/test_bspline_knot_insertion.py``.

    Args:
        case (_Case): The shape to build and how to refine it.
        dtype (npt.DTypeLike): The storage format.
        subdivide (bool): Whether to subdivide rather than insert explicit knots.
    """
    if not _clamped(case):
        pytest.skip(
            "the identity does not describe a truncated band, so an unclamped vector is "
            "checked row by row in cpp/tests/test_bspline_refinement.cpp instead"
        )
    with use_backend(Backend.CPP):
        refined = _subdivide(case, dtype) if subdivide else _insert(case, dtype)
    expected = _marsden_net(refined.space.spaces)
    if subdivide:
        num_refined = sum(1 for count in case.counts if count is not None and count > 1)
    else:
        num_refined = sum(1 for values in case.insertions if values)
    assert_accuracy(
        np.asarray(refined.control_points, dtype=np.float64),
        expected.values,
        _accuracy_claim(refined, expected.magnitude, num_refined),
        context=f"{'subdivide' if subdivide else 'insert_knots'} on {case.label} at {dtype}",
    )


def test_the_marsden_oracle_is_not_vacuous() -> None:
    """The closed form disagrees with a wrong net, so agreeing with it says something.

    The control for the check above. Marsden's coefficients of ``u**r`` over the
    *refined* knots differ from those over the *coarse* ones -- if they did not, a
    refinement that returned its input unchanged would pass -- and they differ by far
    more than the bound.
    """
    case = CASES[4]
    with use_backend(Backend.CPP):
        field, coarse = _make_field(case, np.float64)
        refined = _insert(case, np.float64)
    fine = _marsden_net(refined.space.spaces)
    assert fine.values.shape != coarse.values.shape, (
        "the refinement did not change the net's shape, so the two closed forms could "
        "not be told apart by shape either"
    )
    # The coarse net padded with its own last row is the crudest wrong answer a
    # refinement could give; the closed form must reject it.
    missing = fine.values.shape[0] - coarse.values.shape[0]
    padded = np.concatenate([coarse.values, np.repeat(coarse.values[-1:], missing, axis=0)])
    worst = float(np.abs(padded - fine.values).max())
    bound = float(np.asarray(_accuracy_claim(refined, fine.magnitude, 1).bound).max())
    assert worst > 1.0e6 * bound, (
        f"the wrong net is only {worst:.3e} from the closed form against a bound of "
        f"{bound:.3e}, so the accuracy check would not have rejected it"
    )
    assert field.rank == refined.rank


# ---------------------------------------------------------------------------
# The wide sweep
# ---------------------------------------------------------------------------


def _sweep_cases() -> list[_Case]:
    """Ten times the shipped case count, over the same claim.

    Built rather than listed, from every combination of a knot vector, a subdivision
    factor and a regularity the degree admits. Non-dyadic vectors and non-dyadic
    factors are both present, which is what makes the accuracy bound bind: a dyadic
    refinement of a dyadic vector rounds nothing at all, and a sweep of only those
    would report agreement without ever asking the bound a question.

    Returns:
        list[_Case]: The sweep, in a fixed order.
    """
    vectors: tuple[tuple[tuple[float, ...], int], ...] = (
        (_CONSTANT, 0),
        (_LINEAR, 1),
        (_QUADRATIC, 2),
        (_QUADRATIC_THIRDS, 2),
        (_CUBIC, 3),
        (_QUARTIC, 4),
        (_SHIFTED_LINEAR, 1),
    )
    cases: list[_Case] = []
    for knots, degree in vectors:
        lo = float(knots[degree])
        hi = float(knots[-degree - 1])
        for factor in (2, 3, 4, 5, 7):
            for regularity in (*range(-1, degree), None):
                # An interior value that is not a breakpoint and not representable.
                inserted = lo + (hi - lo) * _THIRD
                cases.append(
                    _Case(
                        (knots,),
                        (degree,),
                        ((inserted,),),
                        (factor,),
                        regularity,
                        bool(len(cases) % 3 == 0) and degree > 0,
                        f"degree {degree} by {factor} at regularity {regularity}",
                    )
                )
    # A handful of surfaces, so the sweep is not one-dimensional throughout.
    for factor in (2, 3):
        for regularity in (0, None):
            cases.append(
                _Case(
                    (_QUADRATIC_THIRDS, _SHIFTED_LINEAR),
                    (2, 1),
                    ((0.5,), (11.5,)),
                    (factor, factor),
                    regularity,
                    False,
                    f"a surface by {factor} at regularity {regularity}",
                )
            )
    return cases


@pytest.mark.slow
def test_the_claims_hold_over_a_ten_times_sweep() -> None:
    """Both claims hold over more than ten times the shipped parametrization.

    FELIGN/pantr#398 asks that the bound be verified over a sweep at least ten times the
    one that ships, on the reasoning that a bound checked only by its own shipped sweep
    has not been checked. Two guards keep the sweep from passing trivially: it must be
    ten times the shipped comparison count, and the accuracy bound must actually be
    **approached** somewhere -- a sweep on which every coefficient is exact says nothing
    about a bound.
    """
    if contraction_may_fuse():
        pytest.skip(_FUSING)
    shipped = len(CASES) * len(DTYPES) * 2  # the two operations
    cases = _sweep_cases()
    swept = len(cases) * len(DTYPES) * 2
    assert swept >= 10 * shipped, (
        f"the sweep is {swept} comparisons against {shipped} shipped, which is under "
        f"the ten-times floor of {10 * shipped}"
    )

    differing_bits = 0
    worst_ratio = 0.0
    approached = 0
    for case in cases:
        for dtype in DTYPES:
            for subdivide in (False, True):
                py, cpp = _both(case, dtype, subdivide=subdivide)
                if not np.array_equal(
                    np.asarray(py.control_points).view(
                        np.uint32 if np.dtype(dtype) == np.float32 else np.uint64
                    ),
                    np.asarray(cpp.control_points).view(
                        np.uint32 if np.dtype(dtype) == np.float32 else np.uint64
                    ),
                ) or not np.array_equal(py.space.spaces[0].knots, cpp.space.spaces[0].knots):
                    differing_bits += 1
                expected = _marsden_net(cpp.space.spaces)
                if subdivide:
                    num_refined = sum(1 for c in case.counts if c is not None and c > 1)
                else:
                    num_refined = sum(1 for values in case.insertions if values)
                claim = _accuracy_claim(cpp, expected.magnitude, num_refined)
                deviation = assert_accuracy(
                    np.asarray(cpp.control_points, dtype=np.float64),
                    expected.values,
                    claim,
                    context=f"sweep: {case.label} at {dtype}",
                )
                worst_ratio = max(worst_ratio, deviation.max_ratio_to_bound)
                approached += int(deviation.num_differing > 0)

    assert differing_bits == 0, f"{differing_bits} of {swept} comparisons differed bitwise"
    assert worst_ratio > 0.0, (
        "no coefficient in the whole sweep differed from the closed form, so the "
        "accuracy bound was never asked a question. Add a non-dyadic vector or factor."
    )
    assert approached > swept // 10, (
        f"only {approached} of {swept} comparisons had any deviation from the closed "
        f"form, so the sweep is mostly exact arithmetic and is not exercising the bound"
    )


# ---------------------------------------------------------------------------
# The declared boundaries
# ---------------------------------------------------------------------------


def _periodic_field(dtype: npt.DTypeLike) -> Bspline:
    """A surface whose first direction is periodic and whose second is clamped.

    Args:
        dtype (npt.DTypeLike): The storage format.

    Returns:
        ~pantr.bspline.Bspline: The field, in the active backend's implementation.
    """
    space = BsplineSpace(
        [
            BsplineSpace1D(np.asarray(_PERIODIC_UNIFORM, dtype=dtype), 2, periodic=True),
            BsplineSpace1D(np.asarray(_SHIFTED_LINEAR, dtype=dtype), 1),
        ]
    )
    net = np.arange(math.prod(space.num_basis) * 2, dtype=dtype).reshape(*space.num_basis, 2)
    return Bspline(space, net)


def _binding() -> Any:
    """The compiled extension, imported late so a missing one skips rather than errors.

    Returns:
        Any: The :mod:`pantr._pantr_cpp` module.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    return _pantr_cpp


def test_a_periodic_direction_is_refused_by_the_binding_and_served_by_the_oracle() -> None:
    """The C++ half refuses a periodic direction; the public API still refines it.

    The declared boundary in both halves. Refining a periodic direction needs the
    open-to-periodic conversions, which ``cpp/include/pantr/bspline/bspline.hpp`` lists
    as their own port; :mod:`pantr.bspline._refinement_backend` therefore keeps the
    oracle for such a field.

    The two assertions together are what say the routing happened: the binding refuses
    the call, and the public method that would otherwise make it succeeds. Neither alone
    would -- a result is a result whichever path produced it.
    """
    with use_backend(Backend.CPP):
        field = _periodic_field(np.float64)
        with pytest.raises(ValueError, match="open-to-periodic conversion"):
            _binding().insert_bspline_knots(field._impl, [np.array([0.25]), np.empty(0)])
        refined = field.insert_knots([[0.25], None])

    assert refined.space.spaces[0].periodic, "the periodic direction lost its wrap"
    assert refined.space.spaces[0].knots.size == field.space.spaces[0].knots.size + 1
    # The clamped direction is untouched and shares its wrapper, which is the same
    # contract the all-C++ path keeps.
    assert refined.space.spaces[1] is field.space.spaces[1]


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_periodic_refinement_agrees_between_the_backends(dtype: npt.DTypeLike) -> None:
    """A periodic direction refines identically under either backend -- common mode.

    Said out loud rather than counted as evidence: both backends run the *same*
    implementation here, since :mod:`pantr.bspline._refinement_backend` routes a
    periodic direction to the oracle. What this pins is that the routing produces a
    complete, correct field under the C++ backend -- the refined periodic space, the
    carried-over clamped one and the coefficients -- and not that two implementations
    agree.

    Args:
        dtype (npt.DTypeLike): The storage format.
    """
    with use_backend(Backend.PYTHON):
        py = _periodic_field(dtype).insert_knots([[0.25], [11.5]])
    with use_backend(Backend.CPP):
        cpp = _periodic_field(dtype).insert_knots([[0.25], [11.5]])
    assert_object_parity(
        py=py, cpp=cpp, fields=_fields(2), context=f"a periodic surface at {dtype}"
    )


def test_an_unrefined_periodic_direction_goes_through_the_cpp_path() -> None:
    """A periodic direction given no knots does not push the call onto the oracle.

    The other half of the routing rule, and the one an over-broad condition would get
    wrong: only a periodic direction *receiving* knots is a problem, so a field with an
    untouched periodic direction still refines in C++. Pinned by the binding accepting
    the same call the public method makes.
    """
    with use_backend(Backend.CPP):
        field = _periodic_field(np.float64)
        through_the_binding = _binding().insert_bspline_knots(
            field._impl, [np.empty(0), np.array([11.5])]
        )
        refined = field.insert_knots([None, [11.5]])

    assert np.array_equal(refined.control_points, through_the_binding.control_points)
    assert refined.space.spaces[0] is field.space.spaces[0], (
        "the untouched periodic direction did not keep its wrapper"
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_near_duplicate_insertion_is_snapped_but_refined_unsnapped(
    dtype: npt.DTypeLike,
) -> None:
    """A knot within tolerance of an existing one collapses onto it, after the sweep.

    The one place the refined *space* and the vector the *matrix* was built from are
    not the same array, and it is easy to get wrong in the tidy direction: the merged
    vector carries the near-duplicate as its own distinct value, the two-scale matrix is
    built from **that**, and only then does ``BsplineSpace1D`` snap it onto its class.
    Building the matrix from the snapped vector instead would be a different matrix and
    a different geometry, so the order is load-bearing rather than incidental.

    A mutation that swapped the refined space's snapping for ``as_given`` survived the
    rest of this file, because nothing else here inserts a knot close enough to an
    existing one for snapping to do anything. This is that case, and it is the reason
    the field list's ``knots`` claim is bitwise rather than bounded: one ulp is exactly
    the difference at stake.

    Args:
        dtype (npt.DTypeLike): The storage format.
    """
    resolved = np.dtype(dtype)
    knots = np.asarray((0.0, 0.0, 0.0, 0.25, 0.75, 1.0, 1.0, 1.0), dtype=resolved)
    quarter = np.asarray(0.25, dtype=resolved)
    near = np.nextafter(quarter, np.asarray(1.0, dtype=resolved))
    net = np.arange(5 * 2, dtype=resolved).reshape(5, 2)

    refined: list[Bspline] = []
    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            space = BsplineSpace([BsplineSpace1D(knots, 2)])
            assert float(near - quarter) < space.spaces[0].tolerance, (
                "one ulp is not inside this space's tolerance, so nothing here is a "
                "near duplicate and the test would assert snapping that cannot happen"
            )
            refined.append(Bspline(space, net).insert_knots(np.asarray([near], dtype=resolved)))

    for field in refined:
        stored = field.space.spaces[0].knots
        assert np.count_nonzero(stored == quarter) == 2, (
            f"the near duplicate was not collapsed onto its class: {stored!r}"
        )
        assert not np.any(stored == near), (
            f"the refined space stored the near duplicate as its own knot: {stored!r}"
        )
    assert_object_parity(
        py=refined[0],
        cpp=refined[1],
        fields=_fields(1),
        context=f"a near-duplicate insertion at {resolved}",
    )


def test_the_backend_entry_points_agree_when_nothing_is_refined() -> None:
    """Called directly with nothing to refine, both branches return an unrefined copy.

    Not reachable through the public methods, which refuse an all-empty argument first.
    It is asserted because :mod:`pantr.bspline._refinement_backend`'s two entry points
    are private symbols in a package whose private symbols a downstream consumer
    already imports, and because the two branches would otherwise disagree: the C++
    entry points restate the wrapper's "at least one direction" refusal for a C++
    caller, while the oracle returns a copy.
    :func:`~pantr.bspline._refinement_backend._the_cpp_backend_can_take_it` keeps the
    case on the oracle path so that the dispatcher has one answer rather than two.
    """
    case = CASES[7]
    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            field, _ = _make_field(case, np.float64)
            for result in (
                insert_knots_into_field(field, [None, np.empty(0)]),
                subdivide_field(field, [1, None], None),
            ):
                assert result.space.num_basis == field.space.num_basis
                assert np.array_equal(result.control_points, field.control_points)
                for direction in range(field.dim):
                    assert result.space.spaces[direction] is field.space.spaces[direction]

    # And the binding still refuses it, which is the asymmetry being routed around
    # rather than removed: a C++ caller with no wrapper in front of it gets the
    # wrapper's own message.
    with use_backend(Backend.CPP):
        field, _ = _make_field(case, np.float64)
        with pytest.raises(ValueError, match="At least one direction"):
            _binding().insert_bspline_knots(field._impl, [np.empty(0), np.empty(0)])


def test_wrap_over_refuses_a_dimension_it_cannot_match_positionally() -> None:
    """The space wrapper's reuse is positional, so a changed dimension is refused.

    ``BsplineSpace._wrap_over`` reuses direction ``d``'s wrapper when the new
    implementation still holds direction ``d``'s implementation, which is only sound
    for an operation that preserves the directions and their order. Refinement does; a
    later boundary extraction or permutation would not, and would otherwise hand back a
    wrapper for a different direction while every value comparison agreed. The
    precondition is checked, and this is the check.
    """
    with use_backend(Backend.CPP):
        surface, _ = _make_field(CASES[7], np.float64)
        one_d, _ = _make_field(CASES[2], np.float64)
        with pytest.raises(ValueError, match="one prior wrapper per direction"):
            BsplineSpace._wrap_over(surface.space._impl, one_d.space.spaces)


def test_a_zero_size_insertion_is_skipped_whatever_its_rank() -> None:
    """An empty array skips its direction even when its shape is not 1-D.

    The regression test for a divergence a review round found, and the exact input that
    elicited it: ``[np.empty((0, 3)), [0.5]]``. The oracle skips on ``nk.size == 0`` in
    ``_insert_knots_bspline``, one level *above* the rank check in
    ``_compute_inserted_knot_vector_1d``, so a zero-size array never reaches that check
    and the refinement succeeds. ``_flat_insertions`` tested the rank first and refused
    it -- the port raising where the oracle returns.

    That direction of divergence is the worse one and it is invisible to a comparison of
    refusal texts, which is why it needs its own test rather than a row in the refusal
    table. Both orderings of the pair are exercised, because a fix that skipped on size
    only in the leading position would still be wrong.
    """
    case = _Case(
        (_QUADRATIC_THIRDS, _SHIFTED_LINEAR),
        (2, 1),
        (None, None),
        (1, 1),
        None,
        False,
        "the zero-size input",
    )
    for label, argument, refined_axis in (
        ("leading", [np.empty((0, 3)), np.array([11.5])], 1),
        ("trailing", [np.array([0.5]), np.empty((2, 0))], 0),
    ):
        shapes: list[tuple[int, ...]] = []
        for backend in (Backend.PYTHON, Backend.CPP):
            with use_backend(backend):
                field, _ = _make_field(case, np.float64)
                before = field.control_points.shape
                refined = field.insert_knots(argument)
                shapes.append(refined.control_points.shape)
                assert (
                    refined.space.spaces[1 - refined_axis] is field.space.spaces[1 - refined_axis]
                ), f"{label}: the zero-size direction did not keep its wrapper"
                assert refined.control_points.shape[refined_axis] == before[refined_axis] + 1, (
                    f"{label}: the other direction was not refined"
                )
        assert shapes[0] == shapes[1], f"{label}: {shapes[0]} against {shapes[1]}"

    # A zero-size array of any rank still counts as empty for the "at least one
    # direction" refusal, so the two backends refuse the all-empty call identically.
    messages: list[str] = []
    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            field, _ = _make_field(case, np.float64)
            with pytest.raises(ValueError) as raised:
                field.insert_knots([np.empty((0, 3)), np.empty(0)])
            messages.append(str(raised.value))
    assert messages[0] == messages[1], messages
    assert "At least one direction" in messages[0]


def test_the_refusal_order_divergence_is_exactly_one_input() -> None:
    """The two backends disagree on which of two bad arguments they report, and only there.

    The one divergence :mod:`pantr.bspline._refinement_backend` records. The oracle
    checks an insertion array's rank inside its per-direction loop, so with direction 0
    out of domain *and* direction 1 not 1D it reports the domain; the C++ path checks
    every rank before the call, so it reports the rank. Both texts are the oracle's and
    both are ``ValueError``.

    Pinned in both directions rather than described, because a divergence nothing
    asserts is a divergence that will be "fixed" by accident in either direction.
    """
    case = _Case(
        (_QUADRATIC_THIRDS, _SHIFTED_LINEAR),
        (2, 1),
        (None, None),
        (1, 1),
        None,
        False,
        "the divergent input",
    )
    bad = [np.array([5.0]), np.array([[11.5, 11.5]])]

    with use_backend(Backend.PYTHON):
        field, _ = _make_field(case, np.float64)
        with pytest.raises(ValueError, match="outside the domain") as oracle:
            field.insert_knots(bad)
    with use_backend(Backend.CPP):
        field, _ = _make_field(case, np.float64)
        with pytest.raises(ValueError, match="must be a 1D array-like") as port:
            field.insert_knots(bad)

    assert "1D array-like" not in str(oracle.value)
    assert "outside the domain" not in str(port.value)

    # With only the rank at fault the two agree, text for text, which is what says the
    # divergence is the *order* and not the message.
    only_rank = [None, np.array([[11.5, 11.5]])]
    messages = []
    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            field, _ = _make_field(case, np.float64)
            with pytest.raises(ValueError) as raised:
                field.insert_knots(only_rank)
            messages.append(str(raised.value))
    assert messages[0] == messages[1], messages


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------


def _split_rendered_list(message: str) -> tuple[str, list[float]]:
    """Split a refusal that ends in a rendered list into its text and its numbers.

    Args:
        message (str): The refusal text, ending in ``": [a b c]"``.

    Returns:
        tuple[str, list[float]]: Everything before the list, and the values in it.
    """
    prefix, _, listed = message.rpartition(": ")
    return prefix, [float(token) for token in listed.strip("[]").split()]


class _Refusal(NamedTuple):
    """One bad call and where its refusal comes from.

    Attributes:
        label (str): A short name for the failure message.
        call (str): ``"insert_knots"`` or ``"subdivide"``.
        argument (Any): What to pass.
        regularity (int | None): The second argument to ``subdivide``.
        common_mode (bool): Whether the refusal is
            :class:`pantr.bspline.Bspline`'s own, above the backend branch, and
            therefore not evidence that two implementations agree.
        lists_values (bool): Whether the message ends in a rendered list of the
            offending values. Those are compared as *numbers* plus the text in front
            of them, because the oracle renders the list with ``numpy``'s repr --
            which chooses a shared precision across the elements and pads them -- and
            ``cpp/include/pantr/bspline/knot_insertion.hpp`` states that reproducing
            that is a formatting port nobody needs. So ``[5.]`` against ``[5.0]`` is
            the intended difference and the values behind them are the claim.
    """

    label: str
    call: str
    argument: Any
    regularity: int | None
    common_mode: bool
    lists_values: bool = False


_REFUSALS: Final = (
    _Refusal("a sequence of the wrong length", "insert_knots", [[0.5]], None, True),
    _Refusal("every direction empty", "insert_knots", [[], []], None, True),
    _Refusal("a knot outside the domain", "insert_knots", [[5.0], None], None, False, True),
    _Refusal("a knot array that is not 1D", "insert_knots", [[[0.5, 0.5]], None], None, False),
    _Refusal(
        "a multiplicity above degree + 1",
        "insert_knots",
        [[_THIRD, _THIRD, _THIRD, _THIRD], None],
        None,
        False,
    ),
    _Refusal("a count sequence of the wrong length", "subdivide", [2], None, True),
    _Refusal("a count below one", "subdivide", [0, 2], None, True),
    _Refusal("no direction above one", "subdivide", [1, 1], None, True),
    _Refusal("a regularity above degree - 1", "subdivide", [2, 2], 2, True),
    _Refusal("a regularity below minus one", "subdivide", [2, 2], -2, True),
)
"""Every refusal both operations can produce, and whether it is common mode.

The four that are not are the ones a cross-backend comparison actually decides: the
domain check, the multiplicity check and the rank check are made by
``pantr/bspline/refinement.hpp`` and ``_refinement_backend`` on one side and by
``_compute_inserted_knot_vector_1d`` on the other, so the texts really are two
independent literals. The six marked common mode are
:class:`pantr.bspline.Bspline`'s own, above the branch, and are compared anyway --
cheaply, and so that a future cut that moves one below the branch does not silently
stop being checked.
"""


@pytest.mark.parametrize("refusal", _REFUSALS, ids=lambda refusal: refusal.label)
def test_the_refusals_read_the_same_under_both_backends(refusal: _Refusal) -> None:
    """A bad call is refused with the same text on either backend.

    Args:
        refusal (_Refusal): The bad call.
    """
    case = _Case(
        (_QUADRATIC_THIRDS, _SHIFTED_LINEAR),
        (2, 1),
        (None, None),
        (1, 1),
        None,
        False,
        "a surface",
    )
    messages: list[str] = []
    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            field, _ = _make_field(case, np.float64)
            with pytest.raises(ValueError) as raised:
                if refusal.call == "insert_knots":
                    field.insert_knots(refusal.argument)
                else:
                    field.subdivide(refusal.argument, refusal.regularity)
            messages.append(str(raised.value))
    if refusal.lists_values:
        oracle_prefix, oracle_values = _split_rendered_list(messages[0])
        port_prefix, port_values = _split_rendered_list(messages[1])
        assert oracle_prefix == port_prefix, (
            f"{refusal.label}: the oracle says {oracle_prefix!r} and the port says {port_prefix!r}"
        )
        assert oracle_values == port_values, (
            f"{refusal.label}: the oracle listed {oracle_values} and the port listed {port_values}"
        )
        return
    assert messages[0] == messages[1], (
        f"{refusal.label}: the oracle says {messages[0]!r} and the port says {messages[1]!r}"
    )


def test_a_field_from_the_other_backend_is_refused() -> None:
    """Refining a Python-built field under the C++ backend refuses rather than converts.

    ``design/cross_backend_types.md`` forbids reconciling two implementations of one
    type by converting between them, and :meth:`pantr.bspline.Bspline._mutate` already
    refuses the same mismatch for the three ``in_place=`` methods. The refusal is a
    property of *taking the C++ route*, which is
    :func:`pantr.bezier._bezier_backend._cpp_handle`'s asymmetry: it fires under the C++
    backend and not under the Python one, where the oracle runs happily over a C++
    field's read-only control points.
    """
    case = CASES[2]
    with use_backend(Backend.PYTHON):
        python_field, _ = _make_field(case, np.float64)
    with use_backend(Backend.CPP):
        cpp_field, _ = _make_field(case, np.float64)
        with pytest.raises(TypeError, match="different backend"):
            python_field.insert_knots([0.5])
        with pytest.raises(TypeError, match="different backend"):
            python_field.subdivide(2)

    # The other direction is not a refusal, and that is the asymmetry rather than an
    # oversight: the oracle can read a C++ field's coefficients.
    with use_backend(Backend.PYTHON):
        refined = cpp_field.insert_knots([0.5])
    assert refined.control_points.shape[0] == cpp_field.control_points.shape[0] + 1


# ---------------------------------------------------------------------------
# The wire format and the wrapper contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", CASES[:4], ids=lambda case: case.label)
def test_a_refined_field_survives_a_pickle_round_trip_across_the_backends(
    case: _Case, dtype: npt.DTypeLike
) -> None:
    """A refined field pickles under either backend and loads under either.

    The refined field is the new picklable object this cut introduces -- it is built by
    ``Bspline._wrap_over`` rather than by the constructor, so its space wrapper comes
    from ``BsplineSpace._wrap_over`` and its directions from ``BsplineSpace1D._wrap``,
    none of which the existing round-trip tests reach. A pickle written under one
    backend has to load under the other, or the backend switch would silently become a
    data-format switch.

    Args:
        case (_Case): The shape to build and refine.
        dtype (npt.DTypeLike): The storage format.
    """
    for writer in (Backend.PYTHON, Backend.CPP):
        with use_backend(writer):
            refined = _insert(case, dtype)
            wire = pickle.dumps(refined)
        for reader in (Backend.PYTHON, Backend.CPP):
            with use_backend(reader):
                restored = pickle.loads(wire)
            assert type(restored) is Bspline
            assert restored.degree == refined.degree
            assert restored.rank == refined.rank
            assert restored.is_rational == refined.is_rational
            assert np.dtype(restored.dtype) == np.dtype(refined.dtype)
            assert np.array_equal(restored.control_points, refined.control_points), (
                f"{writer.name} -> {reader.name}: the coefficients moved"
            )
            for direction in range(len(case.degrees)):
                assert np.array_equal(
                    restored.space.spaces[direction].knots,
                    refined.space.spaces[direction].knots,
                )


def test_the_refined_field_shares_the_space_its_implementation_holds() -> None:
    """The wrapper in front of a refined field presents its own implementation's space.

    ``Bspline._wrap_over`` exists so that the field's implementation and the space
    wrapper in front of it are one answer rather than two computations of one. Under the
    C++ backend that is checkable directly: the handle the wrapper's space holds must be
    the very handle the field's implementation holds.
    """
    with use_backend(Backend.CPP):
        refined = _insert(CASES[7], np.float64)
        assert refined.space._impl is refined._impl.space
        for direction, one_d in enumerate(refined._impl.space.spaces):
            assert refined.space.spaces[direction]._impl is one_d


def test_a_refined_field_has_a_cold_derived_block() -> None:
    """A refinement shares no memo with the field it came from.

    ``design/bspline_derived_caches.md`` asks that the derived block be replaced
    wholesale rather than invalidated piece by piece, and ``Bspline._take`` is the only
    writer. A field built by ``_wrap_over`` goes through it, so both memos start cold --
    which matters because a refined field is a *different geometry* and a Bézier
    decomposition or a point-inversion context carried over from its parent would be
    silently wrong rather than merely stale.
    """
    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            refined = _insert(CASES[2], np.float64)
        assert refined._derived.beziers is None
        assert refined._derived.locate is None


def test_the_wrapper_is_still_immutable_after_a_refinement() -> None:
    """A refined field refuses attribute assignment, like any other.

    ``_wrap_over`` fills the slots through ``object.__setattr__``, which is the only
    door; a wrapper that had somehow acquired a writable one would be a second way to
    reseat a value.
    """
    with use_backend(Backend.CPP):
        refined = _insert(CASES[2], np.float64)
    with pytest.raises(AttributeError, match="immutable"):
        refined._impl = None  # type: ignore[assignment]
    with pytest.raises(AttributeError, match="immutable"):
        refined.space = None  # type: ignore[misc, assignment]


# ---------------------------------------------------------------------------
# The case table
# ---------------------------------------------------------------------------


def test_the_case_table_earns_its_keep() -> None:
    """Each case is asymmetric where a symmetric one would hide a defect.

    Four properties the module docstring and ``CASES`` rely on, asserted rather than
    believed: every multi-direction case differs between its directions in the degree,
    the basis count and the domain; some case is rational; some carries a knot no binary
    format represents; and the operations really do refine, i.e. no case is a no-op.
    """
    multi = [case for case in CASES if len(case.degrees) > 1]
    assert multi, "no multi-direction case, so nothing here could see a transposition"
    for case in multi:
        assert len(set(case.degrees)) == len(case.degrees), f"{case.label}: repeated degree"
        counts = [
            len(knots) - degree - 1 for knots, degree in zip(case.knots, case.degrees, strict=True)
        ]
        assert len(set(counts)) == len(counts), f"{case.label}: repeated basis count"
        # Not *all* distinct: what makes a transposition detectable is that no
        # permutation of the net's shape is another admissible shape, which the
        # distinct basis counts above already give. What a repeated domain would hide
        # is a scale-dependent slip, so it is enough that the directions do not all
        # share one.
        domains = [
            (knots[degree], knots[-degree - 1])
            for knots, degree in zip(case.knots, case.degrees, strict=True)
        ]
        assert len(set(domains)) > 1, f"{case.label}: every direction has one domain"

    assert any(case.is_rational for case in CASES), "no rational case"
    assert any(not _clamped(case) for case in CASES), "no unclamped case"
    assert any(
        any(knot not in (0.0, 0.25, 0.5, 0.75, 1.0) for knots in case.knots for knot in knots)
        for case in CASES
    ), "every knot in the table is dyadic, so no refinement here has to round"

    with use_backend(Backend.PYTHON):
        for case in CASES:
            field, _ = _make_field(case, np.float64)
            for refined in (_insert(case, np.float64), _subdivide(case, np.float64)):
                assert refined.control_points.size > field.control_points.size, (
                    f"{case.label}: the refinement produced no new coefficients"
                )
