"""Parity for the two n-dimensional Bézier evaluation entry points.

**Two claims, one per entry point, and neither is bitwise.** That is the whole shape
of this file, and it follows from a measurement rather than from a preference:
``scripts/measure_bezier_nd_widths.py`` runs rival models against each oracle and
reports how often they disagree, so a match cannot come from a check that could not
fail. What it found, at ``1c5fba7`` with numpy 2.4.6:

* the ``np.einsum`` contraction of ``_evaluate_bezier_nd_pts_array`` accumulates in
  the Bézier's **own storage format**, not in ``float64`` -- 336 of 336 against 0 of
  336 at ``float32``, fully discriminated. This is ``design/backend_parity.md``
  Rule 9 biting: the house ``accumulator_t<float> == double`` policy exists because
  *Numba* promotes a ``float64`` scalar against a ``float32`` array, and nothing on
  this path is Numba. Inheriting it would have made the claim be about arithmetic
  nobody performs;
* an ascending-index transliteration reproduces that ``einsum`` bit for bit wherever
  the contraction's trailing block holds two or more elements, and **stops** where
  that block is a single element -- which is every scalar-valued non-rational
  Bézier, because numpy then dispatches a vectorised reduction whose summation tree
  is a property of the host rather than of the expression;
* the ``np.tensordot`` contraction of ``_evaluate_bezier_nd_lattice`` matches no
  width model at any shape swept, because it reshapes to a matrix product and
  reaches BLAS.

So one C++ kernel cannot be exact against both oracles, and this file states two
bounded claims rather than one covering both.

The derivation, which is shared
-------------------------------

Both entry points contract ``dim`` times in sequence. Direction ``d`` sums ``n_d``
terms, and a sum of ``n`` products commits at most ``n`` roundings per term **in any
summation order**, so by Higham's standard dot-product result each stage contributes
a relative perturbation bounded by ``gamma_{n_d}``. Composing them,
``(1 + theta_a)(1 + theta_b) = 1 + theta_{a+b}``, gives ``gamma_N`` with
``N = sum_d n_d`` -- which is the ``stages`` field below, one accumulator rounding
each and no narrowing store, since the accumulator *is* the storage format.

The magnitude that turns that into an absolute bound is the **absolute-value
companion**: the same contraction run on ``|c|``. It is exact rather than merely
valid here, because a Bernstein basis is non-negative, so the companion is the
magnitude actually reachable at each output element rather than an over-estimate of
it. ``max|c|`` was refused before it was written, for the reason Rule 10 records: on
a net spanning decades whose output cancels, a flat amplification is a bound the
harness has to reject as vacuous.

The harness doubles the result, and here that factor is a derivation rather than a
margin: neither backend is exact, each sits within its own one-sided bound, and
their difference is bounded by the sum.

**What differs between the two claims is not the formula but the mechanism it
covers.** The pts-array claim covers a summation tree numpy chooses per output
shape, on a host whose vectorised reduction is IFUNC-dispatched. The lattice claim
covers BLAS: blocking, a different traversal order, and a fused multiply-add this
project does not control. The two are separately derived below because a single
``why`` would name one mechanism and be quoted at a failure of the other.

Three consequences worth stating
--------------------------------

**The bound does not depend on ``__fp_contract__``.** A fused multiply-add removes a
rounding rather than adding one, so a fusing build stays inside a budget written for
a non-fusing one. Unlike ``tests/parity/test_bezier_arithmetic.py``, no claim here
has a conditional arm, and none of these tests skips on a fusing build.

**A one-dimensional Bézier is not covered here.** ``_evaluate_bezier`` branches on
``dim == 1`` before reaching either contraction and calls the fused 1-D kernel, whose
bitwise claim ``tests/parity/test_bezier_arithmetic.py`` already carries.
``test_a_one_dimensional_bezier_still_delegates`` pins that the branch is still taken
rather than restating the claim.

**The Bernstein bases are common mode, and only on a compiled oracle.** Both backends
tabulate the basis before contracting, and the two tabulations are bit-identical --
``tests/parity/test_basis_tabulations.py`` asserts it -- so the whole budget above is
the contraction's. That premise fails under ``NUMBA_DISABLE_JIT=1``, where the oracle
seeds its ratio recurrence with numpy's ``np.power`` instead of numba's and the two
differ by an ulp (Rule 12), so every claim here calls
:func:`demand_a_compiled_seed`. A bounded claim usually need not be gated; this one
must, because its budget is written for the contraction alone.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from math import comb
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pytest

from pantr._backend import Backend, use_backend
from pantr.bezier import Bezier
from pantr.quad import PointsLattice
from tests._parity_harness import (
    ParityClaim,
    Roundings,
    absolute_tolerance,
    assert_accuracy,
    assert_parity,
    bounded_parity,
    demand_a_compiled_seed,
    demand_the_compiled_kernel,
    demand_the_reference_host,
    derived_accuracy,
    unit_roundoff,
)

DTYPES: Final = (np.float64, np.float32)
"""Both storage formats. The float32 half is where a width error is visible at all."""

DEGREES: Final = ((1, 1), (2, 3), (3, 2), (5, 4), (0, 4), (2, 2, 2), (4, 3, 2))
"""Degree tuples spanning dim 2 and dim 3, including a degree-0 direction.

A degree-0 direction contracts one term, which is the shortest chain the schedule
can take and the one where an off-by-one in the stage count would be invisible
everywhere else.
"""

RANKS: Final = (1, 3)
"""Output ranks. Rank 1 is the one that matters for a non-rational Bézier: its raw
result has a trailing block of one element, which is where numpy leaves the naive
summation order and the claim stops being an equality."""

_LATTICE_POINTS: Final = 4
"""Points per direction in the lattice sweep. Four rather than three so the grid is
not square with the smallest degrees, which would let an axis mix-up pass."""

_TINY: Final = float(np.finfo(np.float64).tiny)
"""Floor for an amplification, so a tolerance is never identically zero."""

_PTS_ARRAY_WHY: Final = (
    "the pts-array schedule contracts one direction at a time with np.einsum, and "
    "sum_d (degree_d + 1) is the summed contraction length. A sum of n products "
    "commits at most n roundings per term in ANY summation order, so each direction "
    "contributes gamma_{n_d} and the directions compose to gamma_N by Higham's "
    "(1+theta_a)(1+theta_b) = 1+theta_{a+b}. The accumulator is the storage format, "
    "measured rather than assumed: scripts/measure_bezier_nd_widths.py separates a "
    "narrow model from a float64 one at float32, 336/336 against 0/336. The "
    "amplification is the same contraction run on |c|, which is exact rather than "
    "conservative because a Bernstein basis is non-negative. What this covers and "
    "the lattice claim does not: numpy's own vectorised reduction, which it "
    "dispatches on the output's trailing extent and whose summation tree is a "
    "property of the host"
)

_LATTICE_WHY: Final = (
    "the lattice schedule contracts one AXIS at a time with np.tensordot, which "
    "reshapes to a matrix product and reaches BLAS. The rounding budget is the same "
    "gamma_N over the summed contraction lengths, for the same reason -- the bound "
    "holds in any summation order -- but what it covers is different: BLAS blocking, "
    "a traversal order this project does not choose, and a fused multiply-add it "
    "does not control. A fused site removes a rounding rather than adding one, so "
    "the budget covers a fusing build without a conditional arm. The amplification "
    "is the same contraction run on |c|, exact for the same reason. Measured: no "
    "width model reproduces tensordot at any shape swept, so unlike the pts-array "
    "claim this one is never an equality even in principle"
)

_RATIONAL_TAIL: Final = (
    ". For a rational Bezier one stage is added for the projection's division, and "
    "the amplification carries the quotient rule: with N and D the numerator and "
    "denominator companions and R the projected value, |d(N/D)| <= gamma (A_N + "
    "|R| A_D)/|D| + u|R|, and u <= gamma so the |R| term rides the same growth "
    "factor"
)


def _mixed_control_points(
    shape: tuple[int, ...],
    dtype: npt.DTypeLike,
    seed: int,
    exponents: tuple[int, int] = (-6, 7),
) -> npt.NDArray[np.float32 | np.float64]:
    """Control points spanning many magnitudes, so the contraction has cancelling to do.

    A net of uniform magnitude is the easy case: every partial sum is the size of the
    answer and a change of summation order does not reach the output. Scaling each
    entry by a random power of ten is what makes the two schedules separate.

    Args:
        shape (tuple[int, ...]): Shape of the control net, components last.
        dtype (npt.DTypeLike): Storage format.
        seed (int): Generator seed.
        exponents (tuple[int, int]): Half-open range of decimal exponents. Twelve
            orders by default, which an operation whose output is the size of its
            input can carry at float32. Defaults to (-6, 7).

    Returns:
        npt.NDArray[np.float32 | np.float64]: The control points.
    """
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(shape) * 10.0 ** rng.integers(*exponents, shape)
    return np.ascontiguousarray(values, dtype=dtype)


def _weighted_net(
    degrees: tuple[int, ...],
    rank: int,
    dtype: npt.DTypeLike,
    seed: int,
    *,
    rational: bool,
) -> npt.NDArray[np.float32 | np.float64]:
    """Build a control net, giving a rational one weights bounded away from zero.

    A weight near zero is not a parity question but a conditioning one: the projected
    value diverges and every bound with it, so the sweep would be measuring the
    quotient's condition number rather than the two backends' agreement.

    Args:
        degrees (tuple[int, ...]): Degree per parametric direction.
        rank (int): Number of value components.
        dtype (npt.DTypeLike): Storage format.
        seed (int): Generator seed.
        rational (bool): Whether to append a weight column.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The control net.
    """
    components = rank + 1 if rational else rank
    shape = (*(degree + 1 for degree in degrees), components)
    net = _mixed_control_points(shape, dtype, seed)
    if rational:
        rng = np.random.default_rng(seed + 1)
        weights = rng.uniform(0.5, 2.0, net.shape[:-1])
        net[..., -1] = np.asarray(weights, dtype=dtype)
    return net


def _adversarial_parameters(dtype: npt.DTypeLike, count: int) -> npt.NDArray[Any]:
    """Parameters reaching the branches a uniform sweep does not.

    Both endpoints, either side of the mirror threshold the Bernstein tabulation
    switches on, and a value small enough that ``1 - (1 - u)`` loses it outright.
    Padded to ``count`` with a deterministic sweep across the unit interval.

    Args:
        dtype (npt.DTypeLike): Storage format, which sets what "next to one" means.
        count (int): How many parameters to return.

    Returns:
        npt.NDArray[Any]: The parameters, in the storage format.
    """
    one = np.array(1.0, dtype=dtype)
    half = np.array(0.5, dtype=dtype)
    special = [
        0.0,
        1e-8,
        0.25,
        0.5,
        float(np.nextafter(half, one)),
        0.75,
        float(np.nextafter(one, np.array(0.0, dtype=dtype))),
        1.0,
    ]
    padding = [float(k) / float(count + 1) for k in range(1, count + 1)]
    return np.asarray((special + padding)[:count], dtype=dtype)


def _point_array(
    degrees: tuple[int, ...], dtype: npt.DTypeLike, count: int
) -> npt.NDArray[np.float32 | np.float64]:
    """An explicit array of evaluation points, one column per direction.

    Each direction gets the adversarial list rotated by its index, so no point sits
    on the diagonal of the parameter cube: a rotation is what stops an axis mix-up
    from producing the right answer.

    Args:
        degrees (tuple[int, ...]): Degree per direction, for the direction count.
        dtype (npt.DTypeLike): Storage format.
        count (int): Number of points.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Points of shape ``(count, dim)``.
    """
    base = _adversarial_parameters(dtype, count)
    columns = [np.roll(base, direction) for direction in range(len(degrees))]
    return np.ascontiguousarray(np.stack(columns, axis=1), dtype=dtype)


def _lattice(degrees: tuple[int, ...], dtype: npt.DTypeLike) -> PointsLattice:
    """A lattice whose directions carry different parameters.

    Args:
        degrees (tuple[int, ...]): Degree per direction, for the direction count.
        dtype (npt.DTypeLike): Storage format.

    Returns:
        PointsLattice: One column of ``_LATTICE_POINTS`` parameters per direction.
    """
    base = _adversarial_parameters(dtype, _LATTICE_POINTS)
    return PointsLattice([np.roll(base, direction) for direction in range(len(degrees))])


def _companion(
    net: npt.NDArray[Any],
    points: npt.NDArray[Any] | PointsLattice,
) -> npt.NDArray[np.float64]:
    """Evaluate one non-rational net through the oracle's own schedule.

    Used both for the absolute-value companion and for the computed denominator of a
    rational projection. It runs under the Python backend deliberately: the
    amplification must not be produced by the implementation under test.

    Args:
        net (npt.NDArray[Any]): A non-rational control net.
        points (npt.NDArray[Any] | PointsLattice): Where to evaluate. The schedule
            follows from which of the two this is, which is exactly the distinction
            the two claims are about.

    Returns:
        npt.NDArray[np.float64]: The values, with the component axis kept.
    """
    with use_backend(Backend.PYTHON):
        values = np.asarray(Bezier(net).evaluate(points), dtype=np.float64)
    return values if net.shape[-1] > 1 else values[..., np.newaxis]


def _claim(
    net: npt.NDArray[Any],
    points: npt.NDArray[Any] | PointsLattice,
    reference: npt.NDArray[Any],
    *,
    rational: bool,
    why: str,
) -> ParityClaim:
    """State the bounded claim for one evaluation.

    Args:
        net (npt.NDArray[Any]): The control net, weight column included.
        points (npt.NDArray[Any] | PointsLattice): Where it was evaluated.
        reference (npt.NDArray[Any]): The oracle's result, for the quotient rule's
            ``|R|`` term. Ignored for a non-rational Bézier.
        rational (bool): Whether the last component is a homogeneous weight.
        why (str): The entry point's derivation.

    Returns:
        ParityClaim: A BOUNDED claim over ``sum_d (degree_d + 1)`` stages, plus one
        for the division when rational.
    """
    dtype = net.dtype
    stages = sum(extent for extent in net.shape[:-1])

    if not rational:
        amplification = _companion(np.abs(net), points)
    else:
        stages += 1
        numerator = _companion(np.abs(net[..., :-1]), points)
        denominator = _companion(np.abs(net[..., -1:]), points)
        computed = np.abs(_companion(net[..., -1:], points))
        value = np.abs(np.asarray(reference, dtype=np.float64))
        if value.ndim < numerator.ndim:
            value = value[..., np.newaxis]
        amplification = (numerator + (value * denominator)) / computed + value

    shaped = np.reshape(np.maximum(amplification, _TINY), np.shape(reference))
    return bounded_parity(
        roundings=Roundings(stages=stages, accumulator_per_stage=1, storage_per_stage=0),
        accumulator=dtype,
        storage=dtype,
        amplification=np.ascontiguousarray(shaped, dtype=np.float64),
        why=why + (_RATIONAL_TAIL if rational else ""),
    )


def _both_backends(
    net: npt.NDArray[Any], points: npt.NDArray[Any] | PointsLattice, *, rational: bool
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Evaluate the same Bézier under each backend.

    Args:
        net (npt.NDArray[Any]): The control net.
        points (npt.NDArray[Any] | PointsLattice): Where to evaluate.
        rational (bool): Whether the last component is a homogeneous weight.

    Returns:
        tuple[npt.NDArray[Any], npt.NDArray[Any]]: ``(actual, reference)``, C++
        first, matching :func:`assert_parity`'s own argument order.
    """
    with use_backend(Backend.PYTHON):
        reference = np.asarray(Bezier(net, is_rational=rational).evaluate(points))
    with use_backend(Backend.CPP):
        actual = np.asarray(Bezier(net, is_rational=rational).evaluate(points))
    return actual, reference


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degrees", DEGREES)
@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("rational", [False, True])
def test_the_pts_array_entry_point_is_bounded(
    cpp_backend: None,
    degrees: tuple[int, ...],
    rank: int,
    dtype: npt.DTypeLike,
    *,
    rational: bool,
) -> None:
    """The einsum schedule and its C++ counterpart agree inside the contraction budget."""
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    demand_a_compiled_seed()

    net = _weighted_net(degrees, rank, dtype, seed=20260830, rational=rational)
    points = _point_array(degrees, dtype, count=9)

    actual, reference = _both_backends(net, points, rational=rational)
    assert_parity(
        actual,
        reference,
        _claim(net, points, reference, rational=rational, why=_PTS_ARRAY_WHY),
        context=f"evaluate degrees {degrees} rank {rank} rational {rational} {dtype}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degrees", DEGREES)
@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("rational", [False, True])
def test_the_lattice_entry_point_is_bounded(
    cpp_backend: None,
    degrees: tuple[int, ...],
    rank: int,
    dtype: npt.DTypeLike,
    *,
    rational: bool,
) -> None:
    """The tensordot schedule and its C++ counterpart agree inside the same budget."""
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    demand_a_compiled_seed()

    net = _weighted_net(degrees, rank, dtype, seed=20260831, rational=rational)
    lattice = _lattice(degrees, dtype)

    actual, reference = _both_backends(net, lattice, rational=rational)
    assert_parity(
        actual,
        reference,
        _claim(net, lattice, reference, rational=rational, why=_LATTICE_WHY),
        context=f"evaluate_on_lattice degrees {degrees} rank {rank} rational {rational} {dtype}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_one_dimensional_bezier_still_delegates(cpp_backend: None, dtype: npt.DTypeLike) -> None:
    """``dim == 1`` never reaches either contraction, so it keeps the 1-D bitwise claim.

    Pinning the branch rather than restating the claim: if the n-d path ever absorbed
    the one-dimensional case, this file's two bounded claims would silently replace a
    bitwise one that ``tests/parity/test_bezier_arithmetic.py`` still asserts, and
    nothing else would report it.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    demand_a_compiled_seed()

    net = _mixed_control_points((6, 3), dtype, seed=20260832)
    points = _adversarial_parameters(dtype, 8)

    actual, reference = _both_backends(net, points, rational=False)
    assert np.array_equal(actual, reference), (
        "a one-dimensional Bezier is evaluated by the fused 1-D kernel on both "
        "backends, whose claim is bitwise. A difference here means the dim == 1 "
        "branch was removed and the contraction absorbed it, which changes which "
        "claim covers the case rather than merely moving a last bit."
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degrees", [(3, 2), (2, 2, 2)])
def test_the_two_entry_points_are_not_the_same_arithmetic(
    cpp_backend: None, degrees: tuple[int, ...], dtype: npt.DTypeLike
) -> None:
    """The reason there are two claims is still true on this host.

    A liveness guard in the sense of ``design/backend_parity.md`` Rule 7, and gated
    like one: that ``einsum`` and ``tensordot`` disagree is a fact about numpy's
    dispatch and the BLAS this machine links, not about pantr. If they ever agreed
    everywhere the two claims could be merged -- but discovering that from a red
    build on somebody else's laptop is exactly the intermittent failure Rule 7 exists
    to prevent.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    demand_a_compiled_seed()
    demand_the_reference_host(
        "einsum and tensordot disagree",
        "measured by scripts/measure_bezier_nd_widths.py on the calibrated host",
    )

    net = _weighted_net(degrees, 3, dtype, seed=20260833, rational=False)
    lattice = _lattice(degrees, dtype)
    columns = list(lattice.pts_per_dir)
    grid = np.stack([axis.ravel() for axis in np.meshgrid(*columns, indexing="ij")], axis=1).astype(
        dtype
    )

    with use_backend(Backend.PYTHON):
        bezier = Bezier(net)
        from_lattice = np.asarray(bezier.evaluate(lattice)).reshape(-1, 3)
        from_points = np.asarray(bezier.evaluate(np.ascontiguousarray(grid)))

    assert not np.array_equal(from_lattice, from_points), (
        "the two entry points produced bit-identical results over the whole sweep, "
        "so the premise of stating two claims instead of one no longer holds here. "
        "Re-run scripts/measure_bezier_nd_widths.py before merging them: the likely "
        "cause is a numpy release routing both through the same loop."
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degrees", [(3, 2), (2, 2, 2)])
@pytest.mark.parametrize("rational", [False, True])
def test_the_bound_refuses_a_perturbed_result(
    cpp_backend: None, degrees: tuple[int, ...], dtype: npt.DTypeLike, *, rational: bool
) -> None:
    """A bound that accepts anything is not a bound.

    The complement of the two tests above: they show the claim is satisfied, this
    shows it is *satisfiable only by the right answer*. The perturbation is one part
    in ``1e-4`` of each value, which is far above any rounding budget at either dtype
    and far below the magnitude spread of the net, so it cannot be absorbed by the
    amplification either.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    demand_a_compiled_seed()

    net = _weighted_net(degrees, 3, dtype, seed=20260834, rational=rational)
    points = _point_array(degrees, dtype, count=9)
    actual, reference = _both_backends(net, points, rational=rational)
    claim = _claim(net, points, reference, rational=rational, why=_PTS_ARRAY_WHY)

    perturbed = np.asarray(actual * np.asarray(1.0 + 1e-4, dtype=dtype), dtype=dtype)
    with pytest.raises(AssertionError):
        assert_parity(perturbed, reference, claim, context="deliberately perturbed")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degrees", [(3, 2), (2, 2, 2)])
def test_the_bound_is_not_larger_than_the_values_it_compares(
    cpp_backend: None, degrees: tuple[int, ...], dtype: npt.DTypeLike
) -> None:
    """Rule 3, asserted here rather than left to the harness's own guard.

    ``assert_parity`` refuses a vacuous bound and would fail the tests above if one
    were ever formed, but it fails them with a message about the *comparison*. This
    one names the quantity, so a future net whose output cancels harder reports that
    the amplification has outgrown the values rather than that parity broke.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)

    net = _weighted_net(degrees, 3, dtype, seed=20260835, rational=False)
    points = _point_array(degrees, dtype, count=9)
    with use_backend(Backend.PYTHON):
        reference = np.asarray(Bezier(net).evaluate(points))

    tolerance = absolute_tolerance(
        _claim(net, points, reference, rational=False, why=_PTS_ARRAY_WHY)
    )
    largest = float(np.max(np.abs(np.asarray(reference, dtype=np.float64))))
    assert float(np.max(tolerance)) < largest, (
        f"the amplification reached {float(np.max(tolerance)):.3g} against values of "
        f"at most {largest:.3g}, so this claim admits any answer including zero. "
        f"Rule 3 refuses it; the fix is a narrower sweep, not a tighter constant."
    )


@pytest.mark.slow
@pytest.mark.parametrize("dtype", DTYPES)
def test_each_bound_holds_over_a_sweep_ten_times_the_shipped_one(
    cpp_backend: None, dtype: npt.DTypeLike
) -> None:
    """A bound checked only by the sweep that ships with it has not been checked.

    ``design/backend_parity.md`` records the case: the affine-map bound held over 500
    draws and failed 88 times at 60000. This runs several hundred independent nets
    per entry point against both claims, more than ten times the shipped
    parametrization, and reports the worst ratio to the bound so a later reader can
    see whether it is being approached or merely satisfied.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    demand_a_compiled_seed()

    worst = {"pts": 0.0, "lattice": 0.0}
    seed = 0
    for degrees in DEGREES:
        for rank in RANKS:
            for rational in (False, True):
                for draw in range(6):
                    seed += 1
                    net = _weighted_net(degrees, rank, dtype, seed=90000 + seed, rational=rational)
                    points = _point_array(degrees, dtype, count=7 + draw)
                    lattice = _lattice(degrees, dtype)
                    for label, where, why in (
                        ("pts", points, _PTS_ARRAY_WHY),
                        ("lattice", lattice, _LATTICE_WHY),
                    ):
                        actual, reference = _both_backends(net, where, rational=rational)
                        claim = _claim(net, where, reference, rational=rational, why=why)
                        deviation = assert_parity(
                            actual,
                            reference,
                            claim,
                            context=f"{label} sweep {degrees} rank {rank} rational {rational}",
                        )
                        worst[label] = max(worst[label], deviation.max_ratio_to_bound)

    for label, ratio in worst.items():
        assert ratio > 0.0, (
            f"the {label} entry point agreed bit for bit over the whole sweep, so "
            f"nothing here evaluated the bound. That is not a pass: re-run "
            f"scripts/measure_bezier_nd_widths.py, because a schedule that used to "
            f"differ and no longer does has changed under this claim."
        )


def _exact_bernstein(degree: int, parameter: Fraction) -> list[Fraction]:
    """The Bernstein basis of one degree at one parameter, in exact rational arithmetic.

    Args:
        degree (int): Polynomial degree.
        parameter (Fraction): Where to evaluate.

    Returns:
        list[Fraction]: ``degree + 1`` exact basis values, summing to exactly one.
    """
    return [
        Fraction(comb(degree, i)) * parameter**i * (1 - parameter) ** (degree - i)
        for i in range(degree + 1)
    ]


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_result_is_accurate_against_exact_rational_arithmetic(
    cpp_backend: None, dtype: npt.DTypeLike
) -> None:
    """The independent check every ported module owes, since parity cannot see a shared error.

    ``design/backend_parity.md`` opens with it: parity says the two backends agree,
    not that either is right, so a defect injected into both is invisible to every
    test above. The oracle here is the tensor-product contraction carried out in
    :class:`~fractions.Fraction`, which is the exact answer rather than a second
    implementation of the approximate one.

    The bound is the same ``gamma_N`` against the same companion, but **one-sided**:
    this compares one computation against the truth, not two computations against
    each other, so the harness's factor of two does not apply and is not used.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    demand_a_compiled_seed()

    degrees = (3, 2)
    # Dyadic parameters so the exact rational and the float64 point array denote the
    # same number: a non-representable parameter would put the difference between two
    # *different* polynomial arguments inside the comparison.
    parameters = (Fraction(1, 4), Fraction(5, 8), Fraction(1, 2))
    net = _mixed_control_points((4, 3, 2), dtype, seed=20260836, exponents=(-2, 3))

    points = np.asarray([[float(a), float(b)] for a in parameters for b in parameters], dtype=dtype)
    with use_backend(Backend.CPP):
        computed = np.asarray(Bezier(net).evaluate(points), dtype=np.float64)

    exact_rows: list[list[float]] = []
    exact_net = [[[Fraction(float(v)) for v in row] for row in plane] for plane in net]
    for first in parameters:
        basis_0 = _exact_bernstein(degrees[0], first)
        for second in parameters:
            basis_1 = _exact_bernstein(degrees[1], second)
            row = [
                sum(
                    basis_0[i] * basis_1[j] * exact_net[i][j][component]
                    for i, j in product(range(degrees[0] + 1), range(degrees[1] + 1))
                )
                for component in range(2)
            ]
            exact_rows.append([float(value) for value in row])
    exact = np.asarray(exact_rows, dtype=np.float64)

    stages = sum(degree + 1 for degree in degrees)
    unit = unit_roundoff(dtype)
    growth = stages * unit / (1.0 - (stages * unit))
    companion = _companion(np.abs(net), points)
    assert_accuracy(
        computed,
        exact,
        derived_accuracy(
            bound=np.ascontiguousarray(np.maximum(growth * companion, _TINY), dtype=np.float64),
            why=(
                "one-sided gamma_N against the absolute-value companion, with N the "
                "summed contraction lengths, comparing one computation against the "
                "exact rational answer rather than two computations against each "
                "other -- so the harness's one-sided-to-two-sided factor of two is "
                "deliberately absent. The parameters are dyadic, so the float and "
                "the Fraction denote the same argument and the only difference the "
                "comparison sees is the contraction's own rounding"
            ),
        ),
        context=f"exact rational oracle, degrees {degrees}, {np.dtype(dtype).name}",
    )
