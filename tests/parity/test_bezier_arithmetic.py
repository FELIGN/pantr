r"""Parity of the eight Bézier arithmetic kernels against their Numba oracles.

`cpp/include/pantr/bezier/bezier.hpp` and
`cpp/include/pantr/core/reduction_operator.hpp` name this file as the place their
parity claims are measured.

Every claim here is **bitwise**, and that is a first for this port. `quad` could
not claim it because Golub-Welsch and a companion-matrix eigensolver are different
algorithms; `change_basis` could not because a dense solve sits in the middle. Here
both backends run the same expressions in the same order over `+`, `-`, `*` and
`/`, each of which IEEE 754 pins to one correctly rounded result, so the two agree
to the last bit.

Where exactness was not free
----------------------------

Three things had to be got right, and each one is invisible at float64.

**The accumulation widths are not uniform, and are not this port's to choose.** The
four de Casteljau kernels compute each step in double and round once on the store,
because numba promotes their float64 scalars against a float32 workspace. The
derivative kernel is the exception: it opens with ``dtype = pts.dtype`` and
allocates every workspace in it, so at float32 the whole recursion is float32.
Elevation and the product mix, their coefficient tables being float64
unconditionally. Accumulating narrow where the oracle accumulates wide moved 125 of
630 values in the measurement that found it.

**The evaluation kernel's two branches seed from bases of different width.** Above
the mirror threshold the oracle raises ``u``, which is the point array's own dtype;
below it it raises ``1 - u``, which the literal ``1.0`` has already promoted to
float64. A single value in a whole-kernel sweep caught that, at degree 17 and
``u = 0.75``, where ``0.75^17 = 3^17 / 2^34`` needs 27 significand bits and the wide
seed survives where the narrow one rounds. Computing the mirrored seed in double is
still the natural port, and ``scripts/measure_bezier_parity.py`` reports 1305
differences out of 1280256 if you do it.

**The seed is a ``pow``, so its claim is observed rather than derived.** Neither C
nor IEEE 754 requires ``pow`` to be correctly rounded, and the mirrored branch raises
``u`` at storage width, so at float32 the C++ calls ``powf``. Measured over 1280256
pairs, degrees 1 to 64 across the whole mirrored range: it and numba's ``np.power``
agree on every one. `bernstein.hpp` records the same open question with the same
answer.

**Every figure in this docstring is reproducible**, by
``scripts/measure_bezier_parity.py``. That is a script rather than a test because
these are counts over particular grids, and a reader deciding whether to believe one
wants the grid rather than a green tick. It exists because an earlier version of this
file quoted numbers whose only artifact was a scratch directory that no longer
exists.

A build that can fuse, and why the claims are conditional
---------------------------------------------------------

Unlike the Bernstein tabulation, these kernels contain ``a * b + c * d`` sites, so
``-ffp-contract`` has something to fuse and the exactness above is a property of the
**build**, not of the code. Each claim below is therefore selected at run time:
bitwise where the target ISA has no fused multiply-add, bounded where it has one.
Rule 7 of design/backend_parity.md is what licenses that shape, and Rule 10 states
the bound.

The bound is one budget for all eight kernels and eight different amplifications.
At a fused site the oracle computes ``fl(a + fl(b*c))`` and this backend
``fl(a + b*c)``, so the two differ by at most ``u|b*c| + 2u|a + b*c|``: three
accumulator roundings per stage. Where the storage is narrower the two pre-store
values can fall either side of a rounding boundary, which costs one further ulp, so
two storage roundings per stage. What differs per kernel is the stage count and the
magnitude the relative budget multiplies, and those are what the ``_amplification``
helpers below carry.

Fourteen sites fuse, and they were enumerated by disassembling a ``-march=native``
build rather than by reading the source: two in ``evaluate``, five in
``evaluate_deriv``, two in ``restrict`` and one each in the other five kernels.
``scripts/measure_bezier_fma_bound.py`` reproduces that enumeration, the movement
counts, and the slack each bound carries.

Three tests still skip on a fusing build, each for a reason the bound cannot cover:
``minimize_degree`` because its verdict is a **discrete** degree rather than a
coefficient, so no tolerance saves a flipped decision; the factorial-overflow test
because a wrapped falling factorial no longer describes the quantity the
amplification is derived from; and ``compose`` because the product kernel's operands
are formed inside the composition and are not observable from the public surface, so
no tight amplification can be built for them. The last is covered instead by a test
that drives the product kernel at its own entry point, where the operands are known.
"""

from __future__ import annotations

from collections.abc import Callable
from math import comb, prod
from typing import Any, Final, NamedTuple

import numpy as np
import numpy.typing as npt
import pytest

from pantr._backend import Backend, use_backend
from pantr.bezier import Bezier
from pantr.bezier._bezier_degree import _interpolating_reduction_operator
from tests._parity_harness import (
    ParityClaim,
    Roundings,
    absolute_tolerance,
    assert_parity,
    bitwise_parity,
    bounded_parity,
    contraction_may_fuse,
    demand_the_compiled_kernel,
)

_TINY: Final = float(np.finfo(np.float64).tiny)
"""Smallest positive double, used so an amplification is never exactly zero."""

DTYPES: Final = (np.float64, np.float32)
"""The two storage formats the Bézier layer accepts."""

DEGREES: Final = (0, 1, 2, 3, 5, 8, 13, 17, 25)
"""Degrees swept by the univariate tests.

0 and 1 are the two branches the evaluation kernel short-circuits; 17 is the degree
at which the mirrored seed's width was caught, and is kept for that reason; 25 is
past anything pantr's own builders reach.
"""

_DE_CASTELJAU_WHY = (
    "the triangle is +, -, * and / only, each pinned by IEEE 754 to one correctly "
    "rounded result, evaluated in the oracle's order and rounded through the "
    "workspace at the oracle's width rather than carried in a register. No fused "
    "multiply-add is available on this build, so the one operation that could "
    "differ cannot occur"
)

_EVALUATE_WHY = (
    "every operation but the seed is +, -, * or /, evaluated in the oracle's order, "
    "with the running term carried in a register exactly as the oracle carries it. "
    "The seed is pow, which neither C nor IEEE 754 requires to be correctly rounded, "
    "so this claim is observed rather than derived: over 1280256 pairs, reproducible "
    "by scripts/measure_bezier_parity.py, the platform powf and numba's np.power "
    "agree on every argument these degrees form, at both widths the two branches "
    "seed at. No fused multiply-add on this build"
)

_BINOMIAL_WHY = (
    "the coefficient tables are built from an exact-integer binomial recurrence that "
    "is the same recurrence on both sides, and every later operation is +, -, * or / "
    "in the oracle's order. No fused multiply-add on this build"
)

_REDUCTION_WHY = (
    "the operator is assembled once in exact rational arithmetic on the Python side "
    "and crosses as float64, so both backends multiply the same matrix; the apply "
    "accumulates in float64 in the same order on both sides and rounds once on the "
    "write. No fused multiply-add on this build"
)

_ACCUMULATOR_ROUNDINGS_PER_STAGE: Final = 3
"""Accumulator roundings by which the two backends may differ, per stage.

At a fused site the oracle evaluates ``fl(a + fl(b*c))`` and this backend
``fl(a + b*c)``. Writing the first as ``(a + b*c(1 + d1))(1 + d2)`` and the second as
``(a + b*c)(1 + d3)`` with every ``|d| <= u``, their difference is

    ``|b*c| * u * (1 + u)  +  |a + b*c| * 2u``,

so three roundings' worth, of which the first is charged against the **product** and
the last two against the stage's own value. Both are folded into one count because
``amplification`` carries a magnitude that dominates each: for every kernel here the
amplification bounds the partial sums, hence the products, as well as the result.
"""

_STORAGE_ROUNDINGS_PER_STAGE: Final = 2
"""Storage roundings by which the two backends may differ, per stage.

Two values differing by less than one ulp of the storage format usually round to the
same stored value and occasionally straddle a boundary and do not. A straddle costs
one ulp, which is ``2u`` in the harness's unit-roundoff convention. The harness
charges this at zero when the storage format is the accumulator's own, which is why
one budget serves both dtypes.

This term is what the float32 measurements are about: for seven of the eight kernels
the fusion happens in a float64 accumulator and only a straddle can carry it to the
output, so float32 barely moves. ``evaluate_deriv`` is the exception and contracts at
storage width, which is Rule 9 of design/backend_parity.md reappearing.
"""

_FUSED_PREFIX: Final = (
    "this build can fuse a multiply-add, so the two backends commit different numbers "
    "of roundings at the fused sites. Isolated to contraction and nothing else: "
    "rebuilt with -march=native, 91 of 1260 de Casteljau values and 472 of 3616 "
    "whole-kernel values move, and -ffp-contract=off on top of the same -march "
    "restores bit-identity exactly. The oracle never fuses: numba targets this host's "
    "ISA and emits no FMA without fastmath, which no pantr kernel sets. "
)

_DISCRETE_VERDICT: Final = (
    "this build can fuse, and the quantity under test is a discrete degree rather "
    "than a coefficient, so no tolerance covers a flipped decision. The greedy search "
    "accepts or rejects each trial by comparing a round-trip error against a "
    "tolerance; a contraction difference can move that comparison and change the "
    "resulting degree by one, which is not a bounded disagreement."
)

_WRAPPED_FACTORIAL: Final = (
    "this build can fuse, and the falling factorial has wrapped, so the amplification "
    "the bound is derived from -- p!/(p-k)! -- no longer describes the quantity the "
    "kernel computes. Both backends wrap identically on a non-fusing build, which is "
    "what this test exists to pin; bounding the fused case would need an "
    "amplification derived from the wrapped value, which is not a magnitude of "
    "anything."
)

_OPERANDS_NOT_OBSERVABLE: Final = (
    "this build can fuse, and the product kernel's operands are formed inside "
    "compose, so no tight amplification can be built from the public surface: the "
    "only bound available from outside is max|outer| * (1 + 2 max|inner|)^p, which is "
    "1e18 at the degrees swept here and would accept anything. "
    "test_the_product_kernel_is_bounded_under_contraction covers the same kernel at "
    "its own entry point, where the operands are known and the amplification is the "
    "exact convex sum."
)


_DE_CASTELJAU_FUSED_WHY: Final = (
    "every stage is a convex combination (1-t)a + tb with t in [0, 1], so a "
    "divergence introduced at one stage passes through the remaining ones without "
    "growing in the max norm and the per-stage injections merely add. max|c| bounds "
    "the whole triangle, and it is the right magnitude rather than |result|: the "
    "triangle can cancel to near zero while the error that reached it does not"
)


def demand_a_bound_the_claim_can_carry(reason: str) -> None:
    """Skip on a fusing build where the derived bound does not describe the quantity.

    Three tests reach this and each names its own reason. Everything else states a
    bounded claim instead of skipping.

    Args:
        reason (str): Why a bound cannot be formed for this quantity.

    Raises:
        Skipped: On a build whose target ISA offers a fused multiply-add.
    """
    if contraction_may_fuse():
        pytest.skip(reason)


def _net_magnitude(*arrays: npt.NDArray[Any]) -> float:
    """Return the largest absolute entry across some control nets.

    Every kernel whose stages are convex combinations amplifies a perturbation by at
    most this, because a convex combination of entries is bounded by the largest of
    them and so is every partial sum along the way.

    Args:
        *arrays (npt.NDArray[Any]): The nets.

    Returns:
        float: The largest absolute entry, or the smallest positive double if every
            entry is zero, so that a tolerance is never identically zero.
    """
    largest = max((float(np.max(np.abs(np.float64(a)))) for a in arrays if a.size), default=0.0)
    return largest if largest > 0.0 else float(np.finfo(np.float64).tiny)


class _Budget(NamedTuple):
    """The two things that differ from kernel to kernel in the fused claim.

    Attributes:
        stages (int): Length of the dependency chain the fused sites sit on.
        amplification (npt.NDArray[np.float64]): Elementwise magnitude the relative
            budget multiplies, matching the compared arrays' shape.
    """

    stages: int
    amplification: npt.NDArray[np.float64]


def _sliced_control_points(
    ctrl: npt.NDArray[Any], value: float
) -> npt.NDArray[np.float32 | np.float64]:
    """Slice a net and return the coefficients, whatever ``slice`` chose to return.

    :meth:`Bezier.slice` returns a bare array for a univariate Bézier and a
    :class:`Bezier` otherwise, so a caller that only wants the numbers has to branch.

    Args:
        ctrl (npt.NDArray[Any]): The control net.
        value (float): Parameter to slice at.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The sliced coefficients.
    """
    sliced = Bezier(ctrl).slice(0, value)
    return np.asarray(getattr(sliced, "control_points", sliced))


def _split_half(value: float, half: int) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[Any]]:
    """Return a driver that splits a net and hands back one of the two halves.

    A named factory rather than a lambda with a default argument, because the latter
    is what the loop over the two halves would otherwise need and mypy cannot infer
    its type.

    Args:
        value (float): Parameter to split at.
        half (int): 0 for the left half, 1 for the right.

    Returns:
        Callable: Takes a float64 net and returns that half's coefficients.
    """

    def build(net: npt.NDArray[np.float64]) -> npt.NDArray[Any]:
        return np.asarray(Bezier(net).split(0, value)[half].control_points)

    return build


def _absolute_companion(
    build: Callable[[npt.NDArray[np.float64]], npt.NDArray[Any]], ctrl: npt.NDArray[Any]
) -> npt.NDArray[np.float64]:
    """Bound each output element by running the same operation on ``|c|``.

    The tight amplification for a recurrence whose weights are non-negative and sum
    to one: the magnitude reachable at output element ``i`` is exactly what the
    operation produces from the absolute values of the net, because every weight
    passes through the absolute value unchanged. It is what the harness's own
    vacuity message prescribes, and it is licensed here and **only** here -- the same
    trick applied to an oscillatory recurrence diverges, which is why
    ``evaluate_deriv`` and ``reduce_degree`` derive their amplification instead.

    ``max|c|`` was the first thing tried and it is correct but useless: on a net
    spanning twelve decades whose output cancels to order one, it bounds a float32
    de Casteljau by 25 against values of 0.75, and the harness refuses that as a
    bound satisfied by any result. Seven cases failed that way, all float32, all at
    degree 25 or at a parameter within 1e-8 of an endpoint.

    Args:
        build (Callable): Runs the operation on a float64 net and returns the result.
        ctrl (npt.NDArray[Any]): The control net.

    Returns:
        npt.NDArray[np.float64]: Elementwise magnitude bound, never exactly zero.
    """
    with use_backend(Backend.PYTHON):
        companion = np.abs(np.asarray(build(np.abs(np.asarray(ctrl, dtype=np.float64)))))
    return np.asarray(np.maximum(companion, _TINY), dtype=np.float64)


def _parity_claim(
    *,
    bitwise_why: str,
    fused_why: str,
    budget: _Budget,
    storage: npt.DTypeLike,
    accumulator: npt.DTypeLike = np.float64,
) -> ParityClaim:
    """State the parity claim for one kernel, conditioned on what the build can do.

    Bitwise where the target ISA has no fused multiply-add, which is the shipped
    configuration; bounded where it has one. Copied in shape from
    ``tests/parity/test_basis_cardinal_bspline.py``, which is the precedent for a
    selector, and from ``tests/parity/test_quad_gauss_legendre.py``, which is the
    precedent for the bounded half being probed on a host that cannot reach it.

    Args:
        bitwise_why (str): Justification for the exact claim.
        fused_why (str): What the stage count and amplification are, for the bounded
            claim. Prefixed with the shared contraction argument.
        budget (_Budget): This kernel's stage count and amplification.
        storage (npt.DTypeLike): Format the output array holds.
        accumulator (npt.DTypeLike): Format intermediates accumulate in. float64 for
            every kernel but ``evaluate_deriv``, which accumulates at storage width.

    Returns:
        ParityClaim: BITWISE or BOUNDED, whichever this build supports.
    """
    if not contraction_may_fuse():
        return bitwise_parity(why=bitwise_why)
    return _fused_claim(
        fused_why=fused_why, budget=budget, storage=storage, accumulator=accumulator
    )


def _fused_claim(
    *,
    fused_why: str,
    budget: _Budget,
    storage: npt.DTypeLike,
    accumulator: npt.DTypeLike = np.float64,
) -> ParityClaim:
    """Build the bounded claim unconditionally, whatever this build can do.

    Split out from :func:`_parity_claim` so the probes below can exercise the branch
    no host in this project reaches. That is the whole reason the quad port's fused
    claims survived review: a bound only this file's author has ever evaluated is a
    bound nobody has checked.

    Args:
        fused_why (str): What the stage count and amplification are.
        budget (_Budget): This kernel's stage count and amplification.
        storage (npt.DTypeLike): Format the output array holds.
        accumulator (npt.DTypeLike): Format intermediates accumulate in.

    Returns:
        ParityClaim: The BOUNDED claim.
    """
    return bounded_parity(
        roundings=Roundings(
            stages=max(budget.stages, 1),
            accumulator_per_stage=_ACCUMULATOR_ROUNDINGS_PER_STAGE,
            storage_per_stage=_STORAGE_ROUNDINGS_PER_STAGE,
        ),
        accumulator=accumulator,
        storage=storage,
        amplification=budget.amplification,
        why=f"{_FUSED_PREFIX}{fused_why}",
    )


def _flat_amplification(shape: tuple[int, ...], magnitude: float) -> npt.NDArray[np.float64]:
    """Spread one magnitude over an output's shape.

    Args:
        shape (tuple[int, ...]): Shape of the compared arrays.
        magnitude (float): The magnitude.

    Returns:
        npt.NDArray[np.float64]: A full array, since the harness reports the worst
            element and wants an amplification it can index alongside the values.
    """
    return np.full(shape, magnitude, dtype=np.float64)


def _derivative_scale(degree: int, order: int, magnitude: float) -> float:
    """Amplification of the derivative kernel at one derivative order.

    See :func:`_derivative_amplification` for the derivation; this is the scalar it
    is built from, and the public API returns one order at a time rather than the
    kernel's stack.

    Args:
        degree (int): Degree of the curve.
        order (int): Derivative order.
        magnitude (float): ``max|c|``.

    Returns:
        float: The amplification, never zero so that a tolerance never vanishes.
    """
    if degree < order:
        return float(np.finfo(np.float64).tiny)
    falling = prod(range(degree - order + 1, degree + 1)) if order else 1
    scale = falling * (2.0**order) * magnitude
    return scale if scale > 0.0 else float(np.finfo(np.float64).tiny)


def _derivative_amplification(
    shape: tuple[int, ...], degree: int, magnitude: float
) -> npt.NDArray[np.float64]:
    """Amplification of the derivative kernel, per derivative order.

    The ``k``-th derivative of a Bernstein form is
    ``p!/(p-k)! * sum_j (Delta^k c)_j B_{j,p-k}``, and ``Delta = shift - identity``
    has infinity norm 2, so ``|Delta^k c| <= 2^k max|c|`` and the order-``k`` block is
    bounded by ``p!/(p-k)! * 2^k * max|c|``.

    **The bound is attained, not merely valid.** Driving the kernel with the identity
    net gives the basis derivatives themselves, and
    ``max_s sum_j |B^(k)_{j,p}(s)| / (p!/(p-k)! * 2^k)`` measures exactly 1.0000 for
    ``k`` up to 4 over nine degrees. ``scripts/measure_bezier_fma_bound.py`` reports
    that ratio; a value below 1 would mean this amplification is loose and a value
    above 1 would refute it.

    Args:
        shape (tuple[int, ...]): Shape of the compared arrays, ``(pts, order + 1,
            rank)``.
        degree (int): Degree of the curve.
        magnitude (float): ``max|c|``.

    Returns:
        npt.NDArray[np.float64]: The amplification, one value per derivative order.
    """
    amplification = np.empty(shape, dtype=np.float64)
    for order in range(shape[1]):
        amplification[:, order, :] = _derivative_scale(degree, order, magnitude)
    return amplification


def _reduction_amplification(
    ctrl: npt.NDArray[Any], degree: int, decrement: int
) -> npt.NDArray[np.float64]:
    """Amplification of the reduction-operator apply, per output element.

    The reduction operator is **not** a convex combination -- it has negative entries
    and reproduces the endpoints exactly while approximating the interior -- so
    ``max|c|`` is not a magnitude for its output. The absolute row action
    ``|R| @ |c|`` is, and it is what a dot product's forward-error bound carries
    anyway.

    Args:
        ctrl (npt.NDArray[Any]): The control net being reduced.
        degree (int): Its degree.
        decrement (int): Degrees removed.

    Returns:
        npt.NDArray[np.float64]: The row action, shaped like the reduced net.
    """
    operator = _interpolating_reduction_operator(degree, decrement)
    action = np.abs(operator) @ np.abs(np.float64(ctrl))
    return np.asarray(np.maximum(action, np.finfo(np.float64).tiny), dtype=np.float64)


def _mixed_control_points(
    shape: tuple[int, ...],
    dtype: npt.DTypeLike,
    seed: int = 20260821,
    exponents: tuple[int, int] = (-6, 7),
) -> npt.NDArray[np.float32 | np.float64]:
    """Control points spanning many magnitudes, so the triangle has cancellation to do.

    A net of uniform magnitude is the easy case: every partial sum is the size of
    the answer and nothing cancels. Scaling each entry by a random power of ten
    between 1e-6 and 1e6 is what makes a difference in accumulation width or in
    operation order actually reach the output.

    Args:
        shape (tuple[int, ...]): Shape of the control net, rank last.
        dtype (npt.DTypeLike): Storage format.
        seed (int): Generator seed. Defaults to 20260821.
        exponents (tuple[int, int]): Half-open range of decimal exponents to draw
            from. The default spans twelve orders of magnitude, which is right for
            an operation whose output is the size of its input. An operation that
            raises its input to a power needs a narrower range or it overflows
            float32 before any kernel is at fault. Defaults to (-6, 7).

    Returns:
        npt.NDArray[np.float32 | np.float64]: The control points.
    """
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(shape) * 10.0 ** rng.integers(*exponents, shape)
    return np.ascontiguousarray(values, dtype=dtype)


def _adversarial_parameters(dtype: npt.DTypeLike) -> list[float]:
    """Parameters reaching the branches a uniform sweep does not.

    Both endpoints, either side of the mirror threshold, a value small enough that
    ``1 - (1 - u)`` loses it outright, and both neighbours of one, where the
    unmirrored seed underflows at high degree.

    Args:
        dtype (npt.DTypeLike): Storage format, which sets what "next to one" means.

    Returns:
        list[float]: The parameters, ascending.
    """
    one = np.array(1.0, dtype=dtype)
    half = np.array(0.5, dtype=dtype)
    return [
        0.0,
        1e-20,
        1e-8,
        0.25,
        0.5,
        float(np.nextafter(half, one)),
        0.75,
        1.0 - 1e-8,
        float(np.nextafter(one, np.array(0.0, dtype=dtype))),
        1.0,
    ]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("rank", [1, 3])
def test_evaluate_matches_the_oracle(
    cpp_backend: None, degree: int, rank: int, dtype: npt.DTypeLike
) -> None:
    """The two backends evaluate a curve identically at every adversarial parameter."""
    del cpp_backend
    demand_the_compiled_kernel(dtype)

    ctrl = _mixed_control_points((degree + 1, rank), dtype)
    points = np.array(_adversarial_parameters(dtype), dtype=dtype)

    with use_backend(Backend.PYTHON):
        reference = Bezier(ctrl).evaluate(points)
    with use_backend(Backend.CPP):
        actual = Bezier(ctrl).evaluate(points)

    assert_parity(
        actual,
        reference,
        _parity_claim(
            bitwise_why=_EVALUATE_WHY,
            fused_why=(
                "the ratio recurrence that builds the basis contains no addition, so "
                "it cannot fuse and both backends form bit-identical basis values; "
                "the disassembly confirms it, with an FMA on the two accumulation "
                "lines and nowhere else. The single chain is therefore the "
                "contraction, degree stages of a convex sum, and max|c| bounds every "
                "partial sum along it as well as the result"
            ),
            budget=_Budget(
                degree,
                _absolute_companion(
                    lambda net: np.asarray(
                        Bezier(net).evaluate(np.asarray(points, dtype=np.float64))
                    ),
                    ctrl,
                ),
            ),
            storage=dtype,
        ),
        context=f"evaluate degree {degree} rank {rank} {np.dtype(dtype).name}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degree", [0, 1, 3, 8, 17])
@pytest.mark.parametrize("order", [0, 1, 2, 4])
def test_evaluate_derivatives_matches_the_oracle(
    cpp_backend: None, degree: int, order: int, dtype: npt.DTypeLike
) -> None:
    """The two backends agree on every derivative order, including past the degree.

    ``order`` runs above ``degree`` on purpose: A2.3's index bounds go negative
    there and the recursion takes branches a well-matched order never reaches.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)

    ctrl = _mixed_control_points((degree + 1, 2), dtype)
    points = np.array(_adversarial_parameters(dtype), dtype=dtype)

    with use_backend(Backend.PYTHON):
        reference = Bezier(ctrl).evaluate_derivatives(points, order)
    with use_backend(Backend.CPP):
        actual = Bezier(ctrl).evaluate_derivatives(points, order)

    assert_parity(
        actual,
        reference,
        _parity_claim(
            bitwise_why=_DE_CASTELJAU_WHY,
            fused_why=(
                "four of this kernel's five fused sites emit a storage-width FMA, "
                "because the oracle allocates every workspace at the point dtype and "
                "this port follows it. So the accumulator IS the storage here, and it "
                "is the one kernel whose float32 path a wide accumulator does not "
                "protect. The ndu table is a de Casteljau triangle and is "
                "non-expansive; the A2.3 recursion differences it, which is where the "
                "2^k in the amplification comes from"
            ),
            budget=_Budget(
                degree + 1,
                _flat_amplification(
                    np.shape(reference), _derivative_scale(degree, order, _net_magnitude(ctrl))
                ),
            ),
            storage=dtype,
            accumulator=dtype,
        ),
        context=f"evaluate_derivatives degree {degree} order {order} {np.dtype(dtype).name}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("degree", "order"), [(25, 16), (30, 14), (30, 25), (40, 13)])
def test_derivatives_agree_past_the_factorial_overflow(
    cpp_backend: None, degree: int, order: int, dtype: npt.DTypeLike
) -> None:
    """The two backends agree where A2.3's falling factorial overflows its integer.

    ``fac`` accumulates ``p (p-1) ... (p-k+1)`` in an ``int64`` in both backends, and
    it overflows: at degree 30 from order 14, at degree 61 from order 11. Every case
    below is past that point, and every one is reachable through
    :meth:`Bezier.evaluate_derivatives`, which imposes no ceiling on ``orders``.

    The oracle wraps, because numba does not trap integer overflow. The port must
    wrap identically, which is why its accumulator is **unsigned**: signed overflow is
    undefined in C++, and the natural translation was undefined behaviour rather than
    a merely wrong answer. Confirmed under ``-fsanitize=undefined``, which this
    repository's ``gcc-debug`` preset enables, so the sanitizer build was aborting on
    a case the parity suite did not reach.

    Neither backend's answer is *correct* here in any useful sense. What is under test
    is that they are the same, and that the C++ reaches it by a defined route.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    demand_a_bound_the_claim_can_carry(_WRAPPED_FACTORIAL)

    ctrl = _mixed_control_points((degree + 1, 2), dtype, exponents=(-1, 2))
    points = np.array([0.0, 0.25, 0.5, 1.0], dtype=dtype)

    with use_backend(Backend.PYTHON):
        reference = Bezier(ctrl).evaluate_derivatives(points, order)
    with use_backend(Backend.CPP):
        actual = Bezier(ctrl).evaluate_derivatives(points, order)

    assert_parity(
        actual,
        reference,
        bitwise_parity(why=_DE_CASTELJAU_WHY),
        context=f"derivatives degree {degree} order {order} {np.dtype(dtype).name}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("increment", [1, 2, 5])
def test_elevate_degree_matches_the_oracle(
    cpp_backend: None, degree: int, increment: int, dtype: npt.DTypeLike
) -> None:
    """The two backends elevate identically, coefficient table included."""
    del cpp_backend
    demand_the_compiled_kernel(dtype)

    ctrl = _mixed_control_points((degree + 1, 3), dtype)

    with use_backend(Backend.PYTHON):
        reference = Bezier(ctrl).elevate_degree(increment).control_points
    with use_backend(Backend.CPP):
        actual = Bezier(ctrl).elevate_degree(increment).control_points

    assert_parity(
        actual,
        reference,
        _parity_claim(
            bitwise_why=_BINOMIAL_WHY,
            fused_why=(
                "the coefficient table is built from an exact-integer recurrence and "
                "no addition of it fuses, so the one fused site is the accumulation. "
                "Its weights are C(p,j)C(t,i-j)/C(p+t,i), non-negative and summing to "
                "one by Vandermonde's identity, so the output is a convex combination "
                "of the input net and max|c| bounds the partial sums with it"
            ),
            budget=_Budget(
                degree + 1,
                _absolute_companion(
                    lambda net: Bezier(net).elevate_degree(increment).control_points, ctrl
                ),
            ),
            storage=dtype,
        ),
        context=f"elevate_degree {degree} by {increment} {np.dtype(dtype).name}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degree", [2, 3, 5, 8, 13, 20])
@pytest.mark.parametrize("decrement", [1, 2])
def test_reduce_degree_matches_the_oracle(
    cpp_backend: None, degree: int, decrement: int, dtype: npt.DTypeLike
) -> None:
    """The two backends apply the same reduction operator to the same result."""
    del cpp_backend
    demand_the_compiled_kernel(dtype)

    ctrl = _mixed_control_points((degree + 1, 3), dtype)

    with use_backend(Backend.PYTHON):
        reference = Bezier(ctrl).reduce_degree(decrement).control_points
    with use_backend(Backend.CPP):
        actual = Bezier(ctrl).reduce_degree(decrement).control_points

    assert_parity(
        actual,
        reference,
        _parity_claim(
            bitwise_why=_REDUCTION_WHY,
            fused_why=(
                "one fused site, the dot product of an operator row with the net, so "
                "the chain is one stage per input coefficient. The operator is the "
                "only one of the eight that is not a convex combination, which is why "
                "the amplification is the absolute row action |R| @ |c| rather than "
                "max|c|"
            ),
            budget=_Budget(degree + 1, _reduction_amplification(ctrl, degree, decrement)),
            storage=dtype,
        ),
        context=f"reduce_degree {degree} by {decrement} {np.dtype(dtype).name}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degree", [3, 4, 7, 12])
@pytest.mark.parametrize("is_rational", [False, True])
def test_minimize_degree_is_bitwise(
    cpp_backend: None, degree: int, is_rational: bool, dtype: npt.DTypeLike
) -> None:
    """The greedy degree search takes the same decisions on both backends.

    This is the one consumer that reaches two dispatched kernels within a single
    call, which is why `pantr.bezier._bezier_backend` hands out a record for them.
    It is also the only test here whose output is *discrete*: the search accepts or
    rejects each trial by comparing a round-trip error against a tolerance, so a
    disagreement in either kernel can change the resulting degree rather than the
    last bit of a coefficient. The degree is asserted before the coefficients for
    that reason.

    **The rational parametrization is not decoration, and its absence was a real
    gap.** An earlier version swept only the non-rational branch while claiming the
    test proved the two backends "took the same path". They do not take the same
    path on a rational net: that branch grades in projected space, so it samples on
    a tensor Gauss grid and builds a Bernstein collocation matrix, and
    ``_bernstein_collocation_1d`` takes its nodes from the dispatched
    :func:`~pantr.quad.get_gauss_legendre_1d` while tabulating them with a kernel
    imported directly from ``pantr.basis._basis_core``, bypassing that package's
    catalogue. So under ``PANTR_BACKEND=cpp`` the accept/reject verdict rests on a
    matrix that is half one backend and half the other.

    That bypass predates this port and is not fixed here. What this parametrization
    does is make it visible: if the mixed matrix ever moves a verdict, this is the
    test that says so.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    demand_a_bound_the_claim_can_carry(_DISCRETE_VERDICT)

    # A net that is genuinely reducible, so the search has something to find: a
    # quadratic elevated to `degree`, which is exactly recoverable. The lowest
    # degree swept is 3 because elevating by zero is refused.
    # A rational net needs a weight column, and the weights are kept near one so the
    # projected geometry stays well conditioned and the search decides on the curve
    # rather than on a near-singular divide.
    rank = 3 if is_rational else 2
    base = _mixed_control_points((3, rank), dtype, exponents=(-1, 2))
    if is_rational:
        base[:, -1] = np.array([1.0, 0.8, 1.2], dtype=dtype)
    ctrl = Bezier(base, is_rational=is_rational).elevate_degree(degree - 2).control_points

    with use_backend(Backend.PYTHON):
        reference = Bezier(ctrl, is_rational=is_rational).minimize_degree()
    with use_backend(Backend.CPP):
        actual = Bezier(ctrl, is_rational=is_rational).minimize_degree()

    assert actual.degree == reference.degree, (
        f"minimize_degree from {degree} in {np.dtype(dtype).name}: the backends "
        f"stopped at different degrees, {actual.degree} against {reference.degree}"
    )
    assert_parity(
        actual.control_points,
        reference.control_points,
        bitwise_parity(why=_REDUCTION_WHY),
        context=f"minimize_degree from {degree} {np.dtype(dtype).name}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("value", [0.0, 1e-20, 0.25, 0.5, 0.75, 1.0])
def test_slice_matches_the_oracle(
    cpp_backend: None, degree: int, value: float, dtype: npt.DTypeLike
) -> None:
    """The two backends run the same de Casteljau triangle to the same last bit."""
    del cpp_backend
    demand_the_compiled_kernel(dtype)

    ctrl = _mixed_control_points((degree + 1, degree + 1, 2), dtype)

    with use_backend(Backend.PYTHON):
        reference = Bezier(ctrl).slice(0, value)
    with use_backend(Backend.CPP):
        actual = Bezier(ctrl).slice(0, value)

    assert isinstance(actual, Bezier) and isinstance(reference, Bezier)
    assert_parity(
        actual.control_points,
        reference.control_points,
        _parity_claim(
            bitwise_why=_DE_CASTELJAU_WHY,
            fused_why=_DE_CASTELJAU_FUSED_WHY,
            budget=_Budget(
                degree,
                _absolute_companion(
                    lambda net: np.asarray(_sliced_control_points(net, value)), ctrl
                ),
            ),
            storage=dtype,
        ),
        context=f"slice degree {degree} at {value!r} {np.dtype(dtype).name}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("value", [1e-20, 1e-8, 0.25, 0.5, 0.75, 1.0 - 1e-8])
def test_split_matches_the_oracle(
    cpp_backend: None, degree: int, value: float, dtype: npt.DTypeLike
) -> None:
    """Both halves of a split agree bit for bit.

    The endpoints are absent because Layer 1 refuses them: :meth:`Bezier.split`
    requires a value strictly inside ``(0, 1)``. The kernel itself has no such
    shortcut and would run the full triangle at either end, unlike
    :meth:`~pantr.bezier.Bezier.slice`, so 1e-20 and ``1 - 1e-8`` are as close to
    the ends as this test can legitimately get.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)

    ctrl = _mixed_control_points((degree + 1, 3), dtype)

    with use_backend(Backend.PYTHON):
        ref_left, ref_right = Bezier(ctrl).split(0, value)
    with use_backend(Backend.CPP):
        got_left, got_right = Bezier(ctrl).split(0, value)

    for half, name, actual, reference in (
        (0, "left", got_left, ref_left),
        (1, "right", got_right, ref_right),
    ):
        assert_parity(
            actual.control_points,
            reference.control_points,
            _parity_claim(
                bitwise_why=_DE_CASTELJAU_WHY,
                fused_why=_DE_CASTELJAU_FUSED_WHY,
                budget=_Budget(
                    degree,
                    _absolute_companion(
                        _split_half(value, half),
                        ctrl,
                    ),
                ),
                storage=dtype,
            ),
            context=f"split {name} degree {degree} at {value!r} {np.dtype(dtype).name}",
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize(
    "bounds",
    [(0.1, 0.9), (0.0, 1e-8), (1.0 - 1e-8, 1.0), (0.25, 0.75), (0.9, 1.0), (0.0, 0.1)],
)
def test_restrict_matches_the_oracle(
    cpp_backend: None, degree: int, bounds: tuple[float, float], dtype: npt.DTypeLike
) -> None:
    """Both orderings of the two-pass restriction agree bit for bit.

    The bounds list straddles the ``|upper| >= |lower - 1|`` test that chooses
    which pass runs first, so both branches are exercised rather than whichever
    one a symmetric interval happens to pick.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)

    ctrl = _mixed_control_points((degree + 1, 3), dtype)

    with use_backend(Backend.PYTHON):
        reference = Bezier(ctrl).restrict(bounds).control_points
    with use_backend(Backend.CPP):
        actual = Bezier(ctrl).restrict(bounds).control_points

    assert_parity(
        actual,
        reference,
        _parity_claim(
            bitwise_why=_DE_CASTELJAU_WHY,
            fused_why=(
                f"{_DE_CASTELJAU_FUSED_WHY}. Twice the stages, because this is two "
                f"passes; the second is convex as well, since the ordering forces the "
                f"divisor to be at least one half and tau2 therefore lands in [0, 1], "
                f"which the kernel header derives"
            ),
            budget=_Budget(
                2 * degree,
                _absolute_companion(lambda net: Bezier(net).restrict(bounds).control_points, ctrl),
            ),
            storage=dtype,
        ),
        context=f"restrict degree {degree} to {bounds} {np.dtype(dtype).name}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("outer_degree", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("inner_degree", [1, 2, 4])
def test_compose_is_bitwise(
    cpp_backend: None, outer_degree: int, inner_degree: int, dtype: npt.DTypeLike
) -> None:
    """The Bernstein product agrees bit for bit, binomial scaling included.

    Driven through :meth:`Bezier.compose` and **not** through
    :meth:`Bezier.multiply`, which is the route a first draft of this test took and
    which exercises none of the ported code. For a 1D Bézier ``multiply`` goes to
    ``_bernstein_product_coefficients`` (and to its ``_nd`` sibling above 1D), both
    pure-numpy helpers that are not dispatched at all; the scalar 1D product kernel
    is reached only from ``compose``, and only when the inner map is univariate. The
    mistake was caught by mutation: reassociating the kernel's accumulation left the
    ``multiply`` version passing.

    A composition runs the kernel many times over -- once per Bernstein basis power
    of the inner map, then again for each tensor term -- so a single case here
    carries far more products than a single multiply would have.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    demand_a_bound_the_claim_can_carry(_OPERANDS_NOT_OBSERVABLE)

    # `inner.rank` must equal `outer.dim`, so a univariate outer map takes a
    # rank-1 inner one. That is exactly the case `use_1d_kernel` selects.
    # A composition of degree `outer_degree` raises the inner map to that power, so
    # the twelve-decade default range overflows float32 well before any kernel is
    # at fault: measured, 1e6 to the eighth is 1e48 against a float32 ceiling near
    # 3.4e38. Three decades still spans enough scale for cancellation to bite.
    spread = (-1, 2)
    outer = Bezier(_mixed_control_points((outer_degree + 1, 2), dtype, 11, spread))
    inner = Bezier(_mixed_control_points((inner_degree + 1, 1), dtype, 22, spread))

    with use_backend(Backend.PYTHON):
        reference = outer.compose(inner).control_points
    with use_backend(Backend.CPP):
        actual = outer.compose(inner).control_points

    assert_parity(
        actual,
        reference,
        bitwise_parity(why=_BINOMIAL_WHY),
        context=(f"compose degree {outer_degree} with {inner_degree} {np.dtype(dtype).name}"),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_strided_out_reaches_the_callers_array(cpp_backend: None, dtype: npt.DTypeLike) -> None:
    """A non-contiguous ``out`` is filled, and filled identically, on both backends.

    The C++ binding refuses a strided array, because ``.noconvert()`` is what stops
    nanobind from filling a temporary and discarding it, and the Python adapter
    absorbs that by buffering and copying back. An adapter that dropped the copy
    would return the right answer and leave the caller's array untouched, with no
    exception anywhere, which is the worst failure shape available.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)

    ctrl = _mixed_control_points((6, 3), dtype)
    points = np.linspace(0.0, 1.0, 9, dtype=dtype)

    results = {}
    for backend in (Backend.PYTHON, Backend.CPP):
        holder = np.zeros((3, points.size), dtype=dtype)
        view = holder.T
        assert not view.flags["C_CONTIGUOUS"]
        with use_backend(backend):
            Bezier(ctrl).evaluate(points, out=view)
        assert np.any(holder != 0.0), f"{backend.name}: the caller's array was not written"
        results[backend] = holder.copy()

    assert_parity(
        results[Backend.CPP],
        results[Backend.PYTHON],
        _parity_claim(
            bitwise_why=f"{_EVALUATE_WHY}; buffering a strided out adds no arithmetic",
            fused_why=(
                "the same claim as evaluate at degree 5; buffering a strided out adds "
                "a copy and no arithmetic, so it changes neither the stage count nor "
                "the amplification"
            ),
            budget=_Budget(
                5,
                _absolute_companion(
                    lambda net: np.asarray(
                        Bezier(net).evaluate(np.asarray(points, dtype=np.float64))
                    ).T,
                    ctrl,
                ),
            ),
            storage=dtype,
        ),
        context=f"strided out, {np.dtype(dtype).name}",
    )


def test_the_split_binding_refuses_its_outputs_positionally(cpp_backend: None) -> None:
    """``out_left`` and ``out_right`` are keyword-only, so they cannot be exchanged.

    They share a dtype and a shape, so nothing in the type system separates them
    and a positional call would silently return the two halves the wrong way round.
    This asserts the guard exists rather than trusting that it was written.
    """
    del cpp_backend
    from pantr import _pantr_cpp  # noqa: PLC0415

    ctrl = np.ascontiguousarray(np.linspace(0.0, 1.0, 8).reshape(4, 2))
    left = np.empty((4, 2))
    right = np.empty((4, 2))

    with pytest.raises(TypeError):
        _pantr_cpp.split_bezier_1d(ctrl, 0.5, left, right)  # type: ignore[misc]

    _pantr_cpp.split_bezier_1d(ctrl, 0.5, out_left=left, out_right=right)
    assert not np.array_equal(left, right), (
        "a split at the midpoint of a non-symmetric net must give two different halves"
    )


# ---------------------------------------------------------------------------
# The bounded branch, exercised on a host that cannot reach it
# ---------------------------------------------------------------------------
#
# `contraction_may_fuse()` is false on the shipped build, so every claim above takes
# its bitwise branch and the bounded one ships unevaluated. These three tests build
# it anyway and check the two things that make a bound a bound: that it permits a
# displacement inside itself, and that it refuses one outside. Copied from
# `tests/parity/test_quad_gauss_legendre.py`, which does the same for its FMA claims.


def _probe_budgets() -> list[tuple[str, _Budget, npt.DTypeLike, npt.DTypeLike]]:
    """Return one representative budget per kernel, for the probes.

    The magnitudes are the ones a real net produces, so a tolerance that came out
    absurdly small or large here would show up as a failing probe rather than as a
    number nobody looked at.

    Returns:
        list: Name, budget, storage and accumulator, one entry per kernel.
    """
    degree = 8
    ctrl = _mixed_control_points((degree + 1, 3), np.float64)
    magnitude = _net_magnitude(ctrl)
    flat = _flat_amplification((degree + 1, 3), magnitude)
    return [
        ("evaluate", _Budget(degree, flat), np.float64, np.float64),
        ("slice", _Budget(degree, flat), np.float64, np.float64),
        ("split", _Budget(degree, flat), np.float64, np.float64),
        ("restrict", _Budget(2 * degree, flat), np.float64, np.float64),
        ("degree_elevate", _Budget(degree + 1, flat), np.float64, np.float64),
        (
            "reduce_degree",
            _Budget(degree + 1, _reduction_amplification(ctrl, degree, 1)),
            np.float64,
            np.float64,
        ),
        (
            "evaluate_deriv float32",
            _Budget(
                degree + 1,
                _flat_amplification((degree + 1, 3), _derivative_scale(degree, 2, magnitude)),
            ),
            np.float32,
            np.float32,
        ),
        ("evaluate float32", _Budget(degree, flat), np.float32, np.float64),
    ]


def test_the_fused_bound_admits_a_displacement_inside_itself() -> None:
    """Each kernel's bounded claim accepts three quarters of its own tolerance.

    Needs no C++ backend: what is under test is the claim's own arithmetic.
    """
    for name, budget, storage, accumulator in _probe_budgets():
        claim = _fused_claim(
            fused_why=f"probe for {name}",
            budget=budget,
            storage=storage,
            accumulator=accumulator,
        )
        tolerance = absolute_tolerance(claim)
        assert float(np.max(tolerance)) > 0.0, (
            f"{name}: the bounded branch derived a zero tolerance"
        )
        reference = np.asarray(budget.amplification, dtype=storage)
        deviation = assert_parity(
            np.asarray(reference + 0.75 * tolerance, dtype=storage),
            reference,
            claim,
            context=f"{name} fused probe, inside the bound",
        )
        assert deviation.max_ratio_to_bound > 0.5, (
            f"{name}: a displacement of three quarters of the bound registered as "
            f"{deviation.max_ratio_to_bound:.3g} of it, so the perturbation did not "
            f"survive its own rounding and this probe tested nothing"
        )


def test_the_fused_bound_refuses_a_displacement_past_itself() -> None:
    """Each kernel's bounded claim rejects one and a half times its own tolerance.

    The half that decides whether the bound is a bound at all: a tolerance nothing
    can exceed would pass the probe above unchanged.
    """
    for name, budget, storage, accumulator in _probe_budgets():
        claim = _fused_claim(
            fused_why=f"probe for {name}",
            budget=budget,
            storage=storage,
            accumulator=accumulator,
        )
        reference = np.asarray(budget.amplification, dtype=storage)
        perturbed = np.asarray(reference + 1.5 * absolute_tolerance(claim), dtype=storage)
        with pytest.raises(AssertionError, match="more than the derived bound"):
            assert_parity(
                perturbed, reference, claim, context=f"{name} fused probe, past the bound"
            )


def test_the_oracle_does_not_contract_a_multiply_add() -> None:
    """Numba emits no FMA without fastmath, which is what makes the bound one-sided.

    **This is the premise the whole fused bound rests on**, and it is a property of
    numba's code generation rather than of pantr. Numba compiles for the host CPU, so
    on any machine with an FMA the oracle *could* fuse; it does not, because LLVM does
    not contract unless fastmath is on, and no pantr kernel sets it. If that default
    ever changes the bound stops being one-sided in the direction it assumes, and this
    test is what says so.

    Deliberately probed on a throwaway function rather than on the Bezier kernels:
    ``inspect_asm`` returns a warning and invalid output for code loaded from numba's
    on-disk cache, which every pantr kernel is. A ``cache=False`` function is the only
    spelling that reports the compiler's real output.
    """
    import re  # noqa: PLC0415

    from numba import njit  # noqa: PLC0415

    fused = re.compile(r"\bvf(n?)m(add|sub)[a-z0-9]*\b")

    @njit(cache=False, fastmath=False)
    def without_fastmath(a: float, b: float, c: float) -> float:
        return a * b + c

    @njit(cache=False, fastmath=True)
    def with_fastmath(a: float, b: float, c: float) -> float:
        return a * b + c

    without_fastmath(1.0, 2.0, 3.0)
    with_fastmath(1.0, 2.0, 3.0)

    plain = sum(len(fused.findall(asm)) for asm in without_fastmath.inspect_asm().values())
    fast = sum(len(fused.findall(asm)) for asm in with_fastmath.inspect_asm().values())

    if fast == 0:
        pytest.skip(
            "this host's numba target has no fused multiply-add, so the check cannot "
            "discriminate: a zero from the fastmath=False build would mean nothing"
        )
    assert plain == 0, (
        f"numba emitted {plain} fused multiply-adds for a * b + c without fastmath, on "
        f"a target where fastmath produces {fast}. The Bezier parity bound assumes the "
        f"oracle never fuses and this backend may, so a fusing oracle makes the bound "
        f"describe the wrong difference"
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("p_degree", "q_degree"), [(1, 1), (3, 2), (8, 5), (17, 8), (25, 13)])
def test_the_product_kernel_matches_the_oracle_at_its_own_entry(
    cpp_backend: None, p_degree: int, q_degree: int, dtype: npt.DTypeLike
) -> None:
    """The Bernstein product agrees, driven where its operands are visible.

    ``test_compose_is_bitwise`` reaches this kernel through the public surface, which
    is the right level for an exactness claim and the wrong one for a bounded claim:
    the operands are formed inside the composition, so the only amplification
    available from outside is ``max|outer| * (1 + 2 max|inner|)^p``, which is around
    1e18 at those degrees and would accept anything.

    Driven at the Layer 3 entry the operands are known and the amplification is the
    exact convex sum ``sum_i C(p,i) C(q,k-i) |a_i| |b_{k-i}| / C(p+q,k)``, whose
    weights are non-negative and sum to one by Vandermonde's identity. That is what
    lets this kernel keep a real bound on a fusing build while ``compose`` skips.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)

    from pantr.bezier import _bezier_backend as backend  # noqa: PLC0415

    total = p_degree + q_degree
    left = _mixed_control_points((p_degree + 1,), dtype, seed=20260822)
    right = _mixed_control_points((q_degree + 1,), dtype, seed=20260823)

    results = {}
    for which in (Backend.PYTHON, Backend.CPP):
        out = np.zeros(total + 1, dtype=dtype)
        backend.product_kernel(which)(left, right, out)
        results[which] = out

    weights = np.array(
        [
            sum(
                comb(p_degree, i)
                * comb(q_degree, k - i)
                * abs(float(left[i]))
                * abs(float(right[k - i]))
                for i in range(max(0, k - q_degree), min(p_degree, k) + 1)
            )
            / comb(total, k)
            for k in range(total + 1)
        ],
        dtype=np.float64,
    )

    assert_parity(
        results[Backend.CPP],
        results[Backend.PYTHON],
        _parity_claim(
            bitwise_why=_BINOMIAL_WHY,
            fused_why=(
                "one fused site, the accumulation of a_i * b_j scaled by the two "
                "binomials, so the chain is min(p, q) + 1 stages long. Dividing the "
                "finished row by C(p+q,k) makes the weights a partition of unity, by "
                "Vandermonde, so the amplification is the convex sum of |a_i b_j| and "
                "the accumulator is float64 at both storage widths because the "
                "binomial table is float64 unconditionally"
            ),
            budget=_Budget(min(p_degree, q_degree) + 1, np.maximum(weights, _TINY)),
            storage=dtype,
        ),
        context=f"scalar product {p_degree} x {q_degree} {np.dtype(dtype).name}",
    )
