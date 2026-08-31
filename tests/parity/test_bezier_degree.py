"""Parity for the n-dimensional Bézier degree operations.

**Three claims, and they are not the same kind.** Elevation and reduction compose
over kernels ``tests/parity/test_bezier_arithmetic.py`` already claims bitwise, and
everything this port adds around them -- the axis permutation, the buffer handling,
the ``Bezier`` reconstruction -- moves values without computing on them. So on a
build that cannot fuse a multiply-add they are **bitwise**, measured over 56
configurations spanning both dtypes, dim 1 to 3, ranks 1 and 3, rational and not,
with zero mismatches. On a fusing build they inherit ``design/backend_parity.md``
Rule 10's budget for the kernels underneath them, which is why the two rounding
constants come from that file rather than being spelled again here: one derivation
of Rule 10 in the tree, not two that can drift.

``degree_reduction_error`` is **bounded unconditionally**, and for a reason that has
nothing to do with fusion. Its oracle takes the Bernstein-Gram quadratic form
through :func:`numpy.tensordot`, which reshapes to a matrix product and reaches
BLAS, and finishes with :func:`numpy.sum`, whose pairwise summation order is not
reproducible. No transliteration is bit-exact against either, so there is no bitwise
arm to condition on and the claim carries no branch.

What the operators do not contribute
------------------------------------

Both backends are handed the **same** reduction operators and Gram matrices,
assembled once above the backend branch in exact rational and exact integer
arithmetic respectively. That is what makes these comparisons comparisons of the
*operation*: an operator assembled twice would put two roundings of an exact
rational inside a claim about a contraction.
``cpp/include/pantr/bezier/degree.hpp`` records why that assembly is not ported and
what porting it would cost.

The claim the error does not make
---------------------------------

``degree_reduction_error`` is documented as returning the true ``||f - g||`` rather
than an estimate, and **that is a statement about the formula, not the arithmetic**.
The formula is exact: elevation back to the original degrees is an exact operation on
polynomials, and the Bernstein-Gram matrix is the exact inner product of the basis
(Farouki and Rajan, *Computer Aided Geometric Design* 5, 1988). The computed value
carries the roundings of a reduction, an elevation and a quadratic form, which is
what the bound below is. Nothing here asserts the formula's exactness; the tests that
would are the C++ ones, which check it against a known answer.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pytest

from pantr._backend import Backend, use_backend
from pantr.bezier import Bezier
from pantr.bezier._bezier_degree import (
    _reduction_operators,
    _squared_l2_norm,
)
from tests._parity_harness import (
    ParityClaim,
    Roundings,
    assert_parity,
    bitwise_parity,
    bounded_parity,
    contraction_may_fuse,
    demand_the_compiled_kernel,
)

# Rule 10's per-stage budget, imported rather than restated. The derivation lives in
# the sibling file next to the kernels it was derived for, and these operations are
# compositions of exactly those kernels, so a second spelling here would be a second
# thing to keep true.
from tests.parity.test_bezier_arithmetic import (
    _ACCUMULATOR_ROUNDINGS_PER_STAGE,
    _FUSED_PREFIX,
    _STORAGE_ROUNDINGS_PER_STAGE,
)

DTYPES: Final = (np.float64, np.float32)
"""Both storage formats."""

DEGREES: Final = ((3,), (1, 1), (2, 3), (5, 4), (0, 4), (2, 2, 2), (4, 3, 2))
"""Degree tuples spanning dim 1 to 3, including a degree-0 direction.

A degree-0 direction cannot be reduced, so it exercises the branch that leaves a
direction untouched -- the one an off-by-one in the per-direction loop would break
without changing any other case.
"""

RANKS: Final = (1, 3)
"""Output ranks."""

_TINY: Final = float(np.finfo(np.float64).tiny)
"""Floor for an amplification, so a tolerance is never identically zero."""

_ELEVATE_BITWISE_WHY: Final = (
    "elevation composes over degree_elevate_bezier_1d, whose bitwise claim "
    "tests/parity/test_bezier_arithmetic.py carries, and adds nothing that computes: "
    "the axis permutation moves values, and reconstructing the Bezier copies them. "
    "On a build that cannot fuse a multiply-add the two backends therefore run the "
    "same operations on the same values in the same order. Measured over 56 "
    "configurations spanning both dtypes, dim 1 to 3, ranks 1 and 3, rational and "
    "not: zero mismatches"
)

_REDUCE_BITWISE_WHY: Final = (
    "reduction composes over core::apply_reduction_operator, whose bitwise claim "
    "tests/parity/test_bezier_arithmetic.py carries, over an operator BOTH backends "
    "are handed rather than each assembling. The permutation and the reconstruction "
    "compute nothing, so on a non-fusing build the two run the same operations on the "
    "same values in the same order. Measured over the same 56 configurations: zero "
    "mismatches"
)

_ELEVATE_FUSED_WHY: Final = (
    "the elevation chain accumulating into out[i] runs j from max(0, i - t) to "
    "min(p, i), so it is min(p, t) + 1 stages and not p + 1 -- Rule 10 records that "
    "charging p + 1 was 13x too loose at p = 25. The stages of the directions "
    "actually elevated add, since each direction's output feeds the next. Every "
    "weight is non-negative and they sum to one, so the absolute-value companion -- "
    "the same elevation run on |c| -- is the magnitude reachable at each output "
    "element, and is tight rather than merely valid"
)

_REDUCE_FUSED_WHY: Final = (
    "the reduction apply is p + 1 stages per reduced direction, and narrows ONCE per "
    "output element rather than once per stage, because it accumulates into a float64 "
    "local outside the loop over the operator row -- Rule 10 records that charging a "
    "store per stage over-counted it by p + 1, up to 52x at float32. The operator has "
    "negative entries, so the absolute-value companion is NOT available and the "
    "amplification is |R| @ |c| applied per direction, which is the row action of the "
    "finished operator"
)

_ERROR_WHY: Final = (
    "unconditionally bounded, and not because of fusion: the oracle's quadratic form "
    "goes through np.tensordot, which reaches BLAS, and np.sum, whose pairwise order "
    "is not reproducible, so no transliteration is bit-exact against it and there is "
    "no bitwise arm to condition on. The budget is one rounding per accumulation "
    "step: the Gram contraction over sum_a n_a terms per axis, the inner product over "
    "every coefficient -- n and not log n, because this backend sums sequentially "
    "where numpy sums pairwise -- one per component summed, and the final square "
    "root. The reduction's and elevation's own stages are added ONLY on a fusing "
    "build: elsewhere both are bitwise, so the coefficient difference is identical on "
    "the two sides and charging for it would charge for a difference of exactly zero. "
    "The "
    "amplification is the quotient rule for a square root -- |d sqrt(S)| <= "
    "(gamma A) / (2 sqrt(S)) + u sqrt(S), with A the Gram form evaluated on |c| and S "
    "on c -- and A >= S entrywise because a Bernstein Gram matrix has no negative "
    "entry"
)


def _mixed_control_points(
    shape: tuple[int, ...], dtype: npt.DTypeLike, seed: int
) -> npt.NDArray[np.float32 | np.float64]:
    """Control points spanning many magnitudes, so the operations have cancelling to do.

    Args:
        shape (tuple[int, ...]): Shape of the control net, components last.
        dtype (npt.DTypeLike): Storage format.
        seed (int): Generator seed.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The control points.
    """
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(shape) * 10.0 ** rng.integers(-4, 5, shape)
    return np.ascontiguousarray(values, dtype=dtype)


def _net(
    degrees: tuple[int, ...], rank: int, dtype: npt.DTypeLike, seed: int, *, rational: bool
) -> npt.NDArray[np.float32 | np.float64]:
    """Build a control net, giving a rational one weights bounded away from zero.

    Args:
        degrees (tuple[int, ...]): Degree per direction.
        rank (int): Number of value components.
        dtype (npt.DTypeLike): Storage format.
        seed (int): Generator seed.
        rational (bool): Whether to append a weight column.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The control net.
    """
    components = rank + 1 if rational else rank
    net = _mixed_control_points((*(d + 1 for d in degrees), components), dtype, seed)
    if rational:
        rng = np.random.default_rng(seed + 1)
        net[..., -1] = np.asarray(rng.uniform(0.5, 2.0, net.shape[:-1]), dtype=dtype)
    return net


def _increments(degrees: tuple[int, ...]) -> tuple[int, ...]:
    """Elevate the even directions by one and the odd ones by two.

    Uneven on purpose: a per-direction loop that used one direction's increment for
    all of them would still pass with a uniform vector.

    Args:
        degrees (tuple[int, ...]): Degree per direction, for the direction count.

    Returns:
        tuple[int, ...]: One increment per direction.
    """
    return tuple(1 if d % 2 == 0 else 2 for d in range(len(degrees)))


def _decrements(degrees: tuple[int, ...]) -> tuple[int, ...]:
    """Drop one degree in every direction that has one to drop.

    Args:
        degrees (tuple[int, ...]): Degree per direction.

    Returns:
        tuple[int, ...]: One decrement per direction, zero where the degree is zero.
    """
    return tuple(1 if p >= 1 else 0 for p in degrees)


def _fused(
    *,
    stages: int,
    stores: int,
    amplification: npt.NDArray[np.float64],
    why: str,
    dtype: npt.DTypeLike,
) -> ParityClaim:
    """Build the bounded claim a fusing build needs.

    Args:
        stages (int): Length of the dependency chain the fused sites sit on.
        stores (int): How many times that chain narrows into the storage format.
        amplification (npt.NDArray[np.float64]): Elementwise magnitude.
        why (str): The derivation.
        dtype (npt.DTypeLike): Storage format; the accumulator is float64.

    Returns:
        ParityClaim: The BOUNDED claim.
    """
    roundings = (
        Roundings(
            stages=stages,
            accumulator_per_stage=_ACCUMULATOR_ROUNDINGS_PER_STAGE,
            storage_per_stage=_STORAGE_ROUNDINGS_PER_STAGE,
        )
        if stores == stages
        # Collapsed to one stage because the two counts no longer share a denominator,
        # which `Roundings` documents as the way to spell exactly that. The chain is
        # still `stages` long and `why` says so.
        else Roundings(
            stages=1,
            accumulator_per_stage=_ACCUMULATOR_ROUNDINGS_PER_STAGE * stages,
            storage_per_stage=_STORAGE_ROUNDINGS_PER_STAGE * stores,
        )
    )
    return bounded_parity(
        roundings=roundings,
        accumulator=np.float64,
        storage=dtype,
        amplification=amplification,
        why=f"{_FUSED_PREFIX}{why}",
    )


def _companion(values: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """Floor an amplification so a tolerance is never identically zero.

    Args:
        values (npt.NDArray[Any]): The magnitudes.

    Returns:
        npt.NDArray[np.float64]: The same, floored.
    """
    return np.ascontiguousarray(
        np.maximum(np.abs(np.asarray(values, dtype=np.float64)), _TINY), dtype=np.float64
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degrees", DEGREES)
@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("rational", [False, True])
def test_degree_elevation_matches_the_oracle(
    cpp_backend: None,
    degrees: tuple[int, ...],
    rank: int,
    dtype: npt.DTypeLike,
    *,
    rational: bool,
) -> None:
    """Elevation is bitwise where nothing fuses, and inside Rule 10's budget where it does."""
    del cpp_backend
    demand_the_compiled_kernel(dtype)

    net = _net(degrees, rank, dtype, seed=20260831, rational=rational)
    increments = _increments(degrees)

    with use_backend(Backend.PYTHON):
        reference = np.asarray(
            Bezier(net, is_rational=rational).elevate_degree(increments).control_points
        )
        magnitude = np.asarray(
            Bezier(np.abs(net), is_rational=rational).elevate_degree(increments).control_points
        )
    with use_backend(Backend.CPP):
        actual = np.asarray(
            Bezier(net, is_rational=rational).elevate_degree(increments).control_points
        )

    stages = sum(min(degrees[d], increments[d]) + 1 for d in range(len(degrees)))
    claim = (
        bitwise_parity(why=_ELEVATE_BITWISE_WHY)
        if not contraction_may_fuse()
        else _fused(
            stages=stages,
            stores=stages,
            amplification=_companion(magnitude),
            why=_ELEVATE_FUSED_WHY,
            dtype=dtype,
        )
    )
    assert_parity(
        actual,
        reference,
        claim,
        context=f"elevate_degree {degrees} rank {rank} rational {rational} {dtype}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degrees", DEGREES)
@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("rational", [False, True])
def test_degree_reduction_matches_the_oracle(
    cpp_backend: None,
    degrees: tuple[int, ...],
    rank: int,
    dtype: npt.DTypeLike,
    *,
    rational: bool,
) -> None:
    """Reduction is bitwise where nothing fuses, and inside Rule 10's budget where it does."""
    del cpp_backend
    demand_the_compiled_kernel(dtype)

    net = _net(degrees, rank, dtype, seed=20260832, rational=rational)
    decrements = _decrements(degrees)
    if not any(decrements):
        pytest.skip("every direction has degree 0, so there is nothing to reduce")

    with use_backend(Backend.PYTHON):
        bezier = Bezier(net, is_rational=rational)
        reference = np.asarray(bezier.reduce_degree(decrements).control_points)
        # The row action of the finished operator, per direction: |R| @ |c|. The
        # absolute-value companion is NOT available here, because a reduction
        # operator has negative entries and its partial sums can exceed what
        # survives to the output.
        operators = _reduction_operators(bezier, decrements)
        magnitude = np.abs(np.asarray(net, dtype=np.float64))
        for d, decrement in enumerate(decrements):
            if decrement:
                magnitude = np.moveaxis(
                    np.tensordot(
                        np.abs(operators[d]), np.moveaxis(magnitude, d, 0), axes=([1], [0])
                    ),
                    0,
                    d,
                )
    with use_backend(Backend.CPP):
        actual = np.asarray(
            Bezier(net, is_rational=rational).reduce_degree(decrements).control_points
        )

    stages = sum(degrees[d] + 1 for d in range(len(degrees)) if decrements[d])
    reduced_directions = sum(1 for decrement in decrements if decrement)
    claim = (
        bitwise_parity(why=_REDUCE_BITWISE_WHY)
        if not contraction_may_fuse()
        else _fused(
            stages=stages,
            stores=reduced_directions,
            amplification=_companion(magnitude),
            why=_REDUCE_FUSED_WHY,
            dtype=dtype,
        )
    )
    assert_parity(
        actual,
        reference,
        claim,
        context=f"reduce_degree {degrees} rank {rank} rational {rational} {dtype}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degrees", DEGREES)
@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("rational", [False, True])
def test_the_reduction_error_matches_the_oracle(
    cpp_backend: None,
    degrees: tuple[int, ...],
    rank: int,
    dtype: npt.DTypeLike,
    *,
    rational: bool,
) -> None:
    """The reduction error agrees inside the quadratic form's own rounding budget."""
    del cpp_backend
    demand_the_compiled_kernel(dtype)

    net = _net(degrees, rank, dtype, seed=20260833, rational=rational)
    decrements = _decrements(degrees)
    if not any(decrements):
        pytest.skip("every direction has degree 0, so there is nothing to reduce")

    with use_backend(Backend.PYTHON):
        bezier = Bezier(net, is_rational=rational)
        reference = bezier.degree_reduction_error(decrements)
        difference = np.asarray(
            bezier.reduce_degree(decrements).elevate_degree(decrements).control_points
        ) - np.asarray(net)
    with use_backend(Backend.CPP):
        actual = Bezier(net, is_rational=rational).degree_reduction_error(decrements)

    components = difference.shape[-1]
    squared = sum(_squared_l2_norm(difference[..., r]) for r in range(components))
    magnitude = sum(_squared_l2_norm(np.abs(difference[..., r])) for r in range(components))

    coefficients = int(np.prod(difference.shape[:-1]))
    gram_stages = sum(p + 1 for p in degrees)
    stages = gram_stages + coefficients + components + 1
    if contraction_may_fuse():
        # Only where the build can fuse does the coefficient difference itself move:
        # on a non-fusing build the reduction and the elevation are bitwise, which the
        # two tests above assert, so `difference` is identical on both sides and
        # charging for it would be charging for a difference that is exactly zero.
        stages += sum(min(degrees[d], decrements[d]) + 1 for d in range(len(degrees)))
        stages += sum(degrees[d] + 1 for d in range(len(degrees)) if decrements[d])

    root = np.sqrt(squared)
    amplification = np.asarray(
        [magnitude / (2.0 * root) + root if root > 0.0 else _TINY], dtype=np.float64
    )

    assert_parity(
        np.asarray([actual], dtype=np.float64),
        np.asarray([reference], dtype=np.float64),
        bounded_parity(
            roundings=Roundings(stages=stages, accumulator_per_stage=1, storage_per_stage=0),
            accumulator=np.float64,
            storage=np.float64,
            amplification=amplification,
            why=_ERROR_WHY,
        ),
        context=f"degree_reduction_error {degrees} rank {rank} rational {rational} {dtype}",
    )


_SWEEP_DRAWS: Final = 10
"""Independent nets per configuration in the ten-times sweep.

Sized against the shipped parametrization rather than picked: each claim ships 28
cases per dtype (7 degree tuples x 2 ranks x 2 rationalities), so ten draws of each
gives 280 per claim per dtype, which is the factor of ten the ticket asks for. The
arithmetic is written down because the test passes at any number of draws, which is
exactly the property that lets a sweep quietly stop being the thing it claims to be.
"""


@pytest.mark.slow
@pytest.mark.parametrize("dtype", DTYPES)
def test_each_claim_holds_over_a_sweep_ten_times_the_shipped_one(
    cpp_backend: None, dtype: npt.DTypeLike
) -> None:
    """A bound checked only by the sweep that ships with it has not been checked.

    ``design/backend_parity.md`` records the case: the affine-map bound held over 500
    draws and failed 88 times at 60000. This runs ten independent nets per
    configuration against all three claims and asserts, per claim, that the sweep
    actually exercised it -- a claim that agreed bit for bit everywhere would mean
    nothing here evaluated its bound, and for the two bitwise claims it means the
    opposite, that a difference would have been noticed.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)

    exercised = {"elevate": 0, "reduce": 0, "error": 0}
    seed = 0
    for degrees in DEGREES:
        for rank in RANKS:
            for rational in (False, True):
                for _draw in range(_SWEEP_DRAWS):
                    seed += 1
                    net = _net(degrees, rank, dtype, seed=70000 + seed, rational=rational)
                    increments = _increments(degrees)
                    decrements = _decrements(degrees)

                    with use_backend(Backend.PYTHON):
                        source = Bezier(net, is_rational=rational)
                        elevated = np.asarray(source.elevate_degree(increments).control_points)
                    with use_backend(Backend.CPP):
                        actual = np.asarray(
                            Bezier(net, is_rational=rational)
                            .elevate_degree(increments)
                            .control_points
                        )
                    assert np.array_equal(actual, elevated), (
                        f"elevation is claimed bitwise on a non-fusing build and this "
                        f"draw differs: degrees {degrees} rank {rank} rational "
                        f"{rational} {np.dtype(dtype).name}"
                    )
                    exercised["elevate"] += 1

                    if not any(decrements):
                        continue
                    with use_backend(Backend.PYTHON):
                        reduced = np.asarray(source.reduce_degree(decrements).control_points)
                        reference_error = source.degree_reduction_error(decrements)
                    with use_backend(Backend.CPP):
                        target = Bezier(net, is_rational=rational)
                        actual_reduced = np.asarray(target.reduce_degree(decrements).control_points)
                        actual_error = target.degree_reduction_error(decrements)
                    assert np.array_equal(actual_reduced, reduced), (
                        f"reduction is claimed bitwise on a non-fusing build and this "
                        f"draw differs: degrees {degrees} rank {rank} rational "
                        f"{rational} {np.dtype(dtype).name}"
                    )
                    exercised["reduce"] += 1

                    # The error's own bound is asserted by the shipped test; what this
                    # sweep adds is breadth, so it re-derives the same amplification
                    # rather than a looser stand-in.
                    difference = np.asarray(
                        Bezier(net, is_rational=rational)
                        .reduce_degree(decrements)
                        .elevate_degree(decrements)
                        .control_points
                    ) - np.asarray(net)
                    components = difference.shape[-1]
                    squared = sum(_squared_l2_norm(difference[..., r]) for r in range(components))
                    magnitude = sum(
                        _squared_l2_norm(np.abs(difference[..., r])) for r in range(components)
                    )
                    root = np.sqrt(squared)
                    stages = (
                        sum(p + 1 for p in degrees)
                        + int(np.prod(difference.shape[:-1]))
                        + components
                        + 1
                    )
                    bound = (
                        2.0
                        * (stages * np.finfo(np.float64).eps / 2.0)
                        * (magnitude / (2.0 * root) + root if root > 0.0 else _TINY)
                    )
                    assert abs(actual_error - reference_error) <= bound, (
                        f"the reduction error left its bound: degrees {degrees} rank "
                        f"{rank} rational {rational} {np.dtype(dtype).name}"
                    )
                    exercised["error"] += 1

    for claim, count in exercised.items():
        assert count >= 280, (
            f"the {claim} claim saw {count} comparisons, short of the 280 per dtype "
            f"that is ten times its shipped 28. The sweep has stopped being ten times "
            f"the shipped one."
        )
