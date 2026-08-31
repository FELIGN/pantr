r"""Parity of the Bézier product and composition against their NumPy oracles.

`cpp/include/pantr/bezier/product.hpp` names this file as the place its parity
claims are measured.

Both claims are **equalities**, and for `multiply` that is not a happy accident:
the oracle is `_bernstein_product_coefficients` and its `_nd` sibling, pure NumPy
helpers whose operation order the port reproduces term for term. Reassociating any
one of the three orderings `product.hpp` enumerates changes the last bits, and the
mutation section below records what each is worth.

The oracle is NumPy, not Numba, and that changes two things
----------------------------------------------------------

**The arithmetic runs at the storage width.** NumPy computes a ``float32``
expression in ``float32``, so every intermediate in the port is ``T`` and the
accumulator of these claims is the storage format rather than ``float64``. That
inverts `test_bezier_arithmetic.py`, whose kernels promote their ``float64``
coefficient tables against a narrow array and so accumulate wide. The practical
consequence for the bound is that Rule 10's *storage* roundings are charged at zero
here -- there is no narrowing store to straddle a boundary, because the accumulator
is the output array itself -- and only the three accumulator roundings remain.

**No claim here needs :func:`demand_the_compiled_kernel`, except one.**
``NUMBA_DISABLE_JIT=1`` replaces the *Numba* kernels with interpreted Python, which
is Rule 12's whole subject; it does nothing at all to NumPy. `multiply` reaches no
Numba, so its ``float32`` claim stands under interpretation. `compose` reaches the
Numba product kernel and only for a **univariate** inner map, so exactly that
parametrization is gated and the n-dimensional one is not.

`p + q = 80`, which is the point of the tables crossing as data
--------------------------------------------------------------

`test_the_product_stays_exact_where_the_kernel_leaves_its_envelope` multiplies two
degree-40 curves. The result's binomials reach ``C(80, 40)``, which is nineteen
above ``_BINCOEFF_MAX_N``: `scalar_bernstein_product_1d` is undefined there and the
NumPy helper is not, because :func:`math.comb` is arbitrary precision. That
difference of domain is one of the three reasons
``design/cross_backend_types.md`` gives for `multiply` never routing through the
kernel, and it is why the C++ port is handed its tables rather than computing them.
A port that had assembled them from ``core::bincoeff`` would agree on every case in
this file but that one.

Where the composition's fused-build coverage actually comes from
---------------------------------------------------------------

On a build with a fused multiply-add the composition's parity test skips, for the
reason `test_bezier_arithmetic.py`'s ``_OPERANDS_NOT_OBSERVABLE`` already records
and which the whole-composition port does not improve: the products' operands are
formed inside the composition, so the only amplification available from the public
surface is the companion run on absolute values, and against an outer net spanning
three decades whose result cancels that bound is one the harness refuses as
vacuous.

What covers the arithmetic there instead, stated so the gap is not larger than it
looks: the composition's products are the same two implementations `multiply` uses,
and both have a bounded arm here;
``test_the_product_kernel_matches_the_oracle_at_its_own_entry`` in
`test_bezier_arithmetic.py` covers the univariate kernel at its own entry. What is
**not** covered on a fusing build is the composition's own final accumulation,
``result += coef * basis``, and that is a real residual gap rather than an
oversight.

The mutation record
-------------------

AC5 of FELIGN/pantr#392 makes a mutation check mandatory here rather than
customary, because this is the exact place a parity test in this repo once
exercised none of the ported code. Seven mutations were applied one at a time to
`cpp/include/pantr/bezier/product.hpp`, the extension rebuilt for each, and this
file run against it; the baseline is **80 passed**.

Five are caught:

1. **Accumulate the univariate product in ``j``-major order** (swap the two loops of
   ``bernstein_product_1d``): 9 failed.
2. **Reverse the n-dimensional convolution's multi-index order** (count ``alpha``
   down instead of up): 25 failed.
3. **Reassociate ``(coeff * f_i) * g_j`` to ``coeff * (f_i * g_j)``**: 20 failed.
4. **Compute the univariate term in ``accumulator_t<T>``** rather than at the storage
   width: 9 failed, every one of them ``float32``, which is the whole point of Rule 9.
5. **Swap the operands of the power recurrence** in ``scalar_net_powers``
   (``g * g^(k-1)`` for ``g^(k-1) * g``): 8 failed. This one reaches only through
   ``compose``, so it is what pins that the composition is ported rather than merely
   dispatched.

Two survive, and that is a fact about the arithmetic rather than a gap here:

6. **Accumulate ``d[k] + term`` in ``accumulator_t<T>``**: 80 passed.
7. **Weight the n-dimensional operands in ``accumulator_t<T>``**: 80 passed.

Both widen a **single** operation and narrow it straight back, which cannot move the
result: a product of two ``float32`` values is exact in ``double``, and a sum of two
of them is subject to the double-rounding theorem `product.hpp`'s width section
cites. Mutation 3 and mutation 4 are the ones that widen a *chain*, and both are
caught. Nothing here would notice a widening that is provably a no-op, and nothing
should.

The counts are in this docstring rather than beside an assertion because they are
counts over a particular parametrization, and a reader deciding whether to believe
one wants the parametrization.
"""

from __future__ import annotations

import math
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import pytest

from pantr._backend import Backend, use_backend
from pantr.bezier import Bezier
from pantr.bspline._bspline_degree_core import _BINCOEFF_MAX_N
from tests._parity_harness import (
    ParityClaim,
    Roundings,
    assert_parity,
    bitwise_parity,
    bounded_parity,
    contraction_may_fuse,
    demand_the_compiled_kernel,
)
from tests.parity.test_bezier_arithmetic import (
    _ACCUMULATOR_ROUNDINGS_PER_STAGE,
    _FUSED_PREFIX,
    _STORAGE_ROUNDINGS_PER_STAGE,
    demand_a_bound_the_claim_can_carry,
)

DTYPES: Final = (np.float64, np.float32)
"""Both storage formats."""

DEGREE_PAIRS: Final = (
    ((3,), (2,)),
    ((1,), (5,)),
    ((0,), (4,)),
    ((2, 2), (1, 3)),
    ((3, 1), (0, 2)),
    ((1, 1, 1), (2, 1, 2)),
)
"""Operand degree tuples, dim 1 to 3, including a degree-0 direction on either side.

A degree-0 direction is not decoration: it is the case where one operand contributes
a single coefficient along an axis, so an off-by-one in the convolution's extents
still produces an array of the right total size in one dimension and the wrong one
above.
"""

RANKS: Final = (1, 3)
"""Output ranks: a scalar field and a vector one."""

_TINY: Final = float(np.finfo(np.float64).tiny)
"""Floor for an amplification, so a tolerance is never identically zero."""

_PRODUCT_WHY: Final = (
    "both backends run the same NumPy expression in the same order over +, * and /, "
    "each pinned by IEEE 754 to one correctly rounded result: the reciprocal binomial "
    "folded into the coefficient before any accumulation for a curve and applied to "
    "the finished convolution above one, the two control values multiplied in left to "
    "right, and each coefficient's sum taken in the order np.add.at applies its index "
    "array, which is i-major. The binomial tables cross as data, assembled once by "
    "math.comb and rounded to the storage format by the same expression the oracle "
    "uses, so neither backend assembles its own. No fused multiply-add on this build"
)

_PRODUCT_FUSED_WHY: Final = (
    "one fused site per accumulated term, `d[k] + coeff * f_i * g_j`, so the chain is "
    "as long as the number of terms reaching one coefficient. The weights "
    "C(p,i) C(q,j) / C(p+q,k) are non-negative and sum to one over the terms of "
    "coefficient k, by Vandermonde's identity, so the amplification is the exact "
    "convex sum of |f_i g_j| -- obtained by running the same product on |c|, which is "
    "that sum term for term. The ACCUMULATOR IS THE STORAGE FORMAT here, unlike every "
    "other Bezier claim: the oracle is NumPy rather than Numba and accumulates "
    "directly into the output array, so there is no narrowing store and Rule 10's "
    "storage roundings are charged at zero rather than at two"
)

_COMPOSITION_WHY: Final = (
    "the composition is a fixed sequence of Bernstein products, binomial scalings and "
    "accumulations, and both backends run it in the same order: the powers of the "
    "inner map built by repeated multiplication by g rather than by squaring, the "
    "bases assembled as B_0 then the scaled interior then B_m, the tensor terms "
    "multiplied in ascending direction, and the result accumulated in row-major order "
    "over the outer control points. Each product is dispatched exactly as the oracle "
    "dispatches it -- the Numba scalar kernel for a univariate inner map, the NumPy "
    "n-dimensional helper above that -- and those two differ in accumulation width and "
    "in where they normalise, so the branch is part of this claim rather than an "
    "implementation detail. No fused multiply-add on this build"
)

_COMPOSITION_OPERANDS_NOT_OBSERVABLE: Final = (
    "this build can fuse, and every product's operands are formed inside the "
    "composition. The only amplification the public surface offers is the companion "
    "run on absolute values, which the harness refuses as vacuous against a net whose "
    "result cancels. test_bezier_arithmetic.py records the same conclusion for the "
    "same reason; porting the whole composition rather than only its kernel does not "
    "make the intermediates observable. The module docstring names what covers the "
    "arithmetic on such a build and what is left uncovered."
)


def _net(
    degrees: tuple[int, ...], rank: int, dtype: npt.DTypeLike, seed: int, *, rational: bool
) -> npt.NDArray[np.float32 | np.float64]:
    """Build a control net spanning many magnitudes, weights bounded away from zero.

    Three decades rather than the twelve `_mixed_control_points` defaults to, because
    a product multiplies its operands: two entries of 1e6 make 1e12, and a
    composition raises one operand to the outer degree, so the wide default overflows
    ``float32`` well before any port is at fault.

    Args:
        degrees (tuple[int, ...]): Degree per direction.
        rank (int): Number of value components, weight excluded.
        dtype (npt.DTypeLike): Storage format.
        seed (int): Generator seed.
        rational (bool): Whether to append a weight column.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The control net.
    """
    rng = np.random.default_rng(seed)
    components = rank + 1 if rational else rank
    shape = (*(degree + 1 for degree in degrees), components)
    net = rng.standard_normal(shape) * 10.0 ** rng.integers(-1, 2, shape)
    net = np.ascontiguousarray(net, dtype=dtype)
    if rational:
        net[..., -1] = np.asarray(rng.uniform(0.5, 2.0, net.shape[:-1]), dtype=dtype)
    return net


def _reparametrization(
    degrees: tuple[int, ...], rank: int, dtype: npt.DTypeLike, seed: int
) -> npt.NDArray[np.float32 | np.float64]:
    """Build an inner map whose coefficients live in the unit interval.

    A genuine reparametrization, and the only kind whose powers stay bounded: the
    composition raises the inner map to the outer degree, so a coefficient of 10
    composed with a degree-8 outer map is 1e8 before anything is at fault.

    The first two coefficients are pinned to 0.9 and 0.1, neither representable in
    ``float32``. `shape.hpp` records what a power-of-two fraction hides, and the same
    trap applies to a value that reaches the port through a table or a subtraction.

    Args:
        degrees (tuple[int, ...]): Degree per direction.
        rank (int): Number of components, which must be the outer map's dimension.
        dtype (npt.DTypeLike): Storage format.
        seed (int): Generator seed.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The control net.
    """
    rng = np.random.default_rng(seed)
    shape = (*(degree + 1 for degree in degrees), rank)
    net = np.ascontiguousarray(rng.uniform(0.0, 1.0, shape), dtype=dtype)
    # Assigned as Python floats and narrowed by the array's own dtype, which is one
    # rounding and the same one a coefficient read from a file would carry.
    flat = net.reshape(-1)
    flat[0] = 0.9
    if flat.size > 1:
        flat[1] = 0.1
    return net


def _companion(values: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """Floor an amplification so a tolerance is never identically zero.

    Args:
        values (npt.NDArray[Any]): The reachable magnitudes.

    Returns:
        npt.NDArray[np.float64]: The same, floored at the smallest normal double.
    """
    return np.asarray(np.maximum(np.abs(np.asarray(values, dtype=np.float64)), _TINY))


def _storage_width_claim(
    why: str,
    fused_why: str,
    *,
    stages: int,
    amplification: npt.NDArray[np.float64],
    dtype: npt.DTypeLike,
) -> ParityClaim:
    """Bitwise where the build cannot fuse, Rule 10's budget where it can.

    The same shape as `test_bezier_shape.py`'s ``_inherited`` and
    `test_bezier_arithmetic.py`'s ``_parity_claim``, with one deliberate difference:
    the accumulator is the **storage** format, because the oracle here is NumPy and
    accumulates into the output array. That makes the harness charge Rule 10's
    storage roundings at zero, which is correct rather than convenient -- there is no
    narrowing store for two nearby values to straddle.

    Args:
        why (str): The derivation for the bitwise arm.
        fused_why (str): The derivation for the fused arm, prefixed with the shared
            contraction argument.
        stages (int): Length of the dependency chain the fused sites sit on.
        amplification (npt.NDArray[np.float64]): Elementwise magnitude.
        dtype (npt.DTypeLike): Storage format, and the accumulator too.

    Returns:
        ParityClaim: BITWISE or BOUNDED, whichever this build supports.
    """
    if not contraction_may_fuse():
        return bitwise_parity(why=why)
    return bounded_parity(
        roundings=Roundings(
            stages=stages,
            accumulator_per_stage=_ACCUMULATOR_ROUNDINGS_PER_STAGE,
            storage_per_stage=_STORAGE_ROUNDINGS_PER_STAGE,
        ),
        accumulator=dtype,
        storage=dtype,
        amplification=amplification,
        why=f"{_FUSED_PREFIX}{fused_why}",
    )


def _product_stages(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    """The most terms that reach one coefficient of the product.

    ``prod_d (min(p_d, q_d) + 1)``: coefficient ``gamma`` accumulates one term per
    multi-index ``alpha`` with ``alpha <= gamma <= alpha + q``, and the widest such
    range in direction ``d`` has ``min(p_d, q_d) + 1`` entries. The univariate case
    is the same formula at one direction, which is the stage count
    `test_bezier_arithmetic.py` uses for the kernel.

    Args:
        left (tuple[int, ...]): Degrees of the first operand.
        right (tuple[int, ...]): Degrees of the second.

    Returns:
        int: The stage count, at least 1.
    """
    stages = 1
    for p_d, q_d in zip(left, right, strict=True):
        stages *= min(p_d, q_d) + 1
    return stages


def _product_amplification(
    left: npt.NDArray[Any], right: npt.NDArray[Any], *, rational: bool
) -> npt.NDArray[np.float64]:
    """The exact convex sum of ``|f_alpha g_beta|`` reaching each output coefficient.

    Run the same product on the absolute values of both nets, in ``float64`` and on
    the Python backend. The Bernstein product's weights are non-negative and sum to
    one over the terms of a coefficient, so the absolute-value companion is that
    convex sum term for term rather than a bound on it. This is the licence
    `test_bezier_arithmetic.py`'s ``_absolute_companion`` states, applied to an
    operation of two arguments.

    Args:
        left (npt.NDArray[Any]): First control net.
        right (npt.NDArray[Any]): Second control net.
        rational (bool): Whether both nets carry a weight column.

    Returns:
        npt.NDArray[np.float64]: Elementwise magnitude, never exactly zero.
    """
    with use_backend(Backend.PYTHON):
        companion = Bezier(
            np.abs(np.asarray(left, dtype=np.float64)), is_rational=rational
        ) * Bezier(np.abs(np.asarray(right, dtype=np.float64)), is_rational=rational)
    return _companion(companion.control_points)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rational", [False, True])
@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize(("left_degrees", "right_degrees"), DEGREE_PAIRS)
def test_the_product_agrees_with_the_oracle(  # noqa: PLR0913 -- four parametrization axes
    cpp_backend: None,
    left_degrees: tuple[int, ...],
    right_degrees: tuple[int, ...],
    rank: int,
    rational: bool,
    dtype: npt.DTypeLike,
) -> None:
    """The pointwise product agrees bit for bit, reciprocal binomials included.

    Driven through :meth:`Bezier.multiply`, which is the entry point the port
    replaced. That is worth saying because the opposite mistake is on record here:
    `test_bezier_arithmetic.py`'s ``test_compose_is_bitwise`` had to be driven through
    ``compose`` precisely because ``multiply`` reached no ported code at the time.
    It does now, and ``test_the_product_reaches_the_port`` pins that structurally so
    a future refactor cannot quietly undo it.

    The rational arm is a separate computation rather than a variation: the operands
    are promoted to homogeneous form and their numerators and weight columns are
    multiplied independently, so it exercises the product twice at two different
    component counts and then a concatenation.
    """
    del cpp_backend

    left = _net(left_degrees, rank, dtype, seed=20260831, rational=rational)
    right = _net(right_degrees, rank, dtype, seed=20260832, rational=rational)

    with use_backend(Backend.PYTHON):
        reference = np.asarray(
            (
                Bezier(left, is_rational=rational) * Bezier(right, is_rational=rational)
            ).control_points
        )
    with use_backend(Backend.CPP):
        actual = np.asarray(
            (
                Bezier(left, is_rational=rational) * Bezier(right, is_rational=rational)
            ).control_points
        )

    assert_parity(
        actual,
        reference,
        _storage_width_claim(
            _PRODUCT_WHY,
            _PRODUCT_FUSED_WHY,
            stages=_product_stages(left_degrees, right_degrees),
            amplification=_product_amplification(left, right, rational=rational),
            dtype=dtype,
        ),
        context=(
            f"multiply {left_degrees} by {right_degrees} rank {rank} "
            f"rational {rational} {np.dtype(dtype).name}"
        ),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_product_stays_exact_where_the_kernel_leaves_its_envelope(
    cpp_backend: None, dtype: npt.DTypeLike
) -> None:
    """Two degree-40 curves multiply to degree 80, where the 1-D kernel is undefined.

    ``AC4`` of FELIGN/pantr#392. This is the configuration the two one-dimensional
    Bernstein products do not share: the NumPy helper reaches ``C(80, 40)`` through
    :func:`math.comb`, which is arbitrary precision, while
    ``scalar_bernstein_product_1d`` computes its binomials in ``int64`` and is
    undefined past ``C(61, k)``. The assertion on ``_BINCOEFF_MAX_N`` is what keeps
    this test's premise honest: raise that constant above 80 and the test stops being
    about the divergence and should be retargeted rather than left passing.

    So the port cannot assemble its own tables here, and would silently agree
    everywhere else if it did.
    """
    del cpp_backend

    left_degree = 40
    right_degree = 40
    assert left_degree + right_degree > _BINCOEFF_MAX_N, (
        f"this test exists because the summed degree {left_degree + right_degree} is "
        f"outside the exact-integer binomial envelope {_BINCOEFF_MAX_N}; inside it, "
        f"the two 1-D products no longer differ in domain and the test measures "
        f"nothing it claims to"
    )

    left = _net((left_degree,), 1, dtype, seed=20260833, rational=False)
    right = _net((right_degree,), 1, dtype, seed=20260834, rational=False)

    with use_backend(Backend.PYTHON):
        reference = np.asarray((Bezier(left) * Bezier(right)).control_points)
    with use_backend(Backend.CPP):
        actual = np.asarray((Bezier(left) * Bezier(right)).control_points)

    assert actual.shape == (left_degree + right_degree + 1, 1)
    assert_parity(
        actual,
        reference,
        _storage_width_claim(
            _PRODUCT_WHY,
            _PRODUCT_FUSED_WHY,
            stages=_product_stages((left_degree,), (right_degree,)),
            amplification=_product_amplification(left, right, rational=False),
            dtype=dtype,
        ),
        context=f"multiply degree 40 by 40 {np.dtype(dtype).name}",
    )


_COMPOSITIONS: Final = (
    ((3,), (4,), 1),
    ((8,), (2,), 1),
    ((1,), (7,), 1),
    ((0,), (5,), 1),
    ((2, 3), (2,), 2),
    ((1, 1), (4,), 2),
    ((2,), (2, 1), 1),
    ((3,), (1, 2), 1),
    ((1, 2), (1, 1), 2),
    ((2, 1), (2, 2), 2),
)
"""``(outer degrees, inner degrees, outer rank)`` configurations.

The inner map's rank is the outer map's dimension, which the composition requires,
so it is not a free parameter. The first six have a **univariate** inner map and so
route every product through the Numba scalar kernel; the last four have a
two-dimensional one and route through the NumPy n-dimensional helper, which no
parity test reached before this file. ``((0,), (5,), 1)`` and ``((1,), (7,), 1)``
are the two configurations the oracle exempts from its binomial-envelope check,
because neither forms a product at all.
"""


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("outer_degrees", "inner_degrees", "rank"), _COMPOSITIONS)
def test_the_composition_agrees_with_the_oracle(
    cpp_backend: None,
    outer_degrees: tuple[int, ...],
    inner_degrees: tuple[int, ...],
    rank: int,
    dtype: npt.DTypeLike,
) -> None:
    """The composition agrees bit for bit, across both of its product branches.

    `test_bezier_arithmetic.py`'s ``test_compose_is_bitwise`` reaches the univariate
    branch through the same public method and is not duplicated here: what this adds
    is the **n-dimensional** branch, which routes through
    ``_bernstein_product_coefficients_nd`` and which no parity test reached before,
    plus the degree-0 and degree-1 outer maps the oracle exempts from its envelope
    check.

    Gated at ``float32`` only where the oracle reaches Numba, which is the univariate
    branch. Above one dimension the oracle is NumPy end to end and interpretation
    changes nothing, so gating there would skip a claim that holds.
    """
    del cpp_backend
    if len(inner_degrees) == 1:
        demand_the_compiled_kernel(dtype)
    demand_a_bound_the_claim_can_carry(_COMPOSITION_OPERANDS_NOT_OBSERVABLE)

    outer_net = _net(outer_degrees, rank, dtype, seed=20260835, rational=False)
    inner_net = _reparametrization(inner_degrees, len(outer_degrees), dtype, seed=20260836)

    with use_backend(Backend.PYTHON):
        reference = np.asarray(Bezier(outer_net).compose(Bezier(inner_net)).control_points)
    with use_backend(Backend.CPP):
        actual = np.asarray(Bezier(outer_net).compose(Bezier(inner_net)).control_points)

    expected_degree = tuple(sum(outer_degrees) * n for n in inner_degrees)
    assert actual.shape == (*(n + 1 for n in expected_degree), rank)

    assert_parity(
        actual,
        reference,
        bitwise_parity(why=_COMPOSITION_WHY),
        context=(
            f"compose {outer_degrees} with {inner_degrees} rank {rank} {np.dtype(dtype).name}"
        ),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_product_reaches_the_port(cpp_backend: None, dtype: npt.DTypeLike) -> None:
    """The C++ backend's product returns a C++ value, and the catalogue routes to it.

    ``AC2`` asks the test to demonstrate that it reaches the ported code, and this is
    the structural half of that demonstration: the mutation record in the module
    docstring is the empirical half. Both are needed. A mutation proves the code ran
    for the tree it was applied to; this assertion is what fails on a later refactor
    that reroutes :meth:`Bezier.multiply` back to the NumPy helper without touching
    the numbers, which is precisely the failure that went unnoticed here once.

    The composition is asserted alongside it because its adapter is the same shape
    and would fail the same way.
    """
    del cpp_backend

    from pantr import _pantr_cpp  # noqa: PLC0415
    from pantr.bezier import _bezier_backend as backend  # noqa: PLC0415
    from pantr.bezier._bezier_compose import _compose_python  # noqa: PLC0415
    from pantr.bezier._bezier_product import _multiply_python  # noqa: PLC0415

    assert backend.multiply_kernel(Backend.CPP) is backend._cpp_multiply
    assert backend.multiply_kernel(Backend.PYTHON) is _multiply_python
    assert backend.compose_kernel(Backend.CPP) is backend._cpp_compose
    assert backend.compose_kernel(Backend.PYTHON) is _compose_python

    handles = (_pantr_cpp.Bezier32, _pantr_cpp.Bezier64)
    curve = _net((2,), 1, dtype, seed=20260837, rational=False)
    reparametrization = _reparametrization((2,), 1, dtype, seed=20260838)
    with use_backend(Backend.CPP):
        product = Bezier(curve) * Bezier(curve)
        composition = Bezier(curve).compose(Bezier(reparametrization))
        assert isinstance(product._impl, handles)
        assert isinstance(composition._impl, handles)


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_two_spellings_of_a_binomial_agree(dtype: npt.DTypeLike) -> None:
    """The table's entries equal the composition oracle's ``float(math.comb(...))``.

    A hypothesis of ``_binomial_tables``' docstring, and one this code has to satisfy
    rather than one the theorem behind it can be trusted to supply. The oracle reaches
    a binomial two ways: ``np.array([math.comb(n, k) ...], dtype=dtype)`` in the two
    product helpers, and ``float(math.comb(m, i))`` multiplied into an array in
    ``_compute_bernstein_bases``. The first rounds the exact integer to the storage
    format once; the second rounds it to ``float64`` and then, under NEP 50's weak
    scalar rule, to the storage format again.

    Double rounding through an intermediate of at least ``2 * 24 + 2 = 50`` bits
    equals single rounding, and ``float64`` has 53, so the two agree -- but that is a
    theorem about real numbers rounded twice, and what it is applied to here is a
    particular expression in a particular language. This checks the application, over
    the whole range these operations reach and slightly past it.

    Its reciprocal twin needs no such check: both spellings divide in ``float64``
    before any narrowing, so they are the same expression.
    """
    from pantr.bezier._bezier_product import _binomial_tables  # noqa: PLC0415

    order = 96
    binomials, _ = _binomial_tables(order, dtype)
    for n in range(order + 1):
        direct = binomials[n, : n + 1]
        # `float()` first, so the exact integer is rounded to float64 and then to the
        # storage format: the composition oracle's two-step route, spelled out.
        via_double = np.asarray([float(math.comb(n, k)) for k in range(n + 1)], dtype=dtype)
        assert np.array_equal(direct, via_double), (
            f"the two spellings of C({n}, k) differ at {np.dtype(dtype).name}; the "
            f"composition oracle reaches its binomials through float(math.comb(...)) "
            f"and the product helpers through np.array(..., dtype=dtype), and "
            f"_binomial_tables serves both"
        )


@pytest.mark.parametrize("dtype", DTYPES)
def test_both_backends_refuse_the_same_operands(cpp_backend: None, dtype: npt.DTypeLike) -> None:
    """A rejected argument gets one message, whichever backend is active.

    The checks live in Layer 2, above the backend branch, so this is what pins that
    they were not duplicated into the adapter with different wording. The C++ header
    carries its own copies for a caller with no Python, and those are compared
    against the same strings by ``cpp/tests/test_bezier_product.cpp``.
    """
    del cpp_backend

    curve = Bezier(_net((2,), 1, dtype, seed=20260839, rational=False))
    surface = Bezier(_net((2, 2), 1, dtype, seed=20260840, rational=False))
    vector = Bezier(_net((2,), 3, dtype, seed=20260841, rational=False))
    weighted = Bezier(_net((2,), 1, dtype, seed=20260842, rational=True), is_rational=True)

    cases: tuple[tuple[Any, Any, type[Exception], str], ...] = (
        (curve, surface, ValueError, "Operands must have the same dimension."),
        (curve, vector, ValueError, "Operands must have the same rank."),
    )
    for left, right, error, fragment in cases:
        messages = []
        for which in (Backend.PYTHON, Backend.CPP):
            with use_backend(which), pytest.raises(error) as info:
                left.multiply(right)
            messages.append(str(info.value))
        assert fragment in messages[0]
        assert messages[0] == messages[1], (
            f"the two backends worded the same rejection differently: "
            f"{messages[0]!r} against {messages[1]!r}"
        )

    for which in (Backend.PYTHON, Backend.CPP):
        with use_backend(which), pytest.raises(TypeError, match="rational"):
            curve.compose(weighted)


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_binding_refuses_a_table_that_is_too_small(
    cpp_backend: None, dtype: npt.DTypeLike
) -> None:
    """A short table is refused rather than read past, and the tables cannot be swapped.

    Neither is reachable through :class:`Bezier`, which sizes the tables from the
    binding's own accessor and passes them by keyword. Both are reachable by a caller
    that uses the extension directly, which ``cpp/bindings/bezier.cpp`` is explicit
    about protecting: the two tables have the same dtype and the same shape, so
    nothing but keyword-only arguments separates them, and transposing them returns a
    plausible Bézier rather than an error.
    """
    del cpp_backend

    from pantr import _pantr_cpp  # noqa: PLC0415

    curve = _net((3,), 1, dtype, seed=20260843, rational=False)
    # The class is chosen from the array's own dtype one expression earlier, which is a
    # correlation between a value and a type the checker cannot state; `test_bezier_shape.py`
    # stands in for it the same way.
    handle_type = _pantr_cpp.Bezier32 if np.dtype(dtype) == np.float32 else _pantr_cpp.Bezier64
    handle = handle_type(cast("Any", curve))

    order = _pantr_cpp.bezier_product_table_order(handle, handle)
    assert order == 6, "two degree-3 curves multiply to degree 6"

    from pantr.bezier._bezier_product import _binomial_tables  # noqa: PLC0415

    binomials, inverse_binomials = _binomial_tables(order, dtype)
    short, short_inverse = _binomial_tables(order - 1, dtype)

    with pytest.raises(ValueError, match="binomials must have shape at least"):
        _pantr_cpp.multiply_bezier(
            handle, handle, binomials=short, inverse_binomials=inverse_binomials
        )
    with pytest.raises(ValueError, match="inverse_binomials must have shape at least"):
        _pantr_cpp.multiply_bezier(
            handle, handle, binomials=binomials, inverse_binomials=short_inverse
        )
    with pytest.raises(TypeError):
        _pantr_cpp.multiply_bezier(handle, handle, binomials, inverse_binomials)  # type: ignore[misc]
    assert short_inverse.shape == (order, order)


@pytest.mark.slow
@pytest.mark.parametrize("dtype", DTYPES)
def test_the_product_agrees_over_a_random_sweep(cpp_backend: None, dtype: npt.DTypeLike) -> None:
    """Sweep the product over many nets, so one seed's luck cannot carry the claim.

    Twenty configurations per dtype, each with its own net, drawn over dimensions one
    to three and over both rationalities. The per-case parametrization above fixes
    its seeds so a failure is reproducible; this exists because a fixed seed is
    exactly what a port can accidentally be tuned to.
    """
    del cpp_backend

    rng = np.random.default_rng(20260844)
    for trial in range(20):
        dim = int(rng.integers(1, 4))
        left_degrees = tuple(int(rng.integers(0, 5)) for _ in range(dim))
        right_degrees = tuple(int(rng.integers(0, 5)) for _ in range(dim))
        rank = int(rng.integers(1, 4))
        rational = bool(rng.integers(0, 2))

        left = _net(left_degrees, rank, dtype, seed=90000 + trial, rational=rational)
        right = _net(right_degrees, rank, dtype, seed=95000 + trial, rational=rational)

        with use_backend(Backend.PYTHON):
            reference = np.asarray(
                (
                    Bezier(left, is_rational=rational) * Bezier(right, is_rational=rational)
                ).control_points
            )
        with use_backend(Backend.CPP):
            actual = np.asarray(
                (
                    Bezier(left, is_rational=rational) * Bezier(right, is_rational=rational)
                ).control_points
            )

        assert_parity(
            actual,
            reference,
            _storage_width_claim(
                _PRODUCT_WHY,
                _PRODUCT_FUSED_WHY,
                stages=_product_stages(left_degrees, right_degrees),
                amplification=_product_amplification(left, right, rational=rational),
                dtype=dtype,
            ),
            context=(
                f"sweep trial {trial}: multiply {left_degrees} by {right_degrees} "
                f"rank {rank} rational {rational} {np.dtype(dtype).name}"
            ),
        )
