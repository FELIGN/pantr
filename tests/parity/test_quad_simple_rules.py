r"""Parity of the two closed-form quadrature rules against their Python oracle.

`cpp/include/pantr/quad/simple_rules.hpp` names this file as the place its parity
claims are derived. The trapezoidal rule and the modified Chebyshev interpolation
nodes are grouped because they share a header and a shape -- both are given by a
closed form with no iteration -- and because the interesting fact about them is
the contrast: one is bit-identical in both storage formats and the other is not,
for a reason that is visible in the source.

Where the difference comes from
-------------------------------

**Trapezoidal is bitwise in both formats.** `numpy.linspace` forms
``arange(n) * step + start`` in float64 and narrows once at the end, and it
*assigns* the last element rather than computing it. Every operation is `+`, `-`,
`*` or `/`, each of which IEEE 754 pins to one correctly rounded result, so two
implementations evaluating the same expression in the same order agree to the last
bit. Nothing here is a library call.

**Modified Chebyshev is bitwise in float64 and bounded in float32**, and that
asymmetry is the whole content of the comparison. The kernel builds its index
array in the *storage* format, so under NEP 50 the multiply by `pi`, the divide,
the cosine and both halvings all happen in that format. In float64 the two
libraries' `cos` agree on every argument this rule forms. In float32 they do not.

The float32 bound, derived
--------------------------

Write `theta = pi i / (n - 1)` and `node = 0.5 - 0.5 c` with `c = cos(theta)`, all
in binary32, `u = eps32 / 2 = 5.96e-08`.

1. **`theta` is common mode.** It is formed by a multiply and a divide on values
   both sides compute identically, so it is bit-identical. Measured: 0 of 2130
   arguments differ, over n from 2 to 1000. Nothing upstream of the cosine
   contributes to the bound.
2. **The cosine is the only source.** The two libraries differ by at most
   ``k_libm`` units in the last place of the result. Measured on exactly the
   arguments this rule forms: 370 of 2130 differ, worst **1 ulp**. One ulp of `c`
   is `2 u |c| <= 2 u`, so `|dc| <= 2 k_libm u` with `k_libm = 1`.
3. **The halving is exact** -- multiplication by a power of two -- so `0.5 |dc|`
   passes through unrounded, contributing `k_libm u`.
4. **The subtraction adds at most one rounding of its own result**, `u |node|`.

So `|d node| <= k_libm u + u |node| <= 2 u`, and with `|node| <= 1` the elementwise
bound is `u (k_libm + |node|)`. Doubling it for parity, since neither side is the
exact cosine, gives `2 u (k_libm + |node|)`.

**Measured worst deviation: 5.96e-08, which is exactly `1 u`.** The bound at that
element is about `2 u`, so it is reached to within a factor of two -- an honest gap
between a bound and an observation, and not one to close by quoting the
observation.

**`k_libm` is a claim about two libraries on one platform, not a theorem.** It is
therefore measured by a test in this file rather than asserted here, and the
measurement is what the bound is built on.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Final, TypeVar

import numpy as np
import numpy.typing as npt
import pytest

from pantr._backend import Backend, use_backend
from pantr.quad import get_modified_chebyshev_nodes_1d, get_trapezoidal_1d
from tests._parity_harness import (
    FloatArray,
    ParityClaim,
    Roundings,
    assert_parity,
    bitwise_parity,
    bounded_parity,
)

DTYPES: Final = (np.float64, np.float32)
"""The two storage formats the quadrature layer accepts."""

N_PTS: Final = (1, 2, 3, 4, 5, 17, 64, 200, 1000)
"""Point counts swept by the parity tests.

The legal corners of ``tools/adversarial_sweep/_probes_quad.py``'s own
``_N_PTS_FULL``, deliberately the same list so the two cannot drift: 1 and 2 are
the special cases each rule branches on, and 1000 is large enough to stress the
cosine's argument reduction. The sweep's two illegal corners, 0 and -1, are not
here because a rejected call has no result to compare -- that both backends reject
them identically is the validation layer's business and is checked in
``tests/test_quad.py``, which now runs under both.
"""

CHEBYSHEV_N_PTS: Final = tuple(n for n in N_PTS if n >= 2)
"""The modified Chebyshev nodes need at least two points; ``n - 1`` is a divisor."""

_LIBM_COSINE_ULP_BUDGET: Final = 1
"""How far the two libraries' ``cos`` may differ, in units in the last place.

Measured on exactly the arguments this rule forms, not quoted from either
library's documentation: over n from 2 to 1000, 370 of 2130 float32 arguments give
a different result and the worst difference is 1 ulp.
:func:`test_the_cosine_libraries_differ_by_no_more_than_the_budget` re-measures it,
so a platform where it is false fails here rather than inside a bound.
"""


_Result = TypeVar("_Result")


def _both_backends(build: Callable[[], _Result]) -> tuple[_Result, _Result]:
    """Run a rule builder once under each backend.

    Args:
        build (Callable[[], _Result]): A zero-argument callable returning the rule.

    Returns:
        tuple[_Result, _Result]: The Python result and the C++ result, in that
            order, so the reference is always first.
    """
    with use_backend(Backend.PYTHON):
        python = build()
    with use_backend(Backend.CPP):
        cpp = build()
    return python, cpp


_EXACT_BY_ARITHMETIC: Final[str] = (
    "every operation in the trapezoidal rule is +, -, * or / on values both sides "
    "form in the same order, and IEEE 754 pins each of those to one correctly "
    "rounded result. numpy.linspace computes arange(n) * step + start in float64 "
    "and ASSIGNS the last element rather than computing it, which the C++ "
    "reproduces; the narrowing to the storage format is a single rounding of the "
    "same value on both sides. There is no library call anywhere in the rule, so "
    "there is nothing left that could differ."
)

_EXACT_BY_MEASURED_LIBM: Final[str] = (
    "the modified Chebyshev nodes are a closed form whose only non-IEEE operation "
    "is cos. In float64 the two libraries were measured to agree on every argument "
    "this rule forms, so the whole expression is bit-identical. That is a property "
    "of a platform rather than of the source, which is why "
    "test_the_cosine_libraries_differ_by_no_more_than_the_budget measures it."
)


def _chebyshev_claim(nodes: FloatArray, dtype: npt.DTypeLike) -> ParityClaim:
    """State the parity claim for the modified Chebyshev nodes at one storage format.

    Args:
        nodes (FloatArray): The reference nodes, used for their magnitudes.
        dtype (npt.DTypeLike): The storage format the rule was built in.

    Returns:
        ParityClaim: BITWISE in float64, BOUNDED in float32.
    """
    if np.dtype(dtype) == np.float64:
        return bitwise_parity(why=_EXACT_BY_MEASURED_LIBM)

    # One rounding, amplified elementwise. `amplification` carries the whole
    # dimensionless factor the perturbation is multiplied by before it reaches
    # the result, which here is the libm budget plus the subtraction's own
    # rounding of a result of magnitude |node|. The array is in the frame the
    # comparison happens in -- absolute, on a quantity in [0, 1] -- because the
    # nodes approach zero at one end while the error there does not: it is
    # inherited from rounding a quantity of magnitude one half.
    magnitude = np.abs(np.asarray(nodes, dtype=np.float64))
    amplification = _LIBM_COSINE_ULP_BUDGET + magnitude
    return bounded_parity(
        roundings=Roundings(stages=1, accumulator_per_stage=1, storage_per_stage=0),
        accumulator=dtype,
        storage=dtype,
        amplification=amplification,
        why=(
            "theta is bit-identical (measured: 0 of 2130 arguments differ), the "
            "halving by a power of two is exact, and the only source is cos, "
            f"measured to differ by at most {_LIBM_COSINE_ULP_BUDGET} ulp of its own "
            "result. One ulp of c is 2u|c| <= 2u, the exact halving passes 0.5|dc| "
            "through unrounded, and the subtraction adds u|node|. So one rounding "
            "amplified by (libm budget + |node|), doubled by the harness because "
            "neither side is the exact cosine."
        ),
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_pts", N_PTS)
def test_trapezoidal_is_bitwise(n_pts: int, dtype: npt.DTypeLike, cpp_backend: None) -> None:
    """The trapezoidal rule agrees to the last bit, in both storage formats.

    Args:
        n_pts (int): Number of points.
        dtype (npt.DTypeLike): Storage format.
        cpp_backend (None): Requires the compiled extension.
    """
    (py_nodes, py_weights), (cpp_nodes, cpp_weights) = _both_backends(
        functools.partial(get_trapezoidal_1d, n_pts, dtype)
    )
    claim = bitwise_parity(why=_EXACT_BY_ARITHMETIC)
    assert_parity(cpp_nodes, py_nodes, claim, context=f"trapezoidal nodes, n={n_pts}, {dtype}")
    assert_parity(
        cpp_weights, py_weights, claim, context=f"trapezoidal weights, n={n_pts}, {dtype}"
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_pts", CHEBYSHEV_N_PTS)
def test_modified_chebyshev_nodes_agree(
    n_pts: int, dtype: npt.DTypeLike, cpp_backend: None
) -> None:
    """The modified Chebyshev nodes agree, bitwise in float64 and within a bound in float32.

    Args:
        n_pts (int): Number of nodes.
        dtype (npt.DTypeLike): Storage format, which is also the arithmetic format.
        cpp_backend (None): Requires the compiled extension.
    """
    python, cpp = _both_backends(functools.partial(get_modified_chebyshev_nodes_1d, n_pts, dtype))
    claim = _chebyshev_claim(python, dtype)
    assert_parity(cpp, python, claim, context=f"modified Chebyshev nodes, n={n_pts}, {dtype}")


@pytest.mark.parametrize("n_pts", CHEBYSHEV_N_PTS)
def test_the_chebyshev_endpoints_are_exact_in_both_backends(n_pts: int, cpp_backend: None) -> None:
    """Both endpoints land on 0 and 1 exactly, which no tolerance would have checked.

    The rule's docstring promises nodes "starting at 0 and ending at 1". A bound
    of a couple of units of roundoff would accept an endpoint one ulp inside the
    interval, which is a different rule -- and for an interpolation node set, the
    endpoint being exact is the property callers depend on.

    Args:
        n_pts (int): Number of nodes.
        cpp_backend (None): Requires the compiled extension.
    """
    for dtype in DTYPES:
        _, cpp = _both_backends(functools.partial(get_modified_chebyshev_nodes_1d, n_pts, dtype))
        nodes = np.asarray(cpp)
        assert nodes[0] == 0.0, f"the first node is {nodes[0]!r} rather than exactly 0 in {dtype}"
        assert nodes[-1] == 1.0, f"the last node is {nodes[-1]!r} rather than exactly 1 in {dtype}"


def test_the_cosine_libraries_differ_by_no_more_than_the_budget(cpp_backend: None) -> None:
    """Re-measure the one input the float32 Chebyshev bound rests on.

    The bound is built from a claim about two implementations of ``cos`` on this
    platform, and a claim about a platform is not a theorem. This test recovers
    the difference from the rule's own output rather than calling either library
    directly, which is what makes it a check on the quantity the bound actually
    uses.

    Inverting the closed form gives ``c = 1 - 2 node``, and the inversion is exact
    **only where Sterbenz's lemma applies**: ``0.5 c`` must lie in ``[0.25, 1]``,
    that is ``c >= 0.5``, that is ``node <= 0.25``. The measurement is restricted
    to that subset, which is about a third of the nodes at every count and plenty.

    **Off that subset the recovered number is meaningless, and the way it fails is
    worth keeping.** Near ``node = 0.5`` the cosine passes through zero, so
    ``1 - 2 node`` subtracts two quantities of magnitude one to leave a tiny one:
    a one-ulp difference in the node becomes sixteen ulp of the recovered ``c``
    while the node itself moved by half a unit of roundoff. That is
    ``design/backend_parity.md`` Rule 2 -- a relative measure on a quantity that
    vanishes -- appearing inside the test that applies it. Measured: 1 ulp on the
    Sterbenz subset against 16 ulp off it, and the direct C++ measurement agrees
    with the first.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    worst_ulp = 0
    samples = 0
    for n_pts in CHEBYSHEV_N_PTS:
        python, cpp = _both_backends(
            functools.partial(get_modified_chebyshev_nodes_1d, n_pts, np.float32)
        )
        reference = np.asarray(python)
        invertible = reference <= np.float32(0.25)
        if not invertible.any():
            continue
        py_cos = (np.float32(1.0) - np.float32(2.0) * reference[invertible]).astype(np.float32)
        cpp_cos = (np.float32(1.0) - np.float32(2.0) * np.asarray(cpp)[invertible]).astype(
            np.float32
        )
        gap = np.abs(
            py_cos.view(np.int32).astype(np.int64) - cpp_cos.view(np.int32).astype(np.int64)
        )
        worst_ulp = max(worst_ulp, int(gap.max()))
        samples += int(invertible.sum())

    assert samples > 100, (
        f"only {samples} nodes landed in the range where the inversion is exact, "
        f"which is too few to call this a measurement of anything"
    )
    assert worst_ulp <= _LIBM_COSINE_ULP_BUDGET, (
        f"the two cos implementations differ by {worst_ulp} ulp on this platform, "
        f"past the {_LIBM_COSINE_ULP_BUDGET} the float32 Chebyshev bound is derived "
        f"from. The bound is not wrong here, it is unsupported: re-derive it from "
        f"the measured budget rather than widening it until the suite is green."
    )


def test_the_float32_chebyshev_bound_is_reached_rather_than_idle(cpp_backend: None) -> None:
    """The float32 bound is approached, so it is asserting something.

    A tolerance never approached asserts nothing, and the harness has no way to
    tell an over-wide bound from a tight one. The measured worst deviation is one
    unit of roundoff against a bound of about two, so the ratio should sit near a
    half and certainly not near zero.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    worst_ratio = 0.0
    for n_pts in CHEBYSHEV_N_PTS:
        python, cpp = _both_backends(
            functools.partial(get_modified_chebyshev_nodes_1d, n_pts, np.float32)
        )
        deviation = assert_parity(
            cpp,
            python,
            _chebyshev_claim(python, np.float32),
            context=f"modified Chebyshev nodes, n={n_pts}, float32",
        )
        worst_ratio = max(worst_ratio, deviation.max_ratio_to_bound)

    assert worst_ratio > 0.1, (
        f"the float32 Chebyshev bound is never approached more closely than "
        f"{worst_ratio:.3g} of itself, so it is not distinguishing a correct port "
        f"from a wrong one. Either the two cos implementations now agree on this "
        f"platform -- in which case the claim should be BITWISE and say so -- or "
        f"the bound has been widened past what its derivation supports."
    )
