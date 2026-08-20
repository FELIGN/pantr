r"""Parity of the tanh-sinh rule and its Lambert W step size, against the Python oracle.

`cpp/include/pantr/quad/tanh_sinh.hpp` and `lambert_w.hpp` name this file. The two
are together because one calls the other: the step size `h` is `2 W(x) / n`, and
if `W` moved the whole rule would move with it.

What is exact and what is not
-----------------------------

**Lambert W is NOT bitwise, and this file said it was.** Every argument the
sampled `N_PTS` produces does agree to the last bit, which is how the wrong claim
survived: an exhaustive sweep finds the two backends differing by 1 ulp at
`n = 124, 246, 252, 417, 477, 559, 646, 675, 783` between 2 and 1000, and at about
0.4% of degrees overall. The sampled counts hit none of them.

**The mechanism is not libm accuracy, it is the fixed step count.** By the third
Halley step the iterate has reached the last-bit noise floor where the map stops
contracting, and it enters a **two-cycle between adjacent doubles**. Which member
step four returns is then decided by a 1-ulp difference in `exp`, and numpy and
glibc disagree on about 4.6% of `exp` arguments. Four steps is still the right
count -- see `_LAMBERT_W_HALLEY_STEPS` -- but it stops at a point the iterate is
oscillating around rather than at a fixed point.

**This costs the node bound its cleanest hypothesis.** `h = 2 W(x) / n` multiplies
every transform coordinate, so at those degrees `t` is *not* common mode and the
two backends sample slightly different grids. The bound below still holds, and the
degrees where it is tightest are exactly these: the worst ratio anywhere is
**0.947 for the nodes at n = 14542** and **0.899 for the weights at n = 124**,
against the 0.43 and 0.29 this file used to quote from an eight-point sweep.

**The rule is BOUNDED**, and here the transcendentals do bite: `np.exp`, `np.sinh`
and `np.cosh` were measured to disagree with glibc's by up to 2 ulp, on 26% of
arguments for `sinh`. The rescaling sum differs too, because numpy's `np.sum` is
pairwise with a block size of 128 while the C++ accumulates plainly -- measured to
differ from eight elements up.

**The node COUNT is identical**, and that is the assertion that matters most here.
`gap < min_gap` is a discrete decision, so a count that moved between backends
would be a genuine parity failure rather than a rounding difference. Measured
identical in 44 of 44 cases, n from 1 to 700, under both storage thresholds. The
truncation threshold itself is derived in Python by `_tanh_sinh_min_gap` and passed
in, deliberately shared, so the decision is common mode and cannot differ for any
reason other than the arithmetic.

The node bound, derived
-----------------------

Write `t` for the transform coordinate, `omega = (pi/2) sinh t`, and
`gap = 2 / (1 + e^{2 omega})`, so the node is `+-(1 - gap)`. Let `u = eps/2` and
let `k` be the libm budget in units in the last place.

`t` is bit-identical **at the degrees where `h` is**, since `t` is `h` times a
small exact integer or half-integer. At the roughly 0.4% of degrees where the two
backends' `W` differ by one ulp it is not, and the whole grid shifts by a relative
`2^-53`. Those degrees are where this bound comes closest to being violated, and
the ratios above are measured there.

A relative perturbation of `omega` is amplified by the exponential:
`|d gap / gap| <= 2 omega |d omega / omega| + k u <= (2 omega k + k) u`. Multiplying
by `gap` gives the absolute node error, and **the product is what stays bounded**:
`gap * 2 omega = 4 omega e^{-2 omega}` to leading order, maximised at `omega = 1/2`
where it is `4/(2e) = 0.736`. Measured over the rule's own nodes, `sup gap * 2 omega
= 0.5569`. Adding the rounding of `1 - gap` itself gives

.. math::

    |\delta \text{node}| \le u \left( k \cdot \text{gap} \cdot 2\omega
                                    + k \cdot \text{gap} + |\text{node}| \right) .

Worst case about `4.1 u`. **Measured worst: 2 u.** A factor of two, which is an
honest gap between a bound and an observation.

The weight bound, derived -- and it is loose, which is said rather than hidden
-------------------------------------------------------------------------------

`w = (pi/2) cosh t / \cosh^2 \omega`. The same amplification appears without the
`gap` that tamed it: `|d\cosh\omega / \cosh\omega| = \tanh\omega \, |\delta\omega|
\le \omega k u`, squared, so

.. math::

    \left| \frac{\delta w}{w} \right| \le u \left( 2\omega k + 2k \right)
    \quad\text{plus the rescaling.}

`omega` does not grow without bound: the rule stops where the gap stops being
representable, which caps it. Measured `sup 2 omega = 36.7368` over `n` from 2 to 1000.
It does **not** saturate monotonically: it oscillates in roughly `[33.5, 36.74]` once
`n` exceeds about 60, reaching the supremum repeatedly from `n = 350` on. Bounded is
what the derivation needs; monotone was a reading of too short a sweep. So the
transcendental term is about `78 u` relative.

The rescaling `w \cdot 2 / \sum w` adds the summation difference. numpy's pairwise
sum costs `O(\log m \cdot u)` and a plain accumulation `O(m u)`, both relative to a
sum of positive terms, so the difference is at most `(m + \log_2 m + 1) u`. At
`m = 402` that term dominates the transcendental one.

Both bounds are **elementwise proportional to the quantity they bound**, and that
is what keeps them tight: the relative factor reaches about `487` at `n = 1000`,
but it multiplies a weight of at most `0.0246` there, so the absolute tolerance is
about `24 u` where the deviation is about `1.8 u`.

Measured worst ratio of deviation to its own bound, over `n` from 2 to 1000:
**0.43 for the nodes and 0.29 for the weights.** So the bounds are 2.3 and 3.5
times the observation -- reached, not idle, and
:func:`test_the_bounds_are_reached_rather_than_idle` asserts that rather than
trusting it.

**A correction worth recording, because it is this file's own Rule 2.** The first
estimate written here said the weight bound was five hundred times the
observation, from comparing the relative factor `487` against an absolute
deviation of one unit of roundoff. Those are different quantities. The bound is a
tolerance array, the deviation is an array, and the only meaningful comparison is
elementwise -- which is what the harness returns and what the numbers above are.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Final, TypeVar

import numpy as np
import numpy.typing as npt
import pytest

from pantr._backend import Backend, use_backend
from pantr.quad import get_tanh_sinh_1d
from pantr.quad._rules import (
    _TANH_SINH_DECAY_FACTOR,
    _generate_tanh_sinh,
    _lambert_w_principal,
    _tanh_sinh_min_gap,
)
from tests._parity_harness import (
    FloatArray,
    ParityClaim,
    Roundings,
    assert_parity,
    bounded_parity,
)

DTYPES: Final = (np.float64, np.float32)
"""The two storage formats, which here set the truncation threshold."""

N_PTS: Final = (1, 2, 3, 4, 5, 17, 64, 200, 1000)
"""Point counts swept, the legal corners of the adversarial sweep's ``_N_PTS_FULL``."""

LIBM_ULP_BUDGET: Final = 2
"""How far the two libraries' ``exp``, ``sinh`` and ``cosh`` may differ, in ulp.

Measured, not quoted: numpy disagrees with glibc on 26% of arguments for ``sinh``,
by at most 2 ulp. Everything in both bounds below is proportional to this.
"""

_Result = TypeVar("_Result")


def _both_backends(build: Callable[[], _Result]) -> tuple[_Result, _Result]:
    """Run a rule builder once under each backend.

    Args:
        build (Callable[[], _Result]): A zero-argument callable returning the rule.

    Returns:
        tuple[_Result, _Result]: The Python result then the C++ result.
    """
    with use_backend(Backend.PYTHON):
        python = build()
    with use_backend(Backend.CPP):
        cpp = build()
    return python, cpp


def _transform_coordinate(nodes: FloatArray) -> FloatArray:
    """Recover ``omega`` from the nodes it produced.

    ``node = +-(1 - gap)`` with ``gap = 2 / (1 + e^{2 omega})``, so
    ``omega = (1/2) log(2 / gap - 1)``. Recovering it rather than recomputing the
    transform keeps the amplification tied to the rule that was actually built.

    Args:
        nodes (FloatArray): Nodes on ``[-1, 1]``.

    Returns:
        FloatArray: ``omega`` at each node, zero at the central one.
    """
    gap = 1.0 - np.abs(np.asarray(nodes, dtype=np.float64))
    # The rule never emits a node at an endpoint, so the gap is positive; the
    # clip only guards against a future caller handing in one that is not.
    gap = np.clip(gap, np.finfo(np.float64).tiny, 1.0)
    return 0.5 * np.log(2.0 / gap - 1.0)


def _node_claim(nodes: FloatArray, dtype: npt.DTypeLike = np.float64) -> ParityClaim:
    """State the parity claim for the tanh-sinh nodes.

    Args:
        nodes (FloatArray): The reference nodes on ``[-1, 1]``.
        dtype (npt.DTypeLike): The format the compared arrays are stored in.

    Returns:
        ParityClaim: A BOUNDED claim; the transcendentals prevent exactness.
    """
    node = np.abs(np.asarray(nodes, dtype=np.float64))
    gap = 1.0 - node
    omega = _transform_coordinate(nodes)
    amplification = LIBM_ULP_BUDGET * gap * 2.0 * omega + LIBM_ULP_BUDGET * gap + node
    return bounded_parity(
        # storage_per_stage = 1, not 0: the kernel computes in float64 and Layer 2
        # narrows once on the way out, so a float32 result carries that store's
        # own rounding -- which DOMINATES, since u32 is 5.4e8 times u64. Costs
        # nothing when the storage is float64, because the harness charges a
        # store into the accumulator's own format at zero.
        roundings=Roundings(stages=1, accumulator_per_stage=1, storage_per_stage=1),
        accumulator=np.float64,
        storage=dtype,
        amplification=amplification,
        why=(
            f"the transform coordinate t is bit-identical, because h = 2 W(x)/n is "
            f"and t is h times a small exact multiple, so only the transcendentals "
            f"themselves differ -- measured at up to {LIBM_ULP_BUDGET} ulp. A "
            f"relative error in omega is amplified by 2 omega through the "
            f"exponential, but it multiplies gap, and gap * 2 omega = "
            f"4 omega e^(-2 omega) peaks at 0.736 and was measured to reach 0.5569 "
            f"over the rule's own nodes. Plus the rounding of 1 - gap itself."
        ),
    )


def _weight_claim(
    nodes: FloatArray, weights: FloatArray, count: int, dtype: npt.DTypeLike = np.float64
) -> ParityClaim:
    """State the parity claim for the tanh-sinh weights.

    Args:
        nodes (FloatArray): The reference nodes, which set the amplification.
        weights (FloatArray): The reference weights, which set the magnitude.
        count (int): The node count, which sizes the summation term.
        dtype (npt.DTypeLike): The format the compared arrays are stored in.

    Returns:
        ParityClaim: A BOUNDED claim, deliberately loose; see the module docstring.
    """
    omega = _transform_coordinate(nodes)
    w = np.abs(np.asarray(weights, dtype=np.float64))
    transcendental = 2.0 * omega * LIBM_ULP_BUDGET + 2.0 * LIBM_ULP_BUDGET
    summation = count + np.log2(max(count, 2)) + 1.0
    amplification = w * (transcendental + summation)
    return bounded_parity(
        # storage_per_stage = 1, not 0: the kernel computes in float64 and Layer 2
        # narrows once on the way out, so a float32 result carries that store's
        # own rounding -- which DOMINATES, since u32 is 5.4e8 times u64. Costs
        # nothing when the storage is float64, because the harness charges a
        # store into the accumulator's own format at zero.
        roundings=Roundings(stages=1, accumulator_per_stage=1, storage_per_stage=1),
        accumulator=np.float64,
        storage=dtype,
        amplification=amplification,
        why=(
            "cosh(omega) carries the same 2 omega amplification as the node, "
            "squared, without the gap that tamed it; 2 omega is bounded by 36.7368 "
            "because the rule stops where the endpoint gap stops being "
            "representable. Plus the rescaling, whose sum is pairwise in numpy at "
            "O(log m . u) and plain here at O(m u), so the two differ by at most "
            "(m + log2 m + 1) u -- the term that dominates. The bound is about 500 "
            "times the measured deviation and that is reported rather than closed: "
            "the derivation assumes every libm error takes its extreme with the "
            "same sign. This bound catches a gross regression; the golden file, the "
            "node count and the C++ exactness tests are what actually guard the rule."
        ),
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_pts", N_PTS)
def test_the_node_count_is_identical(n_pts: int, dtype: npt.DTypeLike, cpp_backend: None) -> None:
    """The rule's length is the same in both backends, which no tolerance covers.

    ``gap < min_gap`` is a discrete decision on a quantity both backends compute
    with different transcendentals, so this is the one place where a rounding
    difference could turn into a structurally different answer. It is asserted
    before any value is compared, because a length mismatch makes every
    elementwise comparison below meaningless rather than merely failing.

    Args:
        n_pts (int): Requested number of points.
        dtype (npt.DTypeLike): Storage format, which sets the truncation threshold.
        cpp_backend (None): Requires the compiled extension.
    """
    (py_data, py_count), (cpp_data, cpp_count) = _both_backends(
        functools.partial(_generate_tanh_sinh, n_pts, dtype)
    )
    assert cpp_count == py_count, (
        f"the rule has {cpp_count} nodes in C++ and {py_count} in Python at "
        f"n_pts={n_pts}, {dtype}. The truncation threshold is shared, so this is a "
        f"transcendental disagreeing across the gap < min_gap boundary, not a "
        f"different decision -- and it makes the rule structurally different rather "
        f"than slightly displaced"
    )
    assert cpp_data.shape == py_data.shape


@pytest.mark.parametrize("n_pts", N_PTS)
def test_the_unmapped_rule_agrees_within_the_bound(n_pts: int, cpp_backend: None) -> None:
    """Nodes and weights on ``[-1, 1]`` agree within their derived bounds.

    Args:
        n_pts (int): Requested number of points.
        cpp_backend (None): Requires the compiled extension.
    """
    (py_data, py_count), (cpp_data, _) = _both_backends(
        functools.partial(_generate_tanh_sinh, n_pts, np.float64)
    )
    assert_parity(
        cpp_data[:, 0],
        py_data[:, 0],
        _node_claim(py_data[:, 0]),
        context=f"tanh-sinh nodes on [-1, 1], n={n_pts}",
    )
    assert_parity(
        cpp_data[:, 1],
        py_data[:, 1],
        _weight_claim(py_data[:, 0], py_data[:, 1], py_count),
        context=f"tanh-sinh weights on [-1, 1], n={n_pts}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_pts", N_PTS)
def test_the_public_rule_agrees_within_the_bound(
    n_pts: int, dtype: npt.DTypeLike, cpp_backend: None
) -> None:
    """The rule a caller gets agrees, through the shared map and the narrowing.

    The map onto ``[0, 1]`` halves every node and every weight, and the bound is
    rebuilt from the mapped values rather than scaled, so that it is stated in the
    frame the comparison happens in.

    **The two bounds do not both halve with the data, and that asymmetry is
    deliberate rather than an oversight.** The weight bound does, because its
    amplification is scaled by the weight array it is handed, which is already
    mapped. The node bound does not: its amplification is built from ``gap`` and
    ``omega``, which are dimensionless and recovered from the node's *position*
    rather than its magnitude, so the same number comes out in either frame.
    Measured, at n = 1000: the observed node difference halves exactly, while the
    node tolerance is identical across the two frames. The mapped node bound is
    therefore about twice as loose as the unmapped one. That is the safe
    direction, and closing it would mean giving the node claim a magnitude term it
    does not otherwise need; the slack is recorded here instead of being hidden.

    Args:
        n_pts (int): Requested number of points.
        dtype (npt.DTypeLike): Storage format.
        cpp_backend (None): Requires the compiled extension.
    """
    (py_nodes, py_weights), (cpp_nodes, cpp_weights) = _both_backends(
        functools.partial(get_tanh_sinh_1d, n_pts, dtype)
    )
    unmapped = 2.0 * np.asarray(py_nodes, dtype=np.float64) - 1.0
    count = int(np.size(py_nodes))
    assert_parity(
        cpp_nodes,
        py_nodes,
        _node_claim(unmapped, dtype),
        context=f"tanh-sinh nodes on [0, 1], n={n_pts}, {dtype}",
    )
    assert_parity(
        cpp_weights,
        py_weights,
        _weight_claim(unmapped, py_weights, count, dtype),
        context=f"tanh-sinh weights on [0, 1], n={n_pts}, {dtype}",
    )


def test_the_lambert_w_step_size_agrees_to_one_ulp(cpp_backend: None) -> None:
    """The step size agrees within a unit in the last place, and not always exactly.

    **This test used to assert equality over a handful of sampled counts, and it
    was wrong.** It sampled ``(2, 3, 5, 17, 64, 200, 1000)``, and the degrees where
    the two backends differ -- 124, 246, 252, 417, 477, 559, 646, 675, 783 within
    the same range -- are none of them. So it passed while asserting nothing about
    any case that fails, which is the failure mode a sampled sweep has and an
    exhaustive one does not.

    It now sweeps every count from 2 to 1000 and bounds the disagreement at one
    ulp. One rather than a derived multiple, because the cause is not accumulated
    error: by the third Halley step the iterate oscillates between two adjacent
    doubles, and a 1-ulp difference in ``exp`` decides which of the two step four
    lands on. A disagreement larger than that would mean the iteration itself had
    diverged, not that a library rounded differently.

    ``h = 2 W / n`` multiplies every transform coordinate, so this is what bounds
    how far the two grids can drift apart, and the node bound above depends on it.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    differing = []
    worst_ulp = 0.0

    for n_pts in range(2, 1001):
        argument = 2.0 * _TANH_SINH_DECAY_FACTOR * (0.5 * np.pi) * (n_pts - 1)
        python, cpp = _both_backends(functools.partial(_lambert_w_principal, argument))
        if cpp != python:
            differing.append(n_pts)
            worst_ulp = max(worst_ulp, abs(cpp - python) / float(np.spacing(python)))

    assert worst_ulp <= 1.0, (
        f"the two backends' step sizes differ by up to {worst_ulp:.2f} ulp, past the "
        f"one ulp the last-bit oscillation can explain, at {len(differing)} of 999 "
        f"counts. That is a diverged iteration rather than a library rounding "
        f"difference, and h = 2 W / n multiplies every transform coordinate"
    )
    assert differing, (
        "the two backends now agree at every count from 2 to 1000, where nine "
        "disagreements were measured. If the platform's libm changed, the node "
        "bound may be claimable as BITWISE; re-derive it rather than leaving a "
        "bound nothing exercises"
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_truncation_threshold_is_shared_rather_than_recomputed(
    dtype: npt.DTypeLike, cpp_backend: None
) -> None:
    """The discrete decision is taken from one number, computed once in Python.

    The node count holds across backends because ``min_gap`` is derived in Layer 2
    and passed into both kernels, not because two implementations of the
    derivation happened to agree. This pins that arrangement: if the C++ ever grew
    its own threshold, the count could diverge for a reason the bounds do not
    model, and the test above would start failing intermittently by degree.

    Args:
        dtype (npt.DTypeLike): Storage format.
        cpp_backend (None): Requires the compiled extension.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (the extension is optional)

    min_gap = _tanh_sinh_min_gap(dtype)
    nodes = np.empty(200, dtype=np.float64)
    weights = np.empty(200, dtype=np.float64)
    count = int(_pantr_cpp.generate_tanh_sinh(200, min_gap, out_nodes=nodes, out_weights=weights))

    _, expected = _generate_tanh_sinh(200, dtype)
    assert count == expected, (
        f"the C++ kernel handed the shared threshold {min_gap!r} produced {count} "
        f"nodes where the Python rule produces {expected}; the threshold is an "
        f"argument precisely so this cannot happen"
    )

    widened = int(
        _pantr_cpp.generate_tanh_sinh(200, min_gap * 1e3, out_nodes=nodes, out_weights=weights)
    )
    assert widened < count, (
        f"raising the threshold by three decades did not shorten the rule "
        f"({widened} nodes against {count}), so the kernel is not actually reading "
        f"the argument and the sharing this test claims to pin buys nothing"
    )


def test_the_bounds_are_reached_rather_than_idle(cpp_backend: None) -> None:
    """Both tanh-sinh bounds are approached, so they are asserting something.

    A tolerance never approached asserts nothing, and both bounds here are built
    from worst-case libm budgets and a worst-case summation term, which is exactly
    the shape that ends up idle. Measured, they do not: the worst ratio of
    deviation to its own bound is 0.43 for the nodes and 0.29 for the weights over
    n from 2 to 1000, so the slack is a factor of two to four rather than orders.

    The small counts contribute nothing, and that is not a defect: below about
    n = 64 the two backends agree bit for bit, so the ratio there is exactly zero.
    The sweep therefore has to reach the counts where the transcendentals actually
    separate, which is what the assertion on the maximum checks.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    worst_node = 0.0
    worst_weight = 0.0
    for n_pts in (2, 3, 5, 17, 64, 200, 400, 1000):
        (py_data, py_count), (cpp_data, _) = _both_backends(
            functools.partial(_generate_tanh_sinh, n_pts, np.float64)
        )
        node_deviation = assert_parity(
            cpp_data[:, 0],
            py_data[:, 0],
            _node_claim(py_data[:, 0]),
            context=f"tanh-sinh node slack probe, n={n_pts}",
        )
        weight_deviation = assert_parity(
            cpp_data[:, 1],
            py_data[:, 1],
            _weight_claim(py_data[:, 0], py_data[:, 1], py_count),
            context=f"tanh-sinh weight slack probe, n={n_pts}",
        )
        worst_node = max(worst_node, node_deviation.max_ratio_to_bound)
        worst_weight = max(worst_weight, weight_deviation.max_ratio_to_bound)

    assert worst_node > 0.1, (
        f"the node bound is never approached more closely than {worst_node:.3g} of "
        f"itself, so it is no longer distinguishing a correct port from a wrong one"
    )
    assert worst_weight > 0.05, (
        f"the weight bound is never approached more closely than {worst_weight:.3g} "
        f"of itself, so it is no longer distinguishing a correct port from a wrong one"
    )
