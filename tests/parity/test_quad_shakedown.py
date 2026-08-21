r"""Adversarial shakedown of the four ported quadrature kernels and their parity claims.

Everything here came out of an attempt to *break* `cpp/include/pantr/quad/`, and each
test pins one thing that attempt found. The tests marked ``xfail`` state a claim the
shipped code or its documentation makes and that this file shows to be false; the
plain ones pin the mechanism behind it, or pin a measurement the shipped tests record
incorrectly.

What broke
----------

**The BITWISE Lambert W claim is false, and the sweep that checks it misses every
counterexample.** ``test_quad_tanh_sinh.py`` samples seven degrees; over ``n`` from 2
to 1000 the two backends disagree by one unit in the last place at nine of them --
124, 246, 252, 417, 477, 559, 646, 675, 783 -- and none is in the sample. The root
cause is not ``log`` or ``exp`` accuracy but the *fixed step count*: by the third
Halley step the iterate has reached the last-bit noise floor and the map is no longer
contracting there, so it enters a two-cycle between adjacent doubles and the fourth
step returns whichever member of the cycle a one-ulp ``exp`` difference selects.
:func:`test_the_halley_iteration_reaches_a_last_bit_limit_cycle` shows the cycle.

That matters beyond the claim itself. ``h = 2 W(x) / n`` multiplies every transform
coordinate, so at those degrees ``t`` is **not** common mode -- and "``t`` is
bit-identical ... so the argument of every transcendental is common mode and only the
functions themselves differ" is the load-bearing sentence of the node bound's
derivation in ``test_quad_tanh_sinh.py``. At those degrees the bound is not derived,
it merely happens to hold: the largest node ratio found anywhere, 0.947 at n = 14542,
is at one of them.

**The `std::pow(c, 2.0)` in `tanh_sinh.hpp` does not survive the compiler.** The
header spends a paragraph explaining that the square must be libm ``pow`` and not
``c * c``, because numpy's *scalar* ``** 2`` is ``pow`` and the two differ on about
0.08% of arguments. Both GCC (from ``-O1``) and clang (at ``-O2``) fold
``pow(x, 2.0)`` into ``x * x`` with no fast-math flag anywhere, and the shipped
extension carries no ``pow`` relocation at all -- so the intent is not merely
defeated on this build, it is unreachable on any optimised one short of
``-fno-builtin-pow``. The weight parity claim is BOUNDED and absorbs the difference,
which is why nothing failed. Worth adding: ``x * x`` is the *correctly rounded*
square, and glibc's ``pow(x, 2.0)`` is the one that is one ulp out wherever the two
part company (checked against exact rational arithmetic on 166 of 166 disagreements),
so the compiler is improving the answer while breaking the transliteration.

**The float32 half of `_tanh_sinh_min_gap`'s derivation is wrong.** Its docstring
walks three roundings and concludes the node collapses onto 1 when ``gap < 0.75 eps``
"in float64 and in float32 alike, the argument being about the binade boundary at 1
and not about the width of the format". In float32 only one of those roundings
happens -- the kernel returns float64 whatever the caller asked for, and
``_scale_and_cast_nodes_and_weights`` does ``((x + 1) * 0.5).astype(dtype)`` with the
arithmetic in float64 -- so the crossing sits at ``0.5 eps``, not ``0.75 eps``. The
shipped threshold is still safe; it clears the true crossing by a factor 2 in float32
rather than the 4/3 claimed.

**The C++ binding guards `min_gap > 0`, which is not the precondition the certificate
needs.** ``get_tanh_sinh_1d`` advertises that no node reaches an endpoint. That needs
``min_gap`` above the collapse threshold, not merely positive: handed ``1e-300`` the
kernel returns nodes at exactly ``+-1``, and handed a ``min_gap`` above every gap it
returns a rule with no points at all -- silently in C++, and with a divide-by-zero
``RuntimeWarning`` in Python.

**And the node count does move.** ``gap < min_gap`` is a discrete decision on a
quantity the two backends reach through different transcendentals, and the whole
legal range of ``n`` -- 2 to ``INT_MAX``, which is what the binding accepts -- was
scanned at the one or two indices that bracket the crossing. In float64 the two
backends land on opposite sides at **36** degrees, the smallest ``n = 212711125``,
where the Python rule has 39516801 nodes and the C++ rule 39516803. That one was
built end to end and the counts are measured, not predicted. In float32 there is no
such degree anywhere in the range, though the closest approach, 4.4e-16 relative at
``n = 734459294``, is a near miss rather than protection.

The numerical cost is nil -- the two extra nodes carry about ``8e-15`` of a weight
sum of 2 -- but the length of the returned array depends on which backend built it,
which is the one thing the port said could not happen.

What held, and over what range
------------------------------

* The tanh-sinh **node count** holds at every ``n`` the binding accepts in float32,
  and at all but 36 of them in float64 -- see above. Why the smallest is as large as
  ``2 * 10^8`` is quantitative rather than luck: the last retained node's margin from
  the threshold is equidistributed over a range of order ``h``, its minimum over the
  first ``N`` degrees shrinks only like ``log N / N^2``, and the two backends' gaps
  sit at most about 68 ulp apart, so the two first meet at ``N`` of order ``10^8``.
* **Gauss-Legendre is bitwise** over every ``n`` from 1 to 900 and sparsely to 6000,
  1.76e6 values, zero differing; and ``np.cos`` was measured equal to glibc's ``cos``
  on 4e6 Tricomi start arguments and on a further 5.5e7 random ones, so the one
  operation IEEE 754 leaves open is not a threat on this platform.
* **Trapezoidal is bitwise** and the **modified Chebyshev nodes** are bitwise in
  float64 and within a third of their bound in float32, over ``n`` from 2 to 6000 and
  at the float32 integer frontier ``n = 2**24 +- 3``.

Reproducibility across thread count is **not applicable** and was checked rather
than assumed: no kernel in ``cpp/include/pantr/quad/`` is parallel, the Python
oracles carry no Numba at all, and there is no reduction whose order a thread count
could change. Measured anyway, at ``OMP_NUM_THREADS`` of 1, 4 and 20: every rule
came back bit-identical within each backend.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import Final

import numpy as np
import numpy.typing as npt
import pytest

from pantr._backend import Backend, use_backend
from pantr.quad._rules import (
    _generate_tanh_sinh,
    _lambert_w_principal,
    _scale_and_cast_nodes_and_weights,
    _tanh_sinh_min_gap,
)
from pantr.quad._rules_core import (
    _LAMBERT_W_HALLEY_STEPS,
    _TANH_SINH_DECAY_FACTOR,
    _generate_tanh_sinh_core,
)
from tests._parity_harness import (
    absolute_tolerance,
    demand_the_reference_host,
    on_the_reference_host,
)
from tests.parity.test_quad_tanh_sinh import N_PTS as SHIPPED_N_PTS
from tests.parity.test_quad_tanh_sinh import _node_claim, _weight_claim

HALF_PI: Final = 0.5 * math.pi
"""``pi / 2``, formed exactly as both kernels form it."""

LAMBERT_W_OFFENDERS: Final = (124, 246, 252, 417, 477, 559, 646, 675, 783)
"""Degrees below 1000 at which the two backends' Lambert W differed when measured.

Recorded so the message of a failure can say which degrees moved, not asserted as a
platform-independent fact: which arguments numpy's ``exp`` and libm's ``exp`` disagree
on is a property of two libraries on one machine.
"""

COUNT_FLIP_DEGREE: Final = 212711125
"""The smallest degree at which the two backends' node counts differ.

Constructed rather than sampled: the truncation decision `gap < min_gap` can only
flip where the two backends' computed gaps straddle `min_gap`, so every ``n`` from 2
to ``INT_MAX`` -- the whole range the binding accepts -- was scanned at the one or
two indices that bracket the crossing, in both storage formats. In float64, 36
degrees flip, this being the smallest; the next are 378759433, 511428201, 661174396.
In float32 none does.
"""

COUNT_FLIP_INDEX: Final = 19758400
"""Loop index inside :data:`COUNT_FLIP_DEGREE` at which the two verdicts differ."""

COUNT_FLIP_COUNTS: Final = (39516801, 39516803)
"""Node counts the Python and the C++ backend return at :data:`COUNT_FLIP_DEGREE`.

Measured end to end, not predicted: `_generate_tanh_sinh_core` returned 39516801 and
`_pantr_cpp.generate_tanh_sinh` returned 39516803 for the same `n` and the same
`min_gap`. That run needs about 7 GB and a minute, which is why the test below pins
the decision rather than the rule.
"""

WORST_WEIGHT_RATIO_DEGREE: Final = 124
"""Degree carrying the largest observed weight deviation relative to its own bound."""

WORST_NODE_RATIO_DEGREE: Final = 14542
"""Degree carrying the largest observed node deviation relative to its own bound.

Found by evaluating the bound only at the degrees where the step size differs between
backends, which is the mechanism the bound's derivation assumes away.
"""


def _lambert_w_argument(n: int) -> float:
    """Return the Lambert W argument the tanh-sinh rule forms at ``n`` points.

    Args:
        n (int): Number of requested points.

    Returns:
        float: ``2 * decay_factor * (pi / 2) * (n - 1)``, in the kernels' association.
    """
    return 2.0 * _TANH_SINH_DECAY_FACTOR * HALF_PI * (n - 1)


def _halley_iterates(x: float, steps: int) -> list[float]:
    """Run the kernels' Halley iteration in libm and return every iterate.

    Uses :mod:`math`, which is the same libm the extension links, so this reproduces
    the C++ side rather than the numpy one.

    Args:
        x (float): The argument, at or above about 1.61.
        steps (int): How many steps to take.

    Returns:
        list[float]: The iterates, starting after the first step.
    """
    log_x = math.log(x)
    log_log_x = math.log(log_x)
    w = log_x - log_log_x + log_log_x / log_x
    out: list[float] = []
    for _ in range(steps):
        exp_w = math.exp(w)
        residual = w * exp_w - x
        derivative = exp_w * (w + 1.0)
        w = w - residual / (derivative - (w + 2.0) * residual / (2.0 * w + 2.0))
        out.append(w)
    return out


def _halley_iterates_numpy(x: float, steps: int) -> list[float]:
    """Run the same iteration through numpy, which is the Python oracle's path.

    Args:
        x (float): The argument, at or above about 1.61.
        steps (int): How many steps to take.

    Returns:
        list[float]: The iterates, starting after the first step.
    """
    log_x = np.log(x)
    log_log_x = np.log(log_x)
    w = log_x - log_log_x + log_log_x / log_x
    out: list[float] = []
    for _ in range(steps):
        exp_w = np.exp(w)
        residual = w * exp_w - x
        derivative = exp_w * (w + 1.0)
        w = w - residual / (derivative - (w + 2.0) * residual / (2.0 * w + 2.0))
        out.append(float(w))
    return out


def _reconstruct_cpp_rule(
    n: int, min_gap: float, square: Callable[[float], float]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Reproduce ``pantr::generate_tanh_sinh`` in Python, with a chosen square.

    Every transcendental goes through :mod:`math`, which is the libm the extension
    links, and every other operation is written in the kernel's own association, so
    the result is bit-for-bit what the C++ produces **provided** the compiler emitted
    the square this function was handed.  That is the whole point: handing it both
    squares and seeing which one reproduces the extension is what identifies the
    instruction GCC actually emitted for ``std::pow(c, 2.0)``.

    Args:
        n (int): Requested number of points. Must be at least 1.
        min_gap (float): Smallest endpoint distance a node may carry.
        square (Callable[[float], float]): How ``cosh(omega)`` is squared.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Nodes and rescaled
            weights on ``[-1, 1]``, in the kernel's generation order.
    """
    if n == 1:
        return np.array([0.0]), np.array([2.0])

    step = 2.0 * _halley_iterates(_lambert_w_argument(n), _LAMBERT_W_HALLEY_STEPS)[-1] / n
    odd = n % 2 != 0
    nodes: list[float] = []
    weights: list[float] = []
    if odd:
        nodes.append(0.0)
        weights.append(HALF_PI)
    for i in range(n // 2):
        t = (i + 1.0) * step if odd else (i + 0.5) * step
        omega = HALF_PI * math.sinh(t)
        weight = HALF_PI * math.cosh(t) / square(math.cosh(omega))
        gap = 2.0 / (1.0 + math.exp(2.0 * omega))
        if gap < min_gap:
            break
        nodes.append(-(1.0 - gap))
        weights.append(weight)
        nodes.append(1.0 - gap)
        weights.append(weight)

    total = 0.0
    for value in weights:
        total = total + value
    scale = 2.0 / total
    return np.array(nodes), np.array([value * scale for value in weights])


def _by_multiplication(value: float) -> float:
    """Square by a multiplication, which is what GCC emits for ``std::pow(x, 2.0)``.

    Args:
        value (float): The value to square.

    Returns:
        float: ``value * value``, correctly rounded.
    """
    return value * value


def _by_libm_pow(value: float) -> float:
    """Square through libm ``pow``, which is what ``tanh_sinh.hpp`` says it does.

    ``math.pow`` calls the platform's ``pow``, and numpy's *scalar* ``** 2`` was
    measured to agree with it on every one of 200000 arguments, so this is also the
    Python oracle's square.

    Args:
        value (float): The value to square.

    Returns:
        float: ``pow(value, 2.0)``.
    """
    return math.pow(value, 2.0)


def _discriminating_degree(min_gap: float, limit: int = 200) -> int | None:
    """Find the smallest ``n`` at which the two squares give different weights.

    Args:
        min_gap (float): Truncation threshold to build the rule with.
        limit (int): Largest ``n`` to try. Defaults to 200.

    Returns:
        int | None: The smallest such ``n``, or None if the two never separate.
    """
    for n in range(2, limit + 1):
        _, by_mul = _reconstruct_cpp_rule(n, min_gap, _by_multiplication)
        _, by_pow = _reconstruct_cpp_rule(n, min_gap, _by_libm_pow)
        if not np.array_equal(by_mul, by_pow):
            return n
    return None


# ---------------------------------------------------------------------------
# The Lambert W step size, and the last-bit limit cycle behind it
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=on_the_reference_host(),
    reason=(
        "REFUTED. tanh_sinh.hpp and test_quad_tanh_sinh.py both claim the Lambert W "
        "step size is bit-identical across backends; over n from 2 to 1000 it is not, "
        "and the seven degrees the shipped test samples miss every counterexample. An "
        "XPASS here means numpy's exp and this platform's libm exp have stopped "
        "disagreeing, in which case the claim needs re-measuring rather than trusting."
    ),
)
def test_the_lambert_w_step_size_is_bitwise_at_every_degree(cpp_backend: None) -> None:
    """The step size agrees to the last bit at every degree, not just the sampled ones.

    ``h = 2 W(x) / n`` multiplies every transform coordinate, and the tanh-sinh node
    bound is derived on the assumption that ``t`` is therefore common mode. A degree
    where ``W`` moves is a degree where that assumption is false, so the claim has to
    hold everywhere or the derivation has to change.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    offenders = []
    for n in range(2, 1001):
        argument = _lambert_w_argument(n)
        with use_backend(Backend.PYTHON):
            python = _lambert_w_principal(argument)
        with use_backend(Backend.CPP):
            cpp = _lambert_w_principal(argument)
        if python != cpp:
            offenders.append(n)

    assert not offenders, (
        f"W is not bit-identical at {len(offenders)} of 999 degrees: {offenders}. The "
        f"shipped sweep is {SHIPPED_N_PTS} and hits none of them, which is why the "
        f"BITWISE assertion passes. h = 2 W / n multiplies every transform coordinate, "
        f"so at these degrees the node bound's premise that t is common mode is false"
    )


def test_the_halley_iteration_reaches_a_last_bit_limit_cycle(cpp_backend: None) -> None:
    """The fourth Halley iterate is not a fixed point, which is why the claim above fails.

    The step count is fixed at four rather than iterated to a residual, and by the
    third step the iterate has reached the last-bit noise floor, where the map stops
    contracting: it alternates between two adjacent doubles for ever. Which member the
    fourth step returns is then decided by a one-ulp difference in ``exp``, and numpy's
    ``exp`` disagrees with libm's on a few per cent of arguments.

    This is the mechanism, so it is pinned separately from the symptom: a fix that
    only widened the sampled degrees would leave it in place.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    argument = _lambert_w_argument(WORST_WEIGHT_RATIO_DEGREE)
    through_numpy = sorted(set(_halley_iterates_numpy(argument, 12)[4:]))
    through_libm = sorted(set(_halley_iterates(argument, 12)[4:]))

    demand_the_reference_host(
        "whether the numpy Halley iteration reaches a two-value limit cycle",
        "a cycle of exactly two doubles from step 5 on, on this project's build server",
    )
    assert len(through_numpy) == 2, (
        f"the numpy iteration was expected to alternate between two doubles at n="
        f"{WORST_WEIGHT_RATIO_DEGREE}; steps 5 to 12 took {len(through_numpy)} distinct "
        f"values {through_numpy}. Without the cycle there is nothing for a one-ulp exp "
        f"difference to select between and this diagnosis is wrong"
    )
    lower, upper = through_numpy
    gap_in_ulp = int(np.float64(upper).view(np.int64) - np.float64(lower).view(np.int64))
    assert gap_in_ulp == 1, (
        f"the two values of the cycle are {gap_in_ulp} ulp apart, not adjacent, so the "
        f"iteration is not merely at the last-bit noise floor"
    )
    assert len(through_libm) == 1 and through_libm[0] in through_numpy, (
        f"the libm iteration settled on {through_libm}, which is not a single member of "
        f"the numpy cycle {through_numpy}; the two paths then differ for some other "
        f"reason than which member of the cycle the fourth step lands on"
    )

    with use_backend(Backend.PYTHON):
        python = _lambert_w_principal(argument)
    with use_backend(Backend.CPP):
        cpp = _lambert_w_principal(argument)
    assert python != cpp and {python, cpp} == {lower, upper}, (
        f"the two backends returned {python!r} and {cpp!r}; they were expected to "
        f"return the two members of the cycle {through_numpy}, one each"
    )


def test_the_node_count_holds_at_the_degrees_where_the_step_size_differs(
    cpp_backend: None,
) -> None:
    """The discrete truncation decision survives a step size that is not common mode.

    ``gap < min_gap`` decides how many nodes the rule has, and a count that moved
    between backends would be a structural failure rather than a rounding difference.
    The degrees where the step size itself differs are where that is most likely, and
    the shipped corner list contains none of them, so they are checked here. Both
    storage formats, because the threshold is what each one sets.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    for dtype in (np.float64, np.float32):
        for n in LAMBERT_W_OFFENDERS:
            with use_backend(Backend.PYTHON):
                _, python_count = _generate_tanh_sinh(n, dtype)
            with use_backend(Backend.CPP):
                _, cpp_count = _generate_tanh_sinh(n, dtype)
            assert cpp_count == python_count, (
                f"the rule has {cpp_count} nodes in C++ and {python_count} in Python at "
                f"n={n}, {np.dtype(dtype).name} -- a degree where the step size already "
                f"differs by one ulp, so the two backends are not sampling the same grid"
            )


def test_the_tanh_sinh_bounds_are_closer_to_failing_than_the_shipped_sweep_records(
    cpp_backend: None,
) -> None:
    """Both bounds are approached far more closely than their own module records.

    ``test_quad_tanh_sinh.py`` reports "0.43 for the nodes and 0.29 for the weights"
    as the worst ratio "over n from 2 to 1000". Those are the worst over the eight
    degrees its own probe visits, not over the range it names -- 0.29 is exactly the
    value at n = 64, the largest of the eight that contributes. Sweeping
    every degree instead reaches 0.90 on the weights at n = 124 and 0.95 on the nodes
    at n = 14542, and both of those are degrees where the Lambert W step size differs
    -- the mechanism the derivation assumes away. The bounds hold; the slack recorded
    around them is a factor of two to three larger than the slack there is.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    observed: dict[str, float] = {}
    for name, n in (("weights", WORST_WEIGHT_RATIO_DEGREE), ("nodes", WORST_NODE_RATIO_DEGREE)):
        with use_backend(Backend.PYTHON):
            python, count = _generate_tanh_sinh(n, np.float64)
        with use_backend(Backend.CPP):
            cpp, _ = _generate_tanh_sinh(n, np.float64)
        if name == "nodes":
            tolerance = absolute_tolerance(_node_claim(python[:, 0]))
            deviation = np.abs(cpp[:, 0] - python[:, 0])
        else:
            tolerance = absolute_tolerance(_weight_claim(python[:, 0], python[:, 1], count))
            deviation = np.abs(cpp[:, 1] - python[:, 1])
        usable = np.asarray(tolerance) > 0.0
        observed[name] = float(np.max(deviation[usable] / np.asarray(tolerance)[usable]))

    demand_the_reference_host(
        "how closely the recorded weight bound is approached",
        "0.90 of the bound, on this project's build server",
    )
    assert observed["weights"] > 0.85, (
        f"the weight bound was approached to {observed['weights']:.3f} of itself at "
        f"n={WORST_WEIGHT_RATIO_DEGREE}, where 0.90 was measured. Below 0.85 either the "
        f"bound was widened or the platform changed; either way the module's recorded "
        f"slack has to be re-measured rather than left as it is"
    )
    assert observed["nodes"] > 0.90, (
        f"the node bound was approached to {observed['nodes']:.3f} of itself at "
        f"n={WORST_NODE_RATIO_DEGREE}, where 0.95 was measured"
    )
    assert observed["weights"] <= 1.0 and observed["nodes"] <= 1.0, (
        f"a shipped parity bound is EXCEEDED: nodes {observed['nodes']:.3f}, weights "
        f"{observed['weights']:.3f} of their own tolerance"
    )

    # The same module says "below about n = 64 the two backends agree bit for bit, so
    # the ratio there is exactly zero", which is why its sweep skips the small counts.
    # They do not: the weights part company at n = 4 and the nodes at n = 7.
    for n, column, quantity in ((4, 1, "weights"), (7, 0, "nodes")):
        with use_backend(Backend.PYTHON):
            python, _ = _generate_tanh_sinh(n, np.float64)
        with use_backend(Backend.CPP):
            cpp, _ = _generate_tanh_sinh(n, np.float64)
        differing = int(np.count_nonzero(python[:, column] != cpp[:, column]))
        assert differing > 0, (
            f"the two backends now agree bit for bit on the {quantity} at n={n}, where "
            f"{differing} of {python.shape[0]} differed when measured. The claim that "
            f"they agree below about n = 64 was false then; if it has become true, the "
            f"reason is a change in one of the two and is worth finding"
        )


@pytest.mark.xfail(
    strict=on_the_reference_host(),
    reason=(
        "REFUTED, by construction. tanh_sinh.hpp says a count that moved between "
        "backends would be a genuine parity failure rather than a rounding "
        "difference, and test_quad_tanh_sinh.py records the count as identical in 44 "
        "of 44 sampled cases. Scanning every n the binding accepts, at the indices "
        "that bracket the truncation crossing, finds 36 float64 degrees where it "
        "moves; the smallest is n = 212711125, where the Python rule has 39516801 "
        "nodes and the C++ rule 39516803. "
        "An XPASS means this platform's numpy and libm no longer straddle min_gap "
        "there, which makes the count a property of two libraries rather than of the "
        "algorithm and is worth knowing either way."
    ),
)
def test_the_truncation_verdict_agrees_at_every_coordinate_the_rule_forms(
    cpp_backend: None,
) -> None:
    r"""``gap < min_gap`` is decided the same way by both backends.

    The decision is pinned rather than the rule, because building the rule at this
    degree costs about 7 GB and a minute; it was built once and the counts are in
    :data:`COUNT_FLIP_COUNTS`. What is reproduced here is the decision itself, which
    is a pure function of the transform coordinate: ``h`` comes from the two real
    backends, and the two ``gap`` values are formed through numpy and through
    :mod:`math`, which is the libm the extension links and which
    :func:`test_a_libm_reconstruction_reproduces_the_cpp_tanh_sinh_kernel_bitwise`
    shows reproduces the kernel bit for bit.

    The mechanism, and why no smaller ``n`` does it: ``gap`` decreases
    double-exponentially, so the last retained node's distance from the threshold is
    equidistributed over a range of order ``h``, while the two backends' ``gap``
    values sit at most about 68 ulp apart. A flip needs the margin below that, and
    the smallest margin available over the first ``N`` degrees shrinks only like
    ``log N / N^2``. It first gets small enough at ``N`` of order ``10^8``, and from
    there on it recurs: 36 degrees below ``INT_MAX``, every one of them landing on
    the same pair of gaps because only one double ``t`` is close enough to the
    crossing for the two libraries to disagree across it.

    What it costs numerically is almost nothing, and that is worth saying: the two
    extra nodes carry a weight of about ``8e-15`` against a weight sum of 2. What it
    costs is the claim -- a caller who switches backend gets an array of a different
    length.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    min_gap = _tanh_sinh_min_gap(np.float64)
    argument = _lambert_w_argument(COUNT_FLIP_DEGREE)
    with use_backend(Backend.PYTHON):
        python_step = 2.0 * _lambert_w_principal(argument) / COUNT_FLIP_DEGREE
    with use_backend(Backend.CPP):
        cpp_step = 2.0 * _lambert_w_principal(argument) / COUNT_FLIP_DEGREE

    assert COUNT_FLIP_DEGREE % 2 == 1, "the sampling offset below assumes an odd degree"
    index = COUNT_FLIP_INDEX + 1.0
    python_gap = 2.0 / (1.0 + np.exp(2.0 * (HALF_PI * np.sinh(index * python_step))))
    cpp_gap = 2.0 / (1.0 + math.exp(2.0 * (HALF_PI * math.sinh(index * cpp_step))))

    assert (float(python_gap) < min_gap) == (cpp_gap < min_gap), (
        f"at n={COUNT_FLIP_DEGREE}, i={COUNT_FLIP_INDEX} the gap is "
        f"{float(python_gap)!r} in Python and {cpp_gap!r} in C++, and min_gap is "
        f"{min_gap!r}: the two land on opposite sides, so one backend keeps the node "
        f"and the other stops. Measured end to end, the rule then has "
        f"{COUNT_FLIP_COUNTS[0]} nodes in Python and {COUNT_FLIP_COUNTS[1]} in C++"
    )


# ---------------------------------------------------------------------------
# The square the compiler actually emitted
# ---------------------------------------------------------------------------


def test_a_libm_reconstruction_reproduces_the_cpp_tanh_sinh_kernel_bitwise(
    cpp_backend: None,
) -> None:
    """The instrument used by the next test is sound: it reproduces the kernel exactly.

    A strict-xfail test proves nothing if it fails because the reconstruction is wrong,
    so the reconstruction is checked first. It is only claimed to match for *one* of
    the two squares -- which one is the finding, and is the next test's business.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (the extension is optional)

    for dtype in (np.float64, np.float32):
        min_gap = _tanh_sinh_min_gap(dtype)
        for n in (2, 3, 5, 17, 42, 64, 124, 200, 401, 544, 1000):
            nodes = np.empty(n, dtype=np.float64)
            weights = np.empty(n, dtype=np.float64)
            count = int(
                _pantr_cpp.generate_tanh_sinh(n, min_gap, out_nodes=nodes, out_weights=weights)
            )
            matched = False
            for square in (_by_multiplication, _by_libm_pow):
                built_nodes, built_weights = _reconstruct_cpp_rule(n, min_gap, square)
                if (
                    built_nodes.size == count
                    and np.array_equal(nodes[:count], built_nodes)
                    and np.array_equal(weights[:count], built_weights)
                ):
                    matched = True
                    break
            assert matched, (
                f"neither square reproduces the extension at n={n}, "
                f"{np.dtype(dtype).name}, so the reconstruction no longer models the "
                f"kernel and nothing built on it can be trusted"
            )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "REFUTED. tanh_sinh.hpp argues at length that the square must be libm pow "
        "rather than c * c, because numpy's scalar ** 2 is pow and the two differ on "
        "about 0.08% of arguments. GCC folds pow(x, 2.0) into x * x at -O2 with no "
        "fast-math flag, and the shipped extension carries no pow relocation, so the "
        "build does the multiplication the header says it must not. An XPASS means the "
        "compiler stopped folding, at which point the header is right again and this "
        "test should be inverted rather than deleted."
    ),
)
def test_the_cpp_square_is_the_libm_pow_the_header_says_it_is(cpp_backend: None) -> None:
    """``std::pow(cosh(omega), 2.0)`` reaches the binary as a call to libm ``pow``.

    Identified behaviourally rather than by reading the disassembly: at the smallest
    degree where ``pow(c, 2.0)`` and ``c * c`` give different weights, exactly one of
    the two reconstructions can match the extension bit for bit, and which one does
    says which instruction the compiler emitted.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (the extension is optional)

    min_gap = _tanh_sinh_min_gap(np.float64)
    n = _discriminating_degree(min_gap)
    assert n is not None, (
        "no degree below 200 separates pow(c, 2.0) from c * c on this platform, so "
        "the two are indistinguishable here and this test cannot decide anything"
    )

    nodes = np.empty(n, dtype=np.float64)
    weights = np.empty(n, dtype=np.float64)
    count = int(_pantr_cpp.generate_tanh_sinh(n, min_gap, out_nodes=nodes, out_weights=weights))
    _, by_pow = _reconstruct_cpp_rule(n, min_gap, _by_libm_pow)
    _, by_mul = _reconstruct_cpp_rule(n, min_gap, _by_multiplication)

    assert np.array_equal(weights[:count], by_pow), (
        f"at n={n} the extension's weights are reproduced by squaring with "
        f"{'a multiplication' if np.array_equal(weights[:count], by_mul) else 'neither square'}, "
        f"not by libm pow. The Python oracle keeps pow -- numpy's scalar ** 2 is pow -- "
        f"so the transliteration tanh_sinh.hpp documents is not the one that ships, and "
        f"the weights differ by one ulp wherever the two squares do"
    )


# ---------------------------------------------------------------------------
# The endpoint certificate
# ---------------------------------------------------------------------------


def _mapped_node(gap: float, dtype: npt.DTypeLike) -> float:
    """Map a gap through the shipped path and return the node a caller would get.

    Args:
        gap (float): The endpoint distance ``1 - |x|`` the kernel produced.
        dtype (npt.DTypeLike): Storage format the rule is returned in.

    Returns:
        float: The node on ``[0, 1]`` nearest the right endpoint.
    """
    nodes, _ = _scale_and_cast_nodes_and_weights(np.array([1.0 - gap]), np.array([1.0]), dtype)
    return float(nodes[0])


def test_the_endpoint_collapse_threshold_is_half_an_epsilon_in_float32(
    cpp_backend: None,
) -> None:
    """``_tanh_sinh_min_gap``'s float32 derivation is wrong, though its answer is safe.

    Its docstring walks three roundings -- ``1 - gap``, ``+ 1`` tying to even, and an
    exact halving -- and concludes the node collapses onto 1 at ``gap < 0.75 eps``
    "in float64 and in float32 alike, the argument being about the binade boundary at
    1 and not about the width of the format". Only the float64 half of that is true.
    The kernel returns float64 whatever the caller asked for, and
    ``_scale_and_cast_nodes_and_weights`` does the whole map in float64 and rounds
    once at the cast, so in float32 there is one rounding rather than three and the
    crossing sits at ``0.5 eps``.

    The threshold shipped is one epsilon either way, so it clears the true crossing by
    4/3 in float64 and by 2 in float32. Nothing is unsafe; the derivation is.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    expected: tuple[tuple[npt.DTypeLike, float], ...] = ((np.float64, 0.75), (np.float32, 0.5))
    for dtype, multiple in expected:
        eps = float(np.finfo(np.dtype(dtype)).eps)
        low, high = 0.0, 4.0 * eps
        assert _mapped_node(low, dtype) >= 1.0
        assert _mapped_node(high, dtype) < 1.0
        for _ in range(400):
            middle = 0.5 * (low + high)
            if middle in (low, high):
                break
            if _mapped_node(middle, dtype) >= 1.0:
                low = middle
            else:
                high = middle
        crossing = high / eps
        # Loose, deliberately: the map rounds twice in float64 before the cast, so
        # the crossing sits one float64 quantum above the exact multiple of eps.
        assert crossing == pytest.approx(multiple, rel=1e-6), (
            f"in {np.dtype(dtype).name} the mapped node collapses onto 1 below "
            f"{crossing:.6f} eps, not {multiple} eps. _tanh_sinh_min_gap's docstring "
            f"derives 0.75 for both formats; if this moved, the margin its threshold "
            f"claims moved with it"
        )
        assert _tanh_sinh_min_gap(dtype) > high, (
            f"the shipped threshold {_tanh_sinh_min_gap(dtype)!r} does not clear the "
            f"measured crossing {high!r} in {np.dtype(dtype).name}"
        )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "REFUTED. cpp/bindings/quad.cpp checks only that min_gap is finite and "
        "strictly positive, but the certificate get_tanh_sinh_1d advertises -- no node "
        "reaches an endpoint -- needs min_gap above the collapse threshold, which is "
        "0.75 eps of the frame the rule is mapped into. The lambert_w_principal "
        "binding right next to it validates its real precondition for exactly the "
        "reason this one should."
    ),
)
def test_the_binding_refuses_a_min_gap_that_cannot_keep_a_node_off_the_endpoint() -> None:
    """A ``min_gap`` below the collapse threshold is refused rather than obeyed.

    Needs no backend comparison: what is under test is one binding's validation.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (the extension is optional)

    nodes = np.empty(60, dtype=np.float64)
    weights = np.empty(60, dtype=np.float64)
    with pytest.raises(ValueError, match="min_gap"):
        _pantr_cpp.generate_tanh_sinh(60, 1e-300, out_nodes=nodes, out_weights=weights)


def test_a_min_gap_below_the_threshold_puts_nodes_on_the_endpoints(cpp_backend: None) -> None:
    """What the unguarded precondition actually costs, in both backends.

    Pinned as behaviour rather than left implicit in the xfail above, because the
    failure is silent: the call returns a plausible-looking rule of the right length
    whose outermost nodes are the two endpoints, which is precisely the configuration
    a double-exponential rule exists to avoid.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (the extension is optional)

    n = 60
    nodes = np.empty(n, dtype=np.float64)
    weights = np.empty(n, dtype=np.float64)
    count = int(_pantr_cpp.generate_tanh_sinh(n, 1e-300, out_nodes=nodes, out_weights=weights))
    on_endpoint = int(np.count_nonzero(np.abs(nodes[:count]) >= 1.0))
    assert on_endpoint > 0, (
        "1e-300 no longer drives a node onto an endpoint, so either the collapse "
        "threshold moved or the binding grew the guard the xfail above asks for"
    )

    python_data, python_count = _generate_tanh_sinh_core(n, 1e-300)
    assert python_count == count
    assert int(np.count_nonzero(np.abs(python_data[:, 0]) >= 1.0)) == on_endpoint, (
        "the two backends disagree on how many nodes land on an endpoint, which would "
        "make this a parity failure on top of a certificate one"
    )

    mapped, _ = _scale_and_cast_nodes_and_weights(
        nodes[:count].copy(), weights[:count].copy(), np.float64
    )
    assert int(np.count_nonzero((mapped <= 0.0) | (mapped >= 1.0))) >= on_endpoint, (
        "the map onto [0, 1] was expected to carry the endpoint nodes through"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "REFUTED. A min_gap above every gap empties the rule. The C++ kernel divides "
        "2 by a sum of no terms, gets an infinite scale, multiplies nothing by it and "
        "returns 0 without a word; the Python oracle takes the same path and emits a "
        "divide-by-zero RuntimeWarning, which is an error under this suite's "
        "filterwarnings. Same input, two different observable behaviours."
    ),
)
def test_an_emptied_rule_is_reported_the_same_way_by_both_backends(cpp_backend: None) -> None:
    """Both backends treat a rule truncated to nothing alike.

    Not reachable through ``get_tanh_sinh_1d``, whose threshold comes from a dtype and
    is never large enough; reachable through the two Layer 3 entry points, which the
    package documents as the seam the port introduced.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (the extension is optional)

    nodes = np.empty(8, dtype=np.float64)
    weights = np.empty(8, dtype=np.float64)
    with warnings.catch_warnings(record=True) as cpp_warnings:
        warnings.simplefilter("always")
        cpp_count = int(_pantr_cpp.generate_tanh_sinh(8, 1.0, out_nodes=nodes, out_weights=weights))
    with warnings.catch_warnings(record=True) as python_warnings:
        warnings.simplefilter("always")
        _, python_count = _generate_tanh_sinh_core(8, 1.0)

    assert cpp_count == python_count == 0
    assert [str(w.message) for w in cpp_warnings] == [str(w.message) for w in python_warnings], (
        f"C++ warned {[str(w.message) for w in cpp_warnings]} and Python warned "
        f"{[str(w.message) for w in python_warnings]} for the same call. The rescale "
        f"divides 2 by a sum of no terms in both, but only numpy says so"
    )
