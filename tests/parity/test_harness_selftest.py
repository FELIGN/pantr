"""Self-tests of the parity harness in `tests/_parity_harness.py`.

Kernel-agnostic checks that the harness's own machinery does what it claims:
a claim cannot be built without a derivation, a bound that exhausts the format
or dwarfs the values it compares is refused rather than passing vacuously, and
the vacuity guard leaves a legitimate absolute floor alone.

The sensitivity probes for the claim kinds themselves --
`test_bitwise_claim_detects_a_one_ulp_difference` and the two
`test_bounded_branch_*` tests -- stay in `tests/parity/test_basis_cardinal_bspline.py`
instead: they drive the probe through `_tabulate` and the cardinal B-spline's own
point sets, which makes them per-kernel rather than harness-only.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._parity_harness import (
    Roundings,
    absolute_tolerance,
    assert_parity,
    bitwise_parity,
    bounded_parity,
    demand_a_compiled_seed,
    demand_the_compiled_kernel,
    the_jit_is_disabled,
)


def test_a_tolerance_cannot_be_stated_without_a_derivation() -> None:
    """Every way of building a claim without a derivation is refused.

    The harness's whole purpose is that the next five ported modules cannot express
    an underived tolerance. That property is a property of the API, so it is tested
    like any other.
    """
    with pytest.raises(ValueError, match="why"):
        bitwise_parity(why="   ")
    with pytest.raises(ValueError, match="derivation"):
        bounded_parity(
            roundings=Roundings(stages=3, accumulator_per_stage=2, storage_per_stage=0),
            accumulator=np.float64,
            storage=np.float64,
            amplification=np.ones((2, 2)),
            why="",
        )
    with pytest.raises(ValueError, match="bitwise_parity"):
        bounded_parity(
            roundings=Roundings(stages=0, accumulator_per_stage=2, storage_per_stage=0),
            accumulator=np.float64,
            storage=np.float64,
            amplification=np.ones((2, 2)),
            why="zero stages is not a bound",
        )
    with pytest.raises(ValueError, match="finite"):
        bounded_parity(
            roundings=Roundings(stages=3, accumulator_per_stage=2, storage_per_stage=0),
            accumulator=np.float64,
            storage=np.float64,
            amplification=np.array([[np.inf]]),
            why="an overflowed amplification makes the comparison vacuous",
        )


def test_a_budget_that_exhausts_the_format_is_refused() -> None:
    """A bound that accepts every finite result is reported, not returned.

    The failure mode a "derived" tolerance still permits: derive it for a degree so
    large, or a format so narrow, that the bound exceeds 1 in relative terms and the
    comparison stops meaning anything. That is worth an error rather than a pass.
    """
    claim = bounded_parity(
        roundings=Roundings(stages=10**8, accumulator_per_stage=2, storage_per_stage=1),
        accumulator=np.float64,
        storage=np.float32,
        amplification=np.ones((2, 2)),
        why="probe: a budget large enough to exhaust float32",
    )
    with pytest.raises(ValueError, match="vacuous"):
        absolute_tolerance(claim)


def test_a_bound_larger_than_the_values_it_compares_is_refused() -> None:
    """A finite but enormous amplification cannot buy a passing comparison.

    What it catches: an amplification that is finite and non-negative, so
    ``bounded_parity`` accepts it, and large enough that no result could violate
    the bound it produces. The assertion then passes for ever and reports
    agreement that was never measured, which is worse than a failure because
    nothing points at it.

    The amplification used here is not invented. It is what this harness's own
    docstring prescribed until it was corrected: the absolute-value companion of
    the kernel's recurrence, applied to the Legendre three-term recurrence, whose
    two homogeneous solutions are bounded on ``[-1, 1]`` while their absolute-value
    companion grows like ``(1 + sqrt(2))**k``. Measured at degree 700 it reaches
    ``1.7e266``, and the tolerance that follows is ``5.3e253``.
    """
    claim = bounded_parity(
        roundings=Roundings(stages=700, accumulator_per_stage=2, storage_per_stage=0),
        accumulator=np.float64,
        storage=np.float64,
        amplification=np.array([1.7e266, 1.0]),
        why="the absolute-value companion of an oscillatory recurrence, which is not a bound",
    )
    with pytest.raises(AssertionError, match="vacuous"):
        assert_parity(
            np.array([1.0, 0.0]),
            np.array([-1e250, 0.0]),
            claim,
            context="a bound larger than the values it compares",
        )


def test_the_vacuity_guard_leaves_an_absolute_floor_alone() -> None:
    """A legitimate bound on a value that is genuinely near zero still passes.

    The guard compares against the array's largest magnitude rather than each
    element's own, and this is the case that forces that choice: an element whose
    true value is zero is compared under the underflow floor, so its own tolerance
    exceeds its own magnitude by any factor you like. Checking per element would
    reject exactly the case :func:`underflow_floor` exists to serve, which is most
    of a B-spline row.
    """
    claim = bounded_parity(
        roundings=Roundings(stages=4, accumulator_per_stage=2, storage_per_stage=1),
        accumulator=np.float64,
        storage=np.float64,
        amplification=np.array([1.0, 0.0]),
        why="an ordinary claim over a row whose second entry is exactly zero",
    )
    deviation = assert_parity(
        np.array([0.5, 0.0]),
        np.array([0.5, 0.0]),
        claim,
        context="an absolute floor on an exactly zero entry",
    )
    assert deviation.num_differing == 0


def test_a_single_rounding_per_stage_gives_a_bound_that_is_not_zero() -> None:
    """A budget of one float64 rounding per stage must not collapse onto the floor.

    The regression test for a defect the shipped harness carried: the relative
    growth was computed as ``(1 + per_stage)**stages - 1``, and in float64 that is
    **exactly zero** whenever ``per_stage`` is one unit of roundoff. ``1 + eps/2``
    lands on the midpoint between ``1`` and ``1 + eps``, and round-half-to-even
    carries it back to ``1`` because ``1``'s significand is even, so the power is
    ``1.0`` at every stage count and the subtraction gives ``0.0``.

    The consequence was a claim saying BOUNDED while asserting bit-for-bit
    agreement, since all that survived was the underflow floor at about
    ``1e-323``. The existing vacuity guard did not catch it because it tests
    ``per_stage == 0``, which is ``1.11e-16`` here and passes: a budget can be
    non-zero and still produce a zero bound.

    The exact triggering data is ``accumulator_per_stage=1`` with the storage
    format equal to the accumulator, so the narrowing term is zero and the whole
    per-stage budget is a single ``u``. Both stage counts below returned ``0.0``
    before the fix.
    """
    floor_scale = 1e-300
    previous = 0.0

    for stages in (1, 4, 20):
        claim = bounded_parity(
            roundings=Roundings(stages=stages, accumulator_per_stage=1, storage_per_stage=0),
            accumulator=np.float64,
            storage=np.float64,
            amplification=np.ones(1),
            why="one float64 rounding per stage and no narrowing store, the case that "
            "made the power form evaluate to exactly zero",
        )
        tolerance = float(absolute_tolerance(claim)[0])

        assert tolerance > floor_scale, (
            f"at {stages} stages the tolerance is {tolerance:.3g}, down at the underflow "
            f"floor rather than at the rounding scale; the relative term evaluated to "
            f"zero and this BOUNDED claim is asserting bit-for-bit agreement"
        )
        assert tolerance > previous, (
            f"the tolerance did not grow from {previous:.3g} to {tolerance:.3g} when the "
            f"stage count reached {stages}; a bound that ignores its own stage count is "
            f"not accumulating anything"
        )
        previous = tolerance


def test_a_budget_that_reaches_the_runaway_half_is_refused() -> None:
    """Gamma stops bounding anything once the accumulated budget reaches one half.

    ``gamma_m = m u / (1 - m u)`` is a bound only while ``m u`` is small; at one
    half it equals 1 and past it the denominator collapses and then changes sign,
    so the expression stops being an error bound and starts being nonsense. The
    refusal has to happen on the budget rather than on the quotient, because a
    negative quotient looks like a small bound.

    **The budget below is chosen to sit between the old refusal threshold and the
    new one, and that choice is the whole test.** A first version used a stage
    count so enormous that the previous implementation's own guard rejected it
    too, so it passed against the broken code and against the fixed code alike --
    it pinned the new behaviour without discriminating. Here the accumulated
    budget is about 0.6: the current form gives ``gamma = 1.5`` and refuses, while
    the superseded power form gave ``(1 + 8u)^s - 1 = 0.82`` and waved it through
    as a usable bound. So this fails against the old implementation, which is what
    makes it a regression test rather than a description.
    """
    # 8 roundings per stage over this many stages accumulates to 0.6 units of
    # roundoff, derived rather than searched: stages = 0.6 / (8 u).
    stages_reaching_six_tenths = 675539944105574
    claim = bounded_parity(
        roundings=Roundings(
            stages=stages_reaching_six_tenths, accumulator_per_stage=8, storage_per_stage=0
        ),
        accumulator=np.float64,
        storage=np.float64,
        amplification=np.ones(1),
        why="a budget accumulating past the half where gamma stops bounding anything",
    )
    with pytest.raises(ValueError, match="vacuous"):
        absolute_tolerance(claim)


def test_the_seed_gate_fires_at_float64_where_the_width_gate_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two gates own different divergences, and only one of them is about width.

    :func:`demand_the_compiled_kernel` owns storage format: interpreted ``float32``
    intermediates promote in a way that has never been pinned. It is right to let
    ``float64`` past, and an earlier version of this suite concluded from that that
    ``float64`` was safe outright. It is not, for two reasons that are nothing to do
    with width: a seed of ``np.power``, which IEEE 754 does not pin and whose numba
    and numpy implementations disagree by an ulp, and a falling-factorial accumulator
    that wraps at int64 when compiled and grows without bound when interpreted. Past
    that overflow the two paths differ by a factor of about seven, which is why this
    cannot be answered with a tolerance.

    :func:`demand_a_compiled_seed` owns those two, at every width, and is asked for
    per test rather than applied to every bitwise claim: for a kernel of ``+``, ``-``,
    ``*`` and ``/`` alone IEEE 754 pins each result, so interpretation reproduces the
    compiled bits by guarantee. Gating all bitwise claims was measured at 442 of 495
    ``float64`` cases skipped, against 79 for this.
    """
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    assert the_jit_is_disabled()

    demand_the_compiled_kernel(np.float64)
    with pytest.raises(pytest.skip.Exception):
        demand_the_compiled_kernel(np.float32)
    with pytest.raises(pytest.skip.Exception):
        demand_a_compiled_seed()

    monkeypatch.setenv("NUMBA_DISABLE_JIT", "0")
    assert not the_jit_is_disabled()
    demand_the_compiled_kernel(np.float32)
    demand_a_compiled_seed()
