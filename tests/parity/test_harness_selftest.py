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
