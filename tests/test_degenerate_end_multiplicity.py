"""A knot vector whose last knot repeats past ``degree + 1`` is refused.

``BsplineSpace1D`` used to accept ``[0, 0, 0, 1, 2, 3, 3, 3, 3]`` at degree 2: the
interval structure is sound, so neither the snapping rule nor the interval rule owns
it, and the space it built was not one. Measured on both backends:

=============================================  ====================================
call on that space                             behaviour before the refusal
=============================================  ====================================
``tabulate_basis`` at the right endpoint       the basis sums to ``0``, not ``1``
``tabulate_basis_derivatives``, oracle         ``ZeroDivisionError``
``tabulate_basis_derivatives``, port           returns ``NaN``
=============================================  ====================================

The last two rows are the reason the rule lives in the constructor rather than in
``tabulate_basis_derivatives``: the two backends do not merely both fail, they fail
*differently*, and a space that cannot be built removes the divergence without anyone
having to write a parity rule about which failure is the right one.

**The rule is exactly as wide as that measurement**, which is most of what the tests
below are for. Three neighbouring cases were measured and are legal, because none of
them shows the failure:

- an excess at the **first** knot, where the partition of unity holds and the
  derivatives are clean. It adds an identically zero basis function, which the
  ordinary degree-0 vector ``[0, 0, 1, 1]`` also has and this library accepts;
- an excess in the **interior**, the ordinary way to lower continuity;
- **degree 0** at any multiplicity, whose recurrence has no denominator.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest

from pantr._numba_compat import wait_for_jit_warmup
from pantr.bspline import BsplineSpace1D

wait_for_jit_warmup()


_TOO_MANY = re.compile("last knot may repeat")
"""Match the refusal without pinning the whole message."""


def _peak_of_each_basis_function(space: BsplineSpace1D, samples: int = 401) -> Any:
    """Sample the domain and return each basis function's largest absolute value.

    Args:
        space (BsplineSpace1D): The space to tabulate.
        samples (int): How many points to sample. Defaults to 401.

    Returns:
        Any: One non-negative float per basis function.
    """
    lo, hi = space.domain
    basis, first = space.tabulate_basis(np.linspace(lo, hi, samples))
    basis = np.asarray(basis)
    peak = np.zeros(space.num_basis)
    for point, start in enumerate(np.asarray(first)):
        for offset in range(basis.shape[-1]):
            peak[start + offset] = max(peak[start + offset], abs(basis[point, offset]))
    return peak


def test_an_excess_at_the_last_knot_is_refused() -> None:
    with pytest.raises(ValueError, match=_TOO_MANY) as excinfo:
        BsplineSpace1D(np.array([0.0, 0, 0, 1, 2, 3, 3, 3, 3]), 2)

    # The caller has to be able to act without opening our source: the ceiling is
    # the number they have to compare their own vector against.
    assert "at most 3 times" in str(excinfo.value), str(excinfo.value)


def test_a_periodic_space_is_refused_on_the_same_evidence() -> None:
    # Not an exemption: the partition of unity fails at the right endpoint there
    # exactly as it does on a clamped space.
    with pytest.raises(ValueError, match=_TOO_MANY):
        BsplineSpace1D(np.array([0.0, 0, 0, 1, 2, 3, 3, 3, 3]), 2, periodic=True)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_the_ceiling_is_degree_plus_one_at_every_degree_and_dtype(
    degree: int, dtype: type[Any]
) -> None:
    interior = np.arange(1.0, 5.0)
    at_the_ceiling = np.concatenate(
        [np.zeros(degree + 1), interior, np.full(degree + 1, 5.0)]
    ).astype(dtype)
    one_too_many = np.concatenate([at_the_ceiling, np.array([5.0], dtype=dtype)])

    assert BsplineSpace1D(at_the_ceiling, degree).num_basis == len(at_the_ceiling) - degree - 1
    with pytest.raises(ValueError, match=_TOO_MANY):
        BsplineSpace1D(one_too_many, degree)


@pytest.mark.parametrize("multiplicity", [1, 2, 3, 4])
def test_degree_zero_is_not_subject_to_the_rule(multiplicity: int) -> None:
    # Measured before the rule existed: at degree 0 the basis sums to one everywhere
    # and the derivatives are clean at every multiplicity tried, on both backends.
    # The recurrence has no denominator, so the failure the rule guards cannot occur
    # and a ceiling here would refuse spaces that compute correctly.
    knots = np.concatenate([np.zeros(1), np.full(multiplicity, 1.0)])
    space = BsplineSpace1D(knots, 0)

    assert space.num_basis == len(knots) - 1
    basis, _ = space.tabulate_basis(np.array([0.0, 0.5, 1.0]))
    # Exact, and legitimately so: a degree-0 basis function is an indicator, so the
    # sum is a single stored 1.0 rather than the result of any arithmetic.
    np.testing.assert_array_equal(np.asarray(basis).sum(axis=-1), 1.0)


def test_an_excess_at_the_first_knot_stays_legal() -> None:
    # Measured: the partition of unity holds and the derivatives are clean. What it
    # does produce is one identically zero basis function -- a different degeneracy,
    # which `[0, 0, 1, 1]` at degree 0 has as well and this library accepts, so
    # refusing here would be wider than the evidence this rule rests on.
    space = BsplineSpace1D(np.array([0.0, 0, 0, 0, 1, 2, 3, 3, 3]), 2)

    assert space.num_basis == 6
    peak = _peak_of_each_basis_function(space)
    assert int((peak == 0.0).sum()) == 1, peak
    basis, _ = space.tabulate_basis(np.array([0.0, 0.5, 1.5, 3.0]))
    # The recurrence is convex, so each of the `degree + 1` non-zero values carries
    # at most a few ulps and their sum at most one more rounding per addition: an
    # absolute bound of `(degree + 2) * eps` covers both, with no fitted constant.
    partition = (space.degree + 2) * float(np.finfo(np.float64).eps)
    np.testing.assert_allclose(np.asarray(basis).sum(axis=-1), 1.0, rtol=0.0, atol=partition)


def test_an_interior_knot_of_high_multiplicity_stays_legal() -> None:
    # The rule must not widen into the interior: this is how continuity is lowered,
    # and the space it builds is correct.
    space = BsplineSpace1D(np.array([0.0, 0, 0, 1, 1, 1, 2, 3, 3, 3]), 2)

    assert space.num_basis == 7
    basis, _ = space.tabulate_basis(np.array([0.0, 0.5, 1.0, 2.5, 3.0]))
    partition = (space.degree + 2) * float(np.finfo(np.float64).eps)
    np.testing.assert_allclose(np.asarray(basis).sum(axis=-1), 1.0, rtol=0.0, atol=partition)


def test_a_vector_that_fails_two_rules_is_told_about_the_interval_first() -> None:
    # The order is the content of the decision: a mesh with no interval at all is
    # the more useful diagnosis, so this check runs after that one and never
    # shadows it.
    with pytest.raises(ValueError, match="spans no interval"):
        BsplineSpace1D(np.zeros(8), 2)
