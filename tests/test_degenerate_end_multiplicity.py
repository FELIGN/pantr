"""A knot vector whose end knot repeats past ``degree + 1`` is refused.

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

The rule reads the two end classes only. An interior knot of high multiplicity is the
ordinary way to lower continuity and stays legal, and the ceiling has a floor of two
so that degree 0 keeps its ordinary clamped vector ``[0, 0, 1, 1]``, where
``degree + 1`` is 1.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest

from pantr._numba_compat import wait_for_jit_warmup
from pantr.bspline import BsplineSpace1D

wait_for_jit_warmup()


_TOO_MANY = re.compile("end knot may repeat")
"""Match the refusal without pinning the whole message."""


def test_an_excess_at_the_right_end_is_refused() -> None:
    with pytest.raises(ValueError, match=_TOO_MANY) as excinfo:
        BsplineSpace1D(np.array([0.0, 0, 0, 1, 2, 3, 3, 3, 3]), 2)

    # The caller has to be able to act without opening our source: the ceiling is
    # the number they have to compare their own vector against.
    assert "at most 3 times" in str(excinfo.value), str(excinfo.value)


def test_an_excess_at_the_left_end_is_refused() -> None:
    with pytest.raises(ValueError, match=_TOO_MANY):
        BsplineSpace1D(np.array([0.0, 0, 0, 0, 1, 2, 3, 3, 3]), 2)


def test_a_periodic_space_is_refused_on_the_same_evidence() -> None:
    # Not an exemption: the partition of unity fails at the right endpoint there
    # exactly as it does on a clamped space, which is what the message's wording
    # ("end knot", not "clamped end") records.
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


def test_degree_zero_keeps_its_ordinary_clamped_vector() -> None:
    # Where the floor of two earns its place: `degree + 1` is 1 here, so a rule
    # without it would refuse the most ordinary degree-0 space there is.
    assert BsplineSpace1D(np.array([0.0, 0.0, 1.0, 1.0]), 0).num_basis == 3


def test_an_interior_knot_of_high_multiplicity_stays_legal() -> None:
    # The rule must not widen into the interior: this is how continuity is lowered,
    # and the space it builds is correct.
    knots = np.array([0.0, 0, 0, 1, 1, 1, 2, 3, 3, 3])
    space = BsplineSpace1D(knots, 2)

    assert space.num_basis == 7
    basis, _ = space.tabulate_basis(np.array([0.0, 0.5, 1.0, 2.5, 3.0]))
    # The recurrence is convex, so each of the `degree + 1` non-zero values carries
    # at most a few ulps and their sum at most one more rounding per addition: an
    # absolute bound of `(degree + 2) * eps` covers both, with no fitted constant.
    partition = (space.degree + 2) * float(np.finfo(np.float64).eps)
    np.testing.assert_allclose(np.asarray(basis).sum(axis=-1), 1.0, rtol=0.0, atol=partition)


def test_a_vector_that_fails_two_rules_is_told_about_the_interval_first() -> None:
    # The order is the content of the decision: a mesh with no interval at all is
    # the more useful diagnosis, so this check runs after that one and never
    # shadows it.
    with pytest.raises(ValueError, match="spans no interval"):
        BsplineSpace1D(np.zeros(8), 2)
