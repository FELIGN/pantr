"""A B-spline space with no interval is refused at construction (issue #320).

``BsplineSpace1D`` used to accept a knot vector whose domain is a single point --
``BsplineSpace1D(np.full(4, 5.0), 1)`` constructed, reported ``num_intervals == 0``
and a domain of zero extent -- and every consumer then met the degeneracy on its
own terms. Measured at ``4d1c28a``:

===============================================  ==================================
entry point                                      behaviour before the fix
===============================================  ==================================
``pantr.grid.tensor_product_grid``               ``ValueError`` naming the
                                                 breakpoints
``Bspline.evaluate`` away from the point         ``ValueError`` naming the domain
``Bspline.locate``                               ``ValueError`` from ``numpy``'s
                                                 zero-size ``min``
``BsplineSpace1D.tabulate_basis_derivatives``    ``ZeroDivisionError``
``BsplineSpace1D.tabulate_basis``                returned ``[[nan, nan]]``
``Bspline.evaluate`` *at* the point              returned ``0.0``
``Bspline.evaluate_derivatives`` at it           returned ``nan``
===============================================  ==================================

The last three are why the decision went to the constructor rather than to each
consumer: a guard added per entry point has to be added to every one of them, and
the three that returned a value would not have been found by surveying what
raises. One rule at the one place a space comes into being covers all of them, and
it matches ``create_cardinal_knots``, which has always refused
``num_intervals < 1``.

The ticket reported ``tabulate_basis`` raising ``AttributeError: 'int' object has
no attribute 'shape'``. That is unrelated to the degeneracy: its reproduction
called ``space.tabulate_basis(pts, 0)``, and the second positional parameter is
``out_basis``, so the ``0`` was validated as an output array. The same call raises
identically on a healthy space. What ``tabulate_basis`` actually did on a
zero-interval space is the ``NaN`` in the table above, and ``validate=True`` did
not catch it, because the query point does lie in the zero-extent domain.

The predicate is ``num_intervals == 0``, not "every knot is the same value": the
family is wider than the ticket's example. ``BsplineSpace1D([0, 1, 1, 1, 2], 1)``
has a knot vector with three distinct values and a domain of ``(1.0, 1.0)``.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pantr._numba_compat import wait_for_jit_warmup
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
from pantr.bspline._bspline_knots import _knot_tolerance

wait_for_jit_warmup()


TICKET_KNOTS = np.full(4, 5.0)
"""The exact knot vector from the ticket's reproduction."""

TICKET_DEGREE = 1
"""The degree the ticket's reproduction pairs with :data:`TICKET_KNOTS`."""

_NO_INTERVAL = re.compile("spans no interval")
"""Match the new refusal without pinning the whole message."""


def test_the_ticket_reproduction_is_refused_at_construction() -> None:
    """AC1/AC3: the ticket's own space, and the message it must now produce.

    Before the fix this constructed silently and the reproduction went on to fail
    two calls later inside ``numpy``, with ``ValueError: zero-size array to
    reduction operation minimum which has no identity`` -- a message naming
    neither the spline nor the degeneracy.
    """
    with pytest.raises(ValueError, match=_NO_INTERVAL) as excinfo:
        BsplineSpace1D(TICKET_KNOTS, TICKET_DEGREE)

    message = str(excinfo.value)
    # The reader has to be able to act without opening our source: what is wrong,
    # which coordinate the domain collapsed onto, and what to supply instead.
    assert "5.0" in message, message
    assert "degree 1" in message, message
    assert "two consecutive knots" in message, message


def test_the_consumers_the_ticket_named_are_no_longer_reachable() -> None:
    """AC2: every entry point agrees, because none of them can be handed such a space.

    That is the whole content of the decision. ``tabulate_basis`` grows no guard
    of its own and needs none: there is no space to call it on.
    """
    with pytest.raises(ValueError, match=_NO_INTERVAL):
        space = BsplineSpace1D(TICKET_KNOTS, TICKET_DEGREE)
        # Unreachable. Kept so the test states which calls the refusal now stands
        # in front of, rather than only that construction failed.
        spline = Bspline(BsplineSpace([space]), np.array([[0.0], [1.0]]))
        spline.locate(np.array([0.5]))
        space.tabulate_basis(np.array([5.0]))
        space.tabulate_basis_derivatives(np.array([5.0]), 1)
        spline.evaluate(np.array([[5.0]]))
        spline.evaluate_derivatives(np.array([[5.0]]), 1)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("value", [0.0, 1.0, -3.5, 1e6])
@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_a_flat_knot_vector_is_refused_at_every_degree_dtype_and_magnitude(
    degree: int, value: float, dtype: type[Any]
) -> None:
    """The whole of the previously-accepted family, at the shortest length allowed.

    ``2 * degree + 2`` is the minimum ``_validate_input`` requires, so this is the
    smallest flat vector that reaches the new check at each degree. The magnitudes
    span the range over which the snapping tolerance itself varies, and ``0.0`` is
    the one value for which that tolerance is exactly zero.
    """
    knots = np.full(2 * degree + 2, value, dtype=dtype)
    with pytest.raises(ValueError, match=_NO_INTERVAL):
        BsplineSpace1D(knots, degree)


@pytest.mark.parametrize(
    ("knots", "degree"),
    [
        # Distinct knot values, and still no interval: the domain runs from
        # ``knots[degree]`` to ``knots[-degree-1]``, and an interior knot of high
        # enough multiplicity swallows it whole. ``raw[0] == raw[-1]``, the
        # predicate the snapping check uses, is blind to every one of these.
        ([0.0, 1.0, 1.0, 2.0], 1),
        ([0.0, 1.0, 1.0, 1.0, 2.0], 1),
        ([0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0], 2),
    ],
)
def test_an_interior_knot_that_swallows_the_domain_is_refused_too(
    knots: list[float], degree: int
) -> None:
    """The predicate is ``num_intervals == 0``, not "every knot is the same value"."""
    with pytest.raises(ValueError, match=_NO_INTERVAL):
        BsplineSpace1D(np.asarray(knots), degree)


def test_periodic_and_unsnapped_construction_are_covered_as_well() -> None:
    """Neither ``periodic=True`` nor ``snap_knots=False`` is a way back in.

    ``snap_knots=False`` matters because the library uses it internally
    (``_bspline_split``, ``_bspline_restrict``, ``_bspline_roots``); had the check
    ridden on snapping, those paths would have kept the hole open.
    """
    with pytest.raises(ValueError, match=_NO_INTERVAL):
        BsplineSpace1D(np.full(4, 5.0), 1, periodic=True)

    with pytest.raises(ValueError, match=_NO_INTERVAL):
        BsplineSpace1D(np.full(4, 5.0), 1, snap_knots=False)


def test_the_snapping_refusal_still_owns_its_own_case() -> None:
    """The snapping message must not be borrowed for a vector that arrived flat.

    The ticket's Non-goals put the snapping rule out of scope, and the two
    diagnoses are genuinely different: "this mesh is finer than float32 resolves
    here" is actionable, and false of a vector the caller supplied flat. Both
    refuse; each says its own thing.
    """
    degree = 2
    lo, hi = 1e6, 1e6 + 1.0
    collapsing = np.asarray(
        [lo] * (degree + 1) + [lo + 0.5] + [hi] * (degree + 1), dtype=np.float32
    )
    with pytest.raises(ValueError, match="collapsed every knot"):
        BsplineSpace1D(collapsing, degree)

    with pytest.raises(ValueError, match=_NO_INTERVAL):
        BsplineSpace1D(np.full(8, 1e6, dtype=np.float32), degree)


def test_the_message_stays_true_when_the_domain_ends_are_not_close() -> None:
    """A chain of near knots merges two ends that are not near each other.

    The grouping splits on ``knots[i] - knots[i-1] > tol``, so it joins by steps. In
    float32 at ``1e6`` the tolerance is 0.954, the three domain knots here are 0.5
    apart in turn, and all three land in one class although the ends are 1.0 apart --
    more than the tolerance. So a message reading "these two are the same knot at a
    tolerance of 0.954" would be false, and the remedy it implies ("make them differ
    by more than 0.954") already holds. What the message reports is the step.
    """
    lo, hi = 1e6, 1e6 + 1.0
    raw = np.asarray([lo, lo, lo, lo + 0.5, hi, hi, hi], dtype=np.float32)

    with pytest.raises(ValueError, match=_NO_INTERVAL) as excinfo:
        BsplineSpace1D(raw, 2, snap_knots=False)

    message = str(excinfo.value)
    assert "every step between them is at most" in message, message
    # The premise of the test: the ends are genuinely further apart than the
    # tolerance the message quotes, so a span-based claim would have been wrong.
    assert hi - lo > _knot_tolerance(raw)


def test_a_space_with_a_single_interval_is_untouched() -> None:
    """The ticket's invariant: nothing changes for a space that has an interval.

    One interval is the boundary of the new rule, so it is what would break first
    if the predicate were off by one. Every entry point the ticket listed is
    exercised on it.
    """
    space = BsplineSpace1D(np.array([0.0, 0.0, 1.0, 1.0]), 1)
    assert space.num_intervals == 1
    assert space.domain == (0.0, 1.0)

    values, _ = space.tabulate_basis(np.array([0.0, 0.5, 1.0]))
    basis: npt.NDArray[np.float64] = np.asarray(values, dtype=np.float64)
    assert np.allclose(basis.sum(axis=-1), 1.0)
    space.tabulate_basis_derivatives(np.array([0.5]), 1)

    spline = Bspline(BsplineSpace([space]), np.array([[0.0], [1.0]]))
    assert np.allclose(np.asarray(spline.evaluate(np.array([[0.25]]))), 0.25)
    cell_ids, ref_coords = spline.locate(np.array([[0.25]]))
    assert np.array_equal(np.asarray(cell_ids), np.array([0]))
    assert np.allclose(np.asarray(ref_coords).ravel(), 0.25)
