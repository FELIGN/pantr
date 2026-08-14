"""Knot-layer predicates must use an absolute tolerance, not ``np.isclose``.

Every tolerance in :mod:`pantr.tolerance` is documented as **absolute**: a fixed
value per dtype, independent of the magnitude of the values compared. But
``np.isclose(a, b, atol=tol)`` keeps numpy's default ``rtol=1e-5``, so the test it
actually performs is ``|a - b| <= tol + 1e-5 * |b|``. Every knot-layer predicate
written that way therefore widened by ``1e-5`` times the operand magnitude, which
is harmless on a unit domain and catastrophic on a large one.

Two further properties of the leak are worth pinning, because they are what makes
it hard to spot:

* the ``rtol`` leg attaches to the **second** operand only, so the predicate is
  asymmetric in its arguments, and a comparison against an operand that happens to
  be ``0.0`` shows no leak at all;
* the widening is *relative*, so a test on a ``[0, 1]`` domain cannot see it.

Each test below therefore drives the same construction at two or three coordinate
scales and asserts the verdict is scale-invariant. The perturbations are hardcoded
at ``1e-6`` and ``1e-7`` relative, i.e. inside the leaked ``1e-5`` window and far
outside the strict float64 tolerance of ``1e-15``, so a leaky predicate answers
the perturbed case exactly as it answers the exact one.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from numpy import typing as npt

from pantr.bspline import (
    Bspline,
    BsplineSpace,
    BsplineSpace1D,
    THBSplineSpace,
    create_uniform_space,
)
from pantr.bspline._bspline_knots import (
    _find_knot_index_and_multiplicity,
    _is_in_domain_impl,
)
from pantr.bspline._bspline_product import _get_boundary_mults
from pantr.bspline._bspline_restrict import (
    _compute_boundary_knots_to_insert,
    _validate_restrict_bounds,
)
from pantr.grid import hierarchical_grid, uniform_grid

# Coordinate scales exercised throughout. 1.0 is the regime where the leak is
# invisible; 1e6 is where a 1e-5 relative window is 10.0 wide in absolute terms.
SCALES = (1.0, 100.0, 1e6)

# Relative perturbation sitting strictly inside the leaked ``rtol=1e-5`` window and
# far outside the strict float64 knot tolerance (1e-15).
INSIDE_LEAK = 1e-6


class TestMultiplicityCounting:
    """Knot-multiplicity counts must not absorb a near-but-distinct knot."""

    @pytest.mark.parametrize("base", [1.0, 1e6])
    def test_periodic_num_basis_ignores_near_first_knot(self, base: float) -> None:
        """A periodic space's ``num_basis`` must not depend on a ``1e-6``-relative gap.

        ``_get_multiplicity_of_first_knot_in_domain_impl`` counts how many of
        ``knots[:degree+1]`` equal ``knots[degree]``. The leak made two knots a
        relative ``1e-6`` below it count as equal, raising the multiplicity from 1
        to 3, driving the periodic regularity correction from ``degree - 1 == 1``
        to ``-1``, and reporting ``num_basis == 8`` instead of 6.
        """
        knots = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]) * base
        gap = INSIDE_LEAK * base
        knots[0] -= gap
        knots[1] -= gap

        space = BsplineSpace1D(knots, 2, periodic=True, snap_knots=False)

        # len(knots) - degree - 1 == 8 raw, minus (regularity + 1) == 2.
        assert space.num_basis == 6

    @pytest.mark.parametrize("base", SCALES)
    def test_has_open_knots_rejects_a_near_clamped_vector(self, base: float) -> None:
        """A knot vector that is not clamped must not read as clamped at any scale.

        At base 1e6 the leaked window is 10.0 wide, so gaps of a full unit read as
        "the first ``degree + 1`` knots are equal" -- and at a gap of ``0.5 * base``
        it would have accepted half the domain length.
        """
        knots = np.array([1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0]) * base
        knots[0] -= INSIDE_LEAK * base

        space = BsplineSpace1D(knots, 2, snap_knots=False)

        assert space.has_open_knots() is False

    @pytest.mark.parametrize("base", SCALES)
    def test_has_open_knots_accepts_an_exactly_clamped_vector(self, base: float) -> None:
        """The tightened predicate must still accept a genuinely clamped vector."""
        knots = np.array([1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0]) * base

        space = BsplineSpace1D(knots, 2, snap_knots=False)

        assert space.has_open_knots() is True

    @pytest.mark.parametrize("base", SCALES)
    def test_boundary_multiplicities_ignore_a_near_boundary_knot(self, base: float) -> None:
        """``_get_boundary_mults`` must report the true multiplicity at any scale.

        Shared by the product-space knot construction and, through the same
        pattern, by ``to_open_bspline`` / ``to_periodic``.
        """
        knots = np.array([1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0]) * base
        knots[0] -= INSIDE_LEAK * base
        knots[-1] += INSIDE_LEAK * base
        space = BsplineSpace1D(knots, 2, snap_knots=False)

        m_left, m_right = _get_boundary_mults(space, space.tolerance)

        assert (m_left, m_right) == (2, 2)


class TestDomainGate:
    """The in-domain gate must be an absolute, argument-order-independent window."""

    @pytest.mark.parametrize("hi", SCALES)
    def test_overshoot_inside_the_leaked_window_is_rejected(self, hi: float) -> None:
        """A point overshooting the domain end must be rejected at any scale.

        The leak attached ``rtol`` to the second operand, which was ``knot_end`` at
        the right end, so overshoot of ``1e-5 * knot_end`` was accepted: 1.0e-5 on
        ``[0, 1]``, 1.0e-3 on ``[0, 100]`` and 10.0 on ``[0, 1e6]``.
        """
        knots = np.array([0.0, 0.0, 0.0, hi, hi, hi], dtype=np.float64)
        tol = 1e-10
        pts = np.array([hi + INSIDE_LEAK * hi, hi + 10.0])

        assert not np.any(_is_in_domain_impl(knots, 2, pts, tol))

    @pytest.mark.parametrize("hi", SCALES)
    def test_gate_is_symmetric_in_undershoot_and_overshoot(self, hi: float) -> None:
        """The window must be exactly ``tol`` wide at both ends.

        The old asymmetry was an accident of argument order: the left end compared
        ``knot_begin`` against ``pts`` and the right end ``pts`` against
        ``knot_end``, so undershoot at a domain starting at 0 was judged on a bare
        ``atol`` while overshoot at the far end got the full relative window.
        """
        knots = np.array([0.0, 0.0, 0.0, hi, hi, hi], dtype=np.float64)
        tol = 1e-10

        just_in = np.array([-0.5 * tol, hi + 0.5 * tol])
        just_out = np.array([-2.0 * tol, hi + 2.0 * tol])

        np.testing.assert_array_equal(_is_in_domain_impl(knots, 2, just_in, tol), [True, True])
        np.testing.assert_array_equal(_is_in_domain_impl(knots, 2, just_out, tol), [False, False])

    @pytest.mark.parametrize("hi", SCALES)
    def test_evaluate_rejects_an_out_of_domain_point(self, hi: float) -> None:
        """The public evaluation path inherits the tightened gate."""
        knots = np.array([0.0, 0.0, 0.0, hi, hi, hi], dtype=np.float64)
        spline = Bspline(
            BsplineSpace([BsplineSpace1D(knots, 2)]),
            np.array([[0.0], [1.0], [0.0]]),
        )

        with pytest.raises(ValueError, match="outside the knot vector domain"):
            spline.evaluate(np.array([hi + INSIDE_LEAK * hi]))


class TestCardinalClassification:
    """A near-uniform knot vector is not a uniform one."""

    # ``knots[9]`` is moved. The kernel judges interval ``k`` (0-based, over the
    # 8 in-domain intervals of the vector below) on the ``2 * degree - 1 == 5``
    # knot-interval window ``knots[1 + k : 7 + k]``, which contains index 9 exactly
    # for ``k >= 3``. So the first three intervals stay cardinal and the rest do not
    # -- a localized pattern, which a blanket all-``True`` or all-``False`` answer
    # cannot imitate.
    EXPECTED: ClassVar[list[bool]] = [True, True, True, False, False, False, False, False]

    @staticmethod
    def _perturbed_uniform(rel: float) -> BsplineSpace1D:
        """Degree-3 uniform space on ``[3, 11]`` with ``knots[9]`` moved by ``rel``."""
        knots = np.arange(0.0, 15.0)
        knots[9] += rel
        return BsplineSpace1D(knots, 3, snap_knots=False)

    def test_uniform_knots_are_all_cardinal(self) -> None:
        """Baseline: the unperturbed uniform vector really is cardinal throughout."""
        space = self._perturbed_uniform(0.0)

        np.testing.assert_array_equal(space.get_cardinal_intervals(), [True] * 8)

    @pytest.mark.parametrize("rel", [1e-7, INSIDE_LEAK, 3e-6])
    def test_near_uniform_knots_are_not_cardinal(self, rel: float) -> None:
        """Moving one knot by ``rel`` must un-cardinal every window that sees it.

        The leak compared ``lengths`` against a reference of magnitude 1, so the
        leaked window was ``1e-5`` wide and every ``rel`` here was absorbed: all
        eight intervals came back cardinal.
        """
        space = self._perturbed_uniform(rel)

        np.testing.assert_array_equal(space.get_cardinal_intervals(), self.EXPECTED)

    @pytest.mark.parametrize("rel", [1e-7, INSIDE_LEAK, 3e-6])
    def test_near_uniform_operators_are_not_replaced_by_the_identity(self, rel: float) -> None:
        """A misclassified interval had its exact operator discarded for ``np.eye``.

        ``_tabulate_Bspline_cardinal_1D_extraction_impl`` computes the exact
        operator and then overwrites it with the identity on every interval flagged
        cardinal, so a misclassification did not merely mislabel: it substituted a
        wrong operator for a correct one that had already been computed two lines
        earlier.
        """
        space = self._perturbed_uniform(rel)
        ops = space.tabulate_cardinal_extraction_operators()
        identity = np.eye(4)

        is_identity = [bool(np.array_equal(ops[i], identity)) for i in range(ops.shape[0])]
        assert is_identity == self.EXPECTED

    def test_uniform_operators_are_the_identity(self) -> None:
        """Baseline: a genuinely cardinal interval does get the identity."""
        space = self._perturbed_uniform(0.0)
        ops = space.tabulate_cardinal_extraction_operators()

        for i in range(ops.shape[0]):
            np.testing.assert_array_equal(ops[i], np.eye(4))


class TestKnotLookup:
    """Locating a knot by value must not silently pick a neighbour."""

    @pytest.mark.parametrize("base", SCALES)
    def test_near_miss_is_not_matched(self, base: float) -> None:
        """A query a relative ``1e-6`` away from every knot must not be found.

        The leak matched a knot up to ``1e-5 * |knot_value|`` away, so
        ``remove_knots(v)`` removed a *different* knot than requested: up to
        1.0e-5 away at a knot of 1.0, and 10.0 away at a knot of 1e6.
        """
        knots = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0]) * base
        query = 1.0 * base + INSIDE_LEAK * base

        with pytest.raises(ValueError, match="not found in knot vector"):
            _find_knot_index_and_multiplicity(knots, 2, query, 1e-15)

    @pytest.mark.parametrize("base", SCALES)
    def test_exact_hit_is_matched(self, base: float) -> None:
        """The tightened lookup must still find the knot it is given exactly."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0]) * base

        r, s = _find_knot_index_and_multiplicity(knots, 2, 1.0 * base, 1e-15)

        assert (r, s) == (3, 1)

    @pytest.mark.parametrize("base", SCALES)
    def test_remove_knots_rejects_a_near_miss(self, base: float) -> None:
        """Public ``remove_knots`` surfaces the tightened lookup as an error."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0]) * base
        space = BsplineSpace1D(knots, 2, snap_knots=False)
        n = space.num_basis
        ctrl = np.stack([np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n) ** 2], axis=1)
        spline = Bspline(BsplineSpace([space]), ctrl)

        with pytest.raises(ValueError, match="not found in knot vector"):
            spline.remove_knots(1.0 * base + INSIDE_LEAK * base)


class TestSplitMultiplicity:
    """``Bspline.split`` must clamp the two pieces at the split value."""

    @staticmethod
    def _curve(offset: float, degree: int = 2, n_int: int = 6) -> Bspline:
        """Degree-2 curve on ``[offset, offset + n_int]`` with unit knot spacing."""
        knots = np.concatenate(
            [
                np.full(degree + 1, offset),
                np.arange(1.0, n_int) + offset,
                np.full(degree + 1, offset + n_int),
            ]
        )
        space = BsplineSpace1D(knots, degree)
        n = space.num_basis
        ctrl = np.stack([np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n) ** 2], axis=1)
        return Bspline(BsplineSpace([space]), ctrl)

    @pytest.mark.parametrize("offset", [0.0, 1e5, 1e6])
    def test_left_piece_interpolates_its_last_control_point(self, offset: float) -> None:
        """The left piece must end exactly at its own last control point.

        ``_split_bspline_1d_impl`` counts the split value's current multiplicity to
        decide how many knots to insert. The leak inflated that count -- measured 3
        instead of 1 at offset 1e5, 18 at 1e6 and 45 at 1e8 -- so the deficit
        collapsed to zero, no knots were inserted, and the split point kept
        multiplicity 1 instead of ``degree + 1``. The left piece was then not
        clamped there, and stopped interpolating its own last control point (a
        measured 1.22e-2 discrepancy at offset 1e5).
        """
        spline = self._curve(offset)
        value = float(spline.space.spaces[0].domain[0]) + 5.0

        left, _ = spline.split(0, value)

        np.testing.assert_allclose(
            np.asarray(left.evaluate(np.array([value]))),
            np.asarray(left.control_points)[-1],
            atol=1e-12,
            rtol=0.0,
        )

    @pytest.mark.parametrize("offset", [0.0, 1e5, 1e6])
    def test_split_point_reaches_full_multiplicity(self, offset: float) -> None:
        """The split value must end up clamped in both pieces."""
        spline = self._curve(offset)
        value = float(spline.space.spaces[0].domain[0]) + 5.0

        left, right = spline.split(0, value)

        assert left.space.spaces[0].has_open_knots() is True
        assert right.space.spaces[0].has_open_knots() is True


class TestRestrictBounds:
    """Restriction bounds must be snapped and validated absolutely.

    These exercise the Layer-2 helpers rather than ``Bspline.restrict`` itself: the
    public method is independently broken for any bound above roughly 8 in
    magnitude by an unrelated defect at ``_bspline_restrict.py:183-184``, where an
    absolute ``+/- tol`` nudge fed to ``np.searchsorted`` underflows below one ulp
    and the wrong end of a repeated-knot block is selected. That defect predates
    this fix and is untouched by it.
    """

    @staticmethod
    def _knots(base: float) -> npt.NDArray[np.float64]:
        """Clamped degree-2 knot vector on ``[base, 3 * base]``."""
        return np.array([1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0]) * base

    @pytest.mark.parametrize("base", SCALES)
    def test_a_bound_inside_the_domain_is_not_snapped_to_the_end(self, base: float) -> None:
        """A bound a relative ``1e-6`` inside the domain must stay where it is.

        The leak read such a bound as coinciding with the domain endpoint, which
        both snapped it to the wrong value and -- when both bounds snapped -- made
        ``restrict`` raise "Bounds match the full domain and the direction is
        already open" for a perfectly valid request.
        """
        knots = self._knots(base)
        a, b = float(knots[2]), float(knots[-3])
        a_new = a + INSIDE_LEAK * base
        b_new = b - INSIDE_LEAK * base

        snapped_a, snapped_b = _validate_restrict_bounds(knots, 2, 1e-15, a_new, b_new)

        assert snapped_a == a_new
        assert snapped_b == b_new

    @pytest.mark.parametrize("base", SCALES)
    def test_a_bound_at_the_domain_end_is_still_snapped(self, base: float) -> None:
        """The tightened snap must still fire on an exact endpoint."""
        knots = self._knots(base)
        a, b = float(knots[2]), float(knots[-3])

        snapped_a, snapped_b = _validate_restrict_bounds(knots, 2, 1e-15, a, b)

        assert (snapped_a, snapped_b) == (a, b)

    @pytest.mark.parametrize("base", SCALES)
    def test_interior_bounds_still_require_boundary_knots(self, base: float) -> None:
        """Bounds a relative ``1e-6`` inside the domain still need clamping knots.

        With the leak, both bounds were taken to be at the domain ends of an
        already-open vector, so no knots were queued and the caller raised instead.
        """
        knots = self._knots(base)
        a, b = float(knots[2]), float(knots[-3])
        a_new = a + INSIDE_LEAK * base
        b_new = b - INSIDE_LEAK * base

        to_insert = _compute_boundary_knots_to_insert(knots, 2, 1e-15, a_new, b_new)

        # Neither bound coincides with an existing knot, so each needs degree + 1.
        assert to_insert.size == 6
        np.testing.assert_array_equal(to_insert[:3], [a_new] * 3)
        np.testing.assert_array_equal(to_insert[3:], [b_new] * 3)


class TestTHBRootBounds:
    """The THB grid/space domain-consistency check needs a real tolerance."""

    def test_mismatched_root_bounds_are_rejected(self) -> None:
        """A factor-of-two wrong domain must be rejected.

        ``np.allclose`` with no tolerance argument at all applies numpy's defaults
        (``rtol=1e-5``, ``atol=1e-8``). Through the ``atol`` leg, ``[0, 1e-9]``
        against ``[0, 2e-9]`` -- twice the domain -- compared equal.
        """
        root = create_uniform_space(2, 4, domain=(0.0, 1e-9))
        grid = hierarchical_grid(uniform_grid([[0.0, 2e-9]], 4), 2)

        with pytest.raises(ValueError, match="grid root bounds must match"):
            THBSplineSpace(root, grid)

    def test_relatively_mismatched_root_bounds_are_rejected(self) -> None:
        """A 3e-6 relative domain mismatch must be rejected, at any scale."""
        root = create_uniform_space(2, 4, domain=(0.0, 1.0))
        grid = hierarchical_grid(uniform_grid([[0.0, 1.0 + 3e-6]], 4), 2)

        with pytest.raises(ValueError, match="grid root bounds must match"):
            THBSplineSpace(root, grid)

    def test_matching_root_bounds_are_accepted(self) -> None:
        """The tightened check must still accept a consistent pair."""
        root = create_uniform_space(2, 4, domain=(0.0, 1e-9))
        grid = hierarchical_grid(uniform_grid([[0.0, 1e-9]], 4), 2)

        space = THBSplineSpace(root, grid)

        assert space.dim == 1
