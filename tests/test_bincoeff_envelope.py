"""Public paths must not walk off ``_bincoeff``'s exactness envelope.

``_bincoeff`` runs an exact-integer multiplicative recurrence whose largest
intermediate is ``C(n, k) * min(k, n - k)``. That fits in ``int64`` for every ``k``
up to ``n = 61`` (6.98e18 against a ceiling of 9.22e18) and overflows from
``n = 62`` (1.44e19). Numba does not trap integer overflow, so past the envelope a
coefficient wraps silently: ``_bincoeff(62, 31)`` returned ``-1.296e17`` where the
exact value is ``4.654e17``.

The kernel documented that envelope and asserted callers stayed inside it, but
nothing enforced it, and five public entry points reached it uncapped. Degree
elevation and composition are *value-preserving* operations, so the corruption
showed up as silent geometry corruption rather than as an error.

Two of the five reach the envelope without the caller asking for any degree change
at all:

* every derivative of a **rational** B-spline re-elevates internally, so a
  degree-62 NURBS was corrupted by ``derivative()`` alone;
* ``Bezier.minimize_degree`` re-elevates each trial to score it, so a degree-62
  Bézier scored its own round-trip against corrupted coefficients.

And ``Bezier.compose`` has a **multiplicative** envelope, ``sum(outer.degree) *
inner.degree[0]``, which binds at far lower operand degrees than the additive one:
an outer of degree 6 with an inner of degree 11 already asks for ``C(66, k)``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy import typing as npt

from pantr.bezier import Bezier
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
from pantr.bspline._bspline_degree_core import _BINCOEFF_MAX_N, _bincoeff

# One inside the envelope, one outside. Both are exercised at every entry point so a
# test cannot pass by rejecting everything.
INSIDE = _BINCOEFF_MAX_N
OUTSIDE = _BINCOEFF_MAX_N + 1

ENVELOPE_MSG = "beyond the largest upper index"


def _bernstein_line(degree: int) -> npt.NDArray[np.float64]:
    """Control points of the identity map ``f(x) = x`` in the degree-``n`` Bernstein basis."""
    return (np.arange(degree + 1, dtype=np.float64) / degree).reshape(-1, 1)


class TestKernelEnvelope:
    """The envelope constant must match where the kernel actually breaks."""

    def test_exact_up_to_the_declared_bound(self) -> None:
        """``_bincoeff`` agrees with ``math.comb`` for every ``k`` at the bound."""
        for k in range(_BINCOEFF_MAX_N + 1):
            assert _bincoeff(_BINCOEFF_MAX_N, k) == float(math.comb(_BINCOEFF_MAX_N, k))

    def test_wraps_one_past_the_declared_bound(self) -> None:
        """One past the bound a coefficient wraps, and does so negative.

        This is what makes the envelope load-bearing rather than cosmetic: the
        failure is a sign flip, not a rounding error, so it propagates as gross
        geometry corruption.

        The ``errstate`` is deliberate and is itself part of the point: with
        ``NUMBA_DISABLE_JIT=1`` the recurrence runs on numpy scalars, which do warn on
        overflow, but the compiled kernel this test exists for wraps in silence.
        """
        with np.errstate(over="ignore"):
            wrapped = _bincoeff(OUTSIDE, OUTSIDE // 2)

        assert wrapped < 0.0
        assert wrapped != float(math.comb(OUTSIDE, OUTSIDE // 2))

    def test_bound_is_the_int64_limit_not_the_float_limit(self) -> None:
        """The declared bound is the exact-integer limit (61), not the float one (56).

        Callers only ever consume ``_bincoeff`` inside floating-point *ratios*, so a
        correctly rounded operand is all they can use; capping at the float-lossless
        limit would reject usable degrees for no gain.
        """
        int64_max = 2**63 - 1
        worst = max(
            math.comb(_BINCOEFF_MAX_N, k) * min(k, _BINCOEFF_MAX_N - k)
            for k in range(_BINCOEFF_MAX_N + 1)
        )
        worst_past = max(math.comb(OUTSIDE, k) * min(k, OUTSIDE - k) for k in range(OUTSIDE + 1))

        assert worst <= int64_max
        assert worst_past > int64_max
        # And the bound is genuinely past the float-lossless limit of 56.
        assert max(math.comb(_BINCOEFF_MAX_N, k) for k in range(_BINCOEFF_MAX_N + 1)) > 2**53


class TestBsplineElevateDegree:
    """``Bspline.elevate_degree`` must be value-preserving or refuse."""

    @staticmethod
    def _quadratic() -> Bspline:
        """Degree-2 Bézier-like B-spline, so elevation has a closed-form check."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        ctrl = np.array([[0.0], [0.5], [1.0]])
        return Bspline(BsplineSpace([BsplineSpace1D(knots, 2)]), ctrl)

    def test_elevation_inside_the_envelope_preserves_values(self) -> None:
        """Elevating to the envelope bound must not move the curve."""
        curve = self._quadratic()
        pts = np.array([0.3, 0.5, 0.7])

        elevated = curve.elevate_degree((INSIDE - 2,))

        assert elevated.degree == (INSIDE,)
        np.testing.assert_allclose(
            np.asarray(elevated.evaluate(pts)), np.asarray(curve.evaluate(pts)), atol=1e-13
        )

    def test_elevation_past_the_envelope_raises(self) -> None:
        """One increment further silently corrupted the curve by O(1); now it raises.

        Measured before the cap: elevating a degree-2 curve to degree 62 moved its
        values at ``[0.3, 0.5, 0.7]`` by 1.023, on a curve whose whole range is
        ``[0, 1]``.
        """
        curve = self._quadratic()

        with pytest.raises(ValueError, match=ENVELOPE_MSG):
            curve.elevate_degree((OUTSIDE - 2,))

    def test_the_cap_names_the_offending_direction(self) -> None:
        """A 2D spline reports which direction overflowed."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        space = BsplineSpace([BsplineSpace1D(knots, 2), BsplineSpace1D(knots, 2)])
        surface = Bspline(space, np.zeros((3, 3, 1)))

        with pytest.raises(ValueError, match="direction 1"):
            surface.elevate_degree((0, OUTSIDE - 2))


class TestRationalDerivative:
    """A rational B-spline's derivative re-elevates internally, with no opt-in."""

    @staticmethod
    def _rational_line(degree: int) -> Bspline:
        """Degree-``n`` rational B-spline with unit weights, representing ``f(x) = x``."""
        knots = np.concatenate([np.zeros(degree + 1), np.ones(degree + 1)])
        coords = np.arange(degree + 1, dtype=np.float64) / degree
        ctrl = np.stack([coords, np.ones(degree + 1)], axis=1)
        return Bspline(BsplineSpace([BsplineSpace1D(knots, degree)]), ctrl, is_rational=True)

    def test_derivative_inside_the_envelope_is_correct(self) -> None:
        """The derivative of the identity map is 1 everywhere."""
        curve = self._rational_line(INSIDE)

        deriv = curve.derivative(0)

        np.testing.assert_allclose(
            np.asarray(deriv.evaluate(np.array([0.3, 0.5, 0.7]))), 1.0, atol=1e-8
        )

    def test_derivative_past_the_envelope_raises(self) -> None:
        """``derivative()`` on a degree-62 NURBS asked for corrupted coefficients.

        No ``keep_degree`` and no elevation requested: being rational is enough to
        route through the degree-elevation kernel.
        """
        curve = self._rational_line(OUTSIDE)

        with pytest.raises(ValueError, match=ENVELOPE_MSG):
            curve.derivative(0)

    def test_nonrational_derivative_is_unaffected(self) -> None:
        """A non-rational derivative lowers the degree and needs no coefficients."""
        knots = np.concatenate([np.zeros(OUTSIDE + 1), np.ones(OUTSIDE + 1)])
        ctrl = _bernstein_line(OUTSIDE)
        curve = Bspline(BsplineSpace([BsplineSpace1D(knots, OUTSIDE)]), ctrl)

        deriv = curve.derivative(0)

        assert deriv.degree == (OUTSIDE - 1,)


class TestBezierElevateDegree:
    """``Bezier.elevate_degree`` shares the elevation envelope."""

    def test_elevation_inside_the_envelope_preserves_values(self) -> None:
        """Elevating the identity map to the bound leaves it the identity."""
        curve = Bezier(np.array([[0.0], [1.0]]))
        pts = np.array([0.3, 0.5, 0.7])

        elevated = curve.elevate_degree((INSIDE - 1,))

        assert elevated.degree == (INSIDE,)
        np.testing.assert_allclose(np.asarray(elevated.evaluate(pts)), pts, atol=1e-13)

    def test_elevation_past_the_envelope_raises(self) -> None:
        """Measured before the cap: an error of 1.023 on a curve of range ``[0, 1]``."""
        curve = Bezier(np.array([[0.0], [1.0]]))

        with pytest.raises(ValueError, match=ENVELOPE_MSG):
            curve.elevate_degree((OUTSIDE - 1,))


class TestBezierMinimizeDegree:
    """``Bezier.minimize_degree`` re-elevates each trial to score it."""

    def test_minimization_inside_the_envelope_reduces_a_line(self) -> None:
        """A degree-61 representation of a straight line reduces to degree 1."""
        curve = Bezier(_bernstein_line(INSIDE))

        assert curve.minimize_degree().degree == (1,)

    def test_minimization_past_the_envelope_raises(self) -> None:
        """Before the cap this silently failed to reduce at all.

        The round-trip error measure was computed from wrapped coefficients, so the
        exact reduction of a straight line was rejected and degree 62 came back
        unchanged. The verdict itself was untrustworthy, not merely conservative,
        which is why this raises rather than degrading quietly.
        """
        curve = Bezier(_bernstein_line(OUTSIDE))

        with pytest.raises(ValueError, match=ENVELOPE_MSG):
            curve.minimize_degree()


class TestBezierCompose:
    """``Bezier.compose``'s envelope is multiplicative, not additive."""

    @staticmethod
    def _identity(degree: int) -> Bezier:
        """Identity map ``t -> t`` as a degree-``n`` Bézier."""
        return Bezier(_bernstein_line(degree))

    @pytest.mark.parametrize(
        ("outer_degree", "inner_degree"),
        [(INSIDE, 1), (2, 30), (3, 20)],
    )
    def test_composition_inside_the_envelope_is_the_identity(
        self, outer_degree: int, inner_degree: int
    ) -> None:
        """Identity composed with identity is the identity, exactly."""
        pts = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        composed = self._identity(outer_degree).compose(self._identity(inner_degree))

        assert composed.degree == (outer_degree * inner_degree,)
        np.testing.assert_allclose(np.asarray(composed.evaluate(pts)), pts, atol=1e-12)

    @pytest.mark.parametrize(
        ("outer_degree", "inner_degree", "measured_error"),
        [(2, 31, 1.010), (3, 21, 2.898), (6, 11, 30.4), (OUTSIDE, 1, 1.010)],
    )
    def test_composition_past_the_envelope_raises(
        self, outer_degree: int, inner_degree: int, measured_error: float
    ) -> None:
        """The product of the degrees is what binds, not either degree alone.

        ``(2, 31)``, ``(3, 21)`` and ``(6, 11)`` all have small operand degrees and all
        exceed the envelope. ``measured_error`` records what the uncapped code produced
        composing the identity with the identity, on a curve whose range is ``[0, 1]``.
        """
        assert measured_error > 1.0  # documents the magnitude; not itself computed here

        with pytest.raises(ValueError, match=ENVELOPE_MSG):
            self._identity(outer_degree).compose(self._identity(inner_degree))

    @pytest.mark.parametrize("inner_degree", [INSIDE, OUTSIDE, 70])
    def test_a_degree_one_1d_outer_is_never_capped(self, inner_degree: int) -> None:
        """A 1D outer of degree 1 forms no Bernstein product, so it needs no cap.

        The power ladder in ``_compute_scalar_powers`` only multiplies from degree 2 up,
        and with a single direction there is no cross-direction product either, so
        ``_bincoeff`` is never called however large the inner degree is. Verified
        against the uncapped code: degree 1 composed with degree 70 reproduced the
        identity to 1.0e-15. Capping this would reject a correct operation.
        """
        pts = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        composed = self._identity(1).compose(self._identity(inner_degree))

        assert composed.degree == (inner_degree,)
        np.testing.assert_allclose(np.asarray(composed.evaluate(pts)), pts, atol=1e-12)

    def test_a_degree_one_2d_outer_is_capped(self) -> None:
        """A degree-(1, 1) *2D* outer is not exempt: the cross-direction product runs.

        Measured off by 2.019 at a composed degree of 62, against an exact answer of
        ``2t``, so the exemption above is genuinely about a missing product rather than
        about the operand degree being small.
        """
        u = np.arange(2, dtype=np.float64)
        outer = Bezier((u[:, None] + u[None, :])[..., None])
        coords = np.arange(32, dtype=np.float64) / 31
        inner = Bezier(np.stack([coords, coords], axis=1))

        with pytest.raises(ValueError, match=ENVELOPE_MSG):
            outer.compose(inner)
