"""Regression tests for the Cox-de Boor denominator guard (issue #257).

The ``BasisFuncs``/``DerBasisFuncs`` kernels in ``_bspline_basis_core.py`` guard the
Cox-de Boor recurrence denominator against near-zero knot differences. The guard now
compares the denominator to ``tol`` scaled by the knot vector's parametric span
(``_scaled_denom_tol``), so it is invariant under affine reparametrization (shift and
scale) of the knot vector, matching the fact that B-spline basis functions themselves
are affine-invariant.

Note on the (scale, shift) grid below: combining the smallest scale (``1e-12``) with
the largest shift (``1e3``) is deliberately *not* tested as a single combination.
At that combination the knot span (``~1e-12``) is only a few float64 ULPs wide at
magnitude ``1e3`` (ULP there is ``~2.2e-13``), so the knot *values* themselves cannot
be represented distinctly -- a float64 representability limit, not a property of the
guard being tested here. Scale and shift invariance are therefore each tested at a
grid where the other axis does not erode representability.
"""

import numpy as np
import pytest

from pantr.bspline import BsplineSpace1D
from pantr.bspline._bspline_basis_core import _compute_basis_nurbs_book_serial_impl
from pantr.tolerance import get_strict


class TestScaleDependentDenominatorGuard:
    """Former known-bug regression (issue #257): the guard is now affine-invariant."""

    def test_tiny_domain_partition_of_unity(self) -> None:
        """A tiny-but-nonzero domain must still satisfy the partition of unity.

        Knot spans are scaled down to ``1e-16`` while ``tol`` stays at the float64
        strict preset (``1e-15``, see :func:`pantr.tolerance.get_strict`). Every
        local knot span is well below the *unscaled* ``tol`` even though it is not
        zero; before the fix this made the guard incorrectly collapse every
        Cox-de Boor contribution, so the basis functions no longer summed to one.
        """
        scale = 1e-16
        knots = np.array([0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0], dtype=np.float64) * scale
        degree = 2
        tol = get_strict(np.float64)
        pts = np.array([0.4 * scale], dtype=np.float64)

        basis = np.empty((1, degree + 1), dtype=np.float64)
        first_basis = np.empty(1, dtype=np.int_)
        _compute_basis_nurbs_book_serial_impl(knots, degree, False, tol, pts, basis, first_basis)

        np.testing.assert_allclose(basis.sum(axis=-1), 1.0, rtol=1e-13)


# ---------------------------------------------------------------------------
# Shared reference knot structures
# ---------------------------------------------------------------------------

# Degree-3 open knot vector with interior double knots at 0.2 and 0.8 (drops
# continuity from C^2 to C^1 there): exercises the general (non-Bezier-like)
# Cox-de Boor kernel.
_REF_DEGREE = 3
_REF_KNOTS = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.4, 0.6, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0],
    dtype=np.float64,
)
_REF_PTS = np.array(
    [
        0.0,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.4,
        0.5,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        1.0,
    ],
    dtype=np.float64,
)

# Degree-3 open uniform, single-span knot vector: triggers the Bézier-like fast
# path (Bernstein evaluation), which never had the "denom < tol" pattern but is
# required to stay affine-invariant too.
_BEZIER_KNOTS = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
_BEZIER_PTS = np.linspace(0.0, 1.0, 11, dtype=np.float64)

_SCALES = [1e-12, 1e-6, 1.0, 1e6, 1e12]


class TestAffineInvarianceBasis:
    """`tabulate_basis` values must be invariant under affine reparametrization."""

    @pytest.mark.parametrize("scale", _SCALES)
    def test_scale_invariance_general_kernel(self, scale: float) -> None:
        """Basis values agree across five orders of magnitude of domain scale."""
        ref = BsplineSpace1D(_REF_KNOTS, _REF_DEGREE)
        basis_ref, first_ref = ref.tabulate_basis(_REF_PTS)

        scaled = BsplineSpace1D(_REF_KNOTS * scale, _REF_DEGREE)
        basis, first = scaled.tabulate_basis(_REF_PTS * scale)

        np.testing.assert_array_equal(first, first_ref)
        np.testing.assert_allclose(basis, basis_ref, rtol=1e-11, atol=1e-11)

    @pytest.mark.parametrize("shift", [0.0, 1e3])
    @pytest.mark.parametrize("scale", [1.0, 1e6])
    def test_shift_invariance_general_kernel(self, scale: float, shift: float) -> None:
        """Basis values are unaffected by translating the knot vector's origin."""
        ref = BsplineSpace1D(_REF_KNOTS, _REF_DEGREE)
        basis_ref, first_ref = ref.tabulate_basis(_REF_PTS)

        shifted = BsplineSpace1D(_REF_KNOTS * scale + shift, _REF_DEGREE)
        basis, first = shifted.tabulate_basis(_REF_PTS * scale + shift)

        np.testing.assert_array_equal(first, first_ref)
        np.testing.assert_allclose(basis, basis_ref, rtol=1e-11, atol=1e-11)

    @pytest.mark.parametrize("scale", _SCALES)
    def test_scale_invariance_bezier_like_fast_path(self, scale: float) -> None:
        """The Bernstein fast path (Bézier-like knots) is also affine-invariant."""
        ref = BsplineSpace1D(_BEZIER_KNOTS, _REF_DEGREE)
        basis_ref, first_ref = ref.tabulate_basis(_BEZIER_PTS)

        scaled = BsplineSpace1D(_BEZIER_KNOTS * scale, _REF_DEGREE)
        assert scaled.has_Bezier_like_knots()
        basis, first = scaled.tabulate_basis(_BEZIER_PTS * scale)

        np.testing.assert_array_equal(first, first_ref)
        np.testing.assert_allclose(basis, basis_ref, rtol=1e-12, atol=1e-12)

    def test_repeated_knots_unchanged(self) -> None:
        """Exact multiplicities on an O(1) domain reproduce known closed-form values.

        Locks down the guard-change requirement that bitwise-repeated knots (the
        common, already-correct case) must not move at all: a degree-2 open
        uniform knot vector reduces to the quadratic Bernstein basis, whose
        values at these points are well-known closed forms.
        """
        space = BsplineSpace1D([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 2)
        pts = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
        basis, _ = space.tabulate_basis(pts)

        t = pts
        expected = np.stack([(1 - t) ** 2, 2 * t * (1 - t), t**2], axis=-1)
        np.testing.assert_allclose(basis, expected, atol=1e-14)


class TestAffineInvarianceDerivatives:
    """`tabulate_basis_derivatives` must be affine-invariant under reparametrization.

    Invariance holds once the chain-rule scaling factor ``(1/scale)^k`` is divided
    back out of the k-th derivative slice.
    """

    @pytest.mark.parametrize("n_deriv", [0, 1, 2, 3])
    @pytest.mark.parametrize("scale", _SCALES)
    def test_scale_invariance_general_kernel(self, scale: float, n_deriv: int) -> None:
        """d^k/dx^k agrees across scales after undoing the (1/scale)^k chain-rule factor."""
        ref = BsplineSpace1D(_REF_KNOTS, _REF_DEGREE)
        deriv_ref, first_ref = ref.tabulate_basis_derivatives(_REF_PTS, n_deriv)

        scaled = BsplineSpace1D(_REF_KNOTS * scale, _REF_DEGREE)
        deriv, first = scaled.tabulate_basis_derivatives(_REF_PTS * scale, n_deriv)

        # d^k/dx^k f(scale*u) = scale^k * d^k/du^k f(u): divide the scale factor
        # back out before comparing to the reference (scale=1) derivatives.
        corrected = deriv * (scale ** np.arange(n_deriv + 1))[None, :, None]

        np.testing.assert_array_equal(first, first_ref)
        np.testing.assert_allclose(corrected, deriv_ref, rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize("n_deriv", [0, 1, 2, 3])
    @pytest.mark.parametrize("scale", _SCALES)
    def test_scale_invariance_bezier_like_fast_path(self, scale: float, n_deriv: int) -> None:
        """Same chain-rule-corrected invariance check for the Bernstein fast path."""
        ref = BsplineSpace1D(_BEZIER_KNOTS, _REF_DEGREE)
        deriv_ref, first_ref = ref.tabulate_basis_derivatives(_BEZIER_PTS, n_deriv)

        scaled = BsplineSpace1D(_BEZIER_KNOTS * scale, _REF_DEGREE)
        assert scaled.has_Bezier_like_knots()
        deriv, first = scaled.tabulate_basis_derivatives(_BEZIER_PTS * scale, n_deriv)

        corrected = deriv * (scale ** np.arange(n_deriv + 1))[None, :, None]

        np.testing.assert_array_equal(first, first_ref)
        np.testing.assert_allclose(corrected, deriv_ref, rtol=1e-9, atol=1e-9)


class TestConsistencyWithMultiplicity:
    """The kernel guard must agree with the space's own multiplicity computation."""

    def test_near_duplicate_knot_pair_matches_space_multiplicity(self) -> None:
        """Two knots 0.5*tol apart: snapping merges them, so the kernel must too.

        With an interior knot pair separated by half the (float64, strict) merge
        tolerance, :meth:`BsplineSpace1D.get_unique_knots_and_multiplicity` groups
        them into a single knot of multiplicity 2 (construction-time snapping
        makes them bitwise identical). Basis evaluation must then be fully
        consistent with that: no NaN/Inf, and the partition of unity holds at and
        around the (now doubled) knot.
        """
        tol = get_strict(np.float64)
        degree = 2
        knots = np.array(
            [0.0, 0.0, 0.0, 0.5, 0.5 + 0.5 * tol, 0.75, 1.0, 1.0, 1.0], dtype=np.float64
        )
        space = BsplineSpace1D(knots, degree)

        unique_knots, mult = space.get_unique_knots_and_multiplicity()
        idx = int(np.argmin(np.abs(unique_knots - 0.5)))
        expected_multiplicity = 2
        assert mult[idx] == expected_multiplicity
        assert space.knots[3] == space.knots[4]  # snapped to bitwise-identical

        pts = np.array([0.4, 0.5, 0.5 + 0.25 * tol, 0.5 + 0.5 * tol, 0.6], dtype=np.float64)
        basis, _ = space.tabulate_basis(pts)

        assert np.all(np.isfinite(basis))
        np.testing.assert_allclose(basis.sum(axis=-1), 1.0, atol=1e-13)
