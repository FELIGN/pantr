"""Tests for mask operations on Bernstein polynomial subcell grids."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bezier import Bezier
from pantr.bezier._mask import (
    _collapse_mask,
    _intersection_mask,
    _line_intersects_mask,
    _mask_empty,
    _nonzero_mask,
    _point_within_mask,
    _restrict_to_face,
)
from pantr.bezier._mask_core import (
    _orthant_test_base_core,
    _orthant_test_core,
    _restrict_scalar_1d,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scalar_bezier_1d(ctrl: list[float]) -> Bezier:
    """Create a 1D scalar Bézier from a flat list of coefficients."""
    return Bezier(np.array(ctrl, dtype=np.float64))


def _scalar_bezier_2d(ctrl: list[list[float]]) -> Bezier:
    """Create a 2D scalar Bézier from a nested list of coefficients."""
    arr = np.array(ctrl, dtype=np.float64)
    return Bezier(arr[:, :, np.newaxis])


def _scalar_bezier_3d(ctrl: npt.NDArray[np.float64]) -> Bezier:
    """Create a 3D scalar Bézier from a 3D coefficient array."""
    return Bezier(ctrl[:, :, :, np.newaxis])


# ===========================================================================
# TestMaskEmpty
# ===========================================================================


class TestMaskEmpty:
    """Tests for _mask_empty."""

    def test_all_false(self) -> None:
        assert _mask_empty(np.zeros(8, dtype=np.bool_))

    def test_all_true(self) -> None:
        assert not _mask_empty(np.ones(8, dtype=np.bool_))

    def test_single_true(self) -> None:
        m = np.zeros((4, 4), dtype=np.bool_)
        m[2, 3] = True
        assert not _mask_empty(m)

    def test_3d_empty(self) -> None:
        assert _mask_empty(np.zeros((2, 2, 2), dtype=np.bool_))


# ===========================================================================
# TestPointWithinMask
# ===========================================================================


class TestPointWithinMask:
    """Tests for _point_within_mask."""

    def test_1d_center_of_subcell(self) -> None:
        m = np.zeros(8, dtype=np.bool_)
        m[3] = True
        # Center of subcell 3 is at x = 3.5/8 = 0.4375.
        assert _point_within_mask(m, np.array([0.4375]), M=8)
        assert not _point_within_mask(m, np.array([0.0625]), M=8)

    def test_2d_center(self) -> None:
        m = np.zeros((4, 4), dtype=np.bool_)
        m[1, 2] = True
        # Center of (1,2) is (1.5/4, 2.5/4) = (0.375, 0.625).
        assert _point_within_mask(m, np.array([0.375, 0.625]), M=4)
        assert not _point_within_mask(m, np.array([0.125, 0.125]), M=4)

    def test_boundary_clamping(self) -> None:
        m = np.zeros(4, dtype=np.bool_)
        m[0] = True
        m[3] = True
        # x=0.0 → cell 0, x=1.0 → cell 3 (clamped).
        assert _point_within_mask(m, np.array([0.0]), M=4)
        assert _point_within_mask(m, np.array([1.0]), M=4)

    def test_wrong_dimension_raises(self) -> None:
        m = np.zeros((4, 4), dtype=np.bool_)
        with pytest.raises(ValueError, match="components"):
            _point_within_mask(m, np.array([0.5]), M=4)


# ===========================================================================
# TestLineIntersectsMask
# ===========================================================================


class TestLineIntersectsMask:
    """Tests for _line_intersects_mask."""

    def test_2d_line_hits(self) -> None:
        m = np.zeros((4, 4), dtype=np.bool_)
        m[1, 2] = True
        # Line along axis 1 at x[0]=0.375 (cell 1) passes through (1, *).
        assert _line_intersects_mask(m, np.array([0.375]), axis=1, M=4)

    def test_2d_line_misses(self) -> None:
        m = np.zeros((4, 4), dtype=np.bool_)
        m[1, 2] = True
        # Line along axis 1 at x[0]=0.875 (cell 3) — row 3 is all False.
        assert not _line_intersects_mask(m, np.array([0.875]), axis=1, M=4)

    def test_1d_line(self) -> None:
        m = np.array([False, True, False, False], dtype=np.bool_)
        # N=1, any non-empty mask means line intersects.
        assert _line_intersects_mask(m, np.empty(0, dtype=np.float64), axis=0, M=4)

    def test_1d_empty(self) -> None:
        m = np.zeros(4, dtype=np.bool_)
        assert not _line_intersects_mask(m, np.empty(0, dtype=np.float64), axis=0, M=4)

    def test_axis_out_of_range_raises(self) -> None:
        m = np.zeros((4, 4), dtype=np.bool_)
        with pytest.raises(ValueError, match="out of range"):
            _line_intersects_mask(m, np.array([0.5]), axis=2, M=4)


# ===========================================================================
# TestCollapseMask
# ===========================================================================


class TestCollapseMask:
    """Tests for _collapse_mask."""

    def test_2d_collapse_axis0(self) -> None:
        m = np.zeros((4, 4), dtype=np.bool_)
        m[2, 1] = True
        result = _collapse_mask(m, axis=0)
        assert result.shape == (4,)
        assert result[1]
        assert not result[0]
        assert not result[2]

    def test_2d_collapse_axis1(self) -> None:
        m = np.zeros((4, 4), dtype=np.bool_)
        m[2, 1] = True
        result = _collapse_mask(m, axis=1)
        assert result.shape == (4,)
        assert result[2]
        assert not result[0]

    def test_3d_collapse(self) -> None:
        m = np.zeros((3, 3, 3), dtype=np.bool_)
        m[1, 2, 0] = True
        result = _collapse_mask(m, axis=2)
        assert result.shape == (3, 3)
        assert result[1, 2]

    def test_1d_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            _collapse_mask(np.zeros(4, dtype=np.bool_), axis=0)

    def test_axis_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            _collapse_mask(np.zeros((4, 4), dtype=np.bool_), axis=2)


# ===========================================================================
# TestRestrictToFace
# ===========================================================================


class TestRestrictToFace:
    """Tests for _restrict_to_face."""

    def test_2d_lower_face_axis0(self) -> None:
        m = np.zeros((4, 4), dtype=np.bool_)
        m[0, 2] = True
        result = _restrict_to_face(m, axis=0, side=0)
        assert result.shape == (4,)
        assert result[2]
        assert not result[0]

    def test_2d_upper_face_axis0(self) -> None:
        m = np.zeros((4, 4), dtype=np.bool_)
        m[3, 1] = True
        result = _restrict_to_face(m, axis=0, side=1)
        assert result.shape == (4,)
        assert result[1]

    def test_2d_lower_face_axis1(self) -> None:
        m = np.zeros((4, 4), dtype=np.bool_)
        m[2, 0] = True
        result = _restrict_to_face(m, axis=1, side=0)
        assert result.shape == (4,)
        assert result[2]

    def test_3d_face(self) -> None:
        m = np.zeros((3, 3, 3), dtype=np.bool_)
        m[1, 2, 2] = True
        result = _restrict_to_face(m, axis=2, side=1)
        assert result.shape == (3, 3)
        assert result[1, 2]

    def test_invalid_side_raises(self) -> None:
        with pytest.raises(ValueError, match="side must be 0 or 1"):
            _restrict_to_face(np.zeros((4, 4), dtype=np.bool_), axis=0, side=2)

    def test_1d_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            _restrict_to_face(np.zeros(4, dtype=np.bool_), axis=0, side=0)


# ===========================================================================
# TestNonzeroMask
# ===========================================================================


class TestNonzeroMask:
    """Tests for _nonzero_mask."""

    def test_1d_all_positive_gives_empty_mask(self) -> None:
        """Polynomial with all-positive coefficients has no zeros."""
        b = _scalar_bezier_1d([1.0, 2.0, 3.0])
        mask = _nonzero_mask(b, M=8)
        assert mask.shape == (8,)
        assert _mask_empty(mask)

    def test_1d_all_negative_gives_empty_mask(self) -> None:
        b = _scalar_bezier_1d([-1.0, -2.0, -3.0])
        mask = _nonzero_mask(b, M=8)
        assert _mask_empty(mask)

    def test_1d_linear_crossing_zero(self) -> None:
        """Linear Bernstein f(x) = -1 + 2x: zero at x=0.5."""
        b = _scalar_bezier_1d([-1.0, 1.0])
        mask = _nonzero_mask(b, M=8)
        assert not _mask_empty(mask)
        # The zero is at x=0.5 → subcell 4. Nearby subcells might also be True
        # (conservative).
        assert mask[4] or mask[3]

    def test_1d_with_input_mask(self) -> None:
        """Input mask restricts the search region."""
        b = _scalar_bezier_1d([-1.0, 1.0])
        # Only look at the first half.
        fmask = np.zeros(8, dtype=np.bool_)
        fmask[:4] = True
        mask = _nonzero_mask(b, mask=fmask, M=8)
        # No True entries in the upper half.
        assert not np.any(mask[4:])

    def test_1d_constant_zero_gives_full_mask(self) -> None:
        """Zero polynomial: all subcells should be True."""
        b = _scalar_bezier_1d([0.0, 0.0])
        mask = _nonzero_mask(b, M=4)
        assert np.all(mask)

    def test_2d_all_positive(self) -> None:
        b = _scalar_bezier_2d([[1.0, 2.0], [3.0, 4.0]])
        mask = _nonzero_mask(b, M=4)
        assert mask.shape == (4, 4)
        assert _mask_empty(mask)

    def test_2d_with_zero_crossing(self) -> None:
        """2D linear: f(x,y) = -1 + 2x → zero at x=0.5 for all y."""
        ctrl = np.array([[[-1.0], [1.0]]], dtype=np.float64)
        # Shape (1, 2, 1) → degree (0, 1), dim=2.
        # Actually this is degree 0 in axis 0, degree 1 in axis 1.
        # f(x,y) = (1-y)*(-1) + y*1 = -1+2y → zero at y=0.5.
        b = Bezier(ctrl)
        mask = _nonzero_mask(b, M=4)
        assert not _mask_empty(mask)

    def test_3d_all_positive(self) -> None:
        ctrl = np.ones((2, 2, 2), dtype=np.float64)
        b = _scalar_bezier_3d(ctrl)
        mask = _nonzero_mask(b, M=4)
        assert mask.shape == (4, 4, 4)
        assert _mask_empty(mask)

    def test_rational_mixed_sign_weights_raises(self) -> None:
        """Rational Bezier with mixed-sign weights is rejected."""
        b = Bezier(np.array([[1.0, 1.0], [2.0, -1.0], [3.0, 1.0]]), is_rational=True)
        with pytest.raises(TypeError, match="same strict sign"):
            _nonzero_mask(b)

    def test_rational_zero_weight_raises(self) -> None:
        """Rational Bezier with a zero weight is rejected."""
        b = Bezier(np.array([[1.0, 1.0], [2.0, 0.0], [3.0, 1.0]]), is_rational=True)
        with pytest.raises(TypeError, match="same strict sign"):
            _nonzero_mask(b)

    def test_rational_positive_weights_all_positive(self) -> None:
        """Rational Bezier with all-positive numerator and positive weights → empty mask."""
        # Numerator coefficients (w_i * c_i) are [1, 4, 9], all positive → no zeros.
        b = Bezier(np.array([[1.0, 1.0], [4.0, 2.0], [9.0, 3.0]]), is_rational=True)
        mask = _nonzero_mask(b, M=8)
        assert _mask_empty(mask)

    def test_rational_negative_weights_all_positive(self) -> None:
        """Rational Bezier with negative weights but all-positive numerator → empty mask."""
        b = Bezier(np.array([[1.0, -1.0], [4.0, -2.0], [9.0, -3.0]]), is_rational=True)
        mask = _nonzero_mask(b, M=8)
        assert _mask_empty(mask)

    def test_rational_with_zero_crossing(self) -> None:
        """Rational Bezier whose numerator crosses zero → non-empty mask."""
        # Numerator coefficients: [-1, 1], weights: [1, 1] → zero at x=0.5.
        b = Bezier(np.array([[-1.0, 1.0], [1.0, 1.0]]), is_rational=True)
        mask = _nonzero_mask(b, M=8)
        assert not _mask_empty(mask)

    def test_vector_raises(self) -> None:
        b = Bezier(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        with pytest.raises(ValueError, match="rank == 1"):
            _nonzero_mask(b)

    def test_dim_gt_3_raises(self) -> None:
        """Dim > 3 is not supported."""
        ctrl = np.ones((2, 2, 2, 2, 1), dtype=np.float64)
        b = Bezier(ctrl)
        with pytest.raises(ValueError, match="dim <= 3"):
            _nonzero_mask(b)


# ===========================================================================
# TestIntersectionMask
# ===========================================================================


class TestIntersectionMask:
    """Tests for _intersection_mask."""

    def test_2d_no_shared_zeros(self) -> None:
        """Two polynomials with no shared zeros."""
        # f > 0 everywhere, g > 0 everywhere.
        f = _scalar_bezier_2d([[1.0, 2.0], [3.0, 4.0]])
        g = _scalar_bezier_2d([[5.0, 6.0], [7.0, 8.0]])
        fmask = np.ones((4, 4), dtype=np.bool_)
        gmask = np.ones((4, 4), dtype=np.bool_)
        result = _intersection_mask(f, fmask, g, gmask, M=4)
        assert result.shape == (4, 4)
        assert _mask_empty(result)

    def test_2d_with_shared_zero(self) -> None:
        """Two polynomials that share a zero."""
        # f(x,y) = -1 + 2x: zero at x=0.5.
        # g(x,y) = -1 + 2y: zero at y=0.5.
        # They intersect at (0.5, 0.5).
        f = _scalar_bezier_2d([[-1.0, -1.0], [1.0, 1.0]])
        g = _scalar_bezier_2d([[-1.0, 1.0], [-1.0, 1.0]])
        fmask = _nonzero_mask(f, M=4)
        gmask = _nonzero_mask(g, M=4)
        result = _intersection_mask(f, fmask, g, gmask, M=4)
        assert not _mask_empty(result)

    def test_2d_disjoint_masks(self) -> None:
        """Masks don't overlap → empty intersection."""
        f = _scalar_bezier_2d([[-1.0, -1.0], [1.0, 1.0]])
        g = _scalar_bezier_2d([[-1.0, 1.0], [-1.0, 1.0]])
        fmask = np.zeros((4, 4), dtype=np.bool_)
        gmask = np.zeros((4, 4), dtype=np.bool_)
        fmask[0, 0] = True
        gmask[3, 3] = True
        result = _intersection_mask(f, fmask, g, gmask, M=4)
        assert _mask_empty(result)

    def test_dimension_mismatch_raises(self) -> None:
        f = _scalar_bezier_1d([1.0, 2.0])
        g = _scalar_bezier_2d([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="Dimension mismatch"):
            _intersection_mask(
                f,
                np.ones(4, dtype=np.bool_),
                g,
                np.ones((4, 4), dtype=np.bool_),
                M=4,
            )

    def test_1d_raises(self) -> None:
        """intersection_mask requires dim >= 2."""
        f = _scalar_bezier_1d([-1.0, 1.0])
        g = _scalar_bezier_1d([1.0, -1.0])
        with pytest.raises(ValueError, match="dim >= 2"):
            _intersection_mask(
                f,
                np.ones(4, dtype=np.bool_),
                g,
                np.ones(4, dtype=np.bool_),
                M=4,
            )

    def test_3d_all_positive(self) -> None:
        """Both polynomials positive everywhere → empty intersection."""
        ctrl_f = np.ones((2, 2, 2), dtype=np.float64)
        ctrl_g = 2.0 * np.ones((2, 2, 2), dtype=np.float64)
        f = _scalar_bezier_3d(ctrl_f)
        g = _scalar_bezier_3d(ctrl_g)
        fmask = np.ones((4, 4, 4), dtype=np.bool_)
        gmask = np.ones((4, 4, 4), dtype=np.bool_)
        result = _intersection_mask(f, fmask, g, gmask, M=4)
        assert _mask_empty(result)

    def test_dim_gt_3_raises(self) -> None:
        """Dim > 3 is not supported."""
        ctrl = np.ones((2, 2, 2, 2, 1), dtype=np.float64)
        f = Bezier(ctrl)
        g = Bezier(ctrl)
        with pytest.raises(ValueError, match="dim <= 3"):
            _intersection_mask(
                f,
                np.ones((4, 4, 4, 4), dtype=np.bool_),
                g,
                np.ones((4, 4, 4, 4), dtype=np.bool_),
            )

    def test_3d_with_shared_zero(self) -> None:
        """Two 3D planes that intersect along a line produce a non-empty mask."""
        # f(x,y,z) = -1 + 2x: zero at x=0.5 for all y,z.
        # g(x,y,z) = -1 + 2y: zero at y=0.5 for all x,z.
        # They share a common zero on the line x=0.5, y=0.5 for all z.
        ctrl_f = np.array([[[[-1.0], [-1.0]], [[-1.0], [-1.0]]], [[[1.0], [1.0]], [[1.0], [1.0]]]])
        ctrl_g = np.array([[[[-1.0], [-1.0]], [[1.0], [1.0]]], [[[-1.0], [-1.0]], [[1.0], [1.0]]]])
        f = Bezier(ctrl_f)
        g = Bezier(ctrl_g)
        fmask = _nonzero_mask(f, M=4)
        gmask = _nonzero_mask(g, M=4)
        result = _intersection_mask(f, fmask, g, gmask, M=4)
        assert result.shape == (4, 4, 4)
        assert not _mask_empty(result)

    def test_non_default_M(self) -> None:
        """Results are consistent when M differs from the default."""
        f = _scalar_bezier_2d([[-1.0, -1.0], [1.0, 1.0]])
        g = _scalar_bezier_2d([[-1.0, 1.0], [-1.0, 1.0]])
        for M in (4, 6, 16):
            fmask = _nonzero_mask(f, M=M)
            gmask = _nonzero_mask(g, M=M)
            result = _intersection_mask(f, fmask, g, gmask, M=M)
            assert result.shape == (M, M)
            assert not _mask_empty(result)


# ===========================================================================
# TestOrthantTest
# ===========================================================================


class TestOrthantTest:
    """Direct tests for _orthant_test_core and _orthant_test_base_core."""

    def test_all_positive_provably_disjoint(self) -> None:
        """Both f and g positive everywhere: orthant test returns True (disjoint)."""
        f = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        g = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        assert _orthant_test_core(f, g)

    def test_opposite_signs_provably_disjoint(self) -> None:
        """Positive f and negative g: their zeros cannot overlap."""
        f = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        g = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
        assert _orthant_test_core(f, g)

    def test_crossing_not_provably_disjoint(self) -> None:
        """Both polynomials change sign: cannot prove disjointness."""
        f = np.array([-1.0, 1.0], dtype=np.float64)
        g = np.array([-1.0, 1.0], dtype=np.float64)
        assert not _orthant_test_core(f, g)

    def test_base_sign_plus_feasible(self) -> None:
        """sign=+1 with all-positive x and y: alpha=0 satisfies x+0*y > 0."""
        x = np.array([1.0, 2.0], dtype=np.float64)
        y = np.array([0.5, 0.5], dtype=np.float64)
        assert _orthant_test_base_core(x, y, 1)

    def test_base_sign_impossible(self) -> None:
        """y=0 and x*sign <= 0 makes the system infeasible."""
        x = np.array([-1.0, 1.0], dtype=np.float64)
        y = np.array([0.0, 1.0], dtype=np.float64)
        # sign=+1: first constraint is x[0]*1 + alpha*0 > 0 → -1 > 0, impossible.
        assert not _orthant_test_base_core(x, y, 1)

    def test_single_element_positive(self) -> None:
        """Single positive element: trivially disjoint."""
        f = np.array([5.0], dtype=np.float64)
        g = np.array([3.0], dtype=np.float64)
        assert _orthant_test_core(f, g)


# ===========================================================================
# TestRestrictScalar1D
# ===========================================================================


class TestRestrictScalar1D:
    """Direct tests for _restrict_scalar_1d covering both numeric branches."""

    def test_full_interval_identity(self) -> None:
        """Restricting to [0, 1] is the identity."""
        coeffs = [1.0, 3.0, 2.0]
        c = np.array(coeffs, dtype=np.float64)
        out = np.empty(len(coeffs), dtype=np.float64)
        _restrict_scalar_1d(c, 0.0, 1.0, out)
        np.testing.assert_allclose(out, coeffs)

    def test_upper_branch(self) -> None:
        """abs(upper) >= abs(lower-1) branch: upper=0.9, lower=0.1."""
        # Linear Bernstein p(t) = t (coeffs [0, 1]): restriction to [lo, hi]
        # gives endpoint values [lo, hi].
        # abs(0.9) >= abs(0.1 - 1) = 0.9 → upper branch taken.
        lower, upper = 0.1, 0.9
        c = np.array([0.0, 1.0], dtype=np.float64)
        out = np.empty(2, dtype=np.float64)
        _restrict_scalar_1d(c, lower, upper, out)
        np.testing.assert_allclose(out[0], lower, atol=1e-14)
        np.testing.assert_allclose(out[1], upper, atol=1e-14)

    def test_lower_branch(self) -> None:
        """abs(lower-1) > abs(upper) branch: upper=0.1, lower=0.01."""
        # abs(0.1) < abs(0.01 - 1) = 0.99 → lower branch taken.
        lower, upper = 0.01, 0.1
        c = np.array([0.0, 1.0], dtype=np.float64)
        out = np.empty(2, dtype=np.float64)
        _restrict_scalar_1d(c, lower, upper, out)
        np.testing.assert_allclose(out[0], lower, atol=1e-14)
        np.testing.assert_allclose(out[1], upper, atol=1e-14)

    def test_quadratic_endpoints(self) -> None:
        """Restricted polynomial has correct endpoint values for a quadratic."""
        # Bernstein degree-2 coefficients [0, 0.5, 1] represent p(t) = t.
        # (p(t) = 0*(1-t)^2 + 0.5*2t(1-t) + 1*t^2 = t.)
        # Restriction to [0.25, 0.75]: out[0] = p(0.25) = 0.25, out[2] = p(0.75) = 0.75.
        c = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        out = np.empty(3, dtype=np.float64)
        _restrict_scalar_1d(c, 0.25, 0.75, out)
        np.testing.assert_allclose(out[0], 0.25, atol=1e-14)
        np.testing.assert_allclose(out[2], 0.75, atol=1e-14)


# ===========================================================================
# TestNonzeroMaskMValues
# ===========================================================================


class TestNonzeroMaskMValues:
    """Tests for _nonzero_mask with non-default M values."""

    def test_m4_crossing(self) -> None:
        """1D zero-crossing detected at M=4."""
        b = _scalar_bezier_1d([-1.0, 1.0])
        mask = _nonzero_mask(b, M=4)
        assert mask.shape == (4,)
        assert not _mask_empty(mask)

    def test_m16_crossing(self) -> None:
        """1D zero-crossing detected at M=16."""
        b = _scalar_bezier_1d([-1.0, 1.0])
        mask = _nonzero_mask(b, M=16)
        assert mask.shape == (16,)
        assert not _mask_empty(mask)
        # With M=16 the mask is tighter: at most a handful of cells near x=0.5.
        assert np.sum(mask) <= 4  # noqa: PLR2004

    def test_m4_all_positive_empty(self) -> None:
        """All-positive polynomial gives empty mask at M=4."""
        b = _scalar_bezier_1d([1.0, 2.0, 3.0])
        assert _mask_empty(_nonzero_mask(b, M=4))

    def test_2d_m6(self) -> None:
        """2D nonzero mask has correct shape at M=6."""
        b = _scalar_bezier_2d([[1.0, 2.0], [3.0, 4.0]])
        mask = _nonzero_mask(b, M=6)
        assert mask.shape == (6, 6)
        assert _mask_empty(mask)
