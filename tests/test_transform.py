"""Tests for the AffineTransform class and Bspline/Bezier.transform()."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.testing as npt
import pytest

from pantr.transform import AffineTransform

if TYPE_CHECKING:
    from pantr.bezier import Bezier
    from pantr.bspline import Bspline

# ======================================================================
# Helpers
# ======================================================================


def _make_bezier_1d(ctrl: list[list[float]], *, is_rational: bool = False) -> Bezier:
    """Create a 1-D Bezier from a list of control points."""
    from pantr.bezier import Bezier as BezierCls  # noqa: PLC0415

    return BezierCls(np.array(ctrl, dtype=np.float64), is_rational=is_rational)


def _make_bspline_1d(
    knots: list[float],
    degree: int,
    ctrl: list[list[float]],
    *,
    is_rational: bool = False,
) -> Bspline:
    """Create a 1-D Bspline from knots, degree, and control points."""
    from pantr.bspline import Bspline as BsplineCls  # noqa: PLC0415
    from pantr.bspline import BsplineSpace, BsplineSpace1D  # noqa: PLC0415

    kv = np.array(knots, dtype=np.float64)
    space = BsplineSpace([BsplineSpace1D(kv, degree)])
    return BsplineCls(space, np.array(ctrl, dtype=np.float64), is_rational=is_rational)


# ======================================================================
# AffineTransform — construction
# ======================================================================


class TestAffineTransformInit:
    """Tests for AffineTransform constructor."""

    def test_identity_like(self) -> None:
        """Construct from identity matrix and zero translation."""
        n = 3
        t = AffineTransform(np.eye(n))
        assert t.dim == n
        npt.assert_array_equal(t.matrix, np.eye(n))
        npt.assert_array_equal(t.offset, np.zeros(n))

    def test_with_translation(self) -> None:
        """Construct with explicit translation."""
        mat = np.array([[2.0, 0.0], [0.0, 3.0]])
        b = np.array([1.0, -1.0])
        t = AffineTransform(mat, b)
        npt.assert_array_equal(t.matrix, mat)
        npt.assert_array_equal(t.offset, b)

    def test_non_square_raises(self) -> None:
        """Non-square matrix raises ValueError."""
        with pytest.raises(ValueError, match="square"):
            AffineTransform(np.ones((2, 3)))

    def test_translation_shape_mismatch_raises(self) -> None:
        """Translation dimension mismatch raises ValueError."""
        with pytest.raises(ValueError, match=r"shape \(2,\)"):
            AffineTransform(np.eye(2), np.zeros(3))

    def test_1d_matrix_raises(self) -> None:
        """1-D matrix raises ValueError."""
        with pytest.raises(ValueError, match="square"):
            AffineTransform(np.array([1.0, 2.0]))

    def test_matrix_is_readonly(self) -> None:
        """Matrix property returns a read-only array."""
        t = AffineTransform(np.eye(2))
        assert not t.matrix.flags.writeable

    def test_offset_is_readonly(self) -> None:
        """Offset property returns a read-only array."""
        t = AffineTransform(np.eye(2))
        assert not t.offset.flags.writeable


# ======================================================================
# AffineTransform — factory methods
# ======================================================================


class TestAffineTransformIdentity:
    """Tests for AffineTransform.identity."""

    def test_identity_2d(self) -> None:
        """Identity in 2D is the identity matrix with zero translation."""
        t = AffineTransform.identity(2)
        npt.assert_array_equal(t.matrix, np.eye(2))
        npt.assert_array_equal(t.offset, np.zeros(2))

    def test_identity_applies_noop(self) -> None:
        """Applying identity to points returns the same points."""
        t = AffineTransform.identity(3)
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        npt.assert_allclose(t(pts), pts)


class TestAffineTransformTranslation:
    """Tests for AffineTransform.translation."""

    def test_translation_2d(self) -> None:
        """Translation shifts points by the given offset."""
        t = AffineTransform.translation([1.0, -2.0])
        pts = np.array([[0.0, 0.0], [3.0, 4.0]])
        expected = np.array([[1.0, -2.0], [4.0, 2.0]])
        npt.assert_allclose(t(pts), expected)

    def test_translation_matrix_is_identity(self) -> None:
        """Translation has identity matrix."""
        t = AffineTransform.translation([5.0, 6.0, 7.0])
        npt.assert_array_equal(t.matrix, np.eye(3))


class TestAffineTransformScaling:
    """Tests for AffineTransform.scaling."""

    def test_isotropic_with_center(self) -> None:
        """Isotropic scaling about a center point."""
        t = AffineTransform.scaling(2.0, center=[1.0, 1.0])
        # Point at center stays fixed.
        npt.assert_allclose(t(np.array([1.0, 1.0])), [1.0, 1.0])
        # Point at (2, 1) maps to (3, 1).
        npt.assert_allclose(t(np.array([2.0, 1.0])), [3.0, 1.0])

    def test_anisotropic(self) -> None:
        """Anisotropic scaling with per-axis factors."""
        t = AffineTransform.scaling([2.0, 3.0])
        pts = np.array([[1.0, 1.0]])
        npt.assert_allclose(t(pts), [[2.0, 3.0]])

    def test_isotropic_no_center_raises(self) -> None:
        """Isotropic scaling without center raises (dimension unknown)."""
        with pytest.raises(ValueError, match="center"):
            AffineTransform.scaling(2.0)

    def test_anisotropic_with_center(self) -> None:
        """Anisotropic scaling about a center point."""
        t = AffineTransform.scaling([2.0, 3.0], center=[1.0, 0.0])
        npt.assert_allclose(t(np.array([1.0, 0.0])), [1.0, 0.0])
        npt.assert_allclose(t(np.array([2.0, 1.0])), [3.0, 3.0])


class TestAffineTransformRotation2D:
    """Tests for AffineTransform.rotation_2d."""

    def test_90_degrees(self) -> None:
        """90-degree CCW rotation maps (1,0) to (0,1)."""
        t = AffineTransform.rotation_2d(np.pi / 2)
        npt.assert_allclose(t(np.array([1.0, 0.0])), [0.0, 1.0], atol=1e-15)

    def test_360_degrees_is_identity(self) -> None:
        """Full rotation is identity."""
        t = AffineTransform.rotation_2d(2 * np.pi)
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        npt.assert_allclose(t(pts), pts, atol=1e-14)

    def test_rotation_about_center(self) -> None:
        """Rotation about a non-origin center."""
        # Rotate (2,1) by 90 degrees about (1,1) -> (1,2)
        t = AffineTransform.rotation_2d(np.pi / 2, center=[1.0, 1.0])
        npt.assert_allclose(t(np.array([2.0, 1.0])), [1.0, 2.0], atol=1e-15)


class TestAffineTransformRotation3D:
    """Tests for AffineTransform.rotation_3d."""

    def test_z_axis_90(self) -> None:
        """90-degree rotation about z-axis maps (1,0,0) to (0,1,0)."""
        t = AffineTransform.rotation_3d(np.pi / 2, axis=2)
        npt.assert_allclose(t(np.array([1.0, 0.0, 0.0])), [0.0, 1.0, 0.0], atol=1e-15)

    def test_x_axis_90(self) -> None:
        """90-degree rotation about x-axis maps (0,1,0) to (0,0,1)."""
        t = AffineTransform.rotation_3d(np.pi / 2, axis=0)
        npt.assert_allclose(t(np.array([0.0, 1.0, 0.0])), [0.0, 0.0, 1.0], atol=1e-15)

    def test_y_axis_90(self) -> None:
        """90-degree rotation about y-axis maps (0,0,1) to (1,0,0)."""
        t = AffineTransform.rotation_3d(np.pi / 2, axis=1)
        npt.assert_allclose(t(np.array([0.0, 0.0, 1.0])), [1.0, 0.0, 0.0], atol=1e-15)

    def test_arbitrary_axis(self) -> None:
        """Rotation about an arbitrary axis (Rodrigues)."""
        axis = np.array([1.0, 1.0, 0.0])
        t = AffineTransform.rotation_3d(np.pi, axis=axis)
        # 180 deg about (1,1,0)/sqrt(2) maps (1,0,0) to (0,1,0)
        npt.assert_allclose(t(np.array([1.0, 0.0, 0.0])), [0.0, 1.0, 0.0], atol=1e-15)

    def test_invalid_int_axis_raises(self) -> None:
        """Integer axis out of range raises."""
        with pytest.raises(ValueError, match="0, 1, or 2"):
            AffineTransform.rotation_3d(0.5, axis=3)

    def test_zero_axis_raises(self) -> None:
        """Zero-length axis raises."""
        with pytest.raises(ValueError, match="non-zero"):
            AffineTransform.rotation_3d(0.5, axis=[0.0, 0.0, 0.0])

    def test_rotation_about_center(self) -> None:
        """3D rotation about a non-origin center."""
        center = np.array([1.0, 0.0, 0.0])
        t = AffineTransform.rotation_3d(np.pi / 2, axis=2, center=center)
        # (1,0,0) is the center, stays fixed
        npt.assert_allclose(t(center), center, atol=1e-15)
        # (2,0,0) rotated 90 about z through (1,0,0) -> (1,1,0)
        npt.assert_allclose(t(np.array([2.0, 0.0, 0.0])), [1.0, 1.0, 0.0], atol=1e-15)


class TestAffineTransformMirror:
    """Tests for AffineTransform.mirror."""

    def test_mirror_x(self) -> None:
        """Mirror across y-axis (normal = x) negates x."""
        t = AffineTransform.mirror([1.0, 0.0])
        npt.assert_allclose(t(np.array([3.0, 5.0])), [-3.0, 5.0], atol=1e-15)

    def test_mirror_diagonal(self) -> None:
        """Mirror across the line y=x swaps coordinates."""
        # Normal to the line y=x is (1,-1)/sqrt(2)
        t = AffineTransform.mirror([1.0, -1.0])
        npt.assert_allclose(t(np.array([1.0, 0.0])), [0.0, 1.0], atol=1e-15)

    def test_mirror_involution(self) -> None:
        """Mirror applied twice is identity."""
        t = AffineTransform.mirror([0.0, 0.0, 1.0])
        t2 = t @ t
        pts = np.array([[1.0, 2.0, 3.0]])
        npt.assert_allclose(t2(pts), pts, atol=1e-15)

    def test_mirror_with_center(self) -> None:
        """Mirror about a plane through a given center."""
        t = AffineTransform.mirror([1.0, 0.0], center=[2.0, 0.0])
        # (3, 0) reflected across x=2 -> (1, 0)
        npt.assert_allclose(t(np.array([3.0, 0.0])), [1.0, 0.0], atol=1e-15)

    def test_zero_normal_raises(self) -> None:
        """Zero-length normal raises."""
        with pytest.raises(ValueError, match="non-zero"):
            AffineTransform.mirror([0.0, 0.0])


class TestAffineTransformShear:
    """Tests for AffineTransform.shear."""

    def test_shear_2d(self) -> None:
        """Shear x by y in 2D."""
        t = AffineTransform.shear(dim=2, component=0, direction=1, factor=2.0)
        npt.assert_allclose(t(np.array([1.0, 3.0])), [7.0, 3.0])

    def test_same_component_direction_raises(self) -> None:
        """Component == direction raises."""
        with pytest.raises(ValueError, match="differ"):
            AffineTransform.shear(dim=2, component=0, direction=0, factor=1.0)

    def test_out_of_range_raises(self) -> None:
        """Out-of-range component raises."""
        with pytest.raises(ValueError, match="component"):
            AffineTransform.shear(dim=2, component=2, direction=0, factor=1.0)


# ======================================================================
# AffineTransform — composition and inverse
# ======================================================================


class TestAffineTransformCompose:
    """Tests for composition of transforms."""

    def test_compose_translations(self) -> None:
        """Composing two translations gives their sum."""
        t1 = AffineTransform.translation([1.0, 0.0])
        t2 = AffineTransform.translation([0.0, 2.0])
        t12 = t1 @ t2
        npt.assert_allclose(t12.offset, [1.0, 2.0])
        npt.assert_array_equal(t12.matrix, np.eye(2))

    def test_compose_matches_sequential(self) -> None:
        """Composed result matches applying transforms sequentially."""
        t1 = AffineTransform.rotation_2d(np.pi / 4)
        t2 = AffineTransform.translation([1.0, 0.0])
        composed = t1 @ t2
        pts = np.array([[1.0, 2.0], [3.0, -1.0]])
        expected = t1(t2(pts))
        npt.assert_allclose(composed(pts), expected, atol=1e-14)

    def test_compose_dimension_mismatch_raises(self) -> None:
        """Composing transforms of different dimensions raises."""
        t2 = AffineTransform.identity(2)
        t3 = AffineTransform.identity(3)
        with pytest.raises(ValueError, match="dimensions"):
            t2 @ t3

    def test_matmul_non_transform_returns_not_implemented(self) -> None:
        """@ with a non-AffineTransform returns NotImplemented."""
        t = AffineTransform.identity(2)
        assert t.__matmul__(42) is NotImplemented


class TestAffineTransformInverse:
    """Tests for AffineTransform.inverse."""

    def test_inverse_roundtrip(self) -> None:
        """T @ T.inverse is identity."""
        t = AffineTransform(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([5.0, 6.0]),
        )
        identity = t @ t.inverse
        npt.assert_allclose(identity.matrix, np.eye(2), atol=1e-14)
        npt.assert_allclose(identity.offset, np.zeros(2), atol=1e-14)

    def test_inverse_applies_correctly(self) -> None:
        """Inverse undoes the forward transform on points."""
        t = AffineTransform.rotation_2d(np.pi / 3)
        pts = np.array([[1.0, 0.0], [0.0, 1.0]])
        npt.assert_allclose(t.inverse(t(pts)), pts, atol=1e-14)

    def test_singular_raises(self) -> None:
        """Singular matrix raises ValueError."""
        t = AffineTransform(np.zeros((2, 2)))
        with pytest.raises(ValueError, match="singular"):
            _ = t.inverse


# ======================================================================
# AffineTransform — __call__
# ======================================================================


class TestAffineTransformCall:
    """Tests for applying the transform to points."""

    def test_broadcast_shape(self) -> None:
        """Transform works on (..., n) shaped inputs."""
        t = AffineTransform.scaling([2.0, 3.0])
        pts = np.ones((4, 5, 2))
        result = t(pts)
        assert result.shape == (4, 5, 2)
        npt.assert_allclose(result[..., 0], 2.0)
        npt.assert_allclose(result[..., 1], 3.0)

    def test_dimension_mismatch_raises(self) -> None:
        """Points with wrong last dimension raise ValueError."""
        t = AffineTransform.identity(3)
        with pytest.raises(ValueError, match="dimension"):
            t(np.array([1.0, 2.0]))


class TestAffineTransformRepr:
    """Tests for __repr__."""

    def test_repr_contains_dim(self) -> None:
        """Repr mentions dimension."""
        t = AffineTransform.identity(2)
        assert "dim=2" in repr(t)


# ======================================================================
# Bezier.transform
# ======================================================================


class TestBezierTransform:
    """Tests for Bezier.transform()."""

    def test_translation(self) -> None:
        """Translating a Bezier shifts control points."""
        b = _make_bezier_1d([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        t = AffineTransform.translation([10.0, 20.0])
        b2 = b.transform(t)
        expected = np.array([[10.0, 20.0], [11.0, 21.0], [12.0, 20.0]])
        npt.assert_allclose(b2.control_points, expected)

    def test_scaling(self) -> None:
        """Scaling a Bezier scales control points."""
        b = _make_bezier_1d([[1.0, 2.0], [3.0, 4.0]])
        t = AffineTransform.scaling([2.0, 3.0])
        b2 = b.transform(t)
        expected = np.array([[2.0, 6.0], [6.0, 12.0]])
        npt.assert_allclose(b2.control_points, expected)

    def test_rotation_2d(self) -> None:
        """90-degree rotation of a Bezier."""
        b = _make_bezier_1d([[1.0, 0.0], [0.0, 0.0]])
        t = AffineTransform.rotation_2d(np.pi / 2)
        b2 = b.transform(t)
        expected = np.array([[0.0, 1.0], [0.0, 0.0]])
        npt.assert_allclose(b2.control_points, expected, atol=1e-15)

    def test_inplace_modifies_self(self) -> None:
        """in_place=True modifies control points in place and returns None."""
        b = _make_bezier_1d([[0.0, 0.0], [1.0, 1.0]])
        t = AffineTransform.translation([5.0, 5.0])
        result = b.transform(t, in_place=True)
        assert result is None
        npt.assert_allclose(b.control_points[0], [5.0, 5.0])

    def test_inplace_no_extra_alloc(self) -> None:
        """in_place=True writes directly into the existing array."""
        b = _make_bezier_1d([[0.0, 0.0], [1.0, 1.0]])
        cp_id = id(b.control_points)
        t = AffineTransform.translation([5.0, 5.0])
        b.transform(t, in_place=True)
        assert id(b.control_points) == cp_id

    def test_not_inplace_returns_new(self) -> None:
        """Default (in_place=False) returns a new Bezier."""
        b = _make_bezier_1d([[0.0, 0.0], [1.0, 1.0]])
        t = AffineTransform.translation([5.0, 5.0])
        b2 = b.transform(t)
        assert b2 is not b
        npt.assert_allclose(b.control_points[0], [0.0, 0.0])  # original unchanged

    def test_rational(self) -> None:
        """Transform on a rational Bezier preserves weights."""
        # Rational with weight=2 for first point, weight=1 for second
        # Stored: (w*x, w*y, w) = (2, 4, 2), (3, 4, 1)
        ctrl = np.array([[2.0, 4.0, 2.0], [3.0, 4.0, 1.0]])
        from pantr.bezier import Bezier  # noqa: PLC0415

        b = Bezier(ctrl, is_rational=True)
        t = AffineTransform.translation([10.0, 20.0])
        b2 = b.transform(t)
        # Physical: (1,2) -> (11,22), (3,4) -> (13,24)
        # Weighted: (2*11, 2*22, 2) = (22,44,2), (1*13, 1*24, 1) = (13,24,1)
        expected = np.array([[22.0, 44.0, 2.0], [13.0, 24.0, 1.0]])
        npt.assert_allclose(b2.control_points, expected)

    def test_dimension_mismatch_raises(self) -> None:
        """Transform dimension mismatch raises ValueError."""
        b = _make_bezier_1d([[0.0, 0.0], [1.0, 1.0]])
        t = AffineTransform.identity(3)
        with pytest.raises(ValueError, match="rank"):
            b.transform(t)

    def test_roundtrip(self) -> None:
        """Transform then inverse recovers original."""
        b = _make_bezier_1d([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        t = AffineTransform.rotation_2d(0.7)
        b2 = b.transform(t).transform(t.inverse)
        npt.assert_allclose(b2.control_points, b.control_points, atol=1e-14)

    def test_evaluate_matches_transformed_points(self) -> None:
        """Evaluating a transformed Bezier matches transforming evaluated points."""
        b = _make_bezier_1d([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]])
        t = AffineTransform.rotation_2d(np.pi / 6)
        b2 = b.transform(t)
        pts = np.linspace(0.0, 1.0, 20, dtype=np.float64)
        direct = t(b.evaluate(pts))
        via_transform = b2.evaluate(pts)
        npt.assert_allclose(via_transform, direct, atol=1e-14)


# ======================================================================
# Bspline.transform
# ======================================================================


class TestBsplineTransform:
    """Tests for Bspline.transform()."""

    def test_translation(self) -> None:
        """Translating a Bspline shifts control points."""
        # Linear B-spline (degree 1)
        knots = [0.0, 0.0, 0.5, 1.0, 1.0]
        ctrl = [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]
        s = _make_bspline_1d(knots, 1, ctrl)
        t = AffineTransform.translation([10.0, 20.0])
        s2 = s.transform(t)
        expected = np.array([[10.0, 20.0], [11.0, 21.0], [12.0, 20.0]])
        npt.assert_allclose(s2.control_points, expected)

    def test_inplace(self) -> None:
        """in_place=True modifies control points in place and returns None."""
        knots = [0.0, 0.0, 1.0, 1.0]
        ctrl = [[0.0, 0.0], [1.0, 1.0]]
        s = _make_bspline_1d(knots, 1, ctrl)
        t = AffineTransform.translation([5.0, 5.0])
        result = s.transform(t, in_place=True)
        assert result is None
        npt.assert_allclose(s.control_points[0], [5.0, 5.0])

    def test_inplace_no_extra_alloc(self) -> None:
        """in_place=True writes directly into the existing array."""
        knots = [0.0, 0.0, 1.0, 1.0]
        ctrl = [[0.0, 0.0], [1.0, 1.0]]
        s = _make_bspline_1d(knots, 1, ctrl)
        cp_id = id(s.control_points)
        t = AffineTransform.translation([5.0, 5.0])
        s.transform(t, in_place=True)
        assert id(s.control_points) == cp_id

    def test_not_inplace_returns_new(self) -> None:
        """Default returns a new Bspline with same space."""
        knots = [0.0, 0.0, 1.0, 1.0]
        ctrl = [[0.0, 0.0], [1.0, 1.0]]
        s = _make_bspline_1d(knots, 1, ctrl)
        t = AffineTransform.translation([5.0, 5.0])
        s2 = s.transform(t)
        assert s2 is not s
        assert s2.space is s.space  # same space object
        npt.assert_allclose(s.control_points[0], [0.0, 0.0])

    def test_rational(self) -> None:
        """Transform on a rational Bspline (NURBS) preserves weights."""
        knots = [0.0, 0.0, 1.0, 1.0]
        # Rational: stored as (w*x, w*y, w)
        ctrl = [[2.0, 4.0, 2.0], [3.0, 4.0, 1.0]]
        s = _make_bspline_1d(knots, 1, ctrl, is_rational=True)
        t = AffineTransform.translation([10.0, 20.0])
        s2 = s.transform(t)
        expected = np.array([[22.0, 44.0, 2.0], [13.0, 24.0, 1.0]])
        npt.assert_allclose(s2.control_points, expected)

    def test_dimension_mismatch_raises(self) -> None:
        """Transform dimension mismatch raises ValueError."""
        knots = [0.0, 0.0, 1.0, 1.0]
        ctrl = [[0.0, 0.0], [1.0, 1.0]]
        s = _make_bspline_1d(knots, 1, ctrl)
        t = AffineTransform.identity(3)
        with pytest.raises(ValueError, match="rank"):
            s.transform(t)

    def test_roundtrip(self) -> None:
        """Transform then inverse recovers original."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        ctrl = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        s = _make_bspline_1d(knots, 2, ctrl)
        t = AffineTransform.rotation_2d(1.2)
        s2 = s.transform(t).transform(t.inverse)
        npt.assert_allclose(s2.control_points, s.control_points, atol=1e-14)

    def test_evaluate_matches_transformed_points(self) -> None:
        """Evaluating a transformed Bspline matches transforming evaluated points."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        ctrl = [[0.0, 0.0], [0.5, 1.0], [1.0, 0.5], [1.5, 0.0]]
        s = _make_bspline_1d(knots, 2, ctrl)
        t = AffineTransform.scaling([2.0, 3.0])
        s2 = s.transform(t)
        pts = np.linspace(0.0, 1.0, 20, dtype=np.float64)
        direct = t(s.evaluate(pts))
        via_transform = s2.evaluate(pts)
        npt.assert_allclose(via_transform, direct, atol=1e-14)
