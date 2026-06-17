"""Tests for :class:`pantr.geometry.AABB`.

Ported from lepard's AABB suite and extended for general ``ndim`` (PaNTr is not
restricted to 2-D/3-D) and PaNTr's standard-exception convention
(:class:`ValueError` / :class:`TypeError` instead of a library-specific error
type).
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy import typing as npt

from pantr.geometry import AABB
from pantr.transform import AffineTransform


def test_aabb_basic_construction() -> None:
    b = AABB(lo=[0.0, 0.0, 0.0], hi=[1.0, 2.0, 3.0])
    np.testing.assert_array_equal(b.lo, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(b.hi, [1.0, 2.0, 3.0])
    assert b.ndim == 3
    assert b.is_empty() is False


def test_aabb_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="must share shape"):
        AABB(lo=[0.0, 0.0], hi=[1.0, 2.0, 3.0])


def test_aabb_allows_general_ndim() -> None:
    # PaNTr is general-d: a 4-D box is valid (lepard capped this at 2/3).
    b = AABB(lo=[0.0, 0.0, 0.0, 0.0], hi=[1.0, 1.0, 1.0, 1.0])
    assert b.ndim == 4
    assert not b.is_empty()


def test_aabb_allows_1d() -> None:
    b = AABB(lo=[0.0], hi=[1.0])
    assert b.ndim == 1


def test_aabb_rejects_ndim_zero() -> None:
    with pytest.raises(ValueError, match="ndim must be >= 1"):
        AABB(lo=[], hi=[])


def test_aabb_rejects_nan_bounds() -> None:
    with pytest.raises(ValueError, match="must not contain NaN"):
        AABB(lo=[0.0, np.nan, 0.0], hi=[1.0, 1.0, 1.0])


def test_aabb_rejects_non_numeric_dtype() -> None:
    with pytest.raises(TypeError, match="numeric"):
        AABB(lo=[True, False], hi=[True, True])


def test_aabb_arrays_are_readonly() -> None:
    b = AABB(lo=[0.0, 0.0, 0.0], hi=[1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match=r"read-only|assignment destination"):
        b.lo[0] = 99.0


def test_aabb_is_empty_when_lo_exceeds_hi() -> None:
    empty = AABB(lo=[1.0, 0.0, 0.0], hi=[0.5, 1.0, 1.0])
    assert empty.is_empty() is True


def test_aabb_union_of_disjoint_boxes() -> None:
    a = AABB(lo=[0.0, 0.0, 0.0], hi=[1.0, 1.0, 1.0])
    b = AABB(lo=[2.0, 2.0, 2.0], hi=[3.0, 3.0, 3.0])
    u = a.union(b)
    np.testing.assert_array_equal(u.lo, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(u.hi, [3.0, 3.0, 3.0])


def test_aabb_union_empty_neutral() -> None:
    a = AABB(lo=[0.0, 0.0, 0.0], hi=[1.0, 1.0, 1.0])
    empty = AABB(lo=[5.0, 0.0, 0.0], hi=[4.0, 1.0, 1.0])
    np.testing.assert_array_equal(a.union(empty).lo, a.lo)
    np.testing.assert_array_equal(a.union(empty).hi, a.hi)


def test_aabb_intersect_overlapping() -> None:
    a = AABB(lo=[0.0, 0.0, 0.0], hi=[2.0, 2.0, 2.0])
    b = AABB(lo=[1.0, 1.0, 1.0], hi=[3.0, 3.0, 3.0])
    overlap = a.intersect(b)
    assert overlap is not None
    np.testing.assert_array_equal(overlap.lo, [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(overlap.hi, [2.0, 2.0, 2.0])


def test_aabb_intersect_disjoint_returns_none() -> None:
    a = AABB(lo=[0.0, 0.0, 0.0], hi=[1.0, 1.0, 1.0])
    b = AABB(lo=[2.0, 2.0, 2.0], hi=[3.0, 3.0, 3.0])
    assert a.intersect(b) is None


def test_aabb_overlaps() -> None:
    a = AABB(lo=[0.0, 0.0, 0.0], hi=[1.0, 1.0, 1.0])
    touching = AABB(lo=[1.0, 0.0, 0.0], hi=[2.0, 1.0, 1.0])
    far = AABB(lo=[5.0, 0.0, 0.0], hi=[6.0, 1.0, 1.0])
    assert a.overlaps(touching) is True
    assert a.overlaps(far) is False


def test_aabb_ndim_mismatch_errors() -> None:
    a2 = AABB(lo=[0.0, 0.0], hi=[1.0, 1.0])
    a3 = AABB(lo=[0.0, 0.0, 0.0], hi=[1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="dimension mismatch"):
        a2.union(a3)
    with pytest.raises(ValueError, match="dimension mismatch"):
        a2.intersect(a3)
    with pytest.raises(ValueError, match="dimension mismatch"):
        a2.overlaps(a3)


def test_aabb_pad_scalar() -> None:
    b = AABB(lo=[0.0, 0.0, 0.0], hi=[1.0, 1.0, 1.0])
    padded = b.pad(0.5)
    np.testing.assert_array_equal(padded.lo, [-0.5, -0.5, -0.5])
    np.testing.assert_array_equal(padded.hi, [1.5, 1.5, 1.5])


def test_aabb_pad_vector() -> None:
    b = AABB(lo=[0.0, 0.0, 0.0], hi=[1.0, 1.0, 1.0])
    padded = b.pad([0.1, 0.2, 0.3])
    np.testing.assert_allclose(padded.lo, [-0.1, -0.2, -0.3])
    np.testing.assert_allclose(padded.hi, [1.1, 1.2, 1.3])


def test_aabb_pad_rejects_wrong_shape() -> None:
    b = AABB(lo=[0.0, 0.0, 0.0], hi=[1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="scalar or shape"):
        b.pad([0.1, 0.2])


def test_aabb_pad_preserves_inf() -> None:
    b = AABB.unbounded(3)
    padded = b.pad(1.0)
    assert np.all(np.isinf(padded.lo))
    assert np.all(np.isinf(padded.hi))


def test_aabb_transform_translation() -> None:
    b = AABB(lo=[0.0, 0.0, 0.0], hi=[1.0, 2.0, 3.0])
    t = AffineTransform(np.eye(3), [10.0, 20.0, 30.0])
    out = b.transform(t)
    np.testing.assert_allclose(out.lo, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(out.hi, [11.0, 22.0, 33.0])


def test_aabb_transform_rotation_2d_90deg() -> None:
    # [0, 1] x [0, 2] rotated 90 deg about origin -> [-2, 0] x [0, 1].
    b = AABB(lo=[0.0, 0.0], hi=[1.0, 2.0])
    rot = np.array([[0.0, -1.0], [1.0, 0.0]])  # +90 deg rotation matrix.
    out = b.transform(AffineTransform(rot))
    np.testing.assert_allclose(out.lo, [-2.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(out.hi, [0.0, 1.0], atol=1e-12)


def test_aabb_transform_scaling() -> None:
    b = AABB(lo=[-1.0, -1.0, -1.0], hi=[1.0, 1.0, 1.0])
    out = b.transform(AffineTransform(np.diag([2.0, 3.0, 4.0])))
    np.testing.assert_allclose(out.lo, [-2.0, -3.0, -4.0])
    np.testing.assert_allclose(out.hi, [2.0, 3.0, 4.0])


def test_aabb_transform_inf_axis_with_zero_column() -> None:
    # A projection-like transform that drops the infinite axis via a zero
    # column must not turn the result into NaN.
    b = AABB(lo=[-1.0, -1.0, -np.inf], hi=[1.0, 1.0, np.inf])
    t = AffineTransform(np.diag([1.0, 1.0, 0.0]), np.zeros(3))
    out = b.transform(t)
    np.testing.assert_array_equal(out.lo, [-1.0, -1.0, 0.0])
    np.testing.assert_array_equal(out.hi, [1.0, 1.0, 0.0])


def test_aabb_transform_rejects_wrong_ndim() -> None:
    b = AABB(lo=[0.0, 0.0], hi=[1.0, 1.0])
    t = AffineTransform(np.eye(3), [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="affine dim"):
        b.transform(t)


def test_aabb_unbounded() -> None:
    u = AABB.unbounded(3)
    assert np.all(np.isneginf(u.lo))
    assert np.all(np.isposinf(u.hi))
    finite = AABB(lo=[0.0, 0.0, 0.0], hi=[1.0, 1.0, 1.0])
    assert u.overlaps(finite) is True


def test_aabb_unbounded_allows_high_ndim() -> None:
    # General-d: unbounded works for ndim > 3 (lepard rejected this).
    u = AABB.unbounded(5)
    assert u.ndim == 5


def test_aabb_unbounded_rejects_ndim_zero() -> None:
    with pytest.raises(ValueError, match="ndim must be >= 1"):
        AABB.unbounded(0)


def test_aabb_from_bounds_round_trip() -> None:
    b = AABB(lo=[0.0, 0.0, 0.0], hi=[1.0, 2.0, 3.0])
    reconstructed = AABB.from_bounds(b.as_bounds())
    np.testing.assert_array_equal(reconstructed.lo, b.lo)
    np.testing.assert_array_equal(reconstructed.hi, b.hi)


def test_aabb_from_bounds_rejects_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        AABB.from_bounds([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])


def test_aabb_attribute_replacement_raises() -> None:
    b = AABB(lo=[0.0, 0.0, 0.0], hi=[1.0, 1.0, 1.0])
    with pytest.raises(AttributeError, match="immutable"):
        b.lo = np.zeros(3)
    with pytest.raises(AttributeError, match="immutable"):
        b.hi = np.ones(3)


def test_aabb_equality_and_hash() -> None:
    a = AABB(lo=[0.0, 0.0], hi=[1.0, 2.0])
    b = AABB(lo=[0.0, 0.0], hi=[1.0, 2.0])
    c = AABB(lo=[0.0, 0.0], hi=[1.0, 3.0])
    assert a == b
    assert a != c
    assert hash(a) == hash(b)
    assert hash(a) != hash(c)
    assert a != "not an AABB"


def test_aabb_empty_factory() -> None:
    e = AABB.empty(3)
    assert e.is_empty()
    assert e.ndim == 3
    np.testing.assert_array_equal(e.lo, [np.inf, np.inf, np.inf])
    np.testing.assert_array_equal(e.hi, [-np.inf, -np.inf, -np.inf])


def test_aabb_empty_factory_rejects_ndim_zero() -> None:
    with pytest.raises(ValueError, match="ndim must be >= 1"):
        AABB.empty(0)


def test_aabb_empty_is_neutral_for_union() -> None:
    finite = AABB(lo=[1.0, 2.0, 3.0], hi=[4.0, 5.0, 6.0])
    assert AABB.empty(3).union(finite) == finite
    assert finite.union(AABB.empty(3)) == finite


def test_aabb_union_both_empty() -> None:
    assert AABB.empty(2).union(AABB.empty(2)).is_empty()


def test_aabb_rejects_nan_in_hi() -> None:
    with pytest.raises(ValueError, match="must not contain NaN"):
        AABB(lo=[0.0, 0.0], hi=[1.0, np.nan])


def test_aabb_ravel_input() -> None:
    # Any-rank input is ravelled; a (1, 3) array works the same as [lo0, lo1, lo2].
    b = AABB(lo=[[0.0, 0.0, 0.0]], hi=[[1.0, 2.0, 3.0]])
    assert b.ndim == 3
    np.testing.assert_array_equal(b.lo, [0.0, 0.0, 0.0])


def test_aabb_delete_attribute_raises() -> None:
    b = AABB(lo=[0.0, 0.0], hi=[1.0, 1.0])
    with pytest.raises(AttributeError, match="immutable"):
        del b.lo
    with pytest.raises(AttributeError, match="immutable"):
        del b.hi


def test_aabb_overlaps_with_empty() -> None:
    finite = AABB(lo=[0.0, 0.0], hi=[1.0, 1.0])
    empty = AABB.empty(2)
    assert empty.overlaps(finite) is False
    assert finite.overlaps(empty) is False
    assert empty.overlaps(empty) is False


def test_aabb_intersect_with_empty() -> None:
    finite = AABB(lo=[0.0, 0.0], hi=[1.0, 1.0])
    empty = AABB.empty(2)
    assert empty.intersect(finite) is None
    assert finite.intersect(empty) is None
    assert empty.intersect(empty) is None


def test_aabb_pad_negative_shrinks_to_empty() -> None:
    b = AABB(lo=[0.0, 0.0], hi=[1.0, 1.0])
    assert b.pad(-2.0).is_empty()


def test_aabb_transform_empty_stays_empty() -> None:
    assert AABB.empty(2).transform(AffineTransform(np.eye(2))).is_empty()
    assert AABB.empty(3).transform(AffineTransform(np.diag([2.0, 3.0, 4.0]))).is_empty()


def test_aabb_transform_nan_raises() -> None:
    # Row 0 of A has two non-zero entries; axis 1 of the box has lo=hi=+inf,
    # so contrib_min[0,1] = +inf while contrib_min[0,0] = -inf (lo[0]=-inf).
    # np.sum(contrib_min, axis=1)[0] = -inf + (+inf) = NaN → ValueError.
    b = AABB(lo=[-np.inf, np.inf], hi=[1.0, np.inf])
    t = AffineTransform(np.array([[1.0, 1.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="NaN bounds"):
        b.transform(t)


def test_aabb_transform_rejects_non_square_matrix() -> None:
    b = AABB(lo=[0.0, 0.0, 0.0], hi=[1.0, 1.0, 1.0])

    class RectAffine:
        @property
        def dim(self) -> int:
            return 3

        @property
        def matrix(self) -> npt.NDArray[np.float64]:
            return np.eye(2, dtype=np.float64)

        @property
        def offset(self) -> npt.NDArray[np.float64]:
            return np.zeros(3, dtype=np.float64)

    with pytest.raises(ValueError, match="matrix"):
        b.transform(RectAffine())


def test_aabb_contains_point() -> None:
    b = AABB(lo=[0.0, 0.0, 0.0], hi=[1.0, 2.0, 3.0])
    assert b.contains_point([0.5, 1.0, 1.5]) is True
    assert b.contains_point([0.0, 0.0, 0.0]) is True  # boundary
    assert b.contains_point([1.0, 2.0, 3.0]) is True  # boundary
    assert b.contains_point([2.0, 0.0, 0.0]) is False
    assert AABB.empty(3).contains_point([0.5, 0.5, 0.5]) is False
    with pytest.raises(ValueError, match="length"):
        b.contains_point([0.5, 0.5])


def test_aabb_as_bounds_properties() -> None:
    b = AABB(lo=[0.0, 1.0, 2.0], hi=[3.0, 4.0, 5.0])
    bounds = b.as_bounds()
    assert bounds.shape == (3, 2)
    assert bounds.dtype == np.float64
    assert bounds.flags.writeable
    np.testing.assert_array_equal(bounds[:, 0], b.lo)
    np.testing.assert_array_equal(bounds[:, 1], b.hi)


def test_aabb_repr() -> None:
    b = AABB(lo=[0.0, 1.0], hi=[2.0, 3.0])
    assert repr(b) == "AABB(lo=[0.0, 1.0], hi=[2.0, 3.0])"
