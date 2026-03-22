"""Tests for reverse and permute_directions methods on Bezier."""

import numpy as np
import pytest

from pantr.bezier import Bezier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bezier_2d_surface() -> Bezier:
    """Create a 2D Bezier surface with distinct control points.

    Degree (2, 1), rank 3 — a surface in 3D space.
    Shape: (3, 2, 3).
    """
    cp = np.arange(18, dtype=np.float64).reshape(3, 2, 3)
    return Bezier(cp)


def _make_bezier_3d_volume() -> Bezier:
    """Create a 3D Bezier volume with distinct control points.

    Degree (1, 2, 1), rank 1 — a scalar volume.
    Shape: (2, 3, 2, 1).
    """
    cp = np.arange(12, dtype=np.float64).reshape(2, 3, 2, 1)
    return Bezier(cp)


# ===========================================================================
# Bezier — reverse
# ===========================================================================


class TestBezierReverse:
    """Test Bezier.reverse()."""

    def test_reverse_1d(self) -> None:
        """Reversing a 1D Bezier flips control points."""
        cp = np.array([[1.0], [2.0], [3.0]])
        b = Bezier(cp)
        rev = b.reverse()
        np.testing.assert_array_equal(rev.control_points, cp[::-1])
        assert rev is not b

    def test_reverse_2d_direction_0(self) -> None:
        """Reversing direction 0 of a 2D surface flips first axis."""
        b = _make_bezier_2d_surface()
        rev = b.reverse(direction=0)
        expected = np.flip(b.control_points, axis=0)
        np.testing.assert_array_equal(rev.control_points, expected)

    def test_reverse_2d_direction_1(self) -> None:
        """Reversing direction 1 of a 2D surface flips second axis."""
        b = _make_bezier_2d_surface()
        rev = b.reverse(direction=1)
        expected = np.flip(b.control_points, axis=1)
        np.testing.assert_array_equal(rev.control_points, expected)

    def test_reverse_preserves_properties(self) -> None:
        """Reversing preserves degree, dim, rank, dtype, is_rational."""
        b = _make_bezier_2d_surface()
        rev = b.reverse(direction=0)
        assert rev.dim == b.dim
        assert rev.degree == b.degree
        assert rev.rank == b.rank
        assert rev.dtype == b.dtype
        assert rev.is_rational == b.is_rational

    def test_reverse_rational(self) -> None:
        """Reversing a rational Bezier flips control points including weights."""
        cp = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 2.0]])
        b = Bezier(cp, is_rational=True)
        rev = b.reverse()
        np.testing.assert_array_equal(rev.control_points, cp[::-1])
        assert rev.is_rational is True

    def test_reverse_in_place(self) -> None:
        """in_place=True modifies the original and returns None."""
        b = _make_bezier_2d_surface()
        original_cp = b.control_points.copy()
        result = b.reverse(direction=0, in_place=True)
        assert result is None
        expected = np.flip(original_cp, axis=0)
        np.testing.assert_array_equal(b.control_points, expected)

    def test_reverse_double_is_identity(self) -> None:
        """Reversing the same direction twice yields the original."""
        b = _make_bezier_2d_surface()
        original_cp = b.control_points.copy()
        rev2 = b.reverse(direction=1).reverse(direction=1)
        np.testing.assert_array_equal(rev2.control_points, original_cp)

    def test_reverse_invalid_direction(self) -> None:
        """Raises ValueError for out-of-range direction."""
        b = _make_bezier_2d_surface()
        with pytest.raises(ValueError, match="direction must be in"):
            b.reverse(direction=2)
        with pytest.raises(ValueError, match="direction must be in"):
            b.reverse(direction=-1)

    def test_reverse_evaluates_correctly(self) -> None:
        """Reversed Bezier evaluates as b(1-t) in that direction."""
        cp = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]])
        b = Bezier(cp)
        rev = b.reverse()
        pts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        vals_b = b.evaluate(pts)
        vals_rev = rev.evaluate(1.0 - pts)
        np.testing.assert_allclose(vals_b, vals_rev, atol=1e-14)


# ===========================================================================
# Bezier — permute_directions
# ===========================================================================


class TestBezierPermuteDirections:
    """Test Bezier.permute_directions()."""

    def test_permute_2d_swap(self) -> None:
        """Swapping directions of a 2D surface transposes parametric axes."""
        b = _make_bezier_2d_surface()
        perm = b.permute_directions([1, 0])
        expected = np.transpose(b.control_points, (1, 0, 2))
        np.testing.assert_array_equal(perm.control_points, expected)
        assert perm.degree == (b.degree[1], b.degree[0])

    def test_permute_3d_cyclic(self) -> None:
        """Cyclic permutation [1,2,0] on a 3D volume."""
        b = _make_bezier_3d_volume()
        perm = b.permute_directions([1, 2, 0])
        expected = np.transpose(b.control_points, (1, 2, 0, 3))
        np.testing.assert_array_equal(perm.control_points, expected)
        assert perm.degree == (b.degree[1], b.degree[2], b.degree[0])

    def test_permute_identity(self) -> None:
        """Identity permutation [0, 1] is a no-op."""
        b = _make_bezier_2d_surface()
        perm = b.permute_directions([0, 1])
        np.testing.assert_array_equal(perm.control_points, b.control_points)

    def test_permute_preserves_properties(self) -> None:
        """Permutation preserves dim, rank, dtype, is_rational."""
        b = _make_bezier_2d_surface()
        perm = b.permute_directions([1, 0])
        assert perm.dim == b.dim
        assert perm.rank == b.rank
        assert perm.dtype == b.dtype
        assert perm.is_rational == b.is_rational

    def test_permute_in_place(self) -> None:
        """in_place=True modifies the original and returns None."""
        b = _make_bezier_2d_surface()
        original_cp = b.control_points.copy()
        result = b.permute_directions([1, 0], in_place=True)
        assert result is None
        expected = np.transpose(original_cp, (1, 0, 2))
        np.testing.assert_array_equal(b.control_points, expected)

    def test_permute_inverse_is_identity(self) -> None:
        """Applying a permutation then its inverse recovers the original."""
        b = _make_bezier_3d_volume()
        original_cp = b.control_points.copy()
        perm = [1, 2, 0]
        inv_perm = [2, 0, 1]  # inverse of [1, 2, 0]
        result = b.permute_directions(perm).permute_directions(inv_perm)
        np.testing.assert_array_equal(result.control_points, original_cp)

    def test_permute_invalid_permutation(self) -> None:
        """Raises ValueError for invalid permutations."""
        b = _make_bezier_2d_surface()
        with pytest.raises(ValueError, match="permutation must be a permutation"):
            b.permute_directions([0, 0])
        with pytest.raises(ValueError, match="permutation must be a permutation"):
            b.permute_directions([0, 1, 2])
        with pytest.raises(ValueError, match="permutation must be a permutation"):
            b.permute_directions([1])

    def test_permute_1d_identity(self) -> None:
        """1D Bezier with [0] is a no-op."""
        cp = np.array([[1.0], [2.0], [3.0]])
        b = Bezier(cp)
        perm = b.permute_directions([0])
        np.testing.assert_array_equal(perm.control_points, b.control_points)
