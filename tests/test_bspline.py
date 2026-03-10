"""Tests for bspline module."""

import numpy as np
import pytest

from pantr.bspline import Bspline
from pantr.bspline_space_1D import BsplineSpace1D
from pantr.bspline_space_nd import BsplineSpace


class TestBsplineInit:
    """Test Bspline initialization."""

    def test_valid_initialization_1D_scalar(self) -> None:
        """Test valid Bspline initialization with 1D scalar."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        control_points = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        bspline = Bspline(space, control_points)

        assert bspline.dim == 1
        assert bspline.degree == (2,)
        assert bspline.space is space
        assert bspline.is_rational is False
        assert bspline.rank == 1  # shape (3, 1): ndim=2, dim=1, rank=1
        assert bspline.control_points.shape == (3, 1)
        np.testing.assert_array_equal(bspline.control_points, control_points.reshape(3, 1))

    def test_valid_initialization_1D_vector(self) -> None:
        """Test valid Bspline initialization with 1D vector."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        control_points = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        bspline = Bspline(space, control_points)

        assert bspline.dim == 1
        assert bspline.rank == 1
        assert bspline.control_points.shape == (3, 2)
        np.testing.assert_array_equal(bspline.control_points, control_points)

    def test_valid_initialization_2D_scalar(self) -> None:
        """Test valid Bspline initialization with 2D scalar."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space = BsplineSpace([space_1d_1, space_1d_2])
        control_points = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        bspline = Bspline(space, control_points)

        assert bspline.dim == 2  # noqa: PLR2004
        assert bspline.degree == (2, 1)
        assert bspline.rank == 1  # shape (3, 2, 1): ndim=3, dim=2, rank=1
        assert bspline.control_points.shape == (3, 2, 1)
        np.testing.assert_array_equal(bspline.control_points, control_points.reshape(3, 2, 1))

    def test_valid_initialization_2D_vector(self) -> None:
        """Test valid Bspline initialization with 2D vector."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space = BsplineSpace([space_1d_1, space_1d_2])
        control_points = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            dtype=np.float64,
        )
        bspline = Bspline(space, control_points)

        assert bspline.dim == 2  # noqa: PLR2004
        assert bspline.rank == 1
        assert bspline.control_points.shape == (3, 2, 2)
        np.testing.assert_array_equal(bspline.control_points, control_points.reshape(3, 2, 2))

    def test_valid_initialization_3D_scalar(self) -> None:
        """Test valid Bspline initialization with 3D scalar."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        knots3 = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space_1d_3 = BsplineSpace1D(knots3, 3)
        space = BsplineSpace([space_1d_1, space_1d_2, space_1d_3])
        control_points = np.arange(24, dtype=np.float64)
        bspline = Bspline(space, control_points)

        assert bspline.dim == 3  # noqa: PLR2004
        assert bspline.degree == (2, 1, 3)
        assert bspline.rank == 1  # shape (3, 2, 4, 1): ndim=4, dim=3, rank=1
        assert bspline.control_points.shape == (3, 2, 4, 1)

    def test_valid_initialization_rational(self) -> None:
        """Test that rational B-spline with rank 0 raises ValueError."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        # For 1D space, reshape([3, -1]) gives ndim=2, so rank=(2-1)-1=0 -> INVALID
        # With this reshape pattern, rational B-splines always have rank 0
        control_points = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]], dtype=np.float64)

        with pytest.raises(ValueError, match="The B-spline must have at least rank one"):
            Bspline(space, control_points, is_rational=True)

    def test_valid_initialization_rational_rank_1(self) -> None:
        """Test that rational B-spline with 1D space always has rank 0 (invalid)."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        # With reshape([3, -1]), ndim is always 2, so rank=(2-1)-1=0 -> INVALID
        # Rational B-splines with this reshape pattern cannot have rank >= 1
        control_points = np.arange(12, dtype=np.float64)

        with pytest.raises(ValueError, match="The B-spline must have at least rank one"):
            Bspline(space, control_points, is_rational=True)

    def test_valid_initialization_float32(self) -> None:
        """Test valid Bspline initialization with float32 dtype."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        control_points = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        bspline = Bspline(space, control_points)

        assert bspline.control_points.dtype == np.float32
        assert bspline.space.dtype == np.float32

    def test_initialization_control_points_not_multiple_error(self) -> None:
        """Test that control_points not a multiple of num_basis raises ValueError."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        control_points = np.array([1.0, 2.0], dtype=np.float64)  # 2 points, but need 3

        with pytest.raises(ValueError, match="The number of control points must be a multiple"):
            Bspline(space, control_points)

    def test_initialization_dtype_mismatch_error(self) -> None:
        """Test that dtype mismatch between control_points and space raises ValueError."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        control_points = np.array([1.0, 2.0, 3.0], dtype=np.float64)  # Wrong dtype

        with pytest.raises(ValueError, match="The control points must have the same dtype"):
            Bspline(space, control_points)

    def test_initialization_rank_zero_error(self) -> None:
        """Test that rank <= 0 raises ValueError for rational B-splines."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        # For rational scalar: shape (3, 2), ndim=2, dim=1, rank=(2-1)-1=0 -> INVALID
        control_points = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]], dtype=np.float64)

        with pytest.raises(ValueError, match="The B-spline must have at least rank one"):
            Bspline(space, control_points, is_rational=True)

    def test_initialization_rank_negative_error(self) -> None:
        """Test that negative rank raises ValueError."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        # For rational: rank = ndim - dim - 1 = 1 - 1 - 1 = -1, which is invalid
        control_points = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        with pytest.raises(ValueError, match="The B-spline must have at least rank one"):
            Bspline(space, control_points, is_rational=True)

    def test_initialization_control_points_list(self) -> None:
        """Test that control_points can be a list (ArrayLike)."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        control_points = [1.0, 2.0, 3.0]
        bspline = Bspline(space, control_points)

        assert isinstance(bspline.control_points, np.ndarray)
        assert bspline.control_points.shape == (3, 1)

    def test_initialization_control_points_reshaped(self) -> None:
        """Test that control_points are correctly reshaped."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space = BsplineSpace([space_1d_1, space_1d_2])
        control_points = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        bspline = Bspline(space, control_points)

        expected_shape = (3, 2, 1)  # Scalar values get trailing dimension
        assert bspline.control_points.shape == expected_shape
        np.testing.assert_array_equal(bspline.control_points, control_points.reshape(3, 2, 1))


class TestBsplineProperties:
    """Test Bspline properties."""

    def test_dim_property(self) -> None:
        """Test dim property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        control_points = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        bspline = Bspline(space, control_points)

        assert bspline.dim == 1
        assert bspline.dim == space.dim

        # Test 2D
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space_2d = BsplineSpace([space_1d, space_1d_2])
        control_points_2d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        bspline_2d = Bspline(space_2d, control_points_2d)

        assert bspline_2d.dim == 2  # noqa: PLR2004
        assert bspline_2d.dim == space_2d.dim

    def test_degree_property(self) -> None:
        """Test degree property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        control_points = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        bspline = Bspline(space, control_points)

        assert bspline.degree == (2,)
        assert bspline.degree == space.degrees

        # Test 2D
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space_2d = BsplineSpace([space_1d, space_1d_2])
        control_points_2d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        bspline_2d = Bspline(space_2d, control_points_2d)

        assert bspline_2d.degree == (2, 1)
        assert bspline_2d.degree == space_2d.degrees

    def test_space_property(self) -> None:
        """Test space property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        control_points = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        bspline = Bspline(space, control_points)

        assert bspline.space is space
        assert isinstance(bspline.space, BsplineSpace)

    def test_control_points_property(self) -> None:
        """Test control_points property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        control_points = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        bspline = Bspline(space, control_points)

        assert isinstance(bspline.control_points, np.ndarray)
        assert bspline.control_points.shape == (3, 1)
        np.testing.assert_array_equal(bspline.control_points, control_points.reshape(3, 1))

    def test_is_rational_property(self) -> None:
        """Test is_rational property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        control_points = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        bspline_non_rational = Bspline(space, control_points)
        assert bspline_non_rational.is_rational is False

        # For rational, with reshape([3, -1]), ndim=2, rank=(2-1)-1=0 -> INVALID
        control_points_rational = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="The B-spline must have at least rank one"):
            Bspline(space, control_points_rational, is_rational=True)

    def test_rank_property_non_rational_scalar(self) -> None:
        """Test rank property for non-rational scalar B-spline."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        control_points = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        bspline = Bspline(space, control_points)

        # control_points reshaped to (3, 1), so ndim=2, dim=1, rank=2-1=1
        assert bspline.control_points.shape == (3, 1)
        assert bspline.rank == 1  # This is what the code computes

    def test_rank_property_non_rational_vector(self) -> None:
        """Test rank property for non-rational vector B-spline."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        control_points = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        bspline = Bspline(space, control_points)

        # control_points reshaped to (3, 2), so ndim=2, dim=1, rank=2-1=1
        assert bspline.control_points.shape == (3, 2)
        assert bspline.rank == 1

    def test_rank_property_non_rational_higher_rank(self) -> None:
        """Test rank property for non-rational with higher rank values."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        knots3 = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space_1d_3 = BsplineSpace1D(knots3, 3)
        space_3d = BsplineSpace([space_1d_1, space_1d_2, space_1d_3])

        control_points_rank1 = np.arange(24, dtype=np.float64)
        bspline_rank1 = Bspline(space_3d, control_points_rank1)
        assert bspline_rank1.control_points.shape == (3, 2, 4, 1)
        assert bspline_rank1.rank == 1  # ndim=4, dim=3, rank=1

    def test_rank_property_rational_scalar(self) -> None:
        """Test that rational scalar B-spline raises ValueError (rank 0 not allowed)."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        # For rational scalar, shape (3, 2) -> rank = (2-1)-1 = 0 -> INVALID
        control_points = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]], dtype=np.float64)

        with pytest.raises(ValueError, match="The B-spline must have at least rank one"):
            Bspline(space, control_points, is_rational=True)

    def test_rank_property_rational_vector(self) -> None:
        """Test that rational B-spline with 2D space also has rank 0 (invalid)."""
        # With reshape([3, 2, -1]), ndim is always 3, so rank=(3-2)-1=0 -> INVALID
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space_2d = BsplineSpace([space_1d_1, space_1d_2])
        control_points = np.arange(24, dtype=np.float64)

        with pytest.raises(ValueError, match="The B-spline must have at least rank one"):
            Bspline(space_2d, control_points, is_rational=True)

    def test_rank_property_2D_non_rational_scalar(self) -> None:
        """Test rank property for 2D non-rational scalar B-spline."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space = BsplineSpace([space_1d_1, space_1d_2])
        control_points = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        bspline = Bspline(space, control_points)

        # control_points reshaped to (3, 2, 1), so ndim=3, dim=2, rank=3-2=1
        assert bspline.control_points.shape == (3, 2, 1)
        assert bspline.rank == 1

    def test_rank_property_2D_non_rational_vector(self) -> None:
        """Test rank property for 2D non-rational vector B-spline."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space = BsplineSpace([space_1d_1, space_1d_2])
        control_points = np.arange(12, dtype=np.float64).reshape(6, 2)
        bspline = Bspline(space, control_points)

        # control_points reshaped to (3, 2, 2), so ndim=3, dim=2, rank=3-2=1
        assert bspline.control_points.shape == (3, 2, 2)
        assert bspline.rank == 1

    def test_rank_property_2D_rational_scalar(self) -> None:
        """Test that 2D rational scalar B-spline raises ValueError (rank 0 not allowed)."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space = BsplineSpace([space_1d_1, space_1d_2])
        # For rational scalar in 2D: 18 elements -> (3, 2, 3)
        # rank = (3-2)-1 = 0 -> INVALID
        control_points = np.arange(18, dtype=np.float64)

        with pytest.raises(ValueError, match="The B-spline must have at least rank one"):
            Bspline(space, control_points, is_rational=True)


class TestBsplineEdgeCases:
    """Test Bspline edge cases."""

    def test_control_points_immutable_view(self) -> None:
        """Test that control_points returns a view (not a copy)."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        control_points = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        bspline = Bspline(space, control_points)

        # The property returns the array, which is a view of the internal array
        cp = bspline.control_points
        assert isinstance(cp, np.ndarray)

    def test_multiple_ranks_2D(self) -> None:
        """Test Bspline with different ranks in 2D."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space = BsplineSpace([space_1d_1, space_1d_2])

        # Rank 1 (scalar - always has trailing dimension)
        cp0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        bspline0 = Bspline(space, cp0)
        assert bspline0.rank == 1  # shape (3, 2, 1): ndim=3, dim=2, rank=1

        # Rank 1 (vector)
        cp1 = np.arange(12, dtype=np.float64)
        bspline1 = Bspline(space, cp1)
        assert bspline1.rank == 1  # shape (3, 2, 2): ndim=3, dim=2, rank=1

        # With reshape([3, 2, -1]), ndim is always 3, so rank is always 1
        # Can't get rank 2 with this reshape pattern
        cp2 = np.arange(24, dtype=np.float64)
        bspline2 = Bspline(space, cp2)
        assert bspline2.rank == 1  # shape (3, 2, 4): ndim=3, dim=2, rank=1

    def test_control_points_flat_input(self) -> None:
        """Test that flat input is correctly reshaped."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space = BsplineSpace([space_1d_1, space_1d_2])

        # Flat array - reshapes to (3, 2, 1) for scalar
        cp_flat = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        bspline = Bspline(space, cp_flat)
        assert bspline.control_points.shape == (3, 2, 1)

        # Already shaped array - also reshapes to (3, 2, 1)
        cp_shaped = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        bspline2 = Bspline(space, cp_shaped)
        assert bspline2.control_points.shape == (3, 2, 1)

    def test_control_points_3D_space(self) -> None:
        """Test control_points reshaping for 3D space."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        knots3 = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space_1d_3 = BsplineSpace1D(knots3, 3)
        space = BsplineSpace([space_1d_1, space_1d_2, space_1d_3])

        # Scalar: 3 * 2 * 4 = 24 points -> reshapes to (3, 2, 4, 1)
        cp = np.arange(24, dtype=np.float64)
        bspline = Bspline(space, cp)
        assert bspline.control_points.shape == (3, 2, 4, 1)
        assert bspline.rank == 1  # ndim=4, dim=3, rank=1

        # Vector: 24 * 2 = 48 points -> reshapes to (3, 2, 4, 2)
        cp_vec = np.arange(48, dtype=np.float64)
        bspline_vec = Bspline(space, cp_vec)
        assert bspline_vec.control_points.shape == (3, 2, 4, 2)
        assert bspline_vec.rank == 1  # ndim=4, dim=3, rank=1


class TestBsplineEvaluation:
    """Test Bspline evaluation."""

    def test_evaluate_1D_linear(self) -> None:
        """Test evaluation of a 1D linear B-spline."""
        knots = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        space_1d = BsplineSpace1D(knots, 1)
        space = BsplineSpace([space_1d])
        control_points = np.array([0.0, 1.0], dtype=np.float64)
        bspline = Bspline(space, control_points)

        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        values = bspline.evaluate(pts)

        expected = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        np.testing.assert_allclose(values, expected, atol=1e-14)

    def test_evaluate_1D_quadratic(self) -> None:
        """Test evaluation of a 1D quadratic B-spline."""
        # Basis functions on [0,1] for open knot vector [0,0,0,1,1,1] are Bernstein polynomials
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        control_points = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        bspline = Bspline(space, control_points)

        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        values = bspline.evaluate(pts)

        # At 0.5, B2(t) = (1-t)^2 P0 + 2t(1-t) P1 + t^2 P2
        # = 0.25*0 + 0.5*0.5 + 0.25*1 = 0.25 + 0.25 = 0.5
        expected = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        np.testing.assert_allclose(values, expected, atol=1e-14)

        # Non-linear
        control_points = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        bspline = Bspline(space, control_points)
        values = bspline.evaluate(pts)
        # At 0.5: 0.25*0 + 0.5*1 + 0.25*0 = 0.5
        expected = np.array([0.0, 0.5, 0.0], dtype=np.float64)
        np.testing.assert_allclose(values, expected, atol=1e-14)
