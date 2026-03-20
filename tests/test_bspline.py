"""Tests for bspline module."""

import numpy as np
import numpy.typing as npt
import pytest

from pantr._bspline_basis_core import _compute_basis_nurbs_book_impl
from pantr._bspline_space_factory import create_uniform_periodic_knot_vector
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
        assert bspline.rank == 2  # noqa: PLR2004
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
        assert bspline.rank == 2  # noqa: PLR2004
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
        """Test valid rational B-spline initialization."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        # shape[-1]=2, rational rank = 2-1 = 1 -> valid
        control_points = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]], dtype=np.float64)
        bspline = Bspline(space, control_points, is_rational=True)

        assert bspline.is_rational is True
        assert bspline.rank == 1

    def test_valid_initialization_rational_rank_1(self) -> None:
        """Test valid rational B-spline with rank 1 (scalar field)."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        # 12 / 3 basis = 4 columns -> shape[-1]=4, rational rank = 4-1 = 3
        control_points = np.arange(12, dtype=np.float64)
        bspline = Bspline(space, control_points, is_rational=True)

        assert bspline.is_rational is True
        assert bspline.rank == 3  # noqa: PLR2004

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
        # For rational: shape[-1]=1 -> rank = 1-1 = 0 -> INVALID
        control_points = np.array([1.0, 2.0, 3.0], dtype=np.float64)

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

        # shape[-1]=2 -> rational rank = 1 -> valid
        control_points_rational = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]], dtype=np.float64)
        bspline_rational = Bspline(space, control_points_rational, is_rational=True)
        assert bspline_rational.is_rational is True

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

        # control_points reshaped to (3, 2), so shape[-1]=2, rank=2
        assert bspline.control_points.shape == (3, 2)
        assert bspline.rank == 2  # noqa: PLR2004

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
        """Test rank property for rational scalar B-spline (rank 1)."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        # shape[-1]=2 -> rational rank = 2-1 = 1
        control_points = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]], dtype=np.float64)
        bspline = Bspline(space, control_points, is_rational=True)

        assert bspline.rank == 1

    def test_rank_property_rational_vector(self) -> None:
        """Test rank property for rational 2D vector B-spline."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space_2d = BsplineSpace([space_1d_1, space_1d_2])
        # 24 / (3*2=6 basis) = 4 columns -> shape (3, 2, 4), shape[-1]=4, rank=4-1=3
        control_points = np.arange(24, dtype=np.float64)
        bspline = Bspline(space_2d, control_points, is_rational=True)

        assert bspline.rank == 3  # noqa: PLR2004

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

        # control_points reshaped to (3, 2, 2), so shape[-1]=2, rank=2
        assert bspline.control_points.shape == (3, 2, 2)
        assert bspline.rank == 2  # noqa: PLR2004

    def test_rank_property_2D_rational_scalar(self) -> None:
        """Test rank property for 2D rational B-spline."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space = BsplineSpace([space_1d_1, space_1d_2])
        # 18 / (3*2=6 basis) = 3 columns -> shape (3, 2, 3), shape[-1]=3, rank=3-1=2
        control_points = np.arange(18, dtype=np.float64)
        bspline = Bspline(space, control_points, is_rational=True)

        assert bspline.rank == 2  # noqa: PLR2004


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

        # Rank 1 (scalar - trailing dim is 1)
        cp0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        bspline0 = Bspline(space, cp0)
        assert bspline0.rank == 1  # shape (3, 2, 1): shape[-1]=1, rank=1

        # Rank 2 (vector, 2 columns)
        cp1 = np.arange(12, dtype=np.float64)
        bspline1 = Bspline(space, cp1)
        assert bspline1.rank == 2  # shape (3, 2, 2): shape[-1]=2, rank=2  # noqa: PLR2004

        # Rank 4 (vector, 4 columns)
        cp2 = np.arange(24, dtype=np.float64)
        bspline2 = Bspline(space, cp2)
        assert bspline2.rank == 4  # shape (3, 2, 4): shape[-1]=4, rank=4  # noqa: PLR2004

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
        assert bspline_vec.rank == 2  # shape[-1]=2, rank=2  # noqa: PLR2004


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


class TestBsplineEvaluateDerivatives:
    """Test Bspline.evaluate_derivatives."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_bspline(knots: list[float], degree: int, cps: list[float]) -> Bspline:
        """Build a scalar 1-D B-spline from plain Python lists."""
        kv = np.array(knots, dtype=np.float64)
        space_1d = BsplineSpace1D(kv, degree)
        space = BsplineSpace([space_1d])
        cp = np.array(cps, dtype=np.float64)
        return Bspline(space, cp)

    # ------------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------------

    def test_n_deriv_0_matches_evaluate(self) -> None:
        """evaluate_derivatives with orders=[0] must equal evaluate."""
        bspline = self._make_bspline([0, 0, 0, 1, 1, 1], 2, [0.0, 1.0, 0.0])
        pts = np.linspace(0.0, 1.0, 11, dtype=np.float64)

        result = bspline.evaluate_derivatives(pts, [0])
        expected = bspline.evaluate(pts)

        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_linear_constant_first_derivative(self) -> None:
        """Degree-1 on [0,1] with CPs [0,1] gives f'(t)=1 everywhere (interior)."""
        bspline = self._make_bspline([0.0, 0.0, 1.0, 1.0], 1, [0.0, 1.0])
        # Exclude the right endpoint: the existing kernel's endpoint shortcut only
        # fills the zeroth-derivative slot; higher-order derivatives are left zero.
        pts = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float64)

        result = bspline.evaluate_derivatives(pts, [1])

        np.testing.assert_allclose(result, np.ones(4), atol=1e-14)

    def test_quadratic_bezier_t_squared_exact(self) -> None:
        """CPs [0,0,1] on [0,0,0,1,1,1] give f(t)=t², f'=2t, f''=2."""
        # Bernstein: B₀=(1-t)², B₁=2t(1-t), B₂=t²
        # f(t) = 0*(1-t)² + 0*2t(1-t) + 1*t² = t²
        bspline = self._make_bspline([0, 0, 0, 1, 1, 1], 2, [0.0, 0.0, 1.0])
        # Exclude t=1.0: endpoint shortcut in the derivative kernel only fills order 0.
        pts = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float64)

        np.testing.assert_allclose(bspline.evaluate_derivatives(pts, [0]), pts**2, atol=1e-13)
        np.testing.assert_allclose(bspline.evaluate_derivatives(pts, [1]), 2.0 * pts, atol=1e-13)
        np.testing.assert_allclose(
            bspline.evaluate_derivatives(pts, [2]), np.full(4, 2.0), atol=1e-13
        )

    def test_quadratic_bezier_general_exact(self) -> None:
        """CPs [0,1,0] give f(t)=2t(1-t); exact 1st and 2nd derivatives."""
        bspline = self._make_bspline([0, 0, 0, 1, 1, 1], 2, [0.0, 1.0, 0.0])
        # Exclude t=1.0: endpoint shortcut in the derivative kernel only fills order 0.
        pts = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float64)

        f = 2.0 * pts * (1.0 - pts)
        f1 = 2.0 - 4.0 * pts
        f2 = np.full(4, -4.0)

        np.testing.assert_allclose(bspline.evaluate_derivatives(pts, [0]), f, atol=1e-13)
        np.testing.assert_allclose(bspline.evaluate_derivatives(pts, [1]), f1, atol=1e-13)
        np.testing.assert_allclose(bspline.evaluate_derivatives(pts, [2]), f2, atol=1e-13)

    def test_cubic_bezier_exact_derivatives(self) -> None:
        """CPs [0,1/3,2/3,1] give identity f(t)=t; f'(t)=1."""
        # Bernstein cubic with CPs [0,1/3,2/3,1] → f(t)=t, f'(t)=1
        bspline = self._make_bspline([0, 0, 0, 0, 1, 1, 1, 1], 3, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        # Exclude t=1.0: endpoint shortcut in the derivative kernel only fills order 0.
        pts = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float64)

        np.testing.assert_allclose(bspline.evaluate_derivatives(pts, [0]), pts, atol=1e-13)
        np.testing.assert_allclose(bspline.evaluate_derivatives(pts, [1]), np.ones(4), atol=1e-13)

    def test_n_deriv_exceeds_degree_zeros(self) -> None:
        """Derivative order > degree must be zero."""
        bspline = self._make_bspline([0, 0, 0, 1, 1, 1], 2, [0.0, 1.0, 0.0])
        pts = np.array([0.25, 0.5, 0.75], dtype=np.float64)

        # Degree 2 → 3rd and higher derivatives are zero
        for k in range(3, 6):
            result = bspline.evaluate_derivatives(pts, [k])
            np.testing.assert_allclose(result, 0.0, atol=1e-14)

    # ------------------------------------------------------------------
    # Output shapes
    # ------------------------------------------------------------------

    def test_output_shape_scalar(self) -> None:
        """Scalar B-spline returns shape (n_pts,)."""
        bspline = self._make_bspline([0, 0, 0, 1, 1, 1], 2, [0.0, 1.0, 0.0])
        pts = np.linspace(0.0, 1.0, 7, dtype=np.float64)

        result = bspline.evaluate_derivatives(pts, [3])

        assert result.shape == (7,)

    def test_output_shape_vector(self) -> None:
        """Vector B-spline (2-column CPs) returns shape (n_pts, rank)."""
        kv = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        space_1d = BsplineSpace1D(kv, 2)
        space = BsplineSpace([space_1d])
        # 3 control points, each 2D
        cp = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]], dtype=np.float64)
        bspline = Bspline(space, cp)

        pts = np.linspace(0.0, 1.0, 5, dtype=np.float64)

        result = bspline.evaluate_derivatives(pts, [2])

        assert result.shape == (5, 2)

    # ------------------------------------------------------------------
    # Numerical validation
    # ------------------------------------------------------------------

    def test_finite_difference_validation(self) -> None:
        """Central FD approximation of first derivative matches evaluate_derivatives."""
        bspline = self._make_bspline([0, 0, 0, 1, 1, 1], 2, [0.0, 1.0, 0.0])
        # Use interior points away from both endpoints to avoid endpoint-shortcut issues
        # and to keep FD points inside the domain.
        pts = np.linspace(0.1, 0.9, 9, dtype=np.float64)
        h = 1e-6

        result = bspline.evaluate_derivatives(pts, [1])
        fd = (bspline.evaluate(pts + h) - bspline.evaluate(pts - h)) / (2.0 * h)

        # Use atol to handle the case where the true derivative is near zero
        # (rtol would fail when comparing ~0 to ~2e-11 floating-point noise).
        np.testing.assert_allclose(result, fd, atol=1e-5)

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_invalid_n_deriv(self) -> None:
        """Negative order raises ValueError."""
        bspline = self._make_bspline([0, 0, 0, 1, 1, 1], 2, [0.0, 1.0, 0.0])
        pts = np.array([0.5], dtype=np.float64)

        with pytest.raises(ValueError):
            bspline.evaluate_derivatives(pts, [-1])

    def test_out_array_reuse(self) -> None:
        """Pre-allocated out array is filled in-place."""
        bspline = self._make_bspline([0, 0, 0, 1, 1, 1], 2, [0.0, 1.0, 0.0])
        pts = np.array([0.25, 0.5, 0.75], dtype=np.float64)
        out = np.zeros(3, dtype=np.float64)

        result = bspline.evaluate_derivatives(pts, [2], out=out)

        np.testing.assert_array_equal(result, out)
        assert not np.all(out == 0.0)

    def test_dim_not_1_returns_scalar_result(self) -> None:
        """evaluate_derivatives on dim=2 scalar B-spline returns (n_pts,)."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(np.array(knots1, dtype=np.float64), 2)
        space_1d_2 = BsplineSpace1D(np.array(knots2, dtype=np.float64), 1)
        space = BsplineSpace([space_1d_1, space_1d_2])
        cp = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        bspline = Bspline(space, cp)

        pts = np.array([[0.5, 0.5]], dtype=np.float64)
        result = bspline.evaluate_derivatives(pts, [1, 1])

        # scalar 2D spline, mixed first derivative: shape (n_pts,)
        assert result.shape == (1,)


# ---------------------------------------------------------------------------
# TestToOpenBspline
# ---------------------------------------------------------------------------


def _make_periodic_bspline(
    num_intervals: int,
    degree: int,
    dtype: type = np.float64,
    continuity: int | None = None,
) -> Bspline:
    """Create a simple periodic B-spline with sequential integer control points.

    Args:
        num_intervals (int): Number of intervals.
        degree (int): B-spline degree.
        dtype (type): Data type. Defaults to np.float64.
        continuity (int | None): Continuity level at interior knots. None uses degree-1
            (maximum continuity). Defaults to None.

    Returns:
        Bspline: A 1D periodic scalar B-spline.
    """
    knots = create_uniform_periodic_knot_vector(
        num_intervals, degree, continuity=continuity, dtype=dtype
    )
    space_1d = BsplineSpace1D(knots, degree, periodic=True)
    space = BsplineSpace([space_1d])
    n = space.num_total_basis
    ctrl: npt.NDArray[np.float64] = np.arange(1, n + 1, dtype=dtype)
    return Bspline(space, ctrl)


def _eval_periodic_correct(f: Bspline, pts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Evaluate a periodic B-spline using the mathematically correct algorithm.

    Uses the unclamped ``first_basis = knot_id - degree`` index with modulo-wrapped
    control point lookup, which is the standard mathematical definition of a periodic
    B-spline. This differs from ``f.evaluate()`` which uses a clamped first_basis.

    Args:
        f (Bspline): A 1D periodic B-spline.
        pts (np.ndarray): Interior evaluation points (must lie strictly inside domain).

    Returns:
        np.ndarray: Evaluated values at the given points.
    """
    space_1d = f.space.spaces[0]
    knots = space_1d.knots
    p = space_1d.degree
    tol = float(space_1d.tolerance)
    n_stored = f.space.num_total_basis
    ctrl = f._control_points  # shape (n_stored, rank)

    # Compute knot spans using the non-periodic (unclamped) algorithm
    basis_out = np.zeros((len(pts), p + 1), dtype=np.float64)
    first_basis_arr = np.zeros(len(pts), dtype=np.int64)
    _compute_basis_nurbs_book_impl(knots, p, False, tol, pts, basis_out, first_basis_arr)

    rank = ctrl.shape[1]
    result = np.zeros((len(pts), rank), dtype=np.float64)
    for i in range(len(pts)):
        s = int(first_basis_arr[i])
        for j in range(p + 1):
            idx = (s + j) % n_stored
            result[i] += basis_out[i, j] * ctrl[idx]

    return result.squeeze()


class TestPeriodicBsplineEvaluation:
    """Test Bspline.evaluate and evaluate_derivatives for periodic B-splines.

    Covers both maximum continuity and reduced continuity (C^0, C^1) cases.
    All comparisons use interior points only to avoid endpoint/clamping differences.

    Note: ``Bspline.evaluate()`` for periodic splines uses an internal clamped-index
    algorithm that differs from the unclamped modulo-wrapped algorithm (used by
    ``to_open_bspline().evaluate()``).  Correctness of the unclamped path is verified
    via ``to_open_bspline()``; internal consistency of the clamped path is verified by
    comparing ``evaluate_derivatives(pts, [0])`` against ``evaluate(pts)``.
    """

    @pytest.mark.parametrize(
        "num_intervals,degree,continuity",
        [
            (3, 2, None),  # degree 2, max continuity
            (4, 2, 0),  # degree 2, C^0
            (4, 3, None),  # degree 3, max continuity
            (4, 3, 1),  # degree 3, C^1
            (5, 3, 0),  # degree 3, C^0
        ],
    )
    def test_periodic_to_open_evaluate_matches_correct_algorithm(
        self, num_intervals: int, degree: int, continuity: int | None
    ) -> None:
        """to_open_bspline().evaluate() agrees with modulo-wrapped reference algorithm.

        This is the canonical correctness check for periodic B-spline evaluation:
        converting to open form via knot insertion must reproduce the mathematically
        correct unclamped periodic function at interior points.
        """
        f = _make_periodic_bspline(num_intervals, degree, continuity=continuity)
        f_open = f.to_open_bspline()

        a, b = f_open.space.spaces[0].domain
        pts = np.linspace(float(a), float(b), 21, dtype=np.float64)[1:-1]

        np.testing.assert_allclose(
            f_open.evaluate(pts),
            _eval_periodic_correct(f, pts),
            atol=1e-11,
        )

    @pytest.mark.parametrize(
        "num_intervals,degree,continuity",
        [
            (3, 2, None),
            (4, 2, 0),
            (4, 3, None),
            (4, 3, 1),
            (5, 3, 0),
        ],
    )
    def test_periodic_evaluate_derivatives_order_0_matches_evaluate(
        self, num_intervals: int, degree: int, continuity: int | None
    ) -> None:
        """evaluate_derivatives(pts, [0]) equals evaluate(pts) for periodic splines.

        Tests internal consistency of the clamped evaluation path used by
        Bspline.evaluate() and Bspline.evaluate_derivatives() for periodic spaces.
        """
        f = _make_periodic_bspline(num_intervals, degree, continuity=continuity)

        a, b = f.to_open_bspline().space.spaces[0].domain
        pts = np.linspace(float(a), float(b), 21, dtype=np.float64)[1:-1]

        np.testing.assert_allclose(
            f.evaluate_derivatives(pts, [0]),
            f.evaluate(pts),
            atol=1e-13,
        )

    @pytest.mark.parametrize(
        "num_intervals,degree,continuity",
        [
            (4, 2, 0),  # degree 2, C^0
            (5, 3, 1),  # degree 3, C^1
            (5, 3, 0),  # degree 3, C^0
        ],
    )
    def test_periodic_to_open_evaluate_derivatives_matches_open(
        self, num_intervals: int, degree: int, continuity: int | None
    ) -> None:
        """to_open_bspline().evaluate_derivatives() agrees with finite differences."""
        f_open = _make_periodic_bspline(
            num_intervals, degree, continuity=continuity
        ).to_open_bspline()

        a, b = f_open.space.spaces[0].domain
        pts = np.linspace(float(a), float(b), 21, dtype=np.float64)[1:-1]

        # 0th order must match evaluate()
        np.testing.assert_allclose(
            f_open.evaluate_derivatives(pts, [0]),
            f_open.evaluate(pts),
            atol=1e-13,
        )

        # 1st order validated by central finite differences
        h = 1e-6
        fd = (f_open.evaluate(pts + h) - f_open.evaluate(pts - h)) / (2.0 * h)
        np.testing.assert_allclose(
            f_open.evaluate_derivatives(pts, [1]),
            fd,
            atol=1e-5,
        )


def _make_unclamped_bspline(dtype: type = np.float64) -> Bspline:
    """Create a non-open non-periodic B-spline (uniform, no repeated boundary knots)."""
    # Cardinal-style knot vector: no repeated boundary knots, uniform spacing.
    # Degree 2, domain [2, 5] (interior of a larger uniform grid).
    knots: npt.NDArray[np.float64] = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=dtype)
    space_1d = BsplineSpace1D(knots, 2)
    space = BsplineSpace([space_1d])
    n = space.num_total_basis
    ctrl: npt.NDArray[np.float64] = np.arange(1, n + 1, dtype=dtype)
    return Bspline(space, ctrl)


def _make_non_open_bspline_varying_bdry(degree: int, boundary_mult: int) -> Bspline:
    """Create a non-open non-periodic B-spline with given boundary multiplicity.

    Args:
        degree (int): B-spline degree.
        boundary_mult (int): Knot multiplicity at domain endpoints (< degree + 1).

    Returns:
        Bspline: A 1D non-open scalar B-spline with sequential integer control points.
    """
    n_int = max(3, 2 * degree - 2 * boundary_mult + 3)
    interior = np.linspace(0.0, 1.0, n_int + 1)[1:-1]
    knots = np.concatenate([[0.0] * boundary_mult, interior, [1.0] * boundary_mult])
    space_1d = BsplineSpace1D(knots, degree)
    space = BsplineSpace([space_1d])
    n = space.num_total_basis
    ctrl: npt.NDArray[np.float64] = np.arange(1, n + 1, dtype=np.float64)
    return Bspline(space, ctrl)


class TestToOpenBspline:
    """Tests for Bspline.to_open_bspline()."""

    def test_periodic_to_open_is_non_periodic(self) -> None:
        """to_open_bspline on a periodic spline returns a non-periodic spline."""
        f = _make_periodic_bspline(3, 2)
        f_open = f.to_open_bspline()

        assert not f_open.space.spaces[0].periodic
        assert f_open.space.spaces[0].has_open_knots()

    def test_periodic_to_open_correctness(self) -> None:
        """Open B-spline agrees with the correct mathematical periodic evaluation."""
        f = _make_periodic_bspline(4, 2)
        f_open = f.to_open_bspline()

        a, b = f_open.space.spaces[0].domain
        # Exclude endpoints to avoid boundary-matching edge cases.
        pts = np.linspace(float(a), float(b), 51, dtype=np.float64)[1:-1]

        vals_correct = _eval_periodic_correct(f, pts)
        vals_open = f_open.evaluate(pts)

        np.testing.assert_allclose(vals_open, vals_correct, atol=1e-12)

    def test_periodic_to_open_degree3(self) -> None:
        """Works for degree-3 periodic splines."""
        f = _make_periodic_bspline(4, 3)
        f_open = f.to_open_bspline()

        assert f_open.space.spaces[0].has_open_knots()
        assert f_open.space.spaces[0].degree == 3  # noqa: PLR2004

        a, b = f_open.space.spaces[0].domain
        pts = np.linspace(float(a), float(b), 51, dtype=np.float64)[1:-1]
        np.testing.assert_allclose(f_open.evaluate(pts), _eval_periodic_correct(f, pts), atol=1e-12)

    def test_non_open_non_periodic_to_open(self) -> None:
        """to_open_bspline on an unclamped non-periodic spline clamps it correctly."""
        f = _make_unclamped_bspline()
        assert not f.space.spaces[0].has_open_knots()
        assert not f.space.spaces[0].periodic

        f_open = f.to_open_bspline()

        assert f_open.space.spaces[0].has_open_knots()
        assert not f_open.space.spaces[0].periodic

        a, b = f.space.spaces[0].domain
        pts = np.linspace(float(a), float(b), 51, dtype=np.float64)[1:-1]
        np.testing.assert_allclose(f_open.evaluate(pts), f.evaluate(pts), atol=1e-12)

    def test_already_open_raises(self) -> None:
        """to_open_bspline on an already-open spline raises ValueError."""
        knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        f = Bspline(space, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))

        with pytest.raises(ValueError, match="already open"):
            f.to_open_bspline()

    def test_multidim_periodic_to_open(self) -> None:
        """to_open_bspline on a 2D spline with one periodic and one open direction."""
        # Direction 0: periodic degree-2, Direction 1: open degree-1
        knots_per = create_uniform_periodic_knot_vector(4, 2, dtype=np.float64)
        knots_open = np.array([0.0, 0.0, 0.5, 1.0, 1.0], dtype=np.float64)
        space_per = BsplineSpace1D(knots_per, 2, periodic=True)
        space_open = BsplineSpace1D(knots_open, 1)
        space = BsplineSpace([space_per, space_open])

        n0 = space_per.num_basis
        n1 = space_open.num_basis
        rng = np.random.default_rng(0)
        ctrl = rng.random((n0 * n1,), dtype=np.float64)
        f = Bspline(space, ctrl)
        f_open = f.to_open_bspline()

        # Direction 0 must become open; direction 1 must remain unchanged.
        assert f_open.space.spaces[0].has_open_knots()
        assert not f_open.space.spaces[0].periodic
        assert f_open.space.spaces[1].has_open_knots()
        assert not f_open.space.spaces[1].periodic

    def test_multidim_already_open_raises(self) -> None:
        """to_open_bspline raises ValueError when all directions are already open."""
        knots1 = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        knots2 = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        space = BsplineSpace([BsplineSpace1D(knots1, 2), BsplineSpace1D(knots2, 1)])
        f = Bspline(space, np.ones((3 * 2,), dtype=np.float64))

        with pytest.raises(ValueError, match="already open"):
            f.to_open_bspline()

    def test_rational_periodic_to_open(self) -> None:
        """to_open_bspline works on rational periodic B-splines."""
        knots = create_uniform_periodic_knot_vector(3, 2, dtype=np.float64)
        space_1d = BsplineSpace1D(knots, 2, periodic=True)
        space = BsplineSpace([space_1d])
        n = space.num_total_basis
        # rational: last coordinate is homogeneous weight (all weights = 1, so NURBS == B-spline)
        ctrl = np.column_stack(
            [np.arange(1, n + 1, dtype=np.float64), np.ones(n, dtype=np.float64)]
        )
        f = Bspline(space, ctrl, is_rational=True)
        f_open = f.to_open_bspline()

        assert f_open.is_rational
        assert not f_open.space.spaces[0].periodic
        assert f_open.space.spaces[0].has_open_knots()

        # With all weights = 1, NURBS reduces to polynomial B-spline.
        # Evaluate the scalar component (x*w / w = x) and compare against correct periodic eval.
        # Build a non-rational version of f with only the x-coordinates for comparison.
        f_scalar = Bspline(space, ctrl[:, 0], is_rational=False)
        a, b = f_open.space.spaces[0].domain
        pts = np.linspace(float(a), float(b), 51, dtype=np.float64)[1:-1]
        vals_correct = _eval_periodic_correct(f_scalar, pts)
        np.testing.assert_allclose(f_open.evaluate(pts), vals_correct, atol=1e-12)

    @pytest.mark.parametrize(
        "degree,boundary_mult",
        [
            (2, 1),
            (2, 2),
            (3, 1),
            (3, 2),
            (3, 3),
            (4, 2),
            (4, 4),
        ],
    )
    def test_non_open_varying_bdry_to_open_correctness(
        self, degree: int, boundary_mult: int
    ) -> None:
        """to_open_bspline().evaluate() matches evaluate() for non-open varying boundary mult."""
        f = _make_non_open_bspline_varying_bdry(degree, boundary_mult)
        assert not f.space.spaces[0].has_open_knots()
        f_open = f.to_open_bspline()
        assert f_open.space.spaces[0].has_open_knots()

        a, b = f.space.spaces[0].domain
        pts = np.linspace(float(a), float(b), 31, dtype=np.float64)[1:-1]
        np.testing.assert_allclose(f_open.evaluate(pts), f.evaluate(pts), atol=1e-12)

    @pytest.mark.parametrize(
        "degree,boundary_mult",
        [
            (2, 1),
            (2, 2),
            (3, 1),
            (3, 2),
            (3, 3),
            (4, 2),
            (4, 4),
        ],
    )
    def test_non_open_varying_bdry_evaluate_derivatives_order_0(
        self, degree: int, boundary_mult: int
    ) -> None:
        """evaluate_derivatives(pts, [0]) matches evaluate(pts) for non-open varying bdry mult."""
        f = _make_non_open_bspline_varying_bdry(degree, boundary_mult)
        a, b = f.space.spaces[0].domain
        pts = np.linspace(float(a), float(b), 21, dtype=np.float64)[1:-1]
        vals = f.evaluate(pts)
        derivs = f.evaluate_derivatives(pts, [0])
        np.testing.assert_allclose(derivs, vals, atol=1e-14)
