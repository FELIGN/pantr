"""Tests for Bspline initialization, properties, and edge cases."""

import numpy as np
import pytest

from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D


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
    """Test Bspline property accessors."""

    def test_dim_property(self) -> None:
        """Test the dim property for different dimensions."""
        # 1D
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        control_points = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        bspline = Bspline(space, control_points)
        assert bspline.dim == 1

        # 2D
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space2 = BsplineSpace([space_1d, space_1d_2])
        control_points2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        bspline2 = Bspline(space2, control_points2)
        assert bspline2.dim == 2  # noqa: PLR2004

    def test_degree_property(self) -> None:
        """Test the degree property for different spaces."""
        # 1D
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        control_points = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        bspline = Bspline(space, control_points)
        assert bspline.degree == (2,)

        # 2D
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space2 = BsplineSpace([space_1d, space_1d_2])
        control_points2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        bspline2 = Bspline(space2, control_points2)
        assert bspline2.degree == (2, 1)

    def test_space_property(self) -> None:
        """Test the space property returns the correct space."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        control_points = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        bspline = Bspline(space, control_points)
        assert bspline.space is space

    def test_control_points_property(self) -> None:
        """Test the control_points property returns the correct array."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        control_points = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        bspline = Bspline(space, control_points)
        np.testing.assert_array_equal(bspline.control_points, control_points)

    def test_is_rational_property(self) -> None:
        """Test the is_rational property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        control_points = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]], dtype=np.float64)

        bspline_non_rational = Bspline(space, control_points)
        assert bspline_non_rational.is_rational is False

        bspline_rational = Bspline(space, control_points, is_rational=True)
        assert bspline_rational.is_rational is True

    def test_rank_property_non_rational_scalar(self) -> None:
        """Test rank for non-rational scalar B-spline."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        control_points = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        bspline = Bspline(space, control_points)

        # shape (3, 1): ndim=2, dim=1, rank=ndim-dim=1
        assert bspline.rank == 1

    def test_rank_property_non_rational_vector(self) -> None:
        """Test rank for non-rational vector B-spline."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        control_points = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        bspline = Bspline(space, control_points)

        # shape (3, 2): ndim=2, dim=1, rank=2
        assert bspline.rank == 2  # noqa: PLR2004

    def test_rank_property_non_rational_higher_rank(self) -> None:
        """Test rank for non-rational B-spline with higher rank (e.g. matrix-valued)."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        # 3 basis fns, 12/3=4 values -> shape (3, 4) -> rank = ndim - dim = 2-1 = 4
        control_points = np.arange(12, dtype=np.float64)
        bspline = Bspline(space, control_points)

        # shape (3, 4): rank = shape[-1] = 4
        assert bspline.rank == 4  # noqa: PLR2004

    def test_rank_property_rational_scalar(self) -> None:
        """Test rank for rational scalar B-spline."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        control_points = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]], dtype=np.float64)
        bspline = Bspline(space, control_points, is_rational=True)

        # shape (3, 2): rational rank = shape[-1] - 1 = 1
        assert bspline.rank == 1

    def test_rank_property_rational_vector(self) -> None:
        """Test rank for rational vector B-spline."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        # 3 basis, 9/3=3 columns, rational rank = 3-1 = 2
        control_points = np.arange(9, dtype=np.float64).reshape(3, 3)
        bspline = Bspline(space, control_points, is_rational=True)

        # shape (3, 3): rational rank = 3-1 = 2
        assert bspline.rank == 2  # noqa: PLR2004

    def test_rank_property_2D_non_rational_scalar(self) -> None:
        """Test rank for 2D non-rational scalar B-spline."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space = BsplineSpace([space_1d_1, space_1d_2])
        control_points = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        bspline = Bspline(space, control_points)

        # shape (3, 2, 1): ndim=3, dim=2, rank=1
        assert bspline.rank == 1

    def test_rank_property_2D_non_rational_vector(self) -> None:
        """Test rank for 2D non-rational vector B-spline."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space = BsplineSpace([space_1d_1, space_1d_2])
        # 6 basis, 12 values = 6*2, shape = (3, 2, 2), rank = 2
        control_points = np.arange(12, dtype=np.float64)
        bspline = Bspline(space, control_points)

        assert bspline.rank == 2  # noqa: PLR2004

    def test_rank_property_2D_rational_scalar(self) -> None:
        """Test rank for 2D rational scalar B-spline."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space = BsplineSpace([space_1d_1, space_1d_2])
        # 6 basis, 12 values = 6*2, shape = (3, 2, 2), rational rank = 2-1 = 1
        control_points = np.arange(12, dtype=np.float64)
        bspline = Bspline(space, control_points, is_rational=True)

        assert bspline.rank == 1


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
        """Test multiple rank values for 2D B-splines."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space = BsplineSpace([space_1d_1, space_1d_2])

        # num_basis = (3, 2), total = 6
        # Scalar: 6 points -> reshapes to (3, 2, 1), rank=1
        cp_scalar = np.arange(6, dtype=np.float64)
        bspline_scalar = Bspline(space, cp_scalar)
        assert bspline_scalar.control_points.shape == (3, 2, 1)
        assert bspline_scalar.rank == 1

        # Vector: 6 * 3 = 18 points -> reshapes to (3, 2, 3), rank=3
        cp_vec = np.arange(18, dtype=np.float64)
        bspline_vec = Bspline(space, cp_vec)
        assert bspline_vec.control_points.shape == (3, 2, 3)
        assert bspline_vec.rank == 3  # noqa: PLR2004

    def test_control_points_flat_input(self) -> None:
        """Test that flat control_points are correctly reshaped for higher dimensions."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space = BsplineSpace([space_1d_1, space_1d_2])

        # 6 basis, 12 values -> (3, 2, 2)
        control_points = np.arange(12, dtype=np.float64)
        bspline = Bspline(space, control_points)
        assert bspline.control_points.shape == (3, 2, 2)
        # Values should be sequential when flattened
        np.testing.assert_array_equal(
            bspline.control_points.ravel(), np.arange(12, dtype=np.float64)
        )

    def test_control_points_3D_space(self) -> None:
        """Test control_points for 3D B-spline space."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        knots3 = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space_1d_3 = BsplineSpace1D(knots3, 3)
        space = BsplineSpace([space_1d_1, space_1d_2, space_1d_3])

        # num_basis = (3, 2, 4), total = 24
        # Scalar: 24 points -> reshapes to (3, 2, 4, 1)
        cp_scalar = np.arange(24, dtype=np.float64)
        bspline_scalar = Bspline(space, cp_scalar)
        assert bspline_scalar.control_points.shape == (3, 2, 4, 1)
        assert bspline_scalar.rank == 1

        # Vector: 24 * 2 = 48 points -> reshapes to (3, 2, 4, 2)
        cp_vec = np.arange(48, dtype=np.float64)
        bspline_vec = Bspline(space, cp_vec)
        assert bspline_vec.control_points.shape == (3, 2, 4, 2)
        assert bspline_vec.rank == 2  # shape[-1]=2, rank=2  # noqa: PLR2004
