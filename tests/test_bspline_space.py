"""Tests for bspline_space module."""

import numpy as np
import pytest

from pantr.bspline import BsplineSpace, BsplineSpace1D, BsplineSpaceRestriction
from pantr.quad import PointsLattice


class TestBsplineSpaceInit:
    """Test BsplineSpace initialization."""

    def test_valid_initialization_1D(self) -> None:
        """Test valid BsplineSpace initialization with 1D."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        space_1d = BsplineSpace1D(knots, degree)
        space = BsplineSpace([space_1d])

        assert space.dim == 1
        assert space.spaces == (space_1d,)
        assert space.degrees == (2,)

    def test_valid_initialization_2D(self) -> None:
        """Test valid BsplineSpace initialization with 2D."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space = BsplineSpace([space_1d_1, space_1d_2])

        assert space.dim == 2
        assert len(space.spaces) == 2
        assert space.degrees == (2, 1)

    def test_valid_initialization_3D(self) -> None:
        """Test valid BsplineSpace initialization with 3D."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        knots3 = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space_1d_3 = BsplineSpace1D(knots3, 3)
        space = BsplineSpace([space_1d_1, space_1d_2, space_1d_3])

        assert space.dim == 3
        assert len(space.spaces) == 3
        assert space.degrees == (2, 1, 3)

    def test_different_dtype_error(self) -> None:
        """Test that spaces with different dtypes raise ValueError."""
        knots1 = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        knots2 = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        with pytest.raises(ValueError, match="All B-spline spaces must have the same data type"):
            BsplineSpace([space_1d_1, space_1d_2])


class TestBsplineSpaceProperties:
    """Test BsplineSpace properties."""

    def test_dim(self) -> None:
        """Test dim property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])

        assert space.dim == 1

        space_1d_2 = BsplineSpace1D(knots, 2)
        space_2d = BsplineSpace([space_1d, space_1d_2])
        assert space_2d.dim == 2

    def test_spaces(self) -> None:
        """Test spaces property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])

        assert isinstance(space.spaces, tuple)
        assert space.spaces[0] is space_1d

    def test_degrees(self) -> None:
        """Test degrees property."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        knots3 = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space_1d_3 = BsplineSpace1D(knots3, 3)
        space = BsplineSpace([space_1d_1, space_1d_2, space_1d_3])

        assert space.degrees == (2, 1, 3)
        assert isinstance(space.degrees, tuple)

    def test_tolerance(self) -> None:
        """Test tolerance property."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space = BsplineSpace([space_1d_1, space_1d_2])

        tol = max(space_1d_1.tolerance, space_1d_2.tolerance)
        assert space.tolerance == tol

    def test_dtype(self) -> None:
        """Test dtype property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])

        assert space.dtype == space_1d.dtype

    def test_num_basis(self) -> None:
        """Test num_basis property."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space = BsplineSpace([space_1d_1, space_1d_2])

        assert space.num_basis == (3, 2)
        assert isinstance(space.num_basis, tuple)

    def test_num_total_basis(self) -> None:
        """Test num_total_basis property."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space = BsplineSpace([space_1d_1, space_1d_2])

        assert space.num_total_basis == 6
        assert isinstance(space.num_total_basis, int)

    def test_num_intervals(self) -> None:
        """Test num_intervals property."""
        knots1 = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 2)
        space = BsplineSpace([space_1d_1, space_1d_2])

        assert space.num_intervals == (2, 2)
        assert isinstance(space.num_intervals, tuple)

    def test_num_total_intervals(self) -> None:
        """Test num_total_intervals property."""
        knots1 = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 2)
        space = BsplineSpace([space_1d_1, space_1d_2])

        assert space.num_total_intervals == 4
        assert isinstance(space.num_total_intervals, int)

    def test_domain(self) -> None:
        """Test domain property."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [2.0, 2.0, 3.0, 3.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space = BsplineSpace([space_1d_1, space_1d_2])

        domain = space.domain
        assert domain.shape == (2, 2)
        np.testing.assert_array_equal(domain[0, :], [0.0, 1.0])
        np.testing.assert_array_equal(domain[1, :], [2.0, 3.0])


class TestBsplineSpaceMethods:
    """Test BsplineSpace methods."""

    def test_has_Bezier_like_knots_true(self) -> None:
        """Test has_Bezier_like_knots returns True for Bézier-like knots."""
        knots = [1.0, 1.0, 1.0, 3.0, 3.0, 3.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])

        assert space.has_Bezier_like_knots() is True

        # 2D case - both must be Bézier-like
        space_1d_2 = BsplineSpace1D(knots, 2)
        space_2d = BsplineSpace([space_1d, space_1d_2])
        assert space_2d.has_Bezier_like_knots() is True

    def test_has_Bezier_like_knots_false(self) -> None:
        """Test has_Bezier_like_knots returns False for non-Bézier-like knots."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])

        assert space.has_Bezier_like_knots() is False

        # 2D case - if one is not Bézier-like, the whole is not
        knots2 = [1.0, 1.0, 1.0, 3.0, 3.0, 3.0]
        space_1d_2 = BsplineSpace1D(knots2, 2)
        space_2d = BsplineSpace([space_1d, space_1d_2])
        assert space_2d.has_Bezier_like_knots() is False

    def test_tabulate_basis_points_array_1D(self) -> None:
        """Test tabulate_basis with 1D points array."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])

        pts = np.array([[0.0], [0.5], [1.0]], dtype=np.float64)
        basis, first_indices = space.tabulate_basis(pts)

        assert basis.shape == (3, 3)
        assert first_indices.shape == (3, 1)

    def test_tabulate_basis_points_array_2D(self) -> None:
        """Test tabulate_basis with 2D points array."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space = BsplineSpace([space_1d_1, space_1d_2])

        pts = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64)
        basis, first_indices = space.tabulate_basis(pts)

        assert basis.shape == (3, 3, 2)
        assert first_indices.shape == (3, 2)

    def test_tabulate_basis_points_array_3D(self) -> None:
        """Test tabulate_basis with 3D points array."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        knots3 = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space_1d_3 = BsplineSpace1D(knots3, 3)
        space = BsplineSpace([space_1d_1, space_1d_2, space_1d_3])

        pts = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]], dtype=np.float64)
        basis, first_indices = space.tabulate_basis(pts)

        assert basis.shape == (3, 3, 2, 4)
        assert first_indices.shape == (3, 3)

    def test_tabulate_basis_points_lattice_2D(self) -> None:
        """Test tabulate_basis with PointsLattice."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space = BsplineSpace([space_1d_1, space_1d_2])

        pts1 = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        pts2 = np.array([0.0, 1.0], dtype=np.float64)
        lattice = PointsLattice([pts1, pts2])

        basis, first_indices = space.tabulate_basis(lattice)

        assert basis.shape == (3, 2, 3, 2)
        assert first_indices.shape == (3, 2, 2)

    def test_tabulate_basis_points_array_wrong_dimension(self) -> None:
        """Test tabulate_basis with wrong dimension raises ValueError."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])

        pts = np.array([[0.0, 0.0], [0.5, 0.5]], dtype=np.float64)  # 2D points for 1D space
        with pytest.raises(ValueError, match="pts must have 1 columns"):
            space.tabulate_basis(pts)

    def test_tabulate_basis_points_array_not_2D(self) -> None:
        """Test tabulate_basis with non-2D array raises ValueError."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])

        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)  # 1D array
        with pytest.raises(ValueError, match="pts must be a 2D array"):
            space.tabulate_basis(pts)

    def test_tabulate_basis_points_array_outside_domain(self) -> None:
        """Test tabulate_basis with points outside domain raises ValueError."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])

        pts = np.array([[0.0], [1.5]], dtype=np.float64)  # Second point outside domain
        with pytest.raises(ValueError, match="outside the knot vector"):
            space.tabulate_basis(pts)

    def test_tabulate_basis_points_lattice_wrong_dimension(self) -> None:
        """Test tabulate_basis with PointsLattice wrong dimension raises ValueError."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])

        pts1 = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        pts2 = np.array([0.0, 1.0], dtype=np.float64)
        lattice = PointsLattice([pts1, pts2])  # 2D lattice for 1D space
        with pytest.raises(ValueError, match="pts must have 1 columns"):
            space.tabulate_basis(lattice)

    def test_tabulate_basis_points_lattice_outside_domain(self) -> None:
        """Test tabulate_basis with PointsLattice outside domain raises ValueError."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])

        pts = np.array([0.0, 1.5], dtype=np.float64)  # Second point outside domain
        lattice = PointsLattice([pts])
        with pytest.raises(ValueError, match="outside the knot vector"):
            space.tabulate_basis(lattice)


class TestBsplineSpaceEdgeCases:
    """Test BsplineSpace edge cases."""

    def test_empty_spaces_list(self) -> None:
        """Test BsplineSpace with empty spaces list raises error on dtype access."""
        space = BsplineSpace([])
        assert space.dim == 0
        # Empty space will fail on accessing dtype property
        with pytest.raises(IndexError):
            _ = space.dtype

    def test_single_point_in_domain_boundary(self) -> None:
        """Test tabulate_basis with points at domain boundaries."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])

        # Points exactly at boundaries
        pts = np.array([[0.0], [1.0]], dtype=np.float64)
        basis, first_indices = space.tabulate_basis(pts)

        assert basis.shape == (2, 3)
        assert first_indices.shape == (2, 1)

    def test_float32_dtype(self) -> None:
        """Test BsplineSpace with float32 dtype."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])

        assert space.dtype == np.float32

        pts = np.array([[0.0], [0.5], [1.0]], dtype=np.float32)
        basis, _ = space.tabulate_basis(pts)

        assert basis.dtype == np.float32

    def test_different_domains(self) -> None:
        """Test BsplineSpace with spaces having different domains."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [2.0, 2.0, 3.0, 3.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space = BsplineSpace([space_1d_1, space_1d_2])

        domain = space.domain
        np.testing.assert_array_equal(domain[0, :], [0.0, 1.0])
        np.testing.assert_array_equal(domain[1, :], [2.0, 3.0])

        # Points must be in both domains
        pts = np.array([[0.5, 2.5], [0.7, 2.8]], dtype=np.float64)
        basis, _ = space.tabulate_basis(pts)
        assert basis.shape == (2, 3, 2)


class TestBsplineSpaceEvaluation:
    """Regression tests for B-spline basis evaluation.

    These tests store computed values to catch regressions from future changes.
    """

    def test_1D_evaluation_bezier_like(self) -> None:
        """Test 1D evaluation with Bézier-like knots (degree 2)."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])

        pts = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=np.float64)
        basis, first_indices = space.tabulate_basis(pts)

        # Expected values computed from current implementation
        expected_basis = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.5625, 0.375, 0.0625],
                [0.25, 0.5, 0.25],
                [0.0625, 0.375, 0.5625],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        expected_first_indices = np.array([[0], [0], [0], [0], [0]], dtype=np.int_)

        np.testing.assert_allclose(basis, expected_basis, rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(first_indices, expected_first_indices)

    def test_1D_evaluation_multiple_intervals(self) -> None:
        """Test 1D evaluation with multiple intervals."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])

        pts = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=np.float64)
        basis, first_indices = space.tabulate_basis(pts)

        # Hardcoded expected values from current implementation
        expected_basis = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.25, 0.625, 0.125],
                [0.5, 0.5, 0.0],
                [0.125, 0.625, 0.25],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        expected_first_indices = np.array([[0], [0], [1], [1], [1]], dtype=np.int_)

        np.testing.assert_allclose(basis, expected_basis, rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(first_indices, expected_first_indices)

    def test_2D_evaluation_uniform(self) -> None:
        """Test 2D evaluation with uniform knot vectors."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space = BsplineSpace([space_1d_1, space_1d_2])

        pts = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64)
        basis, first_indices = space.tabulate_basis(pts)

        # Hardcoded expected values from current implementation
        expected_basis = np.array(
            [
                [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.125, 0.125], [0.25, 0.25], [0.125, 0.125]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
            ],
            dtype=np.float64,
        )
        expected_first_indices = np.array([[0, 0], [0, 0], [0, 0]], dtype=np.int_)

        np.testing.assert_allclose(basis, expected_basis, rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(first_indices, expected_first_indices)

    def test_2D_evaluation_specific_values(self) -> None:
        """Test 2D evaluation with specific point values."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space = BsplineSpace([space_1d_1, space_1d_2])

        pts = np.array([[0.25, 0.25], [0.75, 0.75]], dtype=np.float64)
        basis, first_indices = space.tabulate_basis(pts)

        # Hardcoded expected values from current implementation
        expected_basis = np.array(
            [
                [
                    [0.421875, 0.140625],
                    [0.28125, 0.09375],
                    [0.046875, 0.015625],
                ],
                [
                    [0.015625, 0.046875],
                    [0.09375, 0.28125],
                    [0.140625, 0.421875],
                ],
            ],
            dtype=np.float64,
        )
        expected_first_indices = np.array([[0, 0], [0, 0]], dtype=np.int_)

        np.testing.assert_allclose(basis, expected_basis, rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(first_indices, expected_first_indices)

    def test_3D_evaluation(self) -> None:
        """Test 3D evaluation."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        knots3 = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space_1d_3 = BsplineSpace1D(knots3, 3)
        space = BsplineSpace([space_1d_1, space_1d_2, space_1d_3])

        pts = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)
        basis, first_indices = space.tabulate_basis(pts)

        # Hardcoded expected values from current implementation (reshaped to (1, 3, 2, 4))
        expected_basis_flat = np.array(
            [
                0.015625,
                0.046875,
                0.046875,
                0.015625,
                0.015625,
                0.046875,
                0.046875,
                0.015625,
                0.03125,
                0.09375,
                0.09375,
                0.03125,
                0.03125,
                0.09375,
                0.09375,
                0.03125,
                0.015625,
                0.046875,
                0.046875,
                0.015625,
                0.015625,
                0.046875,
                0.046875,
                0.015625,
            ],
            dtype=np.float64,
        )
        expected_basis = expected_basis_flat.reshape(1, 3, 2, 4)
        expected_first_indices = np.array([[0, 0, 0]], dtype=np.int_)

        np.testing.assert_allclose(basis, expected_basis, rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(first_indices, expected_first_indices)

    def test_2D_points_lattice_evaluation(self) -> None:
        """Test 2D evaluation with PointsLattice."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space = BsplineSpace([space_1d_1, space_1d_2])

        pts1 = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        pts2 = np.array([0.0, 1.0], dtype=np.float64)
        lattice = PointsLattice([pts1, pts2])

        basis, first_indices = space.tabulate_basis(lattice)

        # Hardcoded expected values from current implementation (reshaped to (3, 2, 3, 2))
        expected_basis_flat = np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.25,
                0.0,
                0.5,
                0.0,
                0.25,
                0.0,
                0.0,
                0.25,
                0.0,
                0.5,
                0.0,
                0.25,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float64,
        )
        expected_basis = expected_basis_flat.reshape(3, 2, 3, 2)
        expected_first_indices_flat = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int_)
        expected_first_indices = expected_first_indices_flat.reshape(3, 2, 2)

        np.testing.assert_allclose(basis, expected_basis, rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(first_indices, expected_first_indices)

    def test_1D_evaluation_degree_1(self) -> None:
        """Test 1D evaluation with degree 1 (linear)."""
        knots = [0.0, 0.0, 0.5, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 1)
        space = BsplineSpace([space_1d])

        pts = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=np.float64)
        basis, first_indices = space.tabulate_basis(pts)

        # Hardcoded expected values from current implementation
        expected_basis = np.array(
            [
                [1.0, 0.0],
                [0.5, 0.5],
                [1.0, 0.0],
                [0.5, 0.5],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )
        expected_first_indices = np.array([[0], [0], [1], [1], [1]], dtype=np.int_)

        np.testing.assert_allclose(basis, expected_basis, rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(first_indices, expected_first_indices)

    def test_1D_evaluation_degree_3(self) -> None:
        """Test 1D evaluation with degree 3 (cubic)."""
        knots = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 3)
        space = BsplineSpace([space_1d])

        pts = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=np.float64)
        basis, first_indices = space.tabulate_basis(pts)

        # Hardcoded expected values from current implementation
        expected_basis = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.421875, 0.421875, 0.140625, 0.015625],
                [0.125, 0.375, 0.375, 0.125],
                [0.015625, 0.140625, 0.421875, 0.421875],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        expected_first_indices = np.array([[0], [0], [0], [0], [0]], dtype=np.int_)

        np.testing.assert_allclose(basis, expected_basis, rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(first_indices, expected_first_indices)

    def test_2D_evaluation_partition_of_unity(self) -> None:
        """Test that 2D basis functions form a partition of unity."""
        knots1 = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 2)
        space = BsplineSpace([space_1d_1, space_1d_2])

        pts = np.array([[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]], dtype=np.float64)
        basis, _ = space.tabulate_basis(pts)

        # Sum over all basis dimensions should equal 1 for each point
        basis_sum = np.sum(basis, axis=(1, 2))
        np.testing.assert_allclose(basis_sum, np.ones(3), rtol=1e-10)

    def test_2D_evaluation_at_boundaries(self) -> None:
        """Test 2D evaluation at domain boundaries."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space = BsplineSpace([space_1d_1, space_1d_2])

        # Test corners of domain
        pts = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        basis, first_indices = space.tabulate_basis(pts)

        # Hardcoded expected values from current implementation (reshaped to (4, 3, 2))
        expected_basis_flat = np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float64,
        )
        expected_basis = expected_basis_flat.reshape(4, 3, 2)
        expected_first_indices = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int_)

        np.testing.assert_allclose(basis, expected_basis, rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(first_indices, expected_first_indices)

    def test_3D_evaluation_partition_of_unity(self) -> None:
        """Test that 3D basis functions form a partition of unity."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        knots3 = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space_1d_3 = BsplineSpace1D(knots3, 3)
        space = BsplineSpace([space_1d_1, space_1d_2, space_1d_3])

        pts = np.array([[0.25, 0.5, 0.5], [0.5, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=np.float64)
        basis, _ = space.tabulate_basis(pts)

        # Sum over all basis dimensions should equal 1 for each point
        basis_sum = np.sum(basis, axis=(1, 2, 3))
        np.testing.assert_allclose(basis_sum, np.ones(3), rtol=1e-10)

    def test_float32_evaluation(self) -> None:
        """Test evaluation with float32 dtype."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])

        pts = np.array([[0.0], [0.5], [1.0]], dtype=np.float32)
        basis, first_indices = space.tabulate_basis(pts)

        # Hardcoded expected values from current implementation
        expected_basis = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.25, 0.5, 0.25],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        expected_first_indices = np.array([[0], [0], [0]], dtype=np.int_)

        np.testing.assert_allclose(basis, expected_basis, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(first_indices, expected_first_indices)

        assert basis.dtype == np.float32
        assert first_indices.dtype == np.int_

    def test_2D_different_domains_evaluation(self) -> None:
        """Test 2D evaluation with spaces having different domains."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [2.0, 2.0, 3.0, 3.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space = BsplineSpace([space_1d_1, space_1d_2])

        pts = np.array([[0.5, 2.5], [0.75, 2.75]], dtype=np.float64)
        basis, first_indices = space.tabulate_basis(pts)

        # Hardcoded expected values from current implementation (reshaped to (2, 3, 2))
        expected_basis_flat = np.array(
            [
                0.125,
                0.125,
                0.25,
                0.25,
                0.125,
                0.125,
                0.015625,
                0.046875,
                0.09375,
                0.28125,
                0.140625,
                0.421875,
            ],
            dtype=np.float64,
        )
        expected_basis = expected_basis_flat.reshape(2, 3, 2)
        expected_first_indices = np.array([[0, 0], [0, 0]], dtype=np.int_)

        np.testing.assert_allclose(basis, expected_basis, rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(first_indices, expected_first_indices)


class TestOutParameter:
    """Test out parameter for BsplineSpace methods."""

    def test_tabulate_basis_out_parameter_1D(self) -> None:
        """Test that out parameter works for tabulate_basis in 1D."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        space_1d = BsplineSpace1D(knots, degree)
        space = BsplineSpace([space_1d])
        pts = np.array([[0.0], [0.5], [1.0]], dtype=np.float64)

        basis1, idx1 = space.tabulate_basis(pts)
        out = np.zeros_like(basis1)
        basis2, idx2 = space.tabulate_basis(pts, out_basis=out)

        np.testing.assert_allclose(basis1, basis2)
        np.testing.assert_array_equal(idx1, idx2)
        np.testing.assert_allclose(out, basis1)
        assert basis2 is out

    def test_tabulate_basis_out_parameter_2D(self) -> None:
        """Test that out parameter works for tabulate_basis in 2D."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        space_1d = BsplineSpace1D(knots, degree)
        space = BsplineSpace([space_1d, space_1d])
        pts = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64)

        basis1, idx1 = space.tabulate_basis(pts)
        out = np.zeros_like(basis1)
        basis2, idx2 = space.tabulate_basis(pts, out_basis=out)

        np.testing.assert_allclose(basis1, basis2)
        np.testing.assert_array_equal(idx1, idx2)
        np.testing.assert_allclose(out, basis1)
        assert basis2 is out

    def test_tabulate_basis_out_wrong_shape(self) -> None:
        """Test that out parameter with wrong shape raises ValueError."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        space_1d = BsplineSpace1D(knots, degree)
        space = BsplineSpace([space_1d])
        pts = np.array([[0.0], [0.5], [1.0]], dtype=np.float64)

        out = np.zeros((3, 2), dtype=np.float64)  # Wrong shape

        with pytest.raises(ValueError, match="Output array has shape"):
            space.tabulate_basis(pts, out_basis=out)

    def test_tabulate_basis_out_wrong_dtype(self) -> None:
        """Test that out parameter with wrong dtype raises ValueError."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        space_1d = BsplineSpace1D(knots, degree)
        space = BsplineSpace([space_1d])
        pts = np.array([[0.0], [0.5], [1.0]], dtype=np.float64)

        basis1, _ = space.tabulate_basis(pts)
        out = np.zeros_like(basis1, dtype=np.float32)  # Wrong dtype

        with pytest.raises(ValueError, match="Output array has dtype"):
            space.tabulate_basis(pts, out_basis=out)

    def test_tabulate_basis_out_first_basis_parameter_1D(self) -> None:
        """Test that out_first_basis parameter works for tabulate_basis in 1D."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        space_1d = BsplineSpace1D(knots, degree)
        space = BsplineSpace([space_1d])
        pts = np.array([[0.0], [0.5], [1.0]], dtype=np.float64)

        basis1, idx1 = space.tabulate_basis(pts)
        out_first = np.zeros_like(idx1)
        basis2, idx2 = space.tabulate_basis(pts, out_first_basis=out_first)

        np.testing.assert_allclose(basis1, basis2)
        np.testing.assert_array_equal(idx1, idx2)
        np.testing.assert_array_equal(out_first, idx1)
        assert idx2 is out_first

    def test_tabulate_basis_out_first_basis_parameter_2D(self) -> None:
        """Test that out_first_basis parameter works for tabulate_basis in 2D."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        space_1d = BsplineSpace1D(knots, degree)
        space = BsplineSpace([space_1d, space_1d])
        pts = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64)

        basis1, idx1 = space.tabulate_basis(pts)
        out_first = np.zeros_like(idx1)
        basis2, idx2 = space.tabulate_basis(pts, out_first_basis=out_first)

        np.testing.assert_allclose(basis1, basis2)
        np.testing.assert_array_equal(idx1, idx2)
        np.testing.assert_array_equal(out_first, idx1)
        assert idx2 is out_first

    def test_tabulate_basis_out_first_basis_wrong_shape(self) -> None:
        """Test that out_first_basis parameter with wrong shape raises ValueError."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        space_1d = BsplineSpace1D(knots, degree)
        space = BsplineSpace([space_1d])
        pts = np.array([[0.0], [0.5], [1.0]], dtype=np.float64)

        out_first = np.zeros((3, 2), dtype=np.int_)  # Wrong shape

        with pytest.raises(ValueError, match="Output array has shape"):
            space.tabulate_basis(pts, out_first_basis=out_first)

    def test_tabulate_basis_out_first_basis_wrong_dtype(self) -> None:
        """Test that out_first_basis parameter with wrong dtype raises ValueError."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        space_1d = BsplineSpace1D(knots, degree)
        space = BsplineSpace([space_1d])
        pts = np.array([[0.0], [0.5], [1.0]], dtype=np.float64)

        out_first = np.zeros((3, 1), dtype=np.float64)  # Wrong dtype (should be int_)

        with pytest.raises(ValueError, match="Output array has dtype"):
            space.tabulate_basis(pts, out_first_basis=out_first)  # type: ignore[arg-type]

    def test_tabulate_basis_out_first_basis_not_writeable(self) -> None:
        """Test that out_first_basis parameter with non-writeable array raises ValueError."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        space_1d = BsplineSpace1D(knots, degree)
        space = BsplineSpace([space_1d])
        pts = np.array([[0.0], [0.5], [1.0]], dtype=np.float64)

        out_first = np.zeros((3, 1), dtype=np.int_)
        out_first.setflags(write=False)

        with pytest.raises(ValueError, match="Output array is not writeable"):
            space.tabulate_basis(pts, out_first_basis=out_first)

    def test_tabulate_basis_points_lattice_1D(self) -> None:
        """Test tabulate_basis with 1D PointsLattice."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        space_1d = BsplineSpace1D(knots, degree)
        space = BsplineSpace([space_1d])
        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        lattice = PointsLattice([pts])

        basis, first_indices = space.tabulate_basis(lattice)

        assert basis.shape == (3, 3)
        assert first_indices.shape == (3, 1)

    def test_tabulate_basis_points_lattice_1D_with_out(self) -> None:
        """Test tabulate_basis with 1D PointsLattice and out parameters."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        space_1d = BsplineSpace1D(knots, degree)
        space = BsplineSpace([space_1d])
        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        lattice = PointsLattice([pts])

        basis1, idx1 = space.tabulate_basis(lattice)
        out_basis = np.zeros_like(basis1)
        out_first = np.zeros_like(idx1)
        basis2, idx2 = space.tabulate_basis(lattice, out_basis=out_basis, out_first_basis=out_first)

        np.testing.assert_allclose(basis1, basis2)
        np.testing.assert_array_equal(idx1, idx2)
        assert basis2 is out_basis
        assert idx2 is out_first

    def test_tabulate_basis_points_lattice_2D_with_out(self) -> None:
        """Test tabulate_basis with 2D PointsLattice and out parameters."""
        knots1 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots2 = [0.0, 0.0, 1.0, 1.0]
        space_1d_1 = BsplineSpace1D(knots1, 2)
        space_1d_2 = BsplineSpace1D(knots2, 1)
        space = BsplineSpace([space_1d_1, space_1d_2])
        pts1 = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        pts2 = np.array([0.0, 1.0], dtype=np.float64)
        lattice = PointsLattice([pts1, pts2])

        basis1, idx1 = space.tabulate_basis(lattice)
        out_basis = np.zeros_like(basis1)
        out_first = np.zeros_like(idx1)
        basis2, idx2 = space.tabulate_basis(lattice, out_basis=out_basis, out_first_basis=out_first)

        np.testing.assert_allclose(basis1, basis2)
        np.testing.assert_array_equal(idx1, idx2)
        assert basis2 is out_basis
        assert idx2 is out_first


def _open_uniform_space_1d(degree: int, n_int: int) -> BsplineSpace1D:
    """Open-uniform 1D space: ``n_int`` unit intervals on ``[0, n_int]``, given degree."""
    knots = (
        [0.0] * (degree + 1) + [float(i) for i in range(1, n_int)] + [float(n_int)] * (degree + 1)
    )
    return BsplineSpace1D(knots, degree)


def _check_restrict(space: BsplineSpace, cell_ids: list[int]) -> BsplineSpaceRestriction:
    """Assert a space restriction reproduces the global basis and DOF map over the window."""
    r = space.restrict(cell_ids)
    sub, l2g = r.space, r.local_to_global_dof
    assert isinstance(r, BsplineSpaceRestriction)
    assert isinstance(sub, BsplineSpace)
    assert not l2g.flags.writeable
    assert l2g.shape == (sub.num_total_basis,)
    assert len(set(l2g.tolist())) == sub.num_total_basis  # bijection into global DOFs

    # Sample the tensor grid of windowed-cell midpoints (inside both windows).
    axis_pts = [
        0.5 * (uk[:-1] + uk[1:])
        for uk in (
            sub.spaces[d].get_unique_knots_and_multiplicity(in_domain=True)[0]
            for d in range(space.dim)
        )
    ]
    grids = np.meshgrid(*axis_pts, indexing="ij")
    pts = np.stack([g.ravel() for g in grids], axis=-1).astype(np.float64)

    gb, gfb = space.tabulate_basis(pts)
    wb, wfb = sub.tabulate_basis(pts)
    np.testing.assert_allclose(wb, gb, atol=1e-12)  # windowed basis == global basis pointwise

    g_nb, w_nb, degs = space.num_basis, sub.num_basis, space.degrees
    for i in range(pts.shape[0]):
        g_axis = [gfb[i, d] + np.arange(degs[d] + 1) for d in range(space.dim)]
        w_axis = [wfb[i, d] + np.arange(degs[d] + 1) for d in range(space.dim)]
        g_flat = np.ravel_multi_index(
            [m.ravel() for m in np.meshgrid(*g_axis, indexing="ij")], g_nb
        )
        w_flat = np.ravel_multi_index(
            [m.ravel() for m in np.meshgrid(*w_axis, indexing="ij")], w_nb
        )
        np.testing.assert_array_equal(l2g[w_flat], g_flat)  # DOF-map ordering correct
    return r


class TestBsplineSpaceRestrict:
    """Tests for BsplineSpace.restrict (windowed sub-space + DOF map)."""

    def test_contiguous_block_2d(self) -> None:
        space = BsplineSpace([_open_uniform_space_1d(2, 4), _open_uniform_space_1d(2, 3)])
        # intervals [1,3) x [0,2); flat C-order over num_intervals (4,3): i*3 + j
        cell_ids = [i * 3 + j for i in (1, 2) for j in (0, 1)]
        r = _check_restrict(space, cell_ids)
        assert r.space.num_intervals == (2, 2)

    def test_full_space_is_bijection(self) -> None:
        space = BsplineSpace([_open_uniform_space_1d(2, 3), _open_uniform_space_1d(1, 4)])
        r = _check_restrict(space, list(range(space.num_total_intervals)))
        assert r.space.num_intervals == space.num_intervals
        assert r.space.num_total_basis == space.num_total_basis
        assert set(r.local_to_global_dof.tolist()) == set(range(space.num_total_basis))

    def test_non_convex_uses_bbox(self) -> None:
        space = BsplineSpace([_open_uniform_space_1d(2, 4), _open_uniform_space_1d(2, 4)])
        corners = [0, 3 * 4 + 3]  # (0,0) and (3,3) over num_intervals (4,4)
        r = _check_restrict(space, corners)
        assert r.space.num_intervals == (4, 4)  # bbox spans everything

    def test_single_cell_1d(self) -> None:
        space = BsplineSpace([_open_uniform_space_1d(3, 5)])
        r = _check_restrict(space, [2])
        assert r.space.num_intervals == (1,)
        assert r.space.num_total_basis == 4  # degree + 1 (Bezier-like window)

    def test_dof_map_matches_global_indices(self) -> None:
        space = BsplineSpace([_open_uniform_space_1d(2, 4)])  # num_basis 6, intervals 4
        r = space.restrict([1, 2])  # intervals [1,3); first basis on interval 1 is index 1
        np.testing.assert_array_equal(r.local_to_global_dof, [1, 2, 3, 4])

    def test_dof_map_explicit_2d(self) -> None:
        # degree-(1,1) space, num_intervals=(3,2), num_basis=(4,3).
        # Restrict to row 1 (all columns): cell_ids [1*2+0, 1*2+1] = [2,3].
        # Per-axis windows: axis-0 -> [1,2), axis-1 -> [0,2) (full).
        # Expected dof_0=[1,2], dof_1=[0,1,2]; meshgrid(ij) -> ravel over (4,3):
        # [1*3+0, 1*3+1, 1*3+2, 2*3+0, 2*3+1, 2*3+2] = [3,4,5,6,7,8].
        space = BsplineSpace([_open_uniform_space_1d(1, 3), _open_uniform_space_1d(1, 2)])
        r = space.restrict([2, 3])
        np.testing.assert_array_equal(r.local_to_global_dof, [3, 4, 5, 6, 7, 8])

    def test_returns_restriction_namedtuple(self) -> None:
        space = BsplineSpace([_open_uniform_space_1d(2, 3), _open_uniform_space_1d(2, 3)])
        r = space.restrict([0])
        assert isinstance(r, BsplineSpaceRestriction)
        assert isinstance(r.space, BsplineSpace)

    def test_empty_raises(self) -> None:
        space = BsplineSpace([_open_uniform_space_1d(2, 3)])
        with pytest.raises(ValueError, match="non-empty"):
            space.restrict([])

    def test_out_of_range_raises(self) -> None:
        space = BsplineSpace([_open_uniform_space_1d(2, 3)])
        with pytest.raises(IndexError):
            space.restrict([space.num_total_intervals])
        with pytest.raises(IndexError):
            space.restrict([-1])

    def test_non_integer_raises(self) -> None:
        space = BsplineSpace([_open_uniform_space_1d(2, 3)])
        with pytest.raises(TypeError, match="integer"):
            space.restrict([0.0])

    def test_periodic_rejected(self) -> None:
        from pantr.bspline import create_uniform_periodic_knots  # noqa: PLC0415

        knots = create_uniform_periodic_knots(num_intervals=4, degree=2)
        per = BsplineSpace1D(knots, 2, periodic=True)
        space = BsplineSpace([per])
        with pytest.raises(ValueError, match="periodic"):
            space.restrict([0])
