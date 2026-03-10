"""Tests for bspline_space_1D module."""

from collections.abc import Callable

import numpy as np
import pytest

from pantr._bspline_basis_core import (
    _compute_basis_Cox_de_Boor_impl,
    _tabulate_Bspline_basis_Bernstein_like_1D,
)
from pantr._bspline_extraction import (
    _tabulate_Bspline_Bezier_1D_extraction_impl,
    _tabulate_Bspline_cardinal_1D_extraction_impl,
    _tabulate_Bspline_Lagrange_1D_extraction_impl,
)
from pantr._bspline_knots import (
    _check_spline_info,
    _get_Bspline_cardinal_intervals_1D_impl,
    _get_Bspline_num_basis_1D_impl,
    _get_last_knot_smaller_equal_impl,
    _get_multiplicity_of_first_knot_in_domain_impl,
    _get_unique_knots_and_multiplicity_impl,
    _is_in_domain_impl,
)
from pantr._bspline_space_factory import (
    create_cardinal_Bspline_knot_vector,
    create_uniform_open_knot_vector,
    create_uniform_periodic_knot_vector,
)
from pantr.basis import (
    LagrangeVariant,
    tabulate_Bernstein_basis_1D,
    tabulate_cardinal_Bspline_basis_1D,
    tabulate_Lagrange_basis_1D,
)
from pantr.bspline_space_1D import BsplineSpace1D
from pantr.change_basis import (
    compute_Bernstein_to_cardinal_change_basis_1D,
    compute_Bernstein_to_Lagrange_change_basis_1D,
    compute_cardinal_to_Bernstein_change_basis_1D,
    compute_Lagrange_to_Bernstein_change_basis_1D,
)
from pantr.tolerance import get_strict_tolerance


class TestBsplineSpace1DInit:
    """Test BsplineSpace1D initialization."""

    def test_valid_initialization(self) -> None:
        """Test valid BsplineSpace1D initialization."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)

        assert spline.degree == 2  # noqa: PLR2004
        assert spline.periodic is False
        np.testing.assert_array_equal(spline.knots, np.array(knots))

    def test_zero_degree_initialization(self) -> None:
        """Test valid BsplineSpace1D initialization."""
        knots = [0.0, 1.0]
        degree = 0
        spline = BsplineSpace1D(knots, degree)

        assert spline.degree == 0
        assert spline.periodic is False
        np.testing.assert_array_equal(spline.knots, np.array(knots))

    def test_periodic_initialization(self) -> None:
        """Test periodic BsplineSpace1D initialization."""
        knots = create_uniform_periodic_knot_vector(num_intervals=3, degree=2, domain=(0.0, 1.0))
        degree = 2
        spline = BsplineSpace1D(knots, degree, periodic=True)

        assert spline.degree == 2  # noqa: PLR2004
        assert spline.periodic is True

    def test_integer_knots_conversion(self) -> None:
        """Test that integer knots are converted to float64."""
        knots = [0, 0, 0, 1, 1, 1]
        degree = 2
        spline = BsplineSpace1D(knots, degree)

        assert spline.dtype == np.float64
        np.testing.assert_array_equal(spline.knots, np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))

    def test_negative_degree_error(self) -> None:
        """Test that negative degree raises ValueError."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        with pytest.raises(ValueError, match="degree must be non-negative"):
            BsplineSpace1D(knots, -1)

    def test_insufficient_knots_error(self) -> None:
        """Test that insufficient knots raise ValueError."""
        knots = [0.0, 1.0]
        with pytest.raises(ValueError, match="knots must have at least"):
            BsplineSpace1D(knots, 2)

    def test_non_decreasing_knots_error(self) -> None:
        """Test that non-decreasing knots raise ValueError."""
        knots = [0.0, 1.0, 0.5, 1.0, 1.0, 1.0]
        with pytest.raises(ValueError, match="knots must be non-decreasing"):
            BsplineSpace1D(knots, 2)

    def test_invalid_knot_type_error(self) -> None:
        """Test that invalid knot type raises TypeError."""
        knots = "invalid"
        with pytest.raises(
            (TypeError, ValueError),
            match=r"knots must be a 1D numpy array or Python list|knots type must be float",
        ):
            BsplineSpace1D(knots, 2)

    def test_snap_knots_disabled(self) -> None:
        """Test initialization with snap_knots disabled."""
        tol = get_strict_tolerance(np.float64)
        knots = [0.0, 0.0, 0.0 + tol, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree, snap_knots=False)

        # Knots should remain unchanged
        np.testing.assert_array_equal(spline.knots, np.array(knots))


class TestBsplineSpace1DProperties:
    """Test BsplineSpace1D properties."""

    def test_degree_property(self) -> None:
        """Test degree property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)
        assert spline.degree == degree

    def test_knots_property(self) -> None:
        """Test knots property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)
        np.testing.assert_array_equal(spline.knots, np.array(knots))

    def test_periodic_property(self) -> None:
        """Test periodic property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree, periodic=True)
        assert spline.periodic is True

    def test_tolerance_property(self) -> None:
        """Test tolerance property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)
        assert spline.tolerance > 0

    def test_dtype_property(self) -> None:
        """Test dtype property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)
        assert spline.dtype == np.float64


class TestBsplineSpace1DMethods:
    """Test BsplineSpace1D methods."""

    def test_get_num_basis_non_periodic(self) -> None:
        """Test get_num_basis for non-periodic spline."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)
        assert spline.num_basis == 3  # noqa: PLR2004

    def test_get_num_basis_periodic(self) -> None:
        """Test get_num_basis for periodic spline."""
        degree = 2
        knots = create_uniform_periodic_knot_vector(
            num_intervals=3, degree=degree, domain=(0.0, 1.0)
        )
        spline = BsplineSpace1D(knots, degree, periodic=True)
        # For periodic splines, the number of basis functions is reduced
        assert spline.num_basis == 3  # noqa: PLR2004

    def test_get_unique_knots_and_multiplicity_full(self) -> None:
        """Test _get_unique_knots_and_multiplicity for full knot vector."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)
        unique_knots, multiplicities = spline.get_unique_knots_and_multiplicity(in_domain=False)

        expected_unique = np.array([0.0, 1.0])
        expected_mults = np.array([3, 3])
        np.testing.assert_array_almost_equal(unique_knots, expected_unique)
        np.testing.assert_array_equal(multiplicities, expected_mults)

    def test_get_unique_knots_and_multiplicity_domain(self) -> None:
        """Test _get_unique_knots_and_multiplicity for domain only."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)
        unique_knots, multiplicities = spline.get_unique_knots_and_multiplicity(in_domain=True)

        expected_unique = np.array([0.0, 1.0])
        expected_mults = np.array([3, 3])
        np.testing.assert_array_almost_equal(unique_knots, expected_unique)
        np.testing.assert_array_equal(multiplicities, expected_mults)

    def test_get_num_intervals(self) -> None:
        """Test get_num_intervals method."""
        num_intervals = 2
        degree = 2
        knots = create_uniform_open_knot_vector(num_intervals, degree)
        spline = BsplineSpace1D(knots, degree)
        assert spline.num_intervals == num_intervals

    def test_get_domain(self) -> None:
        """Test get_domain method."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)
        domain = spline.domain
        np.testing.assert_allclose(domain, (knots[degree], knots[-degree - 1]))

    def test_has_left_end_open_true(self) -> None:
        """Test has_left_end_open returns True for open left end."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)
        assert spline.has_left_end_open() is True

    def test_has_left_end_open_false(self) -> None:
        """Test has_left_end_open returns False for non-open left end."""
        knots = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)
        assert spline.has_left_end_open() is False

    def test_has_right_end_open_true(self) -> None:
        """Test has_right_end_open returns True for open right end."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)
        assert spline.has_right_end_open() is True

    def test_has_right_end_open_false(self) -> None:
        """Test has_right_end_open returns False for non-open right end."""
        knots = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)
        assert spline.has_right_end_open() is False

    def test_has_open_knots_true(self) -> None:
        """Test has_open_knots returns True when both ends are open."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)
        assert spline.has_open_knots() is True

    def test_has_open_knots_false(self) -> None:
        """Test has_open_knots returns False when ends are not open."""
        knots = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)
        assert spline.has_open_knots() is False

    def test_has_Bezier_like_knots_true(self) -> None:
        """Test has_Bezier_like_knots returns True for Bézier-like configuration."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)
        assert bool(spline.has_Bezier_like_knots()) is True

    def test_has_Bezier_like_knots_false(self) -> None:
        """Test has_Bezier_like_knots returns False for non-Bézier-like configuration."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)
        assert bool(spline.has_Bezier_like_knots()) is False

    def test_has_Bezier_like_knots_periodic_false(self) -> None:
        """Test has_Bezier_like_knots returns False for periodic splines."""
        # Use a valid periodic knot vector
        degree = 2
        knots = create_uniform_periodic_knot_vector(
            num_intervals=3, degree=degree, domain=(0.0, 1.0)
        )
        spline = BsplineSpace1D(knots, degree, periodic=True)
        assert bool(spline.has_Bezier_like_knots()) is False

    def test_get_cardinal_intervals(self) -> None:
        """Test get_cardinal_intervals method."""
        knots = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)
        result = spline.get_cardinal_intervals()

        # Should have 4 intervals, middle ones should be cardinal
        expected = np.array([False, True, True, False])
        np.testing.assert_array_equal(result, expected)


class TestBsplineSpace1DWithKnotGenerators:
    """Test BsplineSpace1D with knot vector generators."""

    def test_with_uniform_open_knot_vector(self) -> None:
        """Test BsplineSpace1D with uniform open knot vector."""
        knots = create_uniform_open_knot_vector(num_intervals=2, degree=2, domain=(0.0, 1.0))
        degree = 2
        spline = BsplineSpace1D(knots, degree)

        assert spline.degree == degree
        assert spline.periodic is False
        assert spline.has_open_knots() is True
        assert spline.domain == (knots[degree], knots[-degree - 1])

    def test_with_uniform_periodic_knot_vector(self) -> None:
        """Test BsplineSpace1D with uniform periodic knot vector."""
        degree = 2
        knots = create_uniform_periodic_knot_vector(
            num_intervals=3, degree=degree, domain=(0.0, 1.0)
        )
        spline = BsplineSpace1D(knots, degree, periodic=True)

        assert spline.degree == degree
        assert spline.periodic is True
        assert spline.domain == (knots[degree], knots[-degree - 1])

    def test_with_cardinal_bspline_knot_vector(self) -> None:
        """Test BsplineSpace1D with cardinal B-spline knot vector."""
        degree = 2
        knots = create_cardinal_Bspline_knot_vector(2, degree)
        spline = BsplineSpace1D(knots, degree)

        assert spline.degree == degree
        assert spline.periodic is False
        assert spline.domain == (knots[degree], knots[-degree - 1])


class TestBsplineSpace1DEdgeCases:
    """Test BsplineSpace1D edge cases."""

    def test_degree_zero(self) -> None:
        """Test BsplineSpace1D with degree 0."""
        knots = [0.0, 1.0]
        degree = 0
        spline = BsplineSpace1D(knots, degree)

        assert spline.degree == degree
        assert spline.num_basis == 1
        np.testing.assert_allclose(spline.domain, (knots[degree], knots[-degree - 1]))

    def test_single_interval(self) -> None:
        """Test BsplineSpace1D with single interval."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)

        assert spline.num_intervals == 1
        assert bool(spline.has_Bezier_like_knots()) is True

    def test_high_degree(self) -> None:
        """Test BsplineSpace1D with high degree."""
        degree = 5
        knots = [0.0] * (degree + 1) + [1.0] * (degree + 1)
        spline = BsplineSpace1D(knots, degree)

        assert spline.degree == degree
        assert spline.num_basis == (degree + 1)
        assert bool(spline.has_Bezier_like_knots()) is True

    def test_float32_precision(self) -> None:
        """Test BsplineSpace1D with float32 precision."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
        degree = 2
        spline = BsplineSpace1D(knots, degree)

        assert spline.dtype == np.float32
        assert spline.tolerance > 0


class TestBsplineSpace1DIntegration:
    """Integration tests for BsplineSpace1D."""

    def test_consistency_across_methods(self) -> None:
        """Test consistency across different BsplineSpace1D methods."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)

        # Test that domain indices are consistent
        domain = spline.domain
        unique_knots, _ = spline.get_unique_knots_and_multiplicity(in_domain=True)

        assert domain[0] == unique_knots[0]
        np.testing.assert_array_almost_equal(domain[1], unique_knots[-1])

        # Test that number of intervals is consistent
        num_intervals = spline.num_intervals
        assert num_intervals == len(unique_knots) - 1

    def test_periodic_vs_non_periodic_consistency(self) -> None:
        """Test consistency between periodic and non-periodic versions."""
        # Create equivalent knot vectors
        degree = 2
        knots_open = create_uniform_open_knot_vector(
            num_intervals=3, degree=degree, domain=(0.0, 1.0)
        )
        knots_periodic = create_uniform_periodic_knot_vector(
            num_intervals=3, degree=degree, domain=(0.0, 1.0)
        )

        spline_open = BsplineSpace1D(knots_open, degree, periodic=False)
        spline_periodic = BsplineSpace1D(knots_periodic, degree, periodic=True)

        # Both should have the same domain
        assert spline_open.domain == spline_periodic.domain

        # Both should have the same number of intervals
        assert spline_open.num_intervals == spline_periodic.num_intervals

        # But different number of basis functions
        assert spline_open.num_basis != spline_periodic.num_basis

    def test_knot_snapping_consistency(self) -> None:
        """Test that knot snapping doesn't break consistency."""
        # Create knots with small numerical differences
        knots = [0.0, 0.0, 0.0, 0.5000000001, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree, snap_knots=True)

        # After snapping, should still be valid
        assert spline.num_basis > 0
        assert spline.num_intervals > 0

        # Domain should be well-defined
        domain = spline.domain
        assert domain[0] < domain[1]


class TestAssertSplineInfo:
    """Test the _check_spline_info validation function."""

    def test_valid_inputs(self) -> None:
        """Test that valid inputs don't raise assertions."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        _check_spline_info(knots, degree)

    def test_invalid_degree(self) -> None:
        """Test that negative degree raises ValueError."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = -1
        with pytest.raises(ValueError, match="degree must be non-negative"):
            _check_spline_info(knots, degree)

    def test_insufficient_knots(self) -> None:
        """Test that insufficient knots raise ValueError."""
        knots = np.array([0.0, 1.0], dtype=np.float64)
        degree = 2
        with pytest.raises(ValueError, match="knots must have at least"):
            _check_spline_info(knots, degree)

    def test_non_decreasing_knots(self) -> None:
        """Test that non-decreasing knots raise ValueError."""
        knots = np.array([0.0, 1.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        with pytest.raises(ValueError, match="knots must be non-decreasing"):
            _check_spline_info(knots, degree)


class TestGetMultiplicityOfFirstKnotInDomain:
    """Test the _get_multiplicity_of_first_knot_in_domain_impl function."""

    def test_open_knot_vector(self) -> None:
        """Test multiplicity calculation for open knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        result = _get_multiplicity_of_first_knot_in_domain_impl(knots, degree, tol)
        # First knot in domain (index 2) has multiplicity 3
        assert result == 3  # noqa: PLR2004

    def test_periodic_knot_vector(self) -> None:
        """Test multiplicity calculation for periodic knot vector."""
        knots = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        result = _get_multiplicity_of_first_knot_in_domain_impl(knots, degree, tol)
        assert result == 1  # First knot in domain (index 2) has multiplicity 1


class TestGetUniqueKnotsAndMultiplicity:
    """Test the _get_unique_knots_and_multiplicity_impl function."""

    def test_open_knot_vector_full(self) -> None:
        """Test unique knots extraction for full knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        unique_knots, multiplicities = _get_unique_knots_and_multiplicity_impl(
            knots, degree, tol, in_domain=False
        )
        expected_unique = np.array([0.0, 1.0])
        expected_mults = np.array([3, 3])
        np.testing.assert_array_almost_equal(unique_knots, expected_unique)
        np.testing.assert_array_equal(multiplicities, expected_mults)

    def test_open_knot_vector_domain_only(self) -> None:
        """Test unique knots extraction for domain only."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        unique_knots, multiplicities = _get_unique_knots_and_multiplicity_impl(
            knots, degree, tol, in_domain=True
        )
        expected_unique = np.array([0.0, 1.0])
        expected_mults = np.array([3, 3])
        np.testing.assert_array_almost_equal(unique_knots, expected_unique)
        np.testing.assert_array_equal(multiplicities, expected_mults)

    def test_periodic_knot_vector(self) -> None:
        """Test unique knots extraction for periodic knot vector."""
        knots = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        unique_knots, multiplicities = _get_unique_knots_and_multiplicity_impl(
            knots, degree, tol, in_domain=True
        )
        expected_unique = np.array([0.0, 0.5, 1.0])
        expected_mults = np.array([1, 1, 1])
        np.testing.assert_array_almost_equal(unique_knots, expected_unique)
        np.testing.assert_array_equal(multiplicities, expected_mults)


class TestIsInDomain:
    """Test the _is_in_domain_impl function."""

    def test_points_in_domain(self) -> None:
        """Test that points within domain return True."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        tol = 1e-10
        result = _is_in_domain_impl(knots, degree, pts, tol)
        np.testing.assert_array_equal(result, [True, True, True])

    def test_points_outside_domain(self) -> None:
        """Test that points outside domain return False."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        pts = np.array([-0.1, 1.1], dtype=np.float64)
        tol = 1e-10
        result = _is_in_domain_impl(knots, degree, pts, tol)
        np.testing.assert_array_equal(result, [False, False])

    def test_boundary_points(self) -> None:
        """Test that boundary points return True."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        pts = np.array([0.0, 1.0], dtype=np.float64)
        tol = 1e-10
        result = _is_in_domain_impl(knots, degree, pts, tol)
        np.testing.assert_array_equal(result, [True, True])


class TestComputeNumBasis:
    """Test the compute_num_basis_impl function."""

    def test_non_periodic_open(self) -> None:
        """Test basis count for non-periodic open knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        periodic = False
        tol = 1e-10
        result = _get_Bspline_num_basis_1D_impl(knots, degree, periodic, tol)
        # knots.size - degree - 1 = 6 - 2 - 1 = 3
        assert result == 3  # noqa: PLR2004

    def test_periodic_knot_vector(self) -> None:
        """Test basis count for periodic knot vector."""
        knots = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        degree = 2
        periodic = True
        tol = 1e-10
        result = _get_Bspline_num_basis_1D_impl(knots, degree, periodic, tol)
        # For periodic: num_basis = knots.size - degree - 1 - regularity - 1
        # regularity = degree - multiplicity_of_first_knot_in_domain
        # multiplicity_of_first_knot_in_domain = 1
        # regularity = 2 - 1 = 1
        # num_basis = 7 - 2 - 1 - 1 - 1 = 2
        assert result == 2  # noqa: PLR2004


class TestGetLastKnotSmallerEqual:
    """Test the _get_last_knot_smaller_equal_impl function."""

    def test_basic_functionality(self) -> None:
        """Test basic knot index finding."""
        knots = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        pts = np.array([0.3, 0.7, 1.2, 1.8], dtype=np.float64)
        result = _get_last_knot_smaller_equal_impl(knots, pts)
        expected = np.array([0, 1, 2, 3])  # Indices of knots <= pts
        np.testing.assert_array_equal(result, expected)

    def test_knots_with_repetitions(self) -> None:
        """Test knots with repetitions index finding."""
        knots = np.array([0.0, 0.5, 1.0, 1.0, 1.5, 2.0], dtype=np.float64)
        pts = np.array([0.3, 0.7, 1.2, 1.8], dtype=np.float64)
        result = _get_last_knot_smaller_equal_impl(knots, pts)
        expected = np.array([0, 1, 3, 4])  # Indices of knots <= pts
        np.testing.assert_array_equal(result, expected)

    def test_exact_knot_matches(self) -> None:
        """Test when points exactly match knots."""
        knots = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        result = _get_last_knot_smaller_equal_impl(knots, pts)
        expected = np.array([0, 1, 2])
        np.testing.assert_array_equal(result, expected)


class TestEvaluateBasisCoxDeBoor:
    """Test the _compute_basis_Cox_de_Boor_impl function."""

    def test_bezier_like_evaluation(self) -> None:
        """Test evaluation for Bézier-like knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        periodic = False
        tol = 1e-10
        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)

        basis = np.empty((3, 3), dtype=np.float64)
        first_basis = np.empty(3, dtype=np.int_)
        _compute_basis_Cox_de_Boor_impl(knots, degree, periodic, tol, pts, basis, first_basis)

        # Check shape
        assert basis.shape == (3, 3)
        assert first_basis.shape == (3,)

        # Check partition of unity
        sums = np.sum(basis, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones_like(sums))

    def test_general_knot_vector(self) -> None:
        """Test evaluation for general knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        periodic = False
        tol = 1e-10
        pts = np.array([0.25, 0.75], dtype=np.float64)

        basis = np.empty((2, 3), dtype=np.float64)
        first_basis = np.empty(2, dtype=np.int_)
        _compute_basis_Cox_de_Boor_impl(knots, degree, periodic, tol, pts, basis, first_basis)

        # Check shape
        assert basis.shape == (2, 3)
        assert first_basis.shape == (2,)

        # Check partition of unity
        sums = np.sum(basis, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones_like(sums))

    def test_periodic_evaluation(self) -> None:
        """Test evaluation for periodic knot vector."""
        knots = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        degree = 2
        periodic = True
        tol = 1e-10
        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)

        basis = np.empty((3, 3), dtype=np.float64)
        first_basis = np.empty(3, dtype=np.int_)
        _compute_basis_Cox_de_Boor_impl(knots, degree, periodic, tol, pts, basis, first_basis)

        # Check shape
        assert basis.shape == (3, 3)
        assert first_basis.shape == (3,)

        # Check partition of unity
        sums = np.sum(basis, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones_like(sums))

    def test_out_parameter(self) -> None:
        """Test that C-style output parameters work correctly."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        periodic = False
        tol = 1e-10
        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)

        # Test with float64
        basis1 = np.empty((3, 3), dtype=np.float64)
        first_basis1 = np.empty(3, dtype=np.int_)
        _compute_basis_Cox_de_Boor_impl(knots, degree, periodic, tol, pts, basis1, first_basis1)
        assert basis1.shape == (3, 3)
        assert basis1.dtype == np.float64
        assert first_basis1.shape == (3,)

        # Test with float32
        knots_f32 = knots.astype(np.float32)
        pts_f32 = pts.astype(np.float32)
        basis2 = np.empty((3, 3), dtype=np.float32)
        first_basis2 = np.empty(3, dtype=np.int_)
        _compute_basis_Cox_de_Boor_impl(
            knots_f32, degree, periodic, tol, pts_f32, basis2, first_basis2
        )
        assert basis2.dtype == np.float32

        # Verify results are consistent (within float32 precision)
        np.testing.assert_array_almost_equal(basis1, basis2, decimal=6)
        np.testing.assert_array_equal(first_basis1, first_basis2)


class TestGetCardinalIntervals:
    """Test the _get_Bspline_cardinal_intervals_1D_impl function."""

    def test_uniform_knot_vector(self) -> None:
        """Test cardinal intervals for uniform knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        result = _get_Bspline_cardinal_intervals_1D_impl(knots, degree, tol)

        # Should have 4 intervals, middle ones should be cardinal
        expected = np.array([False, True, True, False])
        np.testing.assert_array_equal(result, expected)

    def test_non_uniform_knot_vector(self) -> None:
        """Test cardinal intervals for non-uniform knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 5.0, 5.0, 5.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        result = _get_Bspline_cardinal_intervals_1D_impl(knots, degree, tol)

        # Should have 4 intervals, some might be cardinal due to uniform spacing in some regions
        expected = np.array([False, True, False, False])
        np.testing.assert_array_equal(result, expected)

    def test_all_multiplicity_greater_than_one(self) -> None:
        """Test when all knots have multiplicity > 1."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        result = _get_Bspline_cardinal_intervals_1D_impl(knots, degree, tol)

        # Should return all False
        expected = np.array([False])
        np.testing.assert_array_equal(result, expected)


class TestCreateBsplineBezierExtractionOperators:
    """Test the _create_bspline_Bezier_extraction_operators_impl function."""

    def test_bezier_like_knot_vector(self) -> None:
        """Test extraction operators for Bézier-like knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        result = _tabulate_Bspline_Bezier_1D_extraction_impl(knots, degree, tol)

        # Should have 1 interval, 3x3 extraction matrix
        assert result.shape == (1, 3, 3)

        # For Bézier-like knots, extraction matrix should be identity
        np.testing.assert_array_almost_equal(result[0], np.eye(3))

    def test_general_knot_vector(self) -> None:
        """Test extraction operators for general knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        result = _tabulate_Bspline_Bezier_1D_extraction_impl(knots, degree, tol)

        # Should have 2 intervals, 3x3 extraction matrices
        assert result.shape == (2, 3, 3)

        # Check that matrices are not identity (since not Bézier-like)
        assert not np.allclose(result[0], np.eye(3))
        assert not np.allclose(result[1], np.eye(3))

    def test_negative_tolerance_error(self) -> None:
        """Test that negative tolerance raises ValueError."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        with pytest.raises(ValueError, match="tol must be positive"):
            _tabulate_Bspline_Bezier_1D_extraction_impl(knots, degree, -1.0)

    def test_public_method(self) -> None:
        """Test the public tabulate_Bezier_extraction_operators method."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)
        result = spline.tabulate_Bezier_extraction_operators()

        # Should have correct shape
        assert result.shape == (2, 3, 3)

        # Should match the implementation
        expected = _tabulate_Bspline_Bezier_1D_extraction_impl(
            np.array(knots, dtype=np.float64), degree, spline.tolerance
        )
        np.testing.assert_array_almost_equal(result, expected)


class TestCreateBsplineLagrangeExtractionOperators:
    """Test the tabulate_Lagrange_extraction_operators method and implementation."""

    def test_bezier_like_knot_vector(self) -> None:
        """Test Lagrange extraction operators for Bézier-like knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        result = _tabulate_Bspline_Lagrange_1D_extraction_impl(knots, degree, tol)

        # Should have 1 interval, 3x3 extraction matrix
        assert result.shape == (1, 3, 3)

        # For Bézier-like knots, extraction matrix should not be identity
        # (since it transforms from Lagrange to B-spline, not Bernstein)
        assert not np.allclose(result[0], np.eye(3))

    def test_general_knot_vector(self) -> None:
        """Test Lagrange extraction operators for general knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        result = _tabulate_Bspline_Lagrange_1D_extraction_impl(knots, degree, tol)

        # Should have 2 intervals, 3x3 extraction matrices
        assert result.shape == (2, 3, 3)

        # Check that matrices are not identity
        assert not np.allclose(result[0], np.eye(3))
        assert not np.allclose(result[1], np.eye(3))

    def test_multiple_intervals(self) -> None:
        """Test Lagrange extraction operators for multiple intervals."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        result = _tabulate_Bspline_Lagrange_1D_extraction_impl(knots, degree, tol)

        # Should have 3 intervals, 3x3 extraction matrices
        assert result.shape == (3, 3, 3)

        # All matrices should be square and of correct size
        for i in range(3):
            assert result[i].shape == (3, 3)

    def test_negative_tolerance_error(self) -> None:
        """Test that negative tolerance raises ValueError."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        with pytest.raises(ValueError, match="tol must be positive"):
            _tabulate_Bspline_Lagrange_1D_extraction_impl(knots, degree, -1.0)

    def test_public_method(self) -> None:
        """Test the public tabulate_Lagrange_extraction_operators method."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)
        result = spline.tabulate_Lagrange_extraction_operators()

        # Should have correct shape
        assert result.shape == (2, 3, 3)

        # Should match the implementation
        expected = _tabulate_Bspline_Lagrange_1D_extraction_impl(
            np.array(knots, dtype=np.float64), degree, spline.tolerance
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_different_degrees(self) -> None:
        """Test Lagrange extraction operators for different degrees."""
        for degree in [1, 2, 3, 4]:
            knots = [0.0] * (degree + 1) + [1.0] * (degree + 1)
            tol = 1e-10
            result = _tabulate_Bspline_Lagrange_1D_extraction_impl(
                np.array(knots, dtype=np.float64), degree, tol
            )

            # Should have 1 interval, (degree+1)x(degree+1) extraction matrix
            assert result.shape == (1, degree + 1, degree + 1)

    def test_float32_precision(self) -> None:
        """Test Lagrange extraction operators with float32 precision."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
        degree = 2
        tol = 1e-6
        result = _tabulate_Bspline_Lagrange_1D_extraction_impl(knots, degree, tol)

        assert result.dtype == np.float32
        assert result.shape == (1, 3, 3)


class TestCreateBsplineCardinalExtractionOperators:
    """Test the tabulate_cardinal_extraction_operators method and implementation."""

    def test_bezier_like_knot_vector(self) -> None:
        """Test cardinal extraction operators for Bézier-like knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        result = _tabulate_Bspline_cardinal_1D_extraction_impl(knots, degree, tol)

        # Should have 1 interval, 3x3 extraction matrix
        assert result.shape == (1, 3, 3)

        # For Bézier-like knots (non-cardinal), extraction matrix should not be identity
        assert not np.allclose(result[0], np.eye(3))

    def test_general_knot_vector(self) -> None:
        """Test cardinal extraction operators for general knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        result = _tabulate_Bspline_cardinal_1D_extraction_impl(knots, degree, tol)

        # Should have 2 intervals, 3x3 extraction matrices
        assert result.shape == (2, 3, 3)

        # Check that matrices are not identity (since not cardinal intervals)
        assert not np.allclose(result[0], np.eye(3))
        assert not np.allclose(result[1], np.eye(3))

    def test_cardinal_intervals_identity(self) -> None:
        """Test that cardinal intervals have identity extraction matrices."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        result = _tabulate_Bspline_cardinal_1D_extraction_impl(knots, degree, tol)

        # Should have 4 intervals
        assert result.shape == (4, 3, 3)

        # Get which intervals are cardinal
        cardinal_intervals = _get_Bspline_cardinal_intervals_1D_impl(knots, degree, tol)

        # Cardinal intervals should have identity matrices
        for i in np.where(cardinal_intervals)[0]:
            np.testing.assert_array_almost_equal(result[i], np.eye(3))

        # Non-cardinal intervals should not be identity
        for i in np.where(~cardinal_intervals)[0]:
            assert not np.allclose(result[i], np.eye(3))

    def test_negative_tolerance_error(self) -> None:
        """Test that negative tolerance raises ValueError."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        with pytest.raises(ValueError, match="tol must be positive"):
            _tabulate_Bspline_cardinal_1D_extraction_impl(knots, degree, -1.0)

    def test_public_method(self) -> None:
        """Test the public tabulate_cardinal_extraction_operators method."""
        knots = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)
        result = spline.tabulate_cardinal_extraction_operators()

        # Should have correct shape
        assert result.shape == (4, 3, 3)

        # Should match the implementation
        expected = _tabulate_Bspline_cardinal_1D_extraction_impl(
            np.array(knots, dtype=np.float64), degree, spline.tolerance
        )
        np.testing.assert_array_almost_equal(result, expected)

        # Verify cardinal intervals are identity
        cardinal_intervals = spline.get_cardinal_intervals()
        for i in np.where(cardinal_intervals)[0]:
            np.testing.assert_array_almost_equal(result[i], np.eye(3))

    def test_different_degrees(self) -> None:
        """Test cardinal extraction operators for different degrees."""
        for degree in [1, 2, 3]:
            knots = [0.0] * (degree + 1) + [1.0, 2.0] + [3.0] * (degree + 1)
            tol = 1e-10
            result = _tabulate_Bspline_cardinal_1D_extraction_impl(
                np.array(knots, dtype=np.float64), degree, tol
            )

            # Should have correct number of intervals
            num_intervals = len(knots) - 2 * (degree + 1) + 1
            assert result.shape == (num_intervals, degree + 1, degree + 1)

    def test_float32_precision(self) -> None:
        """Test cardinal extraction operators with float32 precision."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0], dtype=np.float32)
        degree = 2
        tol = 1e-6
        result = _tabulate_Bspline_cardinal_1D_extraction_impl(knots, degree, tol)

        assert result.dtype == np.float32
        assert result.shape == (3, 3, 3)

    def test_all_intervals_cardinal(self) -> None:
        """Test when all intervals are cardinal (uniform spacing)."""
        # Create uniform knot vector with cardinal intervals
        knots = create_uniform_open_knot_vector(num_intervals=3, degree=2, domain=(0.0, 1.0))
        degree = 2
        tol = 1e-10
        result = _tabulate_Bspline_cardinal_1D_extraction_impl(knots, degree, tol)

        # Check cardinal intervals
        cardinal_intervals = _get_Bspline_cardinal_intervals_1D_impl(knots, degree, tol)

        # All cardinal intervals should be identity
        for i in np.where(cardinal_intervals)[0]:
            np.testing.assert_array_almost_equal(result[i], np.eye(3))


class TestAdditionalEdgeCases:
    """Additional edge-case tests to improve coverage for bspline tools."""

    def test_extraction_non_open_left_end_branch(self) -> None:
        """Cover branch when first knot in domain multiplicity < degree+1."""
        # degree=2, choose first three knots not all equal so multiplicity<3
        knots = np.array([0.0, 0.1, 0.1, 0.5, 1.0, 1.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        Cs = _tabulate_Bspline_Bezier_1D_extraction_impl(knots, degree, tol)
        # Shape sanity
        assert Cs.shape[1:] == (degree + 1, degree + 1)
        # The first element matrix should be modified from identity when mult < degree+1
        assert not np.allclose(Cs[0], np.eye(degree + 1))

    def test_compute_basis_public_shapes_and_domain_error(self) -> None:
        """BsplineSpace1D.tabulate_basis covers scalar/list/ndarray inputs and domain error."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)
        # non Bernstein evaluation path.
        B_scalar, idx_scalar = spline.tabulate_basis(0.0)
        assert B_scalar.shape == (degree + 1,)
        assert np.isscalar(idx_scalar) or np.array(idx_scalar).shape == ()
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)
        # scalar input
        B_scalar, idx_scalar = spline.tabulate_basis(0.0)
        assert B_scalar.shape == (degree + 1,)
        assert np.isscalar(idx_scalar) or np.array(idx_scalar).shape == ()
        # list input
        pts_list = [0.0, 0.5, 1.0]
        B_list, idx_list = spline.tabulate_basis(pts_list)
        assert B_list.shape == (len(pts_list), degree + 1)
        assert idx_list.shape == (len(pts_list),)
        # ndarray input
        pts_arr = np.array([0.25, 0.75], dtype=np.float64)
        B_arr, idx_arr = spline.tabulate_basis(pts_arr)
        assert B_arr.shape == (2, degree + 1)
        assert idx_arr.shape == (2,)
        # outside domain error
        with pytest.raises(ValueError, match="outside the knot vector domain"):
            spline.tabulate_basis([-0.1, 1.1])

    def test_compute_bernstein_like_direct_raises_on_non_bezier(self) -> None:
        """Direct Bernstein-like evaluator should assert on non-Bézier-like splines."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)
        with pytest.raises(ValueError, match="B-spline does not have Bézier-like knots"):
            _tabulate_Bspline_basis_Bernstein_like_1D(spline, np.array([0.0, 0.5, 1.0]))


class TestBsplineSpace1DCoverageTargets:
    """Additional tests to hit uncovered branches in bspline_space_1D.py."""

    def test_validate_input_2d_array_type_error(self) -> None:
        """2D numpy array for knots should raise TypeError at ndim check."""
        knots = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
        with pytest.raises(TypeError, match="knots must be a 1D numpy array or Python list"):
            BsplineSpace1D(knots, 2)

    def test_validate_input_invalid_dtype_value_error(self) -> None:
        """Non-float32/float64 dtype (e.g., float16) should raise ValueError."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float16)
        with pytest.raises(ValueError, match="knots type must be float \\(32 or 64 bits\\)"):
            BsplineSpace1D(knots, 2)

    def test_validate_input_periodic_not_enough_basis(self) -> None:
        """Periodic case with too few basis functions should raise ValueError."""
        # This periodic-like vector yields fewer than degree+1 basis functions for degree=2.
        knots = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        with pytest.raises(ValueError, match="Not enough knots for the specified degree"):
            BsplineSpace1D(knots, 2, periodic=True)

    def test_open_end_checks_are_false_for_periodic(self) -> None:
        """has_left_end_open/has_right_end_open should return False if periodic."""
        degree = 2
        knots = create_uniform_periodic_knot_vector(
            num_intervals=3, degree=degree, domain=(0.0, 1.0)
        )
        spl = BsplineSpace1D(knots, degree, periodic=True)
        assert spl.has_left_end_open() is False
        assert spl.has_right_end_open() is False

    def test_extraction_first_knot_multiplicity_one_branch(self) -> None:
        """Cover branch where first-domain multiplicity == 1 (degree=2 => reg=1)."""
        # degree=2, first three knots are all different so multiplicity at index 2 is 1
        knots = np.array([0.0, 0.1, 0.2, 0.6, 1.0, 1.0], dtype=np.float64)
        Cs = _tabulate_Bspline_Bezier_1D_extraction_impl(knots, 2, 1e-10)
        # At least one coefficient in the first extraction matrix should differ from identity
        assert Cs.shape[1:] == (3, 3)
        assert not np.allclose(Cs[0], np.eye(3))

    def test_check_spline_info_knots_not_1d(self) -> None:
        """_check_spline_info should raise when knots is not 1D."""
        knots = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
        # Numba dispatcher rejects non-1D arrays at signature level
        with pytest.raises(TypeError):
            _check_spline_info(knots, 2)


class TestExtractionOperatorCorrectness:
    """Test correctness of extraction operators.

    Compares transformed basis with B-spline basis.
    """

    @pytest.mark.parametrize(
        "extraction_type",
        ["bezier", "lagrange", "cardinal"],
    )
    @pytest.mark.parametrize(
        "degree",
        [1, 2, 3, 4],
    )
    @pytest.mark.parametrize(
        "knots_factory",
        [
            lambda d: create_uniform_open_knot_vector(num_intervals=2, degree=d),  # 2 intervals
            lambda d: create_uniform_open_knot_vector(num_intervals=3, degree=d),  # 3 intervals
            lambda d: [0.0] * (d + 1) + [0.5, 1.0] + [2.0] * (d + 1),  # 2 intervals, different end
            lambda d: (
                [0.0] * (d + 1)
                + [0.5] * (d - 1)
                + [1.3, 2.7]
                + [3.5] * (d - 1)
                + [4.0, 5.0]
                + [6.0] * (d + 1)
            ),  # 5 intervals with different continuities
            lambda d: create_cardinal_Bspline_knot_vector(
                num_intervals=4, degree=d
            ),  # cardinal intervals
        ],
        ids=[
            "two_intervals",
            "three_intervals",
            "two_intervals_different_end",
            "five_intervals",
            "four_cardinal_intervals",
        ],
    )
    def test_extraction_operator_correctness(
        self,
        extraction_type: str,
        degree: int,
        knots_factory: Callable[[int], list[float]],
    ) -> None:
        """Test extraction operator correctness.

        For each interval, evaluate reference basis (Bernstein/Lagrange/cardinal) in [0,1],
        multiply by extraction operator, and compare with B-spline basis evaluated at
        mapped physical points.

        Args:
            extraction_type: Type of extraction operator ("bezier", "lagrange", "cardinal").
            degree: B-spline degree.
            knots_factory: Function that takes degree and returns knot vector.
        """
        # Generate knot vector from factory
        knots = knots_factory(degree)

        # Validate that knots match degree
        min_knots = 2 * degree + 2
        if len(knots) < min_knots:
            pytest.skip(f"Knot vector too short for degree {degree}")

        spline = BsplineSpace1D(knots, degree)

        # Get extraction operators based on type
        if extraction_type == "bezier":
            C_extraction = spline.tabulate_Bezier_extraction_operators()
        elif extraction_type == "lagrange":
            C_extraction = spline.tabulate_Lagrange_extraction_operators()
        elif extraction_type == "cardinal":
            C_extraction = spline.tabulate_cardinal_extraction_operators()
        else:
            raise ValueError(f"Unknown extraction type: {extraction_type}")

        # Get unique knots to determine intervals
        unique_knots, _ = spline.get_unique_knots_and_multiplicity(in_domain=True)
        num_intervals = len(unique_knots) - 1

        # Evaluation points in reference interval [0, 1] (avoid exact boundaries)
        max_intervals_for_dense = 2
        num_pts = 11 if num_intervals <= max_intervals_for_dense else 7
        xi_ref = np.linspace(0.01, 0.99, num_pts, dtype=spline.dtype)

        # For each interval, test the extraction operator
        for interval_idx in range(num_intervals):
            # Get interval boundaries
            t0 = unique_knots[interval_idx]
            t1 = unique_knots[interval_idx + 1]

            # Map reference points to physical interval
            x_physical = t0 + (t1 - t0) * xi_ref

            # Evaluate reference basis at reference points
            if extraction_type == "bezier":
                B_ref = tabulate_Bernstein_basis_1D(degree, xi_ref)
            elif extraction_type == "lagrange":
                B_ref = tabulate_Lagrange_basis_1D(degree, LagrangeVariant.EQUISPACES, xi_ref)
            elif extraction_type == "cardinal":
                B_ref = tabulate_cardinal_Bspline_basis_1D(degree, xi_ref)
            else:
                raise ValueError(f"Unknown extraction type: {extraction_type}")

            # Transform reference basis using extraction operator
            # C maps reference basis to B-spline basis: N = C @ B_ref
            B_transformed = B_ref @ C_extraction[interval_idx].T

            # Evaluate B-spline basis at physical points
            B_bspline, _ = spline.tabulate_basis(x_physical)

            # Extract the (degree+1) B-spline basis functions for this interval
            B_bspline_extracted = B_bspline[:, :]

            # Compare transformed reference basis with B-spline basis
            np.testing.assert_allclose(B_transformed, B_bspline_extracted, rtol=1e-10, atol=1e-12)


class TestOutParameter:
    """Test out parameter for BsplineSpace1D methods."""

    def test_get_cardinal_intervals_out_parameter(self) -> None:
        """Test that out parameter works for get_cardinal_intervals."""
        knots = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)

        result1 = spline.get_cardinal_intervals()
        out = np.zeros_like(result1)
        result2 = spline.get_cardinal_intervals(out=out)

        np.testing.assert_array_equal(result1, result2)
        np.testing.assert_array_equal(out, result1)
        assert result2 is out

    def test_get_cardinal_intervals_out_wrong_shape(self) -> None:
        """Test that out parameter with wrong shape raises ValueError."""
        knots = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)

        out = np.zeros(3, dtype=np.bool_)  # Wrong shape, should be 4

        with pytest.raises(ValueError, match="Output array has shape"):
            spline.get_cardinal_intervals(out=out)

    def test_get_cardinal_intervals_out_wrong_dtype(self) -> None:
        """Test that out parameter with wrong dtype raises ValueError."""
        knots = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)

        out = np.zeros(4, dtype=np.int_)  # Wrong dtype, should be bool_

        with pytest.raises(ValueError, match="Output array has dtype"):
            spline.get_cardinal_intervals(out=out)  # type: ignore[arg-type]

    def test_get_cardinal_intervals_out_not_writeable(self) -> None:
        """Test that out parameter with non-writeable array raises ValueError."""
        knots = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)

        out = np.zeros(4, dtype=np.bool_)
        out.setflags(write=False)

        with pytest.raises(ValueError, match="Output array is not writeable"):
            spline.get_cardinal_intervals(out=out)

    def test_tabulate_basis_out_parameter(self) -> None:
        """Test that out parameter works for tabulate_basis."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)
        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)

        basis1, idx1 = spline.tabulate_basis(pts)
        out = np.zeros_like(basis1)
        basis2, idx2 = spline.tabulate_basis(pts, out_basis=out)

        np.testing.assert_allclose(basis1, basis2)
        np.testing.assert_array_equal(idx1, idx2)
        np.testing.assert_allclose(out, basis1)
        assert basis2 is out

    def test_tabulate_basis_bezier_like_without_out_first_basis(self) -> None:
        """Test tabulate_basis with Bézier-like knots without out_first_basis parameter."""
        # Bézier-like knots: only two unique knots
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)
        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)

        # Call without out_first_basis to hit line 173
        basis, first_indices = spline.tabulate_basis(pts)

        assert basis.shape == (3, 3)
        assert first_indices.shape == (3,)
        # For Bézier-like knots, first_indices should all be 0
        np.testing.assert_array_equal(first_indices, np.array([0, 0, 0]))

    def test_tabulate_Bspline_basis_Bernstein_like_1D_without_out_first_basis(self) -> None:
        """Test _tabulate_Bspline_basis_Bernstein_like_1D without out_first_basis."""
        # Bézier-like knots: only two unique knots
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)
        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)

        # Call directly without out_first_basis to hit line 173
        basis, first_indices = _tabulate_Bspline_basis_Bernstein_like_1D(spline, pts)

        assert basis.shape == (3, 3)
        assert first_indices.shape == (3,)
        # For Bézier-like knots, first_indices should all be 0
        np.testing.assert_array_equal(first_indices, np.array([0, 0, 0]))

    def test_tabulate_basis_out_wrong_shape(self) -> None:
        """Test that out parameter with wrong shape raises ValueError."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)
        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)

        out = np.zeros((3, 2), dtype=np.float64)  # Wrong shape, should be (3, 3)

        with pytest.raises(ValueError, match="Output array has shape"):
            spline.tabulate_basis(pts, out_basis=out)

    def test_tabulate_basis_out_wrong_dtype(self) -> None:
        """Test that out parameter with wrong dtype raises ValueError."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)
        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)

        out = np.zeros((3, 3), dtype=np.float32)  # Wrong dtype, should be float64

        with pytest.raises(ValueError, match="Output array has dtype"):
            spline.tabulate_basis(pts, out_basis=out)

    def test_tabulate_Bezier_extraction_out_parameter(self) -> None:
        """Test that out parameter works for tabulate_Bezier_extraction_operators."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)

        result1 = spline.tabulate_Bezier_extraction_operators()
        out = np.zeros_like(result1)
        result2 = spline.tabulate_Bezier_extraction_operators(out=out)

        np.testing.assert_allclose(result1, result2)
        np.testing.assert_allclose(out, result1)
        assert result2 is out

    def test_tabulate_Bezier_extraction_out_wrong_shape(self) -> None:
        """Test that out parameter with wrong shape raises ValueError."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)

        out = np.zeros((2, 2, 2), dtype=np.float64)  # Wrong shape, should be (2, 3, 3)

        with pytest.raises(ValueError, match="Output array has shape"):
            spline.tabulate_Bezier_extraction_operators(out=out)

    def test_tabulate_Bezier_extraction_out_wrong_dtype(self) -> None:
        """Test that out parameter with wrong dtype raises ValueError."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)

        out = np.zeros((2, 3, 3), dtype=np.float32)  # Wrong dtype, should be float64

        with pytest.raises(ValueError, match="Output array has dtype"):
            spline.tabulate_Bezier_extraction_operators(out=out)

    def test_tabulate_Bezier_extraction_out_not_writeable(self) -> None:
        """Test that out parameter with non-writeable array raises ValueError."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)

        out = np.zeros((2, 3, 3), dtype=np.float64)
        out.setflags(write=False)

        with pytest.raises(ValueError, match="Output array is not writeable"):
            spline.tabulate_Bezier_extraction_operators(out=out)

    def test_tabulate_Lagrange_extraction_out_parameter(self) -> None:
        """Test that out parameter works for tabulate_Lagrange_extraction_operators."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)

        result1 = spline.tabulate_Lagrange_extraction_operators()
        out = np.zeros_like(result1)
        result2 = spline.tabulate_Lagrange_extraction_operators(out=out)

        np.testing.assert_allclose(result1, result2)
        np.testing.assert_allclose(out, result1)
        assert result2 is out

    def test_tabulate_Lagrange_extraction_out_wrong_shape(self) -> None:
        """Test that out parameter with wrong shape raises ValueError."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)

        out = np.zeros((2, 2, 2), dtype=np.float64)  # Wrong shape, should be (2, 3, 3)

        with pytest.raises(ValueError, match="Output array has shape"):
            spline.tabulate_Lagrange_extraction_operators(out=out)

    def test_tabulate_Lagrange_extraction_out_wrong_dtype(self) -> None:
        """Test that out parameter with wrong dtype raises ValueError."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)

        out = np.zeros((2, 3, 3), dtype=np.float32)  # Wrong dtype, should be float64

        with pytest.raises(ValueError, match="Output array has dtype"):
            spline.tabulate_Lagrange_extraction_operators(out=out)

    def test_tabulate_Lagrange_extraction_out_not_writeable(self) -> None:
        """Test that out parameter with non-writeable array raises ValueError."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)

        out = np.zeros((2, 3, 3), dtype=np.float64)
        out.setflags(write=False)

        with pytest.raises(ValueError, match="Output array is not writeable"):
            spline.tabulate_Lagrange_extraction_operators(out=out)

    def test_tabulate_cardinal_extraction_out_parameter(self) -> None:
        """Test that out parameter works for tabulate_cardinal_extraction_operators."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)

        result1 = spline.tabulate_cardinal_extraction_operators()
        out = np.zeros_like(result1)
        result2 = spline.tabulate_cardinal_extraction_operators(out=out)

        np.testing.assert_allclose(result1, result2)
        np.testing.assert_allclose(out, result1)
        assert result2 is out

    def test_tabulate_cardinal_extraction_out_wrong_shape(self) -> None:
        """Test that out parameter with wrong shape raises ValueError."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)

        out = np.zeros((2, 2, 2), dtype=np.float64)  # Wrong shape, should be (2, 3, 3)

        with pytest.raises(ValueError, match="Output array has shape"):
            spline.tabulate_cardinal_extraction_operators(out=out)

    def test_tabulate_cardinal_extraction_out_wrong_dtype(self) -> None:
        """Test that out parameter with wrong dtype raises ValueError."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)

        out = np.zeros((2, 3, 3), dtype=np.float32)  # Wrong dtype, should be float64

        with pytest.raises(ValueError, match="Output array has dtype"):
            spline.tabulate_cardinal_extraction_operators(out=out)

    def test_tabulate_cardinal_extraction_out_not_writeable(self) -> None:
        """Test that out parameter with non-writeable array raises ValueError."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)

        out = np.zeros((2, 3, 3), dtype=np.float64)
        out.setflags(write=False)

        with pytest.raises(ValueError, match="Output array is not writeable"):
            spline.tabulate_cardinal_extraction_operators(out=out)


class TestBsplineSpace1DChangeBasis:
    """Test change of basis wrappers on BsplineSpace1D."""

    @pytest.mark.parametrize("degree", [1, 2, 3, 4])
    def test_compute_change_basis_wrappers(self, degree: int) -> None:
        """Test that the change of basis methods delegate correctly."""
        knots = create_uniform_open_knot_vector(num_intervals=5, degree=degree, domain=(0.0, 1.0))
        spline = BsplineSpace1D(knots, degree)

        # 1. Lagrange to Bernstein
        res1 = spline.compute_Lagrange_to_Bernstein_change_basis(LagrangeVariant.EQUISPACES)
        exp1 = compute_Lagrange_to_Bernstein_change_basis_1D(
            degree, LagrangeVariant.EQUISPACES, spline.dtype
        )
        np.testing.assert_allclose(res1, exp1)

        # 2. Bernstein to Lagrange
        res2 = spline.compute_Bernstein_to_Lagrange_change_basis(LagrangeVariant.EQUISPACES)
        exp2 = compute_Bernstein_to_Lagrange_change_basis_1D(
            degree, LagrangeVariant.EQUISPACES, spline.dtype
        )
        np.testing.assert_allclose(res2, exp2)

        # 3. Bernstein to Cardinal
        res3 = spline.compute_Bernstein_to_cardinal_change_basis()
        exp3 = compute_Bernstein_to_cardinal_change_basis_1D(degree, spline.dtype)
        np.testing.assert_allclose(res3, exp3)

        # 4. Cardinal to Bernstein
        res4 = spline.compute_cardinal_to_Bernstein_change_basis()
        exp4 = compute_cardinal_to_Bernstein_change_basis_1D(degree, spline.dtype)
        np.testing.assert_allclose(res4, exp4)
