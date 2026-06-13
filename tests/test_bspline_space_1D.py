"""Tests for bspline_space_1D module."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from pantr.basis import (
    LagrangeVariant,
    tabulate_bernstein_1d,
    tabulate_cardinal_bspline_1d,
    tabulate_lagrange_1d,
)
from pantr.basis._basis_core import (
    _PARALLEL_MIN_NUM_PTS,
    _tabulate_Bernstein_basis_1D_core,
    _tabulate_Bernstein_basis_1D_serial_core,
    _tabulate_Bernstein_basis_deriv_1D_core,
    _tabulate_Bernstein_basis_deriv_1D_serial_core,
)
from pantr.bspline import (
    BsplineSpace1D,
    create_cardinal_knots,
    create_uniform_open_knots,
    create_uniform_periodic_knots,
)
from pantr.bspline._bspline_basis_core import (
    _compute_basis_deriv_nurbs_book_impl,
    _compute_basis_deriv_nurbs_book_serial_impl,
    _compute_basis_nurbs_book_impl,
    _compute_basis_nurbs_book_serial_impl,
    _tabulate_Bspline_basis_1D_impl,
    _tabulate_Bspline_basis_Bernstein_like_1D,
    _tabulate_Bspline_basis_deriv_1D_impl,
)
from pantr.bspline._bspline_extraction import (
    _tabulate_Bspline_Bezier_1D_extraction_impl,
    _tabulate_Bspline_cardinal_1D_extraction_impl,
    _tabulate_Bspline_Lagrange_1D_extraction_impl,
)
from pantr.bspline._bspline_knots import (
    _check_spline_info,
    _get_Bspline_cardinal_intervals_1D_impl,
    _get_Bspline_num_basis_1D_impl,
    _get_last_knot_smaller_equal_impl,
    _get_multiplicity_of_first_knot_in_domain_impl,
    _get_unique_knots_and_multiplicity_impl,
    _is_in_domain_impl,
)
from pantr.bspline._bspline_space_1d import _cached_unique_knots_and_multiplicity
from pantr.change_basis import compute_lagrange_to_bernstein_1d
from pantr.tolerance import get_strict


class TestBsplineSpace1DInit:
    """Test BsplineSpace1D initialization."""

    def test_valid_initialization(self) -> None:
        """Test valid BsplineSpace1D initialization."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = BsplineSpace1D(knots, degree)

        assert spline.degree == 2
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
        knots = create_uniform_periodic_knots(num_intervals=3, degree=2, domain=(0.0, 1.0))
        degree = 2
        spline = BsplineSpace1D(knots, degree, periodic=True)

        assert spline.degree == 2
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
        tol = get_strict(np.float64)
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

    def test_knots_immutable(self) -> None:
        """Test that the knots array is read-only to prevent silent cache corruption."""
        spline = BsplineSpace1D([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 2)
        assert not spline.knots.flags.writeable
        with pytest.raises(ValueError):
            spline.knots[0] = 99.0

    def test_unique_knots_cache_is_bounded(self) -> None:
        """Test that the knot-multiplicity cache has a finite maxsize.

        An unbounded cache (functools.cache) would leak memory in long-running
        applications that construct many distinct spline spaces.  Switching to
        lru_cache(maxsize=N) ensures old entries are evicted when the cache is
        full.
        """
        info = _cached_unique_knots_and_multiplicity.cache_info()
        assert info.maxsize is not None, (
            "_cached_unique_knots_and_multiplicity must use lru_cache with a "
            "finite maxsize, not functools.cache, to avoid unbounded memory growth."
        )
        assert info.maxsize > 0

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
        assert spline.num_basis == 3

    def test_get_num_basis_periodic(self) -> None:
        """Test get_num_basis for periodic spline."""
        degree = 2
        knots = create_uniform_periodic_knots(num_intervals=3, degree=degree, domain=(0.0, 1.0))
        spline = BsplineSpace1D(knots, degree, periodic=True)
        # For periodic splines, the number of basis functions is reduced
        assert spline.num_basis == 3

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

    def test_num_intervals(self) -> None:
        """Test num_intervals property."""
        num_intervals = 2
        degree = 2
        knots = create_uniform_open_knots(num_intervals, degree)
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
        knots = create_uniform_periodic_knots(num_intervals=3, degree=degree, domain=(0.0, 1.0))
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
        knots = create_uniform_open_knots(num_intervals=2, degree=2, domain=(0.0, 1.0))
        degree = 2
        spline = BsplineSpace1D(knots, degree)

        assert spline.degree == degree
        assert spline.periodic is False
        assert spline.has_open_knots() is True
        assert spline.domain == (knots[degree], knots[-degree - 1])

    def test_with_uniform_periodic_knot_vector(self) -> None:
        """Test BsplineSpace1D with uniform periodic knot vector."""
        degree = 2
        knots = create_uniform_periodic_knots(num_intervals=3, degree=degree, domain=(0.0, 1.0))
        spline = BsplineSpace1D(knots, degree, periodic=True)

        assert spline.degree == degree
        assert spline.periodic is True
        assert spline.domain == (knots[degree], knots[-degree - 1])

    def test_with_cardinal_bspline_knot_vector(self) -> None:
        """Test BsplineSpace1D with cardinal B-spline knot vector."""
        degree = 2
        knots = create_cardinal_knots(2, degree)
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
        knots_open = create_uniform_open_knots(num_intervals=3, degree=degree, domain=(0.0, 1.0))
        knots_periodic = create_uniform_periodic_knots(
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
        assert result == 3

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
        assert result == 3

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
        assert result == 2


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


class TestEvaluateBasisBasisFuncs:
    """Test the _compute_basis_nurbs_book_impl function."""

    def test_bezier_like_evaluation(self) -> None:
        """Test evaluation for Bézier-like knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        periodic = False
        tol = 1e-10
        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)

        basis = np.empty((3, 3), dtype=np.float64)
        first_basis = np.empty(3, dtype=np.int_)
        _compute_basis_nurbs_book_impl(knots, degree, periodic, tol, pts, basis, first_basis)

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
        _compute_basis_nurbs_book_impl(knots, degree, periodic, tol, pts, basis, first_basis)

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
        _compute_basis_nurbs_book_impl(knots, degree, periodic, tol, pts, basis, first_basis)

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
        _compute_basis_nurbs_book_impl(knots, degree, periodic, tol, pts, basis1, first_basis1)
        assert basis1.shape == (3, 3)
        assert basis1.dtype == np.float64
        assert first_basis1.shape == (3,)

        # Test with float32
        knots_f32 = knots.astype(np.float32)
        pts_f32 = pts.astype(np.float32)
        basis2 = np.empty((3, 3), dtype=np.float32)
        first_basis2 = np.empty(3, dtype=np.int_)
        _compute_basis_nurbs_book_impl(
            knots_f32, degree, periodic, tol, pts_f32, basis2, first_basis2
        )
        assert basis2.dtype == np.float32

        # Verify results are consistent (within float32 precision)
        np.testing.assert_array_almost_equal(basis1, basis2, decimal=6)
        np.testing.assert_array_equal(first_basis1, first_basis2)


class TestNonOpenKnotSpanBoundary:
    """Test basis evaluation at domain endpoints for non-open knot vectors.

    For non-open knot vectors the right domain endpoint knots[-degree-1] is
    not the last knot in the vector.  This class verifies that basis values
    and derivatives are continuous at the right domain endpoint, i.e. that
    evaluation at the boundary matches the left-limit.
    """

    @pytest.mark.parametrize("degree", [1, 2, 3, 4])
    def test_non_open_basis_continuity_at_right_endpoint(self, degree: int) -> None:
        """Test basis continuity at right endpoint for non-open knot vectors."""
        n_intervals = degree + 2
        n_knots = n_intervals + degree + 1
        knots = np.linspace(0.0, 1.0, n_knots, dtype=np.float64)
        space = BsplineSpace1D(knots=knots, degree=degree, periodic=False)
        assert not space.has_open_knots()

        t_end = space.domain[1]
        eps = 1e-12
        pts = np.array([t_end - eps, t_end], dtype=np.float64)
        basis, fb = space.tabulate_basis(pts)

        # first_basis should be the same for both points
        assert fb[0] == fb[1]
        # basis values should be continuous (within tolerance)
        np.testing.assert_allclose(basis[0], basis[1], atol=1e-8)

    @pytest.mark.parametrize("degree", [1, 2, 3, 4])
    def test_non_open_basis_continuity_at_left_endpoint(self, degree: int) -> None:
        """Test basis continuity at left endpoint for non-open knot vectors."""
        n_intervals = degree + 2
        n_knots = n_intervals + degree + 1
        knots = np.linspace(0.0, 1.0, n_knots, dtype=np.float64)
        space = BsplineSpace1D(knots=knots, degree=degree, periodic=False)
        assert not space.has_open_knots()

        t_start = space.domain[0]
        eps = 1e-12
        pts = np.array([t_start, t_start + eps], dtype=np.float64)
        basis, fb = space.tabulate_basis(pts)

        assert fb[0] == fb[1]
        np.testing.assert_allclose(basis[0], basis[1], atol=1e-8)

    @pytest.mark.parametrize("degree", [1, 2, 3, 4])
    def test_periodic_basis_continuity_at_right_endpoint(self, degree: int) -> None:
        """Test basis continuity at right endpoint for periodic knot vectors."""
        knots = create_uniform_periodic_knots(num_intervals=degree + 1, degree=degree)
        space = BsplineSpace1D(knots=knots, degree=degree, periodic=True)

        t_end = space.domain[1]
        eps = 1e-12
        pts = np.array([t_end - eps, t_end], dtype=np.float64)
        basis, fb = space.tabulate_basis(pts)

        assert fb[0] == fb[1]
        np.testing.assert_allclose(basis[0], basis[1], atol=1e-8)

    @pytest.mark.parametrize("degree", [1, 2, 3])
    def test_non_open_derivative_continuity_at_right_endpoint(self, degree: int) -> None:
        """Test derivative continuity at right endpoint for non-open knot vectors."""
        n_intervals = degree + 2
        n_knots = n_intervals + degree + 1
        knots = np.linspace(0.0, 1.0, n_knots, dtype=np.float64)
        space = BsplineSpace1D(knots=knots, degree=degree, periodic=False)

        t_end = space.domain[1]
        eps = 1e-12
        pts = np.array([t_end - eps, t_end], dtype=np.float64)
        deriv, fb = space.tabulate_basis_derivatives(pts, n_deriv=1)

        assert fb[0] == fb[1]
        # 0th derivative (basis values) should be continuous
        np.testing.assert_allclose(deriv[0, 0, :], deriv[1, 0, :], atol=1e-8)

    @pytest.mark.parametrize("degree", [1, 2, 3])
    def test_open_knots_still_correct_at_endpoints(self, degree: int) -> None:
        """Test that open knot vectors still produce correct endpoint values."""
        knots = create_uniform_open_knots(num_intervals=3, degree=degree)
        space = BsplineSpace1D(knots=knots, degree=degree, periodic=False)
        assert space.has_open_knots()

        t_start, t_end = space.domain
        pts = np.array([t_start, t_end], dtype=np.float64)
        basis, _fb = space.tabulate_basis(pts)

        # At left endpoint: first basis function = 1, rest = 0
        expected_left = np.zeros(degree + 1)
        expected_left[0] = 1.0
        np.testing.assert_allclose(basis[0], expected_left, atol=1e-14)

        # At right endpoint: last basis function = 1, rest = 0
        expected_right = np.zeros(degree + 1)
        expected_right[-1] = 1.0
        np.testing.assert_allclose(basis[1], expected_right, atol=1e-14)


class TestNonOpenBoundaryMultiplicityTabulation:
    """Test tabulate_basis and tabulate_basis_derivatives for non-open knot vectors.

    Non-open non-periodic knot vectors have boundary multiplicity < degree+1, which
    controls the continuity at the domain endpoints.  We parametrize over
    boundary_mult ∈ {1, 2, degree} to cover maximum, intermediate, and minimum
    (C^0) continuity at the endpoints.
    """

    @staticmethod
    def _make_knots(degree: int, boundary_mult: int) -> npt.NDArray[np.float64]:
        """Construct a uniform non-open knot vector with given boundary multiplicity.

        Args:
            degree (int): B-spline degree.
            boundary_mult (int): Knot multiplicity at domain endpoints (< degree + 1).

        Returns:
            np.ndarray: Knot vector of dtype float64.
        """
        n_int = max(3, 2 * degree - 2 * boundary_mult + 3)
        interior = np.linspace(0.0, 1.0, n_int + 1)[1:-1]
        return np.concatenate([[0.0] * boundary_mult, interior, [1.0] * boundary_mult])

    @pytest.mark.parametrize(
        "degree,boundary_mult",
        [
            (1, 1),
            (2, 1),
            (2, 2),
            (3, 1),
            (3, 2),
            (3, 3),
            (4, 1),
            (4, 2),
            (4, 4),
        ],
    )
    def test_partition_of_unity(self, degree: int, boundary_mult: int) -> None:
        """Non-open B-spline basis values sum to 1.0 at all interior points."""
        knots = self._make_knots(degree, boundary_mult)
        space = BsplineSpace1D(knots, degree)
        a, b = space.domain
        pts = np.linspace(float(a), float(b), 21, dtype=np.float64)[1:-1]
        basis, _ = space.tabulate_basis(pts)
        np.testing.assert_allclose(basis.sum(axis=-1), np.ones(len(pts)), atol=1e-13)

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
    def test_tabulate_basis_derivatives_sum(self, degree: int, boundary_mult: int) -> None:
        """0th derivative sums to 1; higher-order derivatives sum to 0 at interior points."""
        knots = self._make_knots(degree, boundary_mult)
        space = BsplineSpace1D(knots, degree)
        a, b = space.domain
        pts = np.linspace(float(a), float(b), 15, dtype=np.float64)[1:-1]
        deriv, _ = space.tabulate_basis_derivatives(pts, n_deriv=2)
        np.testing.assert_allclose(deriv[:, 0, :].sum(axis=-1), np.ones(len(pts)), atol=1e-12)
        np.testing.assert_allclose(deriv[:, 1, :].sum(axis=-1), np.zeros(len(pts)), atol=1e-12)
        np.testing.assert_allclose(deriv[:, 2, :].sum(axis=-1), np.zeros(len(pts)), atol=1e-12)

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
    def test_basis_continuity_at_endpoints(self, degree: int, boundary_mult: int) -> None:
        """Basis values are continuous at both domain endpoints."""
        knots = self._make_knots(degree, boundary_mult)
        space = BsplineSpace1D(knots, degree)
        a, b = space.domain
        eps = 1e-12
        # Right endpoint
        pts_r = np.array([b - eps, b], dtype=np.float64)
        basis_r, _ = space.tabulate_basis(pts_r)
        np.testing.assert_allclose(basis_r[0], basis_r[1], atol=1e-8)
        # Left endpoint
        pts_l = np.array([a, a + eps], dtype=np.float64)
        basis_l, _ = space.tabulate_basis(pts_l)
        np.testing.assert_allclose(basis_l[0], basis_l[1], atol=1e-8)


class TestPeriodicBasisTabulation:
    """Test tabulate_basis and tabulate_basis_derivatives for periodic B-spline spaces.

    Covers both maximum continuity (C^{degree-1}) and reduced continuity (e.g., C^0, C^1)
    cases created via create_uniform_periodic_knots with the ``continuity`` parameter.
    """

    @pytest.mark.parametrize(
        "degree,continuity",
        [
            (1, None),  # degree 1, max continuity (C^0)
            (2, None),  # degree 2, max continuity (C^1)
            (2, 0),  # degree 2, C^0
            (3, None),  # degree 3, max continuity (C^2)
            (3, 1),  # degree 3, C^1
            (3, 0),  # degree 3, C^0
            (4, None),  # degree 4, max continuity (C^3)
        ],
    )
    def test_periodic_tabulate_basis_partition_of_unity(
        self, degree: int, continuity: int | None
    ) -> None:
        """Periodic B-spline basis values sum to 1.0 at all interior points."""
        num_intervals = degree + 3
        knots = create_uniform_periodic_knots(
            num_intervals=num_intervals, degree=degree, continuity=continuity
        )
        space = BsplineSpace1D(knots, degree, periodic=True)

        a, b = space.domain
        pts = np.linspace(float(a), float(b), 21, dtype=np.float64)[1:-1]  # interior only
        basis, _ = space.tabulate_basis(pts)

        np.testing.assert_allclose(
            basis.sum(axis=-1),
            np.ones(len(pts), dtype=np.float64),
            atol=1e-13,
        )

    @pytest.mark.parametrize("degree", [1, 2, 3, 4])
    def test_periodic_tabulate_basis_continuity_at_left_endpoint(self, degree: int) -> None:
        """Periodic basis is continuous at the left domain endpoint."""
        knots = create_uniform_periodic_knots(num_intervals=degree + 3, degree=degree)
        space = BsplineSpace1D(knots=knots, degree=degree, periodic=True)

        t_start = space.domain[0]
        eps = 1e-12
        pts = np.array([t_start, t_start + eps], dtype=np.float64)
        basis, fb = space.tabulate_basis(pts)

        assert fb[0] == fb[1]
        np.testing.assert_allclose(basis[0], basis[1], atol=1e-8)

    @pytest.mark.parametrize("degree", [1, 2, 3, 4])
    def test_periodic_C0_basis_continuity_at_right_endpoint(self, degree: int) -> None:
        """Periodic C^0 basis is continuous at the right domain endpoint."""
        knots = create_uniform_periodic_knots(num_intervals=degree + 3, degree=degree, continuity=0)
        space = BsplineSpace1D(knots=knots, degree=degree, periodic=True)

        t_end = space.domain[1]
        eps = 1e-12
        pts = np.array([t_end - eps, t_end], dtype=np.float64)
        basis, fb = space.tabulate_basis(pts)

        assert fb[0] == fb[1]
        np.testing.assert_allclose(basis[0], basis[1], atol=1e-8)

    @pytest.mark.parametrize("degree", [1, 2, 3, 4])
    def test_periodic_C0_basis_continuity_at_left_endpoint(self, degree: int) -> None:
        """Periodic C^0 basis is continuous at the left domain endpoint."""
        knots = create_uniform_periodic_knots(num_intervals=degree + 3, degree=degree, continuity=0)
        space = BsplineSpace1D(knots=knots, degree=degree, periodic=True)

        t_start = space.domain[0]
        eps = 1e-12
        pts = np.array([t_start, t_start + eps], dtype=np.float64)
        basis, fb = space.tabulate_basis(pts)

        assert fb[0] == fb[1]
        np.testing.assert_allclose(basis[0], basis[1], atol=1e-8)

    @pytest.mark.parametrize(
        "degree,continuity",
        [
            (1, None),
            (2, None),
            (2, 0),
            (3, None),
            (3, 1),
            (3, 0),
        ],
    )
    def test_periodic_tabulate_basis_derivatives_sum(
        self, degree: int, continuity: int | None
    ) -> None:
        """Periodic basis derivatives satisfy: 0th order sums to 1, higher orders sum to 0."""
        num_intervals = degree + 3
        knots = create_uniform_periodic_knots(
            num_intervals=num_intervals, degree=degree, continuity=continuity
        )
        space = BsplineSpace1D(knots, degree, periodic=True)

        a, b = space.domain
        pts = np.linspace(float(a), float(b), 15, dtype=np.float64)[1:-1]  # interior only
        deriv, _ = space.tabulate_basis_derivatives(pts, n_deriv=2)

        # 0th derivative: partition of unity
        np.testing.assert_allclose(
            deriv[..., 0, :].sum(axis=-1),
            np.ones(len(pts), dtype=np.float64),
            atol=1e-13,
        )
        # 1st and 2nd derivatives: sum to zero
        np.testing.assert_allclose(
            deriv[..., 1, :].sum(axis=-1),
            np.zeros(len(pts), dtype=np.float64),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            deriv[..., 2, :].sum(axis=-1),
            np.zeros(len(pts), dtype=np.float64),
            atol=1e-8,
        )

    @pytest.mark.parametrize("degree", [2, 3])
    def test_periodic_derivative_continuity_at_right_endpoint(self, degree: int) -> None:
        """Periodic basis derivatives are continuous at the right domain endpoint."""
        knots = create_uniform_periodic_knots(num_intervals=degree + 3, degree=degree)
        space = BsplineSpace1D(knots=knots, degree=degree, periodic=True)

        t_end = space.domain[1]
        eps = 1e-12
        pts = np.array([t_end - eps, t_end], dtype=np.float64)
        deriv, fb = space.tabulate_basis_derivatives(pts, n_deriv=1)

        assert fb[0] == fb[1]
        # 0th derivative (basis values) should be continuous
        np.testing.assert_allclose(deriv[0, 0, :], deriv[1, 0, :], atol=1e-8)
        # 1st derivative should be continuous for C^{degree-1} periodic splines
        np.testing.assert_allclose(deriv[0, 1, :], deriv[1, 1, :], atol=1e-6)

    @pytest.mark.parametrize("degree", [2, 3])
    def test_periodic_derivative_continuity_at_left_endpoint(self, degree: int) -> None:
        """Periodic basis derivatives are continuous at the left domain endpoint."""
        knots = create_uniform_periodic_knots(num_intervals=degree + 3, degree=degree)
        space = BsplineSpace1D(knots=knots, degree=degree, periodic=True)

        t_start = space.domain[0]
        eps = 1e-12
        pts = np.array([t_start, t_start + eps], dtype=np.float64)
        deriv, fb = space.tabulate_basis_derivatives(pts, n_deriv=1)

        assert fb[0] == fb[1]
        np.testing.assert_allclose(deriv[0, 0, :], deriv[1, 0, :], atol=1e-8)
        np.testing.assert_allclose(deriv[0, 1, :], deriv[1, 1, :], atol=1e-6)


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

    def test_periodic_uniform_all_cardinal(self) -> None:
        """Uniform periodic knot vector (max continuity) has all-cardinal intervals."""
        degree = 2
        knots = create_uniform_periodic_knots(num_intervals=4, degree=degree)
        tol = 1e-10
        result = _get_Bspline_cardinal_intervals_1D_impl(knots, degree, tol)

        # All intervals in a uniform periodic knot vector are cardinal since the ghost
        # knots maintain equal spacing throughout.
        assert result.all(), f"Expected all cardinal, got {result}"

    def test_periodic_uniform_C0_none_cardinal(self) -> None:
        """Uniform periodic C^0 knot vector (multiplicity=degree) has no cardinal intervals.

        With multiplicity=degree at every interior knot, the repeated knots break
        the cardinal condition (same length as degree-1 previous and next intervals),
        so no intervals are cardinal.
        """
        degree = 2
        knots = create_uniform_periodic_knots(num_intervals=4, degree=degree, continuity=0)
        tol = 1e-10
        result = _get_Bspline_cardinal_intervals_1D_impl(knots, degree, tol)

        assert not result.any(), f"Expected no cardinal intervals, got {result}"


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


class TestLagrangeExtractionVariants:
    """Test Lagrange extraction operators with different LagrangeVariant values."""

    @pytest.fixture()
    def spline(self) -> BsplineSpace1D:
        """Create a B-spline space with two intervals."""
        return BsplineSpace1D([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], 2)

    @pytest.mark.parametrize("variant", list(LagrangeVariant))
    def test_public_method_matches_impl(
        self, spline: BsplineSpace1D, variant: LagrangeVariant
    ) -> None:
        """Test that the public method delegates correctly for every variant."""
        result = spline.tabulate_Lagrange_extraction_operators(lagrange_variant=variant)
        expected = _tabulate_Bspline_Lagrange_1D_extraction_impl(
            spline.knots, spline.degree, spline.tolerance, lagrange_variant=variant
        )
        np.testing.assert_allclose(result, expected)

    @pytest.mark.parametrize("variant", list(LagrangeVariant))
    def test_shape_and_dtype(self, spline: BsplineSpace1D, variant: LagrangeVariant) -> None:
        """Test that all variants produce correct shape and dtype."""
        result = spline.tabulate_Lagrange_extraction_operators(lagrange_variant=variant)
        assert result.shape == (2, 3, 3)
        assert result.dtype == np.float64

    def test_variants_produce_different_results(self) -> None:
        """Test that different variants produce different extraction operators."""
        # Use degree 3 so equispaced and GLL points differ
        spline_p3 = BsplineSpace1D([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], 3)
        equi = spline_p3.tabulate_Lagrange_extraction_operators(
            lagrange_variant=LagrangeVariant.EQUISPACES
        )
        gll = spline_p3.tabulate_Lagrange_extraction_operators(
            lagrange_variant=LagrangeVariant.GAUSS_LOBATTO_LEGENDRE
        )
        assert not np.allclose(equi, gll)

    def test_default_is_equispaces(self, spline: BsplineSpace1D) -> None:
        """Test that the default variant is EQUISPACES."""
        default_result = spline.tabulate_Lagrange_extraction_operators()
        equi_result = spline.tabulate_Lagrange_extraction_operators(
            lagrange_variant=LagrangeVariant.EQUISPACES
        )
        np.testing.assert_allclose(default_result, equi_result)

    @pytest.mark.parametrize("variant", list(LagrangeVariant))
    def test_out_parameter_with_variant(
        self, spline: BsplineSpace1D, variant: LagrangeVariant
    ) -> None:
        """Test that the out parameter works correctly with all variants."""
        result1 = spline.tabulate_Lagrange_extraction_operators(lagrange_variant=variant)
        out = np.zeros_like(result1)
        result2 = spline.tabulate_Lagrange_extraction_operators(lagrange_variant=variant, out=out)
        np.testing.assert_allclose(result1, result2)
        assert result2 is out

    @pytest.mark.parametrize("variant", list(LagrangeVariant))
    def test_lagrange_is_bezier_times_change_of_basis(self, variant: LagrangeVariant) -> None:
        """Test that Lagrange extraction equals Bézier extraction times change-of-basis.

        The Lagrange extraction operator C_L = C_B @ M, where C_B is the Bézier
        extraction and M is the Lagrange-to-Bernstein change-of-basis matrix.
        """
        degree = 3
        knots_list = [0.0] * (degree + 1) + [0.5] + [1.0] * (degree + 1)
        spline = BsplineSpace1D(knots_list, degree)
        lagr_extraction = spline.tabulate_Lagrange_extraction_operators(lagrange_variant=variant)
        bezier_extraction = spline.tabulate_Bezier_extraction_operators()
        lagr_to_bern = compute_lagrange_to_bernstein_1d(degree, variant, np.float64)
        for i in range(lagr_extraction.shape[0]):
            expected = bezier_extraction[i] @ lagr_to_bern
            np.testing.assert_allclose(lagr_extraction[i], expected, atol=1e-14)


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
        knots = create_uniform_open_knots(num_intervals=3, degree=2, domain=(0.0, 1.0))
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
        knots = create_uniform_periodic_knots(num_intervals=3, degree=degree, domain=(0.0, 1.0))
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
            lambda d: create_uniform_open_knots(num_intervals=2, degree=d),  # 2 intervals
            lambda d: create_uniform_open_knots(num_intervals=3, degree=d),  # 3 intervals
            lambda d: [0.0] * (d + 1) + [0.5, 1.0] + [2.0] * (d + 1),  # 2 intervals, different end
            lambda d: (
                [0.0] * (d + 1)
                + [0.5] * (d - 1)
                + [1.3, 2.7]
                + [3.5] * (d - 1)
                + [4.0, 5.0]
                + [6.0] * (d + 1)
            ),  # 5 intervals with different continuities
            lambda d: create_cardinal_knots(num_intervals=4, degree=d),  # cardinal intervals
            lambda d: create_uniform_periodic_knots(
                num_intervals=d + 3, degree=d
            ),  # periodic, max continuity
            lambda d: create_uniform_periodic_knots(
                num_intervals=d + 3, degree=d, continuity=0
            ),  # periodic, C^0 continuity
            lambda d: np.linspace(0.0, 1.0, 2 * d + 4, dtype=np.float64),  # non-open non-periodic
            lambda d: np.concatenate(  # non-open, boundary multiplicity 2
                [[0.0] * 2, np.linspace(0.0, 1.0, 2 * d + 3)[1:-1], [1.0] * 2]
            ),
            lambda d: np.concatenate(  # non-open, boundary multiplicity = degree (C^0 at endpoints)
                [[0.0] * d, np.linspace(0.0, 1.0, 2 * d + 3)[1:-1], [1.0] * d]
            ),
        ],
        ids=[
            "two_intervals",
            "three_intervals",
            "two_intervals_different_end",
            "five_intervals",
            "four_cardinal_intervals",
            "periodic_max_continuity",
            "periodic_C0_continuity",
            "non_open_uniform",
            "non_open_bdry_mult2",
            "non_open_bdry_mult_degree",
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
                B_ref = tabulate_bernstein_1d(degree, xi_ref)
            elif extraction_type == "lagrange":
                B_ref = tabulate_lagrange_1d(degree, LagrangeVariant.EQUISPACES, xi_ref)
            elif extraction_type == "cardinal":
                B_ref = tabulate_cardinal_bspline_1d(degree, xi_ref)
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


def _open_uniform_1d(degree: int, n_int: int) -> BsplineSpace1D:
    """Open-uniform 1D space: ``n_int`` unit intervals on ``[0, n_int]``."""
    knots = (
        [0.0] * (degree + 1) + [float(i) for i in range(1, n_int)] + [float(n_int)] * (degree + 1)
    )
    return BsplineSpace1D(knots, degree)


def _check_restrict_1d(
    space: BsplineSpace1D, lo: int, hi: int
) -> tuple[BsplineSpace1D, npt.NDArray[np.int64]]:
    """Assert a 1D restriction reproduces the global basis, derivatives, and DOF map."""
    wspace, dof = space.restrict(lo, hi)
    p = space.degree
    assert not dof.flags.writeable
    assert wspace.degree == p
    assert wspace.num_intervals == hi - lo
    assert wspace.num_basis == len(dof)
    # Windowed knots are a pure slice of the global knots (never re-clamped).
    j_lo, j_hi = int(dof[0]), int(dof[-1])
    np.testing.assert_array_equal(wspace.knots, space.knots[j_lo : j_hi + p + 2])
    # Windowed basis and all derivatives equal the global ones over the window cells.
    uk, _ = wspace.get_unique_knots_and_multiplicity(in_domain=True)
    mids = 0.5 * (uk[:-1] + uk[1:])
    gd, gfb = space.tabulate_basis_derivatives(mids, p)
    wd, wfb = wspace.tabulate_basis_derivatives(mids, p)
    np.testing.assert_allclose(wd, gd, atol=1e-11)
    np.testing.assert_array_equal(dof[wfb], gfb)  # windowed first-basis maps to global
    return wspace, dof


class TestBsplineSpace1DRestrict:
    """Tests for BsplineSpace1D.restrict (windowed sub-space + DOF map)."""

    def test_interior_window(self) -> None:
        _check_restrict_1d(_open_uniform_1d(2, 5), 1, 4)

    def test_left_boundary_window(self) -> None:
        _check_restrict_1d(_open_uniform_1d(2, 5), 0, 3)

    def test_interior_window_cubic(self) -> None:
        _check_restrict_1d(_open_uniform_1d(3, 6), 2, 5)

    def test_full_range_is_identity(self) -> None:
        space = _open_uniform_1d(3, 4)
        wspace, dof = _check_restrict_1d(space, 0, space.num_intervals)
        np.testing.assert_array_equal(wspace.knots, space.knots)
        np.testing.assert_array_equal(dof, np.arange(space.num_basis))

    def test_single_interval(self) -> None:
        space = _open_uniform_1d(2, 5)
        wspace, _ = _check_restrict_1d(space, 2, 3)
        assert wspace.num_intervals == 1
        assert wspace.num_basis == space.degree + 1

    def test_interior_window_not_reclamped(self) -> None:
        wspace, _ = _open_uniform_1d(2, 5).restrict(1, 4)
        # Interior window keeps the original (multiplicity-1) boundary knots.
        assert not wspace.has_left_end_open()
        assert not wspace.has_right_end_open()

    def test_dof_range(self) -> None:
        space = _open_uniform_1d(2, 4)  # knots [0,0,0,1,2,3,4,4,4], num_basis 6
        _, dof = space.restrict(1, 3)
        np.testing.assert_array_equal(dof, [1, 2, 3, 4])

    def test_domain_matches_window(self) -> None:
        wspace, _ = _open_uniform_1d(2, 5).restrict(1, 4)
        np.testing.assert_allclose(wspace.domain, (1.0, 4.0))

    def test_periodic_rejected(self) -> None:
        knots = create_uniform_periodic_knots(num_intervals=4, degree=2)
        space = BsplineSpace1D(knots, 2, periodic=True)
        with pytest.raises(ValueError, match="periodic"):
            space.restrict(0, 2)

    def test_invalid_range_raises(self) -> None:
        space = _open_uniform_1d(2, 4)
        with pytest.raises(ValueError, match="interval"):
            space.restrict(2, 2)  # lo == hi
        with pytest.raises(ValueError, match="interval"):
            space.restrict(-1, 2)  # lo < 0
        with pytest.raises(ValueError, match="interval"):
            space.restrict(0, 5)  # hi > num_intervals


class TestSerialParallelKernelTwins:
    """Serial twin kernels must match the parallel kernels exactly."""

    @staticmethod
    def _spaces() -> list[BsplineSpace1D]:
        """Build a mix of open non-uniform and periodic spaces of varied degree."""
        rng = np.random.default_rng(7)
        spaces = []
        for degree in (1, 2, 3, 4):
            interior = np.sort(rng.random(6))
            knots = np.concatenate([np.zeros(degree + 1), interior, np.ones(degree + 1)])
            spaces.append(BsplineSpace1D(knots, degree))
        periodic_knots = create_uniform_periodic_knots(num_intervals=6, degree=2)
        spaces.append(BsplineSpace1D(periodic_knots, 2, periodic=True))
        return spaces

    @pytest.mark.parametrize("num_pts", [1, 3, 100, 5000])
    def test_basis_values_match(self, num_pts: int) -> None:
        """Serial and parallel BasisFuncs kernels agree bitwise on all paths."""
        rng = np.random.default_rng(11)
        for sp in self._spaces():
            lo, hi = (float(sp.domain[0]), float(sp.domain[1]))
            pts = lo + (hi - lo) * rng.random(num_pts)
            order = sp.degree + 1

            basis_par = np.empty((num_pts, order), dtype=np.float64)
            first_par = np.empty(num_pts, dtype=np.int_)
            _compute_basis_nurbs_book_impl(
                sp.knots, sp.degree, sp.periodic, sp.tolerance, pts, basis_par, first_par
            )

            basis_ser = np.empty((num_pts, order), dtype=np.float64)
            first_ser = np.empty(num_pts, dtype=np.int_)
            _compute_basis_nurbs_book_serial_impl(
                sp.knots, sp.degree, sp.periodic, sp.tolerance, pts, basis_ser, first_ser
            )

            np.testing.assert_array_equal(basis_ser, basis_par)
            np.testing.assert_array_equal(first_ser, first_par)

    @pytest.mark.parametrize("num_pts", [1, 3, 100, 5000])
    def test_basis_derivs_match(self, num_pts: int) -> None:
        """Serial and parallel DerBasisFuncs kernels agree bitwise on all paths."""
        rng = np.random.default_rng(13)
        for sp in self._spaces():
            lo, hi = (float(sp.domain[0]), float(sp.domain[1]))
            pts = lo + (hi - lo) * rng.random(num_pts)
            order = sp.degree + 1
            n_deriv = sp.degree + 1  # include the identically-zero row

            der_par = np.empty((num_pts, n_deriv + 1, order), dtype=np.float64)
            first_par = np.empty(num_pts, dtype=np.int_)
            _compute_basis_deriv_nurbs_book_impl(
                sp.knots, sp.degree, sp.periodic, sp.tolerance, n_deriv, pts, der_par, first_par
            )

            der_ser = np.empty((num_pts, n_deriv + 1, order), dtype=np.float64)
            first_ser = np.empty(num_pts, dtype=np.int_)
            _compute_basis_deriv_nurbs_book_serial_impl(
                sp.knots, sp.degree, sp.periodic, sp.tolerance, n_deriv, pts, der_ser, first_ser
            )

            np.testing.assert_array_equal(der_ser, der_par)
            np.testing.assert_array_equal(first_ser, first_par)

    def test_float32_supported(self) -> None:
        """Serial twins compile and agree for float32 inputs."""
        knots = np.array([0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1], dtype=np.float32)
        sp = BsplineSpace1D(knots, 2)
        pts = np.array([0.1, 0.4, 0.9], dtype=np.float32)

        basis_par = np.empty((3, 3), dtype=np.float32)
        first_par = np.empty(3, dtype=np.int_)
        _compute_basis_nurbs_book_impl(
            sp.knots, sp.degree, sp.periodic, sp.tolerance, pts, basis_par, first_par
        )
        basis_ser = np.empty((3, 3), dtype=np.float32)
        first_ser = np.empty(3, dtype=np.int_)
        _compute_basis_nurbs_book_serial_impl(
            sp.knots, sp.degree, sp.periodic, sp.tolerance, pts, basis_ser, first_ser
        )
        np.testing.assert_array_equal(basis_ser, basis_par)
        np.testing.assert_array_equal(first_ser, first_par)

    def test_float32_deriv_supported(self) -> None:
        """Serial derivative twin compiles and agrees for float32 inputs."""
        knots = np.array([0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1], dtype=np.float32)
        sp = BsplineSpace1D(knots, 2)
        pts = np.array([0.1, 0.4, 0.9], dtype=np.float32)
        n_deriv = 2
        order = sp.degree + 1

        der_par = np.empty((3, n_deriv + 1, order), dtype=np.float32)
        first_par = np.empty(3, dtype=np.int_)
        _compute_basis_deriv_nurbs_book_impl(
            sp.knots, sp.degree, sp.periodic, sp.tolerance, n_deriv, pts, der_par, first_par
        )
        der_ser = np.empty((3, n_deriv + 1, order), dtype=np.float32)
        first_ser = np.empty(3, dtype=np.int_)
        _compute_basis_deriv_nurbs_book_serial_impl(
            sp.knots, sp.degree, sp.periodic, sp.tolerance, n_deriv, pts, der_ser, first_ser
        )
        np.testing.assert_array_equal(der_ser, der_par)
        np.testing.assert_array_equal(first_ser, first_par)

    def test_layer2_dispatch_consistent_across_threshold(self) -> None:
        """Layer-2 tabulation gives identical results below and above the threshold."""
        knots = create_uniform_open_knots(num_intervals=16, degree=3)
        sp = BsplineSpace1D(knots, 3)
        rng = np.random.default_rng(17)
        big = rng.random(_PARALLEL_MIN_NUM_PTS + 64)

        basis_big, first_big = _tabulate_Bspline_basis_1D_impl(sp, big)
        small = big[:8]
        basis_small, first_small = _tabulate_Bspline_basis_1D_impl(sp, small)
        np.testing.assert_array_equal(basis_small, basis_big[:8])
        np.testing.assert_array_equal(first_small, first_big[:8])

        # n_deriv=2: standard case
        der_big, dfirst_big = _tabulate_Bspline_basis_deriv_1D_impl(sp, big, 2)
        der_small, dfirst_small = _tabulate_Bspline_basis_deriv_1D_impl(sp, small, 2)
        np.testing.assert_array_equal(der_small, der_big[:8])
        np.testing.assert_array_equal(dfirst_small, dfirst_big[:8])

        # n_deriv=0: degenerate path — only 0th-order row is filled
        der_big0, _ = _tabulate_Bspline_basis_deriv_1D_impl(sp, big, 0)
        der_small0, _ = _tabulate_Bspline_basis_deriv_1D_impl(sp, small, 0)
        np.testing.assert_array_equal(der_small0, der_big0[:8])

    @pytest.mark.parametrize(
        "num_pts", [_PARALLEL_MIN_NUM_PTS - 1, _PARALLEL_MIN_NUM_PTS, _PARALLEL_MIN_NUM_PTS + 1]
    )
    def test_layer2_dispatch_boundary(self, num_pts: int) -> None:
        """Basis values are correct at, one below, and one above the dispatch threshold."""
        knots = create_uniform_open_knots(num_intervals=16, degree=3)
        sp = BsplineSpace1D(knots, 3)
        rng = np.random.default_rng(42)
        pts = rng.random(num_pts)
        basis, _ = _tabulate_Bspline_basis_1D_impl(sp, pts)
        np.testing.assert_allclose(basis.sum(axis=-1), np.ones(num_pts), atol=1e-13)

    def test_layer2_dispatch_periodic_small_batch(self) -> None:
        """Serial path is used for periodic spaces below the threshold."""
        periodic_knots = create_uniform_periodic_knots(num_intervals=8, degree=3)
        sp = BsplineSpace1D(periodic_knots, 3, periodic=True)
        rng = np.random.default_rng(99)
        lo, hi = float(sp.domain[0]), float(sp.domain[1])
        pts_small = lo + (hi - lo) * rng.random(8)
        pts_big = lo + (hi - lo) * rng.random(_PARALLEL_MIN_NUM_PTS + 64)

        basis_small, _ = _tabulate_Bspline_basis_1D_impl(sp, pts_small)
        np.testing.assert_allclose(basis_small.sum(axis=-1), np.ones(8), atol=1e-13)

        basis_big, _ = _tabulate_Bspline_basis_1D_impl(sp, pts_big)
        np.testing.assert_allclose(basis_big.sum(axis=-1), np.ones(len(pts_big)), atol=1e-13)


class TestKnotPredicateCaching:
    """Knot-structure predicates are cached and stable on immutable knots."""

    def test_predicates_stable_and_consistent(self) -> None:
        """Repeated predicate calls return the same values as a fresh instance."""
        knots = create_uniform_open_knots(num_intervals=4, degree=2)
        sp = BsplineSpace1D(knots, 2)
        fresh = BsplineSpace1D(knots.copy(), 2)
        for _ in range(3):
            assert sp.has_left_end_open() == fresh.has_left_end_open() is True
            assert sp.has_right_end_open() == fresh.has_right_end_open() is True
            assert sp.has_open_knots() == fresh.has_open_knots() is True
            assert sp.has_Bezier_like_knots() == fresh.has_Bezier_like_knots() is False

    def test_bezier_like_space(self) -> None:
        """Single-span open space reports Bézier-like on every call."""
        sp = BsplineSpace1D(np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0]), 2)
        assert sp.has_Bezier_like_knots()
        assert sp.has_Bezier_like_knots()  # cached second call

    def test_periodic_space_predicates(self) -> None:
        """Periodic spaces report closed ends and non-Bézier-like knots."""
        knots = create_uniform_periodic_knots(num_intervals=4, degree=2)
        sp = BsplineSpace1D(knots, 2, periodic=True)
        assert not sp.has_left_end_open()
        assert not sp.has_right_end_open()
        assert not sp.has_open_knots()
        assert not sp.has_Bezier_like_knots()

    def test_open_non_bezier_not_bezier_like(self) -> None:
        """Open knots with multiple spans must not report Bézier-like."""
        knots = create_uniform_open_knots(num_intervals=3, degree=2)
        sp = BsplineSpace1D(knots, 2)
        assert sp.has_open_knots()
        assert sp.num_basis > sp.degree + 1
        assert not sp.has_Bezier_like_knots()


class TestBernsteinSerialParallelTwins:
    """Bernstein serial twin cores must match the parallel cores exactly."""

    @pytest.mark.parametrize("num_pts", [1, 3, 100, 5000])
    @pytest.mark.parametrize("degree", [0, 1, 2, 4])
    def test_values_match(self, num_pts: int, degree: int) -> None:
        """Serial and parallel Bernstein value cores agree bitwise (incl. t=1)."""
        rng = np.random.default_rng(19)
        t = rng.random(num_pts)
        t[0] = 1.0  # exercise the t == 1 special case

        out_par = np.empty((num_pts, degree + 1), dtype=np.float64)
        _tabulate_Bernstein_basis_1D_core(np.int32(degree), t, out_par)
        out_ser = np.empty((num_pts, degree + 1), dtype=np.float64)
        _tabulate_Bernstein_basis_1D_serial_core(np.int32(degree), t, out_ser)
        np.testing.assert_array_equal(out_ser, out_par)

    @pytest.mark.parametrize("num_pts", [1, 3, 100, 5000])
    @pytest.mark.parametrize("degree", [0, 2, 3])
    def test_derivs_match(self, num_pts: int, degree: int) -> None:
        """Serial and parallel Bernstein derivative cores agree bitwise."""
        rng = np.random.default_rng(23)
        t = rng.random(num_pts)
        t[-1] = 1.0
        n_deriv = degree + 1

        out_par = np.empty((num_pts, n_deriv + 1, degree + 1), dtype=np.float64)
        _tabulate_Bernstein_basis_deriv_1D_core(np.int32(degree), t, n_deriv, out_par)
        out_ser = np.empty((num_pts, n_deriv + 1, degree + 1), dtype=np.float64)
        _tabulate_Bernstein_basis_deriv_1D_serial_core(np.int32(degree), t, n_deriv, out_ser)
        np.testing.assert_array_equal(out_ser, out_par)

    def test_bezier_like_layer2_dispatch_consistent(self) -> None:
        """Layer-2 results on a Bézier-like space agree below and above the threshold."""
        sp = BsplineSpace1D(np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), 2)
        rng = np.random.default_rng(29)
        big = rng.random(_PARALLEL_MIN_NUM_PTS + 64)

        basis_big, first_big = _tabulate_Bspline_basis_1D_impl(sp, big)
        basis_small, first_small = _tabulate_Bspline_basis_1D_impl(sp, big[:8])
        np.testing.assert_array_equal(basis_small, basis_big[:8])
        np.testing.assert_array_equal(first_small, first_big[:8])

        der_big, dfirst_big = _tabulate_Bspline_basis_deriv_1D_impl(sp, big, 2)
        der_small, dfirst_small = _tabulate_Bspline_basis_deriv_1D_impl(sp, big[:8], 2)
        np.testing.assert_array_equal(der_small, der_big[:8])
        np.testing.assert_array_equal(dfirst_small, dfirst_big[:8])


class TestTabulateValidateFlag:
    """The validate flag skips only the domain check, never changes results."""

    @staticmethod
    def _space() -> BsplineSpace1D:
        knots = create_uniform_open_knots(num_intervals=8, degree=3)
        return BsplineSpace1D(knots, 3)

    def test_values_identical_for_in_domain_points(self) -> None:
        """validate=False returns bitwise-identical results for in-domain points."""
        sp = self._space()
        pts = np.random.default_rng(31).random(50)

        b_val, f_val = sp.tabulate_basis(pts)
        b_no, f_no = sp.tabulate_basis(pts, validate=False)
        np.testing.assert_array_equal(b_no, b_val)
        np.testing.assert_array_equal(f_no, f_val)

        d_val, df_val = sp.tabulate_basis_derivatives(pts, 2)
        d_no, df_no = sp.tabulate_basis_derivatives(pts, 2, validate=False)
        np.testing.assert_array_equal(d_no, d_val)
        np.testing.assert_array_equal(df_no, df_val)

    def test_validate_true_still_raises_out_of_domain(self) -> None:
        """The default validate=True keeps rejecting out-of-domain points."""
        sp = self._space()
        with pytest.raises(ValueError, match="outside"):
            sp.tabulate_basis(np.array([-0.5]))
        with pytest.raises(ValueError, match="outside"):
            sp.tabulate_basis_derivatives(np.array([1.5]), 1)

    def test_validate_false_does_not_raise_out_of_domain_derivatives(self) -> None:
        """validate=False bypasses the domain check in tabulate_basis_derivatives."""
        sp = self._space()
        # Should not raise even though -0.5 is outside [0, 1].
        sp.tabulate_basis_derivatives(np.array([-0.5]), 1, validate=False)

    def test_validate_false_bezier_like_space(self) -> None:
        """validate=False works on a Bézier-like (Bernstein fast-path) space."""
        sp = BsplineSpace1D(np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), 2)
        rng = np.random.default_rng(41)
        pts = rng.random(10)
        b_val, f_val = sp.tabulate_basis(pts)
        b_no, f_no = sp.tabulate_basis(pts, validate=False)
        np.testing.assert_array_equal(b_no, b_val)
        np.testing.assert_array_equal(f_no, f_val)
        # validate=False should not raise for in-domain points on a Bézier-like space.
        sp.tabulate_basis_derivatives(pts, 2, validate=False)
