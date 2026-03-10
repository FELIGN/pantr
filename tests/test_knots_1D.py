"""Tests for knot vector utilities."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.testing as nptest
import numpy.typing as npt
import pytest

from pantr._bspline_knots import (
    _get_knots_ends_and_dtype,
    _validate_knot_input,
)
from pantr._bspline_space_factory import (
    create_cardinal_Bspline_knot_vector,
    create_uniform_open_knot_vector,
    create_uniform_periodic_knot_vector,
)
from pantr.tolerance import get_strict_tolerance


class TestValidateKnotInput:
    """Tests for `_validate_knot_input`."""

    def test_valid_inputs(self) -> None:
        """Accept well-formed parameters without raising."""
        _validate_knot_input(
            num_intervals=2,
            degree=2,
            continuity=1,
            domain=(np.float64(0.0), np.float64(1.0)),
            dtype=np.float64,
        )

    def test_domain_order_error(self) -> None:
        """Reject domains whose start exceeds the end."""
        with pytest.raises(ValueError, match=r"domain\[0\] must be less than domain\[1\]"):
            _validate_knot_input(
                num_intervals=2,
                degree=2,
                continuity=1,
                domain=(np.float64(1.0), np.float64(0.0)),
                dtype=np.float64,
            )

    def test_negative_num_intervals_error(self) -> None:
        """Reject negative interval counts."""
        with pytest.raises(ValueError, match="num_intervals must be non-negative"):
            _validate_knot_input(
                num_intervals=-1,
                degree=2,
                continuity=1,
                domain=(np.float64(0.0), np.float64(1.0)),
                dtype=np.float64,
            )

    def test_negative_degree_error(self) -> None:
        """Reject negative spline degrees."""
        with pytest.raises(ValueError, match="degree must be non-negative"):
            _validate_knot_input(
                num_intervals=2,
                degree=-1,
                continuity=1,
                domain=(np.float64(0.0), np.float64(1.0)),
                dtype=np.float64,
            )

    def test_invalid_continuity_upper_bound_error(self) -> None:
        """Reject continuity higher than degree - 1."""
        with pytest.raises(ValueError, match="Continuity must be between"):
            _validate_knot_input(
                num_intervals=2,
                degree=2,
                continuity=2,
                domain=(np.float64(0.0), np.float64(1.0)),
                dtype=np.float64,
            )

    def test_invalid_continuity_lower_bound_error(self) -> None:
        """Reject continuity less than -1."""
        with pytest.raises(ValueError, match="Continuity must be between"):
            _validate_knot_input(
                num_intervals=2,
                degree=2,
                continuity=-2,
                domain=(np.float64(0.0), np.float64(1.0)),
                dtype=np.float64,
            )

    def test_invalid_dtype_error(self) -> None:
        """Reject non-floating dtype requests."""
        with pytest.raises(ValueError, match="dtype must be float64 or float32"):
            _validate_knot_input(
                num_intervals=2,
                degree=2,
                continuity=1,
                domain=(np.float64(0.0), np.float64(1.0)),
                dtype=np.int32,
            )


class TestGetEndsAndType:
    """Tests for `_get_knots_ends_and_dtype`."""

    def test_defaults(self) -> None:
        """Return default [0, 1] domain in float64."""
        start, end, dtype = _get_knots_ends_and_dtype()
        assert isinstance(start, np.float64)
        assert isinstance(end, np.float64)
        assert start == np.float64(0.0)
        assert end == np.float64(1.0)
        assert dtype == np.dtype(np.float64)

    def test_provided_start_end(self) -> None:
        """Respect provided start and end values."""
        start, end, dtype = _get_knots_ends_and_dtype(start=0.5, end=2.0)
        assert start == np.float64(0.5)
        assert end == np.float64(2.0)
        assert dtype == np.dtype(np.float64)

    def test_provided_dtype(self) -> None:
        """Respect explicit dtype requests."""
        start, end, dtype = _get_knots_ends_and_dtype(dtype=np.float32)
        assert isinstance(start, np.float32)
        assert isinstance(end, np.float32)
        assert start == np.float32(0.0)
        assert end == np.float32(1.0)
        assert dtype == np.dtype(np.float32)

    def test_integer_inputs_promote_to_float64(self) -> None:
        """Promote integer inputs to float64."""
        start, end, dtype = _get_knots_ends_and_dtype(start=0, end=3)
        assert isinstance(start, np.float64)
        assert isinstance(end, np.float64)
        assert start == np.float64(0.0)
        assert end == np.float64(3.0)
        assert dtype == np.dtype(np.float64)

    def test_end_less_than_start_error(self) -> None:
        """Reject decreasing domains."""
        with pytest.raises(ValueError, match="end must be greater than start"):
            _get_knots_ends_and_dtype(start=1.0, end=0.0)

    def test_inferred_dtype_mismatch_error(self) -> None:
        """Reject inconsistent scalar dtypes when inferring."""
        with pytest.raises(ValueError, match="start and end must have the same dtype"):
            _get_knots_ends_and_dtype(start=np.float32(0.0), end=np.float64(1.0))

    def test_requested_dtype_mismatch_error(self) -> None:
        """Reject scalars incompatible with requested dtype."""
        with pytest.raises(ValueError, match="start must be of type dtype float64"):
            _get_knots_ends_and_dtype(
                start=np.float32(0.0),
                end=np.float32(1.0),
                dtype=np.float64,
            )

    def test_invalid_requested_dtype_error(self) -> None:
        """Reject non-floating dtype requests."""
        with pytest.raises(ValueError, match="dtype must be a floating-point type"):
            _get_knots_ends_and_dtype(dtype=np.int32)

    def test_non_scalar_input_error(self) -> None:
        """Reject array-valued inputs when resolving start/end."""
        with pytest.raises(ValueError, match="start must be a scalar value"):
            _get_knots_ends_and_dtype(
                start=cast(Any, np.array([0.0, 1.0], dtype=np.float64)),
            )


class TestCreateUniformOpenKnotVector:
    """Tests for `create_uniform_open_knot_vector`."""

    def test_basic_functionality(self) -> None:
        """Create uniform open knot vector with default continuity."""
        result = create_uniform_open_knot_vector(2, 2, domain=(0.0, 1.0))
        expected: npt.NDArray[np.float64] = np.array(
            [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            dtype=np.float64,
        )
        assert result.dtype == np.float64
        nptest.assert_allclose(result, expected)

    def test_degree_zero(self) -> None:
        """Handle degree 0 splines."""
        result = create_uniform_open_knot_vector(2, 0, domain=(0.0, 1.0))
        expected: npt.NDArray[np.float64] = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        nptest.assert_allclose(result, expected)

    def test_degree_one(self) -> None:
        """Handle degree 1 splines."""
        result = create_uniform_open_knot_vector(2, 1, domain=(0.0, 1.0))
        expected: npt.NDArray[np.float64] = np.array(
            [0.0, 0.0, 0.5, 1.0, 1.0],
            dtype=np.float64,
        )
        nptest.assert_allclose(result, expected)

    def test_single_interval(self) -> None:
        """Handle the single-interval case."""
        result = create_uniform_open_knot_vector(1, 2, domain=(0.0, 1.0))
        expected: npt.NDArray[np.float64] = np.array(
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            dtype=np.float64,
        )
        nptest.assert_allclose(result, expected)

    def test_custom_continuity(self) -> None:
        """Increase interior knot multiplicity when continuity decreases."""
        result = create_uniform_open_knot_vector(2, 2, continuity=0, domain=(0.0, 1.0))
        expected: npt.NDArray[np.float64] = np.array(
            [0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0],
            dtype=np.float64,
        )
        nptest.assert_allclose(result, expected)

    def test_custom_domain(self) -> None:
        """Respect non-default domain."""
        result = create_uniform_open_knot_vector(2, 2, domain=(1.0, 3.0))
        expected: npt.NDArray[np.float64] = np.array(
            [1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0],
            dtype=np.float64,
        )
        nptest.assert_allclose(result, expected)

    def test_float32_dtype(self) -> None:
        """Preserve float32 dtype requests."""
        result = create_uniform_open_knot_vector(2, 2, dtype=np.float32)
        expected: npt.NDArray[np.float32] = np.array(
            [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            dtype=np.float32,
        )
        assert result.dtype == np.float32
        nptest.assert_allclose(result, expected)

    def test_negative_num_intervals_error(self) -> None:
        """Reject negative interval counts."""
        with pytest.raises(ValueError, match="num_intervals must be non-negative"):
            create_uniform_open_knot_vector(-1, 2)

    def test_negative_degree_error(self) -> None:
        """Reject negative degrees."""
        with pytest.raises(ValueError, match="degree must be non-negative"):
            create_uniform_open_knot_vector(2, -1)

    def test_invalid_continuity_error(self) -> None:
        """Reject continuity outside valid range."""
        with pytest.raises(ValueError, match="Continuity must be between"):
            create_uniform_open_knot_vector(2, 2, continuity=2)


class TestCreateUniformPeriodicKnotVector:
    """Tests for `create_uniform_periodic_knot_vector`."""

    def test_basic_functionality(self) -> None:
        """Create periodic knot vector extending beyond the domain."""
        result = create_uniform_periodic_knot_vector(2, 2, domain=(0.0, 1.0))
        expected: npt.NDArray[np.float64] = np.array(
            [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
            dtype=np.float64,
        )
        nptest.assert_allclose(result, expected)

    def test_degree_zero(self) -> None:
        """Handle degree 0 periodic splines."""
        result = create_uniform_periodic_knot_vector(2, 0, domain=(0.0, 1.0))
        expected: npt.NDArray[np.float64] = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        nptest.assert_allclose(result, expected)

    def test_degree_one(self) -> None:
        """Handle degree 1 periodic splines."""
        result = create_uniform_periodic_knot_vector(2, 1, domain=(0.0, 1.0))
        expected: npt.NDArray[np.float64] = np.array(
            [-0.5, 0.0, 0.5, 1.0, 1.5],
            dtype=np.float64,
        )
        nptest.assert_allclose(result, expected)

    def test_single_interval(self) -> None:
        """Handle periodic construction with a single interval."""
        result = create_uniform_periodic_knot_vector(1, 2, domain=(0.0, 1.0))
        expected: npt.NDArray[np.float64] = np.array(
            [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            dtype=np.float64,
        )
        nptest.assert_allclose(result, expected)

    def test_custom_continuity(self) -> None:
        """Respect reduced continuity via increased multiplicity."""
        result = create_uniform_periodic_knot_vector(2, 2, continuity=0, domain=(0.0, 1.0))
        expected: npt.NDArray[np.float64] = np.array(
            [-0.5, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.5],
            dtype=np.float64,
        )
        nptest.assert_allclose(result, expected)

    def test_custom_domain(self) -> None:
        """Respect non-default domains."""
        result = create_uniform_periodic_knot_vector(2, 2, domain=(1.0, 3.0))
        expected: npt.NDArray[np.float64] = np.array(
            [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            dtype=np.float64,
        )
        nptest.assert_allclose(result, expected)

    def test_float32_dtype(self) -> None:
        """Preserve float32 dtype requests."""
        result = create_uniform_periodic_knot_vector(2, 2, dtype=np.float32)
        expected: npt.NDArray[np.float32] = np.array(
            [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
            dtype=np.float32,
        )
        assert result.dtype == np.float32
        nptest.assert_allclose(result, expected)

    def test_negative_num_intervals_error(self) -> None:
        """Reject negative interval counts."""
        with pytest.raises(ValueError, match="num_intervals must be non-negative"):
            create_uniform_periodic_knot_vector(-1, 2)

    def test_negative_degree_error(self) -> None:
        """Reject negative degrees."""
        with pytest.raises(ValueError, match="degree must be non-negative"):
            create_uniform_periodic_knot_vector(2, -1)

    def test_invalid_continuity_error(self) -> None:
        """Reject continuity outside valid range."""
        with pytest.raises(ValueError, match="Continuity must be between"):
            create_uniform_periodic_knot_vector(2, 2, continuity=2)


class TestCreateCardinalBsplineKnotVector:
    """Tests for `create_cardinal_Bspline_knot_vector`."""

    def test_basic_functionality(self) -> None:
        """Create cardinal knot vector with default dtype."""
        result = create_cardinal_Bspline_knot_vector(2, 2)
        expected: npt.NDArray[np.float64] = np.array(
            [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            dtype=np.float64,
        )
        nptest.assert_allclose(result, expected)

    def test_degree_zero(self) -> None:
        """Handle degree 0."""
        result = create_cardinal_Bspline_knot_vector(2, 0)
        expected: npt.NDArray[np.float64] = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        nptest.assert_allclose(result, expected)

    def test_degree_one(self) -> None:
        """Handle degree 1."""
        result = create_cardinal_Bspline_knot_vector(2, 1)
        expected: npt.NDArray[np.float64] = np.array(
            [-1.0, 0.0, 1.0, 2.0, 3.0],
            dtype=np.float64,
        )
        nptest.assert_allclose(result, expected)

    def test_single_interval(self) -> None:
        """Handle single interval."""
        result = create_cardinal_Bspline_knot_vector(1, 2)
        expected: npt.NDArray[np.float64] = np.array(
            [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            dtype=np.float64,
        )
        nptest.assert_allclose(result, expected)

    def test_high_degree(self) -> None:
        """Handle higher degrees."""
        result = create_cardinal_Bspline_knot_vector(2, 5)
        expected: npt.NDArray[np.float64] = np.array(
            [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            dtype=np.float64,
        )
        nptest.assert_allclose(result, expected)

    def test_float32_dtype(self) -> None:
        """Preserve float32 dtype requests."""
        result = create_cardinal_Bspline_knot_vector(2, 2, dtype=np.float32)
        expected: npt.NDArray[np.float32] = np.array(
            [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            dtype=np.float32,
        )
        assert result.dtype == np.float32
        nptest.assert_allclose(result, expected)

    def test_invalid_num_intervals_error(self) -> None:
        """Reject interval counts less than one."""
        with pytest.raises(ValueError, match="num_intervals must be at least 1"):
            create_cardinal_Bspline_knot_vector(0, 2)

    def test_negative_degree_error(self) -> None:
        """Reject negative degrees."""
        with pytest.raises(ValueError, match="degree must be non-negative"):
            create_cardinal_Bspline_knot_vector(2, -1)

    def test_invalid_dtype_error(self) -> None:
        """Reject non-floating dtype requests."""
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            create_cardinal_Bspline_knot_vector(2, 2, dtype=np.int32)


class TestKnotVectorIntegration:
    """Cross-check behaviours shared across knot construction helpers."""

    def test_consistency_across_functions(self) -> None:
        """Ensure outputs are non-decreasing and float64 by default."""
        knots_open = create_uniform_open_knot_vector(3, 2, domain=(0.0, 1.0))
        knots_periodic = create_uniform_periodic_knot_vector(3, 2, domain=(0.0, 1.0))
        knots_cardinal = create_cardinal_Bspline_knot_vector(3, 2)
        for knots in (knots_open, knots_periodic, knots_cardinal):
            diffs = np.diff(knots.astype(np.float64))
            assert np.all(diffs >= 0.0)
            assert knots.dtype == np.float64

    def test_domain_consistency(self) -> None:
        """Verify domain endpoints appear in expected positions."""
        start = np.float64(0.5)
        end = np.float64(2.5)
        degree = 2
        domain = (start, end)
        knots_open = create_uniform_open_knot_vector(2, degree, domain=domain)
        knots_periodic = create_uniform_periodic_knot_vector(2, degree, domain=domain)
        assert knots_open[degree] == start
        assert knots_open[-degree - 1] == end
        assert knots_periodic[degree] == start
        assert knots_periodic[-degree - 1] == end

    def test_continuity_consistency(self) -> None:
        """Check multiplicities align with requested continuity."""
        degree = 3
        continuity = 1
        domain = (0.0, 1.0)
        knots_open = create_uniform_open_knot_vector(
            2, degree, continuity=continuity, domain=domain
        )
        knots_periodic = create_uniform_periodic_knot_vector(
            2, degree, continuity=continuity, domain=domain
        )
        multiplicity = degree - continuity
        for knots in (knots_open, knots_periodic):
            tol = get_strict_tolerance(knots.dtype)
            domain_start = knots[degree]
            domain_end = knots[-degree - 1]
            mask = (knots > domain_start + tol) & (knots < domain_end - tol)
            interior_knots = knots[mask]
            if interior_knots.size == 0:
                continue
            unique_vals = np.unique(interior_knots)
            for val in unique_vals:
                count = np.sum(np.abs(interior_knots - val) < tol)
                assert count == multiplicity

    def test_vector_length(self) -> None:
        """Confirm lengths follow combinatorial expectations."""
        num_intervals = 2
        degree = 2
        continuity = degree - 1
        multiplicity = degree - continuity
        knots_open = create_uniform_open_knot_vector(num_intervals, degree)
        knots_periodic = create_uniform_periodic_knot_vector(num_intervals, degree)
        knots_cardinal = create_cardinal_Bspline_knot_vector(num_intervals, degree)
        expected_open = 2 * (degree + 1) + (num_intervals - 1) * multiplicity
        expected_periodic = 2 * (degree + 1 - multiplicity) + (num_intervals + 1) * multiplicity
        expected_cardinal = expected_periodic
        assert knots_open.size == expected_open
        assert knots_periodic.size == expected_periodic
        assert knots_cardinal.size == expected_cardinal

    def test_dtype_consistency(self) -> None:
        """Respect dtype arguments across generators."""
        dtype = np.dtype(np.float32)
        knots_open = create_uniform_open_knot_vector(2, 2, dtype=dtype)
        knots_periodic = create_uniform_periodic_knot_vector(2, 2, dtype=dtype)
        knots_cardinal = create_cardinal_Bspline_knot_vector(2, 2, dtype=dtype)
        assert knots_open.dtype == dtype
        assert knots_periodic.dtype == dtype
        assert knots_cardinal.dtype == dtype
