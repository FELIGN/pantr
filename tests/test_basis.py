"""Tests for basis function evaluation.

This module contains unit tests for the basis function evaluation functions,
including Bernstein basis polynomials.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any, Literal

import numpy as np
import numpy.testing as nptest
import numpy.typing as npt
import pytest
from scipy.interpolate import BPoly

from pantr.basis import (
    LagrangeVariant,
    tabulate_bernstein,
    tabulate_bernstein_1d,
    tabulate_cardinal_bspline,
    tabulate_cardinal_bspline_1d,
    tabulate_lagrange,
    tabulate_lagrange_1d,
    tabulate_legendre_1d,
)
from pantr.quad import PointsLattice
from pantr.tolerance import get_conservative, get_default

NEGATIVE_TOL: float = get_conservative(np.float64)


class BasisType(str, Enum):
    """Enumeration for basis function types."""

    BERNSTEIN = "bernstein"
    CARDINAL_BSPLINE = "cardinal_bspline"
    LAGRANGE = "lagrange"


def _get_basis_function(
    basis_type: BasisType,
) -> Callable[..., npt.NDArray[np.float32 | np.float64]]:
    """Get the multi-dimensional basis function for the given type."""
    if basis_type == BasisType.BERNSTEIN:
        return tabulate_bernstein
    if basis_type == BasisType.CARDINAL_BSPLINE:
        return tabulate_cardinal_bspline
    if basis_type == BasisType.LAGRANGE:
        return tabulate_lagrange
    raise ValueError(f"Unknown basis type: {basis_type}")


def _get_basis_1d_function(
    basis_type: BasisType,
) -> Callable[..., npt.NDArray[np.float32 | np.float64]]:
    """Get the 1D basis function for the given type."""
    if basis_type == BasisType.BERNSTEIN:
        return tabulate_bernstein_1d
    if basis_type == BasisType.CARDINAL_BSPLINE:
        return tabulate_cardinal_bspline_1d
    if basis_type == BasisType.LAGRANGE:
        return tabulate_lagrange_1d
    raise ValueError(f"Unknown basis type: {basis_type}")


def _call_basis_function(
    basis_type: BasisType,
    degrees: list[int],
    pts: npt.ArrayLike | PointsLattice,
    variant: LagrangeVariant | None = None,
    funcs_order: Literal["C", "F"] = "C",
) -> npt.NDArray[np.float32 | np.float64]:
    """Call the appropriate basis function with the given arguments."""
    func = _get_basis_function(basis_type)
    if basis_type == BasisType.LAGRANGE:
        if variant is None:
            variant = LagrangeVariant.EQUISPACES
        return func(degrees, variant, pts, funcs_order)
    return func(degrees, pts, funcs_order)


def _call_basis_1d_function(
    basis_type: BasisType,
    degree: int,
    pts: npt.ArrayLike,
    variant: LagrangeVariant | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Call the appropriate 1D basis function with the given arguments."""
    func = _get_basis_1d_function(basis_type)
    if basis_type == BasisType.LAGRANGE:
        if variant is None:
            variant = LagrangeVariant.EQUISPACES
        return func(degree, variant, pts)
    return func(degree, pts)


def _compute_with_scipy_bpoly(
    degree: int, pts: npt.ArrayLike, dtype: npt.DTypeLike
) -> npt.NDArray[np.floating[Any]]:
    """Evaluate all Bernstein basis polynomials via SciPy's BPoly on [0, 1]."""
    t = np.array(pts, dtype=dtype, copy=False).ravel()
    num_pts = t.size
    vals = np.empty((num_pts, degree + 1), dtype=np.result_type(dtype, np.float64))
    breaks = np.array([0.0, 1.0], dtype=dtype)

    for i in range(degree + 1):
        coeffs = np.zeros((degree + 1, 1), dtype=dtype)
        coeffs[i, 0] = 1.0
        bp = BPoly(coeffs, breaks, extrapolate=True)
        vals[:, i] = bp(t)

    return vals.reshape((*np.array(pts).shape, degree + 1))


class TestEvalBernsteinBasis1D:
    """Test suite for tabulate_bernstein_1d function."""

    def test_degree_zero(self) -> None:
        """Test Bernstein basis of degree 0."""
        pts = np.array([0.0, 0.5, 1.0])
        result = tabulate_bernstein_1d(0, pts)
        expected = np.ones((3, 1))
        nptest.assert_allclose(result, expected)

    def test_degree_one(self) -> None:
        """Test Bernstein basis of degree 1 (linear)."""
        pts = np.array([0.0, 0.5, 1.0])
        result = tabulate_bernstein_1d(1, pts)
        # B_0,1(t) = 1-t, B_1,1(t) = t
        expected = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        nptest.assert_allclose(result, expected)

    def test_degree_two_example(self) -> None:
        """Test Bernstein basis of degree 2 with example from docstring."""
        pts = np.array([0.0, 0.5, 0.75, 1.0])
        result = tabulate_bernstein_1d(2, pts)
        expected = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.25, 0.5, 0.25],
                [0.0625, 0.375, 0.5625],
                [0.0, 0.0, 1.0],
            ]
        )
        nptest.assert_allclose(result, expected)

    def test_boundary_points(self) -> None:
        """Test Bernstein basis at boundary points t=0 and t=1."""
        degree = 3
        pts = np.array([0.0, 1.0])
        result = tabulate_bernstein_1d(degree, pts)
        # At t=0: only B_0,n(0) = 1, all others are 0
        # At t=1: only B_n,n(1) = 1, all others are 0
        expected = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        nptest.assert_allclose(result, expected)

    def test_scalar_input(self) -> None:
        """Test with scalar input point."""
        result = tabulate_bernstein_1d(2, 0.5)
        expected = np.array([0.25, 0.5, 0.25])
        nptest.assert_allclose(result, expected)

    def test_list_input(self) -> None:
        """Test with list input."""
        result = tabulate_bernstein_1d(2, [0.0, 0.5, 1.0])
        expected = np.array([[1.0, 0.0, 0.0], [0.25, 0.5, 0.25], [0.0, 0.0, 1.0]])
        nptest.assert_allclose(result, expected)

    def test_float32_dtype(self) -> None:
        """Test with float32 input dtype."""
        pts = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = tabulate_bernstein_1d(2, pts)
        assert result.dtype == np.float32
        expected = np.array([[1.0, 0.0, 0.0], [0.25, 0.5, 0.25], [0.0, 0.0, 1.0]], dtype=np.float32)
        nptest.assert_allclose(result, expected)

    def test_float64_dtype(self) -> None:
        """Test with float64 input dtype."""
        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        result = tabulate_bernstein_1d(2, pts)
        assert result.dtype == np.float64
        expected = np.array([[1.0, 0.0, 0.0], [0.25, 0.5, 0.25], [0.0, 0.0, 1.0]], dtype=np.float64)
        nptest.assert_allclose(result, expected)

    def test_int_input_converted_to_float64(self) -> None:
        """Test that integer inputs are converted to float64."""
        pts = np.array([0, 1], dtype=np.int32)
        result = tabulate_bernstein_1d(2, pts)
        assert result.dtype == np.float64

    def test_negative_degree_raises_error(self) -> None:
        """Test that negative degree raises ValueError."""
        pts = np.array([0.0, 0.5, 1.0])
        with pytest.raises(ValueError, match="degree must be non-negative"):
            tabulate_bernstein_1d(-1, pts)

    def test_partition_of_unity(self) -> None:
        """Test that Bernstein basis functions sum to 1 (partition of unity)."""
        degree = 4
        pts = np.linspace(0.0, 1.0, 11)
        result = tabulate_bernstein_1d(degree, pts)
        # Sum over the last dimension should be 1 for each point
        sums = np.sum(result, axis=-1)
        nptest.assert_allclose(sums, 1.0, rtol=get_conservative(np.float64))


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 5, 10])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "pts_factory",
    [
        pytest.param(lambda dt: np.array(0.3, dtype=dt), id="scalar"),
        pytest.param(lambda dt: np.linspace(0.0, 1.0, 11, dtype=dt), id="1d-linspace"),
        pytest.param(
            lambda dt: np.array([[0.0, 0.25, 0.5], [0.75, 1.0, 1 / 3]], dtype=dt),
            id="2d-array",
        ),
    ],
)
def test_matches_bpoly(
    degree: int,
    dtype: npt.DTypeLike,
    pts_factory: Callable[[npt.DTypeLike], npt.ArrayLike],
) -> None:
    """Compare tabulate_bernstein_1d against BPoly for varied degrees/dtypes/shapes."""
    pts = pts_factory(dtype)
    pantr_vals = tabulate_bernstein_1d(degree, pts)
    bpoly_vals = _compute_with_scipy_bpoly(degree, pts, dtype)

    rtol = get_default(dtype)
    nptest.assert_allclose(pantr_vals, bpoly_vals, rtol=rtol, atol=0.0)


@pytest.mark.parametrize("degree", [2, 4, 8])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_outside_unit_interval_matches_bpoly(degree: int, dtype: npt.DTypeLike) -> None:
    """Ensure evaluation outside [0, 1] matches BPoly extrapolation."""
    pts = np.array([-0.5, -0.1, 1.1, 1.5], dtype=dtype)
    pantr_vals = tabulate_bernstein_1d(degree, pts)
    bpoly_vals = _compute_with_scipy_bpoly(degree, pts, dtype)

    rtol = get_default(dtype)
    nptest.assert_allclose(pantr_vals, bpoly_vals, rtol=rtol, atol=0.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_random_points_high_degree(dtype: npt.DTypeLike) -> None:
    """Stress test with higher degree and random points."""
    rng = np.random.default_rng(1234)
    degree = 12
    pts = rng.random(200).astype(dtype)
    pantr_vals = tabulate_bernstein_1d(degree, pts)
    bpoly_vals = _compute_with_scipy_bpoly(degree, pts, dtype)

    rtol = 5.0 * get_default(dtype)
    nptest.assert_allclose(pantr_vals, bpoly_vals, rtol=rtol, atol=0.0)


def test_non_negativity() -> None:
    """Test that Bernstein basis functions are non-negative on [0, 1]."""
    degree = 3
    pts = np.linspace(0.0, 1.0, 21)
    result = tabulate_bernstein_1d(degree, pts)
    # Allow small numerical errors
    assert np.all(result >= -NEGATIVE_TOL)


def test_2d_input_shape_preservation() -> None:
    """Test that 2D input arrays preserve shape correctly."""
    pts = np.array([[0.0, 0.5], [0.25, 0.75]])
    result = tabulate_bernstein_1d(2, pts)
    # Should have shape (2, 2, 3) - original shape + basis dimension
    assert result.shape == (2, 2, 3)


def test_3d_input_shape_preservation() -> None:
    """Test that 3D input arrays preserve shape correctly."""
    pts = np.array([[[0.0], [0.5]], [[0.25], [0.75]]])
    result = tabulate_bernstein_1d(1, pts)
    # Should have shape (2, 2, 1, 2) - original shape + basis dimension
    assert result.shape == (2, 2, 1, 2)


def test_single_point_at_midpoint() -> None:
    """Test evaluation at a single midpoint."""
    result = tabulate_bernstein_1d(3, 0.5)
    # For degree 3 at t=0.5, all basis functions should be symmetric
    # B_0,3(0.5) = B_3,3(0.5) = (1/2)^3 = 0.125
    # B_1,3(0.5) = B_2,3(0.5) = 3 * (1/2)^3 = 0.375
    expected = np.array([0.125, 0.375, 0.375, 0.125])
    nptest.assert_allclose(result, expected)


def test_high_degree() -> None:
    """Test with higher degree to ensure numerical stability."""
    degree = 10
    pts = np.array([0.0, 0.5, 1.0])
    result = tabulate_bernstein_1d(degree, pts)
    # Check shape
    assert result.shape == (3, degree + 1)
    # Check partition of unity
    sums = np.sum(result, axis=-1)
    nptest.assert_allclose(sums, 1.0, rtol=get_conservative(np.float64))
    # Check boundary conditions
    nptest.assert_allclose(result[0, 0], 1.0)
    nptest.assert_allclose(result[0, 1:], 0.0)
    nptest.assert_allclose(result[2, -1], 1.0)
    nptest.assert_allclose(result[2, :-1], 0.0)


def test_points_outside_unit_interval() -> None:
    """Test evaluation at points outside [0, 1]."""
    # Bernstein basis can be evaluated outside [0, 1], though values may be negative
    pts = np.array([-0.5, 1.5])
    result = tabulate_bernstein_1d(2, pts)
    # Should not raise an error and should return valid values
    assert result.shape == (2, 3)
    # Check partition of unity still holds
    sums = np.sum(result, axis=-1)
    nptest.assert_allclose(sums, 1.0, rtol=get_conservative(np.float64))


@pytest.mark.parametrize(
    "basis_type", [BasisType.BERNSTEIN, BasisType.CARDINAL_BSPLINE, BasisType.LAGRANGE]
)
class TestEvalBasis:
    """Test suite for multi-dimensional basis functions.

    Tests Bernstein, cardinal B-spline, and Lagrange basis functions.
    """

    def test_2d_array_single_degree(self, basis_type: BasisType) -> None:
        """Test 2D array input with same degree in both dimensions."""
        degrees = [2, 2]
        pts = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64)
        result = _call_basis_function(basis_type, degrees, pts)
        # Should have shape (n_points, n_basis_functions) = (3, 9) for degree 2 in 2D
        assert result.shape == (3, 9)
        # Verify by comparing with manual computation
        basis_x = _call_basis_1d_function(basis_type, 2, pts[:, 0])
        basis_y = _call_basis_1d_function(basis_type, 2, pts[:, 1])
        expected = np.einsum("pi,pj->pij", basis_x, basis_y).reshape(3, 9)
        if basis_type == BasisType.LAGRANGE:
            rtol = get_default(np.float64)
            nptest.assert_allclose(result, expected, rtol=rtol, atol=0.0)
        else:
            nptest.assert_allclose(result, expected)

    def test_2d_array_different_degrees(self, basis_type: BasisType) -> None:
        """Test 2D array input with different degrees per dimension."""
        degrees = [1, 2]
        pts = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64)
        result = _call_basis_function(basis_type, degrees, pts)
        assert result.shape == (3, 6)
        basis_x = _call_basis_1d_function(basis_type, 1, pts[:, 0])
        basis_y = _call_basis_1d_function(basis_type, 2, pts[:, 1])
        expected = np.einsum("pi,pj->pij", basis_x, basis_y).reshape(3, 6)
        if basis_type == BasisType.LAGRANGE:
            rtol = get_default(np.float64)
            nptest.assert_allclose(result, expected, rtol=rtol, atol=0.0)
        else:
            nptest.assert_allclose(result, expected)

    def test_points_lattice(self, basis_type: BasisType) -> None:
        """Test PointsLattice input."""
        degrees = [2, 2]
        pts_x = np.linspace(0.0, 1.0, 3, dtype=np.float64)
        pts_y = np.linspace(0.0, 1.0, 3, dtype=np.float64)
        lattice = PointsLattice([pts_x, pts_y])
        result = _call_basis_function(basis_type, degrees, lattice)
        assert result.shape == (9, 9)

    def test_points_lattice_mismatched_dimension_raises_error(self, basis_type: BasisType) -> None:
        """Test that mismatched number of degrees vs PointsLattice dimension raises ValueError."""
        degrees = [1, 1, 1]
        pts_x = np.linspace(0.0, 1.0, 3, dtype=np.float64)
        pts_y = np.linspace(0.0, 1.0, 3, dtype=np.float64)
        lattice = PointsLattice([pts_x, pts_y])
        with pytest.raises(
            ValueError, match="The number of evaluators must be equal to the dimension"
        ):
            _call_basis_function(basis_type, degrees, lattice)

    def test_points_lattice_1d(self, basis_type: BasisType) -> None:
        """Test 1D PointsLattice input (to cover return path in _compute_basis_combinator_matrix_for_points_lattice)."""  # noqa: E501
        degrees = [2]
        pts_x = np.linspace(0.0, 1.0, 3, dtype=np.float64)
        lattice = PointsLattice([pts_x])
        result = _call_basis_function(basis_type, degrees, lattice)
        assert result.shape == (3, 3)

    def test_points_lattice_3d(self, basis_type: BasisType) -> None:
        """Test 3D PointsLattice input (to cover loop in _compute_basis_combinator_matrix_for_points_lattice)."""  # noqa: E501
        degrees = [1, 1, 1]
        pts_x = np.linspace(0.0, 1.0, 2, dtype=np.float64)
        pts_y = np.linspace(0.0, 1.0, 2, dtype=np.float64)
        pts_z = np.linspace(0.0, 1.0, 2, dtype=np.float64)
        lattice = PointsLattice([pts_x, pts_y, pts_z])
        result = _call_basis_function(basis_type, degrees, lattice)
        assert result.shape == (8, 8)

    def test_funcs_order_c(self, basis_type: BasisType) -> None:
        """Test C-order (default) for basis functions."""
        degrees = [1, 1]
        pts = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64)
        result = _call_basis_function(basis_type, degrees, pts, funcs_order="C")
        assert result.shape == (3, 4)

    def test_funcs_order_f(self, basis_type: BasisType) -> None:
        """Test F-order for basis functions."""
        degrees = [1, 1]
        pts = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64)
        result_f = _call_basis_function(basis_type, degrees, pts, funcs_order="F")
        result_c = _call_basis_function(basis_type, degrees, pts, funcs_order="C")
        assert result_f.shape == result_c.shape == (3, 4)

    def test_3d_array(self, basis_type: BasisType) -> None:
        """Test 3D array input."""
        degrees = [1, 1, 1]
        pts = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]], dtype=np.float64)
        result = _call_basis_function(basis_type, degrees, pts)
        assert result.shape == (3, 8)

    def test_dtype_preservation_float32(self, basis_type: BasisType) -> None:
        """Test dtype preservation with float32."""
        degrees = [2, 2]
        pts = np.array([[0.0, 0.0], [0.5, 0.5]], dtype=np.float32)
        result = _call_basis_function(basis_type, degrees, pts)
        assert result.dtype == np.float32

    def test_dtype_preservation_float64(self, basis_type: BasisType) -> None:
        """Test dtype preservation with float64."""
        degrees = [2, 2]
        pts = np.array([[0.0, 0.0], [0.5, 0.5]], dtype=np.float64)
        result = _call_basis_function(basis_type, degrees, pts)
        assert result.dtype == np.float64

    def test_negative_degree_raises_error(self, basis_type: BasisType) -> None:
        """Test that negative degree raises ValueError."""
        degrees = [-1, 2]
        pts = np.array([[0.0, 0.0], [0.5, 0.5]], dtype=np.float64)
        with pytest.raises(ValueError, match="All degrees must be non-negative integers"):
            _call_basis_function(basis_type, degrees, pts)

    def test_partition_of_unity(self, basis_type: BasisType) -> None:
        """Test that multi-dimensional basis functions sum to 1."""
        degrees = [2, 2]
        pts = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64)
        result = _call_basis_function(basis_type, degrees, pts)
        sums = np.sum(result, axis=-1)
        if basis_type in (BasisType.LAGRANGE, BasisType.CARDINAL_BSPLINE):
            rtol = get_default(np.float64)
            nptest.assert_allclose(sums, 1.0, rtol=rtol, atol=0.0)
        else:
            nptest.assert_allclose(sums, 1.0, rtol=get_conservative(np.float64))

    def test_single_point(self, basis_type: BasisType) -> None:
        """Test with single point."""
        degrees = [1, 1]
        pts = np.array([[0.5, 0.5]], dtype=np.float64)
        result = _call_basis_function(basis_type, degrees, pts)
        assert result.shape == (1, 4)
        if basis_type in (BasisType.LAGRANGE, BasisType.CARDINAL_BSPLINE):
            rtol = get_default(np.float64)
            nptest.assert_allclose(np.sum(result), 1.0, rtol=rtol, atol=0.0)
        else:
            nptest.assert_allclose(np.sum(result), 1.0)

    def test_non_ndarray_input_list(self, basis_type: BasisType) -> None:
        """Test with list input (converted to ndarray internally)."""
        degrees = [1, 1]
        pts = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
        result = _call_basis_function(basis_type, degrees, pts)
        assert result.shape == (3, 4)

    def test_non_ndarray_input_tuple(self, basis_type: BasisType) -> None:
        """Test with tuple input (converted to ndarray internally)."""
        degrees = [2]
        pts = ((0.0,), (0.5,), (1.0,))
        result = _call_basis_function(basis_type, degrees, pts)
        assert result.shape == (3, 3)

    def test_non_2d_array_raises_error(self, basis_type: BasisType) -> None:
        """Test that non-2D array input raises ValueError."""
        degrees = [1, 1]
        pts_1d = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="Input points must be a 2D array"):
            _call_basis_function(basis_type, degrees, pts_1d)
        pts_3d = np.array([[[0.0, 0.0]]], dtype=np.float64)
        with pytest.raises(ValueError, match="Input points must be a 2D array"):
            _call_basis_function(basis_type, degrees, pts_3d)

    def test_int_dtype_converted_to_float64(self, basis_type: BasisType) -> None:
        """Test that integer dtype is converted to float64."""
        degrees = [1, 1]
        pts = np.array([[0, 0], [1, 1]], dtype=np.int32)
        result = _call_basis_function(basis_type, degrees, pts)
        assert result.dtype == np.float64

    def test_empty_dimension_raises_error(self, basis_type: BasisType) -> None:
        """Test that empty dimension (dim < 1) raises ValueError."""
        degrees = [1]
        pts = np.array([[]], dtype=np.float64).reshape(1, 0)
        with pytest.raises(ValueError, match="The dimension of the points must be at least 1"):
            _call_basis_function(basis_type, degrees, pts)

    def test_mismatched_degrees_dimension_raises_error(self, basis_type: BasisType) -> None:
        """Test that mismatched number of degrees vs dimension raises ValueError."""
        degrees = [1, 1, 1]
        pts = np.array([[0.0, 0.0], [0.5, 0.5]], dtype=np.float64)
        with pytest.raises(
            ValueError, match="The number of evaluators must be equal to the dimension"
        ):
            _call_basis_function(basis_type, degrees, pts)


@pytest.mark.parametrize(
    "variant",
    [
        LagrangeVariant.EQUISPACES,
        LagrangeVariant.GAUSS_LEGENDRE,
        LagrangeVariant.GAUSS_LOBATTO_LEGENDRE,
        LagrangeVariant.CHEBYSHEV_1ST,
        LagrangeVariant.CHEBYSHEV_2ND,
    ],
)
def test_2d_array_single_degree_lagrange_variants(variant: LagrangeVariant) -> None:
    """Test 2D array input with same degree in both dimensions for Lagrange variants."""
    degrees = [2, 2]
    pts = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64)
    result = _call_basis_function(BasisType.LAGRANGE, degrees, pts, variant)
    # Should have shape (n_points, n_basis_functions) = (3, 9) for degree 2 in 2D
    assert result.shape == (3, 9)
    # Verify by comparing with manual computation
    basis_x = _call_basis_1d_function(BasisType.LAGRANGE, 2, pts[:, 0], variant)
    basis_y = _call_basis_1d_function(BasisType.LAGRANGE, 2, pts[:, 1], variant)
    expected = np.einsum("pi,pj->pij", basis_x, basis_y).reshape(3, 9)
    rtol = get_default(np.float64)
    nptest.assert_allclose(result, expected, rtol=rtol, atol=0.0)


@pytest.mark.parametrize(
    "variant",
    [
        LagrangeVariant.EQUISPACES,
        LagrangeVariant.GAUSS_LEGENDRE,
        LagrangeVariant.GAUSS_LOBATTO_LEGENDRE,
        LagrangeVariant.CHEBYSHEV_1ST,
        LagrangeVariant.CHEBYSHEV_2ND,
    ],
)
def test_partition_of_unity_lagrange_variants(variant: LagrangeVariant) -> None:
    """Test that multi-dimensional Lagrange basis functions sum to 1 for all variants."""
    degrees = [2, 2]
    pts = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64)
    result = _call_basis_function(BasisType.LAGRANGE, degrees, pts, variant)
    sums = np.sum(result, axis=-1)
    rtol = get_default(np.float64)
    nptest.assert_allclose(sums, 1.0, rtol=rtol, atol=0.0)


@pytest.mark.parametrize(
    "basis_type", [BasisType.BERNSTEIN, BasisType.CARDINAL_BSPLINE, BasisType.LAGRANGE]
)
@pytest.mark.parametrize("degrees", [[1, 1, 1], [2, 2, 2], [1, 2, 3], [3, 2, 1], [0, 1, 2]])
def test_3d_array_single_degree(basis_type: BasisType, degrees: list[int]) -> None:
    """Test 3D array input with various degree combinations."""
    pts = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [0.25, 0.75, 0.5]],
        dtype=np.float64,
    )
    variant = LagrangeVariant.EQUISPACES if basis_type == BasisType.LAGRANGE else None
    result = _call_basis_function(basis_type, degrees, pts, variant)
    # Expected shape: (n_points, n_basis_functions)
    # n_basis_functions = (degrees[0] + 1) * (degrees[1] + 1) * (degrees[2] + 1)
    expected_n_basis = (degrees[0] + 1) * (degrees[1] + 1) * (degrees[2] + 1)
    assert result.shape == (4, expected_n_basis)
    # Verify by comparing with manual computation using einsum
    basis_x = _call_basis_1d_function(basis_type, degrees[0], pts[:, 0], variant)
    basis_y = _call_basis_1d_function(basis_type, degrees[1], pts[:, 1], variant)
    basis_z = _call_basis_1d_function(basis_type, degrees[2], pts[:, 2], variant)
    expected = np.einsum("pi,pj,pk->pijk", basis_x, basis_y, basis_z).reshape(4, expected_n_basis)
    if basis_type == BasisType.LAGRANGE:
        rtol = get_default(np.float64)
        nptest.assert_allclose(result, expected, rtol=rtol, atol=0.0)
    else:
        nptest.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "variant",
    [
        LagrangeVariant.EQUISPACES,
        LagrangeVariant.GAUSS_LEGENDRE,
        LagrangeVariant.GAUSS_LOBATTO_LEGENDRE,
        LagrangeVariant.CHEBYSHEV_1ST,
        LagrangeVariant.CHEBYSHEV_2ND,
    ],
)
@pytest.mark.parametrize("degrees", [[1, 1, 1], [2, 2, 2], [1, 2, 3], [3, 2, 1]])
def test_3d_array_lagrange_variants(variant: LagrangeVariant, degrees: list[int]) -> None:
    """Test 3D array input with various degree combinations for Lagrange variants."""
    pts = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [0.25, 0.75, 0.5]],
        dtype=np.float64,
    )
    result = _call_basis_function(BasisType.LAGRANGE, degrees, pts, variant)
    # Expected shape: (n_points, n_basis_functions)
    expected_n_basis = (degrees[0] + 1) * (degrees[1] + 1) * (degrees[2] + 1)
    assert result.shape == (4, expected_n_basis)
    # Verify by comparing with manual computation using einsum
    basis_x = _call_basis_1d_function(BasisType.LAGRANGE, degrees[0], pts[:, 0], variant)
    basis_y = _call_basis_1d_function(BasisType.LAGRANGE, degrees[1], pts[:, 1], variant)
    basis_z = _call_basis_1d_function(BasisType.LAGRANGE, degrees[2], pts[:, 2], variant)
    expected = np.einsum("pi,pj,pk->pijk", basis_x, basis_y, basis_z).reshape(4, expected_n_basis)
    rtol = get_default(np.float64)
    nptest.assert_allclose(result, expected, rtol=rtol, atol=0.0)


@pytest.mark.parametrize(
    "basis_type", [BasisType.BERNSTEIN, BasisType.CARDINAL_BSPLINE, BasisType.LAGRANGE]
)
class TestOutParameter1D:
    """Test suite for out parameter in 1D tabulate functions."""

    def test_out_parameter_works(self, basis_type: BasisType) -> None:
        """Test that out parameter works correctly."""
        degree = 2
        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        func = _get_basis_1d_function(basis_type)

        # Compute without out parameter
        if basis_type == BasisType.LAGRANGE:
            result1 = func(degree, LagrangeVariant.EQUISPACES, pts)
        else:
            result1 = func(degree, pts)

        # Compute with out parameter
        out = np.zeros_like(result1)
        if basis_type == BasisType.LAGRANGE:
            result2 = func(degree, LagrangeVariant.EQUISPACES, pts, out=out)
        else:
            result2 = func(degree, pts, out=out)

        # Results should match
        nptest.assert_allclose(result1, result2)
        # Verify that out was used (values match exactly)
        nptest.assert_allclose(out, result1)

    def test_out_parameter_wrong_shape_raises_error(self, basis_type: BasisType) -> None:
        """Test that out parameter with wrong shape raises ValueError."""
        degree = 2
        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        func = _get_basis_1d_function(basis_type)

        # Create out with wrong shape
        out = np.zeros((3, 2), dtype=np.float64)  # Should be (3, 3) for degree 2

        with pytest.raises(ValueError, match="Output array has shape"):
            if basis_type == BasisType.LAGRANGE:
                func(degree, LagrangeVariant.EQUISPACES, pts, out=out)
            else:
                func(degree, pts, out=out)

    def test_out_parameter_wrong_dtype_raises_error(self, basis_type: BasisType) -> None:
        """Test that out parameter with wrong dtype raises ValueError."""
        degree = 2
        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        func = _get_basis_1d_function(basis_type)

        # Create out with wrong dtype
        out = np.zeros((3, 3), dtype=np.float32)  # Should be float64 to match pts

        with pytest.raises(ValueError, match="Output array has dtype"):
            if basis_type == BasisType.LAGRANGE:
                func(degree, LagrangeVariant.EQUISPACES, pts, out=out)
            else:
                func(degree, pts, out=out)

    def test_out_parameter_float32(self, basis_type: BasisType) -> None:
        """Test that out parameter works with float32."""
        degree = 2
        pts = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        func = _get_basis_1d_function(basis_type)

        # Compute without out parameter
        if basis_type == BasisType.LAGRANGE:
            result1 = func(degree, LagrangeVariant.EQUISPACES, pts)
        else:
            result1 = func(degree, pts)

        # Compute with out parameter
        out = np.zeros_like(result1)
        if basis_type == BasisType.LAGRANGE:
            result2 = func(degree, LagrangeVariant.EQUISPACES, pts, out=out)
        else:
            result2 = func(degree, pts, out=out)

        # Results should match
        nptest.assert_allclose(result1, result2)
        assert result2.dtype == np.float32


def test_out_parameter_legendre() -> None:
    """Test that out parameter works for Legendre basis."""
    degree = 2
    pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)

    # Compute without out parameter
    result1 = tabulate_legendre_1d(degree, pts)

    # Compute with out parameter
    out = np.zeros_like(result1)
    result2 = tabulate_legendre_1d(degree, pts, out=out)

    # Results should match
    nptest.assert_allclose(result1, result2)
    # Verify that out was used (values match exactly)
    nptest.assert_allclose(out, result1)

    # Test with wrong shape
    out_wrong = np.zeros((3, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="Output array has shape"):
        tabulate_legendre_1d(degree, pts, out=out_wrong)

    # Test with wrong dtype
    out_wrong_dtype = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="Output array has dtype"):
        tabulate_legendre_1d(degree, pts, out=out_wrong_dtype)

    # Test with read-only array
    out_readonly = np.zeros((3, 3), dtype=np.float64)
    out_readonly.setflags(write=False)
    with pytest.raises(ValueError, match="Output array is not writeable"):
        tabulate_legendre_1d(degree, pts, out=out_readonly)


def test_out_parameter_reuse_result() -> None:
    """Test that a result from one call can be reused as out in another call."""
    degree = 2
    pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)

    # First call without out
    result1 = tabulate_bernstein_1d(degree, pts)
    assert result1.shape == (3, 3)

    # Reuse result1 as out for second call with same input
    result2 = tabulate_bernstein_1d(degree, pts, out=result1)
    # Should work without error and return the same array
    nptest.assert_allclose(result1, result2)

    # Test with scalar input
    result_scalar = tabulate_bernstein_1d(degree, 0.5)
    assert result_scalar.shape == (3,)

    # Reuse scalar result as out
    result_scalar2 = tabulate_bernstein_1d(degree, 0.5, out=result_scalar)
    nptest.assert_allclose(result_scalar, result_scalar2)

    # Test with Lagrange
    result_lagrange = tabulate_lagrange_1d(degree, LagrangeVariant.EQUISPACES, pts)
    result_lagrange2 = tabulate_lagrange_1d(
        degree, LagrangeVariant.EQUISPACES, pts, out=result_lagrange
    )
    nptest.assert_allclose(result_lagrange, result_lagrange2)

    # Test with Legendre
    result_legendre = tabulate_legendre_1d(degree, pts)
    result_legendre2 = tabulate_legendre_1d(degree, pts, out=result_legendre)
    nptest.assert_allclose(result_legendre, result_legendre2)


@pytest.mark.parametrize(
    "basis_type", [BasisType.BERNSTEIN, BasisType.CARDINAL_BSPLINE, BasisType.LAGRANGE]
)
class TestOutParameterMultidimensional:
    """Test suite for out parameter in multidimensional tabulate functions."""

    def test_out_parameter_works(self, basis_type: BasisType) -> None:
        """Test that out parameter works correctly for multidimensional functions."""
        degrees = [2, 3]
        pts = np.array([[0.0, 0.5], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64)
        func = _get_basis_function(basis_type)

        # Compute without out parameter
        if basis_type == BasisType.LAGRANGE:
            result1 = func(degrees, LagrangeVariant.EQUISPACES, pts)
        else:
            result1 = func(degrees, pts)

        # Compute with out parameter
        out = np.zeros_like(result1)
        if basis_type == BasisType.LAGRANGE:
            result2 = func(degrees, LagrangeVariant.EQUISPACES, pts, out=out)
        else:
            result2 = func(degrees, pts, out=out)

        # Results should match
        nptest.assert_allclose(result1, result2)
        # Verify that out was used (values match exactly)
        nptest.assert_allclose(out, result1)

    def test_out_parameter_wrong_shape_raises_error(self, basis_type: BasisType) -> None:
        """Test that out parameter with wrong shape raises ValueError."""
        degrees = [2, 3]
        pts = np.array([[0.0, 0.5], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64)
        func = _get_basis_function(basis_type)

        # Create out with wrong shape
        # Expected shape is (3, 12) = (3 points, (2+1)*(3+1) basis functions)
        out = np.zeros((3, 10), dtype=np.float64)  # Wrong number of basis functions

        with pytest.raises(ValueError, match="Output array has shape"):
            if basis_type == BasisType.LAGRANGE:
                func(degrees, LagrangeVariant.EQUISPACES, pts, out=out)
            else:
                func(degrees, pts, out=out)

    def test_out_parameter_wrong_dtype_raises_error(self, basis_type: BasisType) -> None:
        """Test that out parameter with wrong dtype raises ValueError."""
        degrees = [2, 3]
        pts = np.array([[0.0, 0.5], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64)
        func = _get_basis_function(basis_type)

        # Create out with wrong dtype
        out = np.zeros((3, 12), dtype=np.float32)  # Should be float64 to match pts

        with pytest.raises(ValueError, match="Output array has dtype"):
            if basis_type == BasisType.LAGRANGE:
                func(degrees, LagrangeVariant.EQUISPACES, pts, out=out)
            else:
                func(degrees, pts, out=out)

    def test_out_parameter_float32(self, basis_type: BasisType) -> None:
        """Test that out parameter works with float32."""
        degrees = [2, 3]
        pts = np.array([[0.0, 0.5], [0.5, 0.5], [1.0, 1.0]], dtype=np.float32)
        func = _get_basis_function(basis_type)

        # Compute without out parameter
        if basis_type == BasisType.LAGRANGE:
            result1 = func(degrees, LagrangeVariant.EQUISPACES, pts)
        else:
            result1 = func(degrees, pts)

        # Compute with out parameter
        out = np.zeros_like(result1)
        if basis_type == BasisType.LAGRANGE:
            result2 = func(degrees, LagrangeVariant.EQUISPACES, pts, out=out)
        else:
            result2 = func(degrees, pts, out=out)

        # Results should match
        nptest.assert_allclose(result1, result2)
        assert result2.dtype == np.float32

    def test_out_parameter_points_lattice(self, basis_type: BasisType) -> None:
        """Test that out parameter works with PointsLattice."""
        degrees = [2, 3]
        pts = PointsLattice(
            [
                np.array([0.0, 0.5, 1.0], dtype=np.float64),
                np.array([0.0, 1.0], dtype=np.float64),
            ]
        )
        func = _get_basis_function(basis_type)

        # Compute without out parameter
        if basis_type == BasisType.LAGRANGE:
            result1 = func(degrees, LagrangeVariant.EQUISPACES, pts)
        else:
            result1 = func(degrees, pts)

        # Compute with out parameter
        out = np.zeros_like(result1)
        if basis_type == BasisType.LAGRANGE:
            result2 = func(degrees, LagrangeVariant.EQUISPACES, pts, out=out)
        else:
            result2 = func(degrees, pts, out=out)

        # Results should match
        nptest.assert_allclose(result1, result2)
        nptest.assert_allclose(out, result1)


# ---------------------------------------------------------------------------
# Generator / single-pass iterable regression tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("basis_type", list(BasisType))
def test_generator_degrees_same_as_list(basis_type: BasisType) -> None:
    """Generator input for ``degrees`` must produce the same result as a list.

    Regression test for the double-consumption bug: the old inline validation
    iterated ``degrees`` once (checking types/values) and then the evaluator-
    construction loop iterated it again, silently yielding an empty tuple on
    the second pass when ``degrees`` was a generator.  ``_validate_degrees``
    materialises to a tuple first, so both passes see all elements.
    """
    pts = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float64)
    degrees_list = [2, 3]

    func = _get_basis_function(basis_type)
    if basis_type == BasisType.LAGRANGE:
        result_list = func(degrees_list, LagrangeVariant.EQUISPACES, pts)
        result_gen = func(iter(degrees_list), LagrangeVariant.EQUISPACES, pts)
    else:
        result_list = func(degrees_list, pts)
        result_gen = func(iter(degrees_list), pts)

    assert result_gen.shape == result_list.shape, (
        f"Generator input produced shape {result_gen.shape}, expected {result_list.shape}. "
        "Likely the degrees generator was consumed before the evaluator loop."
    )
    nptest.assert_allclose(result_gen, result_list, atol=1e-14)
