"""Tests for 1D quadrature rules in pantr.quad."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import numpy.testing as nptest
import numpy.typing as npt
import pytest
from numpy.polynomial import chebyshev

from pantr.basis import LagrangeVariant
from pantr.quad import (
    PointsLattice,
    QuadratureRule,
    create_lagrange_points_lattice,
    gauss_legendre_quadrature,
    get_chebyshev_gauss_1st_kind_1d,
    get_chebyshev_gauss_2nd_kind_1d,
    get_gauss_legendre_1d,
    get_gauss_lobatto_legendre_1d,
    get_modified_chebyshev_nodes_1d,
    get_trapezoidal_1d,
    tensor_product_quadrature,
)
from pantr.tolerance import get_conservative, get_default, get_strict


def _integrate_polynomial_on_unit_interval(
    power: int,
    nodes: npt.NDArray[np.floating[Any]],
    weights: npt.NDArray[np.floating[Any]],
) -> np.floating[Any]:
    vals = nodes**power
    result = np.sum(weights * vals, dtype=np.result_type(nodes.dtype, weights.dtype))
    return cast(np.floating[Any], result)


class TestTrapezoidal:
    """Tests for get_trapezoidal_1d."""

    def test_invalid_n_pts_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            get_trapezoidal_1d(0)

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="float32 or float64"):
            get_trapezoidal_1d(2, np.int32)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_npts_one_midpoint_and_unit_weight(self, dtype: npt.DTypeLike) -> None:
        nodes, weights = get_trapezoidal_1d(1, dtype)
        nptest.assert_allclose(nodes, np.array([0.5], dtype=dtype))
        nptest.assert_allclose(weights, np.array([1.0], dtype=dtype))
        assert nodes.dtype == np.dtype(dtype)
        assert weights.dtype == np.dtype(dtype)

    @pytest.mark.parametrize("n_pts", [2, 5, 11])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_partition_and_end_weights(self, n_pts: int, dtype: npt.DTypeLike) -> None:
        nodes, weights = get_trapezoidal_1d(n_pts, dtype)
        # nodes in [0, 1]
        assert np.all((nodes >= 0.0) & (nodes <= 1.0))
        # weights sum to 1
        nptest.assert_allclose(np.sum(weights), np.array(1.0, dtype=dtype))
        if n_pts > 1:
            h = np.array(1.0 / (n_pts - 1), dtype=dtype)
            nptest.assert_allclose(weights[1:-1], h)
            nptest.assert_allclose(weights[[0, -1]], 0.5 * h)


class TestGaussLegendre:
    """Tests for get_gauss_legendre_1d."""

    def test_invalid_n_pts_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            get_gauss_legendre_1d(0)

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="float32 or float64"):
            get_gauss_legendre_1d(2, np.int32)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_basic_properties(self, dtype: npt.DTypeLike) -> None:
        nodes, weights = get_gauss_legendre_1d(4, dtype)
        assert nodes.dtype == np.dtype(dtype)
        assert weights.dtype == np.dtype(dtype)
        assert np.all((nodes >= 0.0) & (nodes <= 1.0))
        assert np.all(weights > 0.0)
        nptest.assert_allclose(np.sum(weights, dtype=np.float64), 1.0, rtol=get_strict(dtype))

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_polynomial_exactness(self, dtype: npt.DTypeLike) -> None:
        # n points should integrate polynomials up to degree 2n-1 exactly
        n = 4
        nodes, weights = get_gauss_legendre_1d(n, dtype)
        rtol = get_default(dtype)
        for p in range(2 * n):  # inclusive upper bound 2n-1
            approx = _integrate_polynomial_on_unit_interval(p, nodes, weights)
            exact = 1.0 / (p + 1)
            nptest.assert_allclose(approx, np.array(exact, dtype=dtype), rtol=rtol, atol=0.0)


class TestGaussLobattoLegendre:
    """Tests for get_gauss_lobatto_legendre_1d."""

    def test_invalid_n_pts_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            get_gauss_lobatto_legendre_1d(1)

    def test_n_pts_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            get_gauss_lobatto_legendre_1d(0)

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="float32 or float64"):
            get_gauss_lobatto_legendre_1d(2, np.int32)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_endpoints_and_sum_weights(self, dtype: npt.DTypeLike) -> None:
        nodes, weights = get_gauss_lobatto_legendre_1d(4, dtype)
        # Endpoints included
        nptest.assert_allclose(nodes[0], np.array(0.0, dtype=dtype))
        nptest.assert_allclose(nodes[-1], np.array(1.0, dtype=dtype))
        # weights positive and sum to 1
        assert np.all(weights > 0.0)
        nptest.assert_allclose(np.sum(weights, dtype=np.float64), 1.0, rtol=get_strict(dtype))

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_polynomial_exactness(self, dtype: npt.DTypeLike) -> None:
        # Degree of exactness: 2n-3
        n = 5
        nodes, weights = get_gauss_lobatto_legendre_1d(n, dtype)
        rtol = get_conservative(dtype)
        for p in range(2 * n - 2):  # inclusive upper bound 2n-3
            approx = _integrate_polynomial_on_unit_interval(p, nodes, weights)
            exact = 1.0 / (p + 1)
            nptest.assert_allclose(approx, np.array(exact, dtype=dtype), rtol=rtol, atol=0.0)


class TestChebyshevGaussFirstKind:
    """Tests for get_chebyshev_gauss_1st_kind_1d."""

    def test_invalid_n_pts_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            get_chebyshev_gauss_1st_kind_1d(0)

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="float32 or float64"):
            get_chebyshev_gauss_1st_kind_1d(2, np.int32)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_npts_one_midpoint_and_weight_sum(self, dtype: npt.DTypeLike) -> None:
        nodes, weights = get_chebyshev_gauss_1st_kind_1d(1, dtype)
        # cheb1 at n=1 returns node 0 on [-1,1] which maps to 0.5
        nptest.assert_allclose(nodes, np.array([0.5], dtype=dtype))
        # Sum of weights equals integral of 1/sqrt(1-x^2) over [0,1] = pi/2
        nptest.assert_allclose(
            np.sum(weights), np.array(np.pi / 2.0, dtype=dtype), rtol=get_strict(dtype)
        )
        assert nodes.dtype == np.dtype(dtype)
        assert weights.dtype == np.dtype(dtype)

    @pytest.mark.parametrize("n_pts", [2, 5, 10])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_nodes_and_total_weight(self, n_pts: int, dtype: npt.DTypeLike) -> None:
        nodes, weights = get_chebyshev_gauss_1st_kind_1d(n_pts, dtype)
        # nodes are mapped chebpts1
        cheb1_t = cast(Callable[[int], npt.NDArray[np.float64]], chebyshev.chebpts1)
        mapped = ((cheb1_t(n_pts) + 1.0) * 0.5).astype(dtype)
        nptest.assert_allclose(nodes, mapped, rtol=get_strict(dtype))
        # weights sum to pi/2 after scaling to [0,1]
        nptest.assert_allclose(
            np.sum(weights), np.array(np.pi / 2.0, dtype=dtype), rtol=get_strict(dtype)
        )
        assert np.all((nodes >= 0.0) & (nodes <= 1.0))


class TestChebyshevGaussSecondKind:
    """Tests for get_chebyshev_gauss_2nd_kind_1d."""

    def test_invalid_n_pts_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            get_chebyshev_gauss_2nd_kind_1d(1)

    def test_n_pts_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            get_chebyshev_gauss_2nd_kind_1d(0)

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="float32 or float64"):
            get_chebyshev_gauss_2nd_kind_1d(2, np.int32)

    @pytest.mark.parametrize("n_pts", [2, 5, 9])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_nodes_endpoints_and_total_weight(self, n_pts: int, dtype: npt.DTypeLike) -> None:
        nodes, weights = get_chebyshev_gauss_2nd_kind_1d(n_pts, dtype)
        # endpoints included for n>=2
        nptest.assert_allclose(nodes[0], np.array(0.0, dtype=dtype))
        nptest.assert_allclose(nodes[-1], np.array(1.0, dtype=dtype))
        # weights sum to (integral of sqrt(1-x^2) over [0,1]) = pi/4 after scaling
        nptest.assert_allclose(
            np.sum(weights), np.array(np.pi / 4.0, dtype=dtype), rtol=get_strict(dtype)
        )
        assert np.all((nodes >= 0.0) & (nodes <= 1.0))


class TestPointsLattice:
    """Tests for PointsLattice class."""

    def test_empty_iterable_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1 dimension"):
            PointsLattice([])

    def test_different_dtypes_raises(self) -> None:
        pts1 = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        pts2 = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="same dtype"):
            PointsLattice([pts1, pts2])

    def test_non_1d_points_raises(self) -> None:
        pts1 = np.array([0.0, 0.5, 1.0])
        pts2 = np.array([[0.0, 0.5], [1.0, 1.5]])
        with pytest.raises(ValueError, match="must be 1D"):
            PointsLattice([pts1, pts2])

    def test_empty_points_raises(self) -> None:
        pts1 = np.array([0.0, 0.5, 1.0])
        pts2 = np.array([])
        with pytest.raises(ValueError, match="at least 1 point"):
            PointsLattice([pts1, pts2])

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_1d_lattice_properties(self, dtype: npt.DTypeLike) -> None:
        pts = np.array([0.0, 0.5, 1.0], dtype=dtype)
        lattice = PointsLattice([pts])
        assert lattice.dim == 1
        assert lattice.dtype == np.dtype(dtype)
        assert len(lattice.pts_per_dir) == 1
        nptest.assert_array_equal(lattice.pts_per_dir[0], pts)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_2d_lattice_properties(self, dtype: npt.DTypeLike) -> None:
        pts_x = np.array([0.0, 0.5, 1.0], dtype=dtype)
        pts_y = np.array([0.0, 1.0], dtype=dtype)
        pts_dir = [pts_x, pts_y]
        lattice = PointsLattice(pts_dir)
        assert lattice.dim == len(pts_dir)
        assert lattice.dtype == np.dtype(dtype)
        assert len(lattice.pts_per_dir) == len(pts_dir)
        nptest.assert_array_equal(lattice.pts_per_dir[0], pts_x)
        nptest.assert_array_equal(lattice.pts_per_dir[1], pts_y)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_3d_lattice_properties(self, dtype: npt.DTypeLike) -> None:
        pts_x = np.array([0.0, 1.0], dtype=dtype)
        pts_y = np.array([0.0, 0.5, 1.0], dtype=dtype)
        pts_z = np.array([0.0, 1.0], dtype=dtype)
        pts_dir = [pts_x, pts_y, pts_z]
        lattice = PointsLattice(pts_dir)
        assert lattice.dim == len(pts_dir)
        assert lattice.dtype == np.dtype(dtype)
        assert len(lattice.pts_per_dir) == len(pts_dir)
        nptest.assert_array_equal(lattice.pts_per_dir[0], pts_x)
        nptest.assert_array_equal(lattice.pts_per_dir[1], pts_y)
        nptest.assert_array_equal(lattice.pts_per_dir[2], pts_z)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_get_all_points_1d_c_order(self, dtype: npt.DTypeLike) -> None:
        pts = np.array([0.0, 0.5, 1.0], dtype=dtype)
        lattice = PointsLattice([pts])
        all_pts = lattice.get_all_points(order="C")
        assert all_pts.shape == (3, 1)
        nptest.assert_array_equal(all_pts[:, 0], pts)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_get_all_points_1d_f_order(self, dtype: npt.DTypeLike) -> None:
        pts = np.array([0.0, 0.5, 1.0], dtype=dtype)
        lattice = PointsLattice([pts])
        all_pts = lattice.get_all_points(order="F")
        assert all_pts.shape == (3, 1)
        nptest.assert_array_equal(all_pts[:, 0], pts)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_get_all_points_2d_c_order(self, dtype: npt.DTypeLike) -> None:
        pts_x = np.array([0.0, 1.0], dtype=dtype)
        pts_y = np.array([0.0, 0.5, 1.0], dtype=dtype)
        lattice = PointsLattice([pts_x, pts_y])
        all_pts = lattice.get_all_points(order="C")
        assert all_pts.shape == (6, 2)
        # C order: last index varies fastest
        expected = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.5],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.5],
                [1.0, 1.0],
            ],
            dtype=dtype,
        )
        nptest.assert_array_equal(all_pts, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_get_all_points_2d_f_order(self, dtype: npt.DTypeLike) -> None:
        pts_x = np.array([0.0, 1.0], dtype=dtype)
        pts_y = np.array([0.0, 0.5, 1.0], dtype=dtype)
        lattice = PointsLattice([pts_x, pts_y])
        all_pts = lattice.get_all_points(order="F")
        assert all_pts.shape == (6, 2)
        # F order: first index varies fastest
        expected = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.5],
                [1.0, 0.5],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=dtype,
        )
        nptest.assert_array_equal(all_pts, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_get_all_points_3d_c_order(self, dtype: npt.DTypeLike) -> None:
        pts_x = np.array([0.0, 1.0], dtype=dtype)
        pts_y = np.array([0.0, 1.0], dtype=dtype)
        pts_z = np.array([0.0, 1.0], dtype=dtype)
        lattice = PointsLattice([pts_x, pts_y, pts_z])
        all_pts = lattice.get_all_points(order="C")
        assert all_pts.shape == (8, 3)
        # C order: last index (z) varies fastest
        assert np.allclose(all_pts[0], [0.0, 0.0, 0.0])
        assert np.allclose(all_pts[1], [0.0, 0.0, 1.0])
        assert np.allclose(all_pts[2], [0.0, 1.0, 0.0])
        assert np.allclose(all_pts[3], [0.0, 1.0, 1.0])
        assert np.allclose(all_pts[4], [1.0, 0.0, 0.0])
        assert np.allclose(all_pts[5], [1.0, 0.0, 1.0])
        assert np.allclose(all_pts[6], [1.0, 1.0, 0.0])
        assert np.allclose(all_pts[7], [1.0, 1.0, 1.0])


class TestCreateLagrangePointsLattice:
    """Tests for create_lagrange_points_lattice function."""

    def test_invalid_n_pts_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            create_lagrange_points_lattice(LagrangeVariant.EQUISPACES, [0])

    def test_invalid_n_pts_in_list_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            create_lagrange_points_lattice(LagrangeVariant.EQUISPACES, [2, 0, 3])

    @pytest.mark.parametrize("variant", list(LagrangeVariant))
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_1d_lattice_creation(self, variant: LagrangeVariant, dtype: npt.DTypeLike) -> None:
        n = [3]
        lattice = create_lagrange_points_lattice(variant, n, dtype)
        assert lattice.dim == len(n)
        assert lattice.dtype == np.dtype(dtype)
        assert len(lattice.pts_per_dir[0]) == n[0]
        # Points should be in [0, 1]
        assert np.all((lattice.pts_per_dir[0] >= 0.0) & (lattice.pts_per_dir[0] <= 1.0))

    @pytest.mark.parametrize("variant", list(LagrangeVariant))
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_2d_lattice_creation(self, variant: LagrangeVariant, dtype: npt.DTypeLike) -> None:
        n = [3, 4]
        lattice = create_lagrange_points_lattice(variant, n, dtype)
        assert lattice.dim == len(n)
        assert lattice.dtype == np.dtype(dtype)
        assert len(lattice.pts_per_dir[0]) == n[0]
        assert len(lattice.pts_per_dir[1]) == n[1]
        # Points should be in [0, 1]
        assert np.all((lattice.pts_per_dir[0] >= 0.0) & (lattice.pts_per_dir[0] <= 1.0))
        assert np.all((lattice.pts_per_dir[1] >= 0.0) & (lattice.pts_per_dir[1] <= 1.0))

    @pytest.mark.parametrize("variant", list(LagrangeVariant))
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_3d_lattice_creation(self, variant: LagrangeVariant, dtype: npt.DTypeLike) -> None:
        n = [2, 3, 4]
        lattice = create_lagrange_points_lattice(variant, n, dtype)
        assert lattice.dim == len(n)
        assert lattice.dtype == np.dtype(dtype)
        assert len(lattice.pts_per_dir[0]) == n[0]
        assert len(lattice.pts_per_dir[1]) == n[1]
        assert len(lattice.pts_per_dir[2]) == n[2]

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_equispaced_points(self, dtype: npt.DTypeLike) -> None:
        lattice = create_lagrange_points_lattice(LagrangeVariant.EQUISPACES, [4], dtype)
        expected = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=dtype)
        nptest.assert_allclose(lattice.pts_per_dir[0], expected, rtol=get_strict(dtype))

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_gauss_lobatto_legendre_endpoints(self, dtype: npt.DTypeLike) -> None:
        lattice = create_lagrange_points_lattice(LagrangeVariant.GAUSS_LOBATTO_LEGENDRE, [4], dtype)
        pts = lattice.pts_per_dir[0]
        # GLL should include endpoints
        nptest.assert_allclose(pts[0], np.array(0.0, dtype=dtype), rtol=get_strict(dtype))
        nptest.assert_allclose(pts[-1], np.array(1.0, dtype=dtype), rtol=get_strict(dtype))

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_get_all_points_from_lattice(self, dtype: npt.DTypeLike) -> None:
        lattice = create_lagrange_points_lattice(LagrangeVariant.EQUISPACES, [2, 3], dtype)
        all_pts = lattice.get_all_points(order="C")
        assert all_pts.shape == (6, 2)
        # Verify all points are in [0, 1]
        assert np.all((all_pts >= 0.0) & (all_pts <= 1.0))


class TestModifiedChebyshevNodes:
    """Tests for get_modified_chebyshev_nodes_1d."""

    def test_invalid_n_pts_raises(self) -> None:
        """n_pts < 2 must raise."""
        with pytest.raises(ValueError, match="at least 2"):
            get_modified_chebyshev_nodes_1d(1)

    def test_n_pts_zero_raises(self) -> None:
        """n_pts=0 must also raise with the min_pts=2 message."""
        with pytest.raises(ValueError, match="at least 2"):
            get_modified_chebyshev_nodes_1d(0)

    def test_invalid_dtype_raises(self) -> None:
        """Non-floating dtype must raise."""
        with pytest.raises(ValueError, match="float32 or float64"):
            get_modified_chebyshev_nodes_1d(3, np.int32)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_endpoints(self, dtype: npt.DTypeLike) -> None:
        """First node is 0, last node is 1."""
        nodes = get_modified_chebyshev_nodes_1d(5, dtype)
        nptest.assert_allclose(nodes[0], 0.0, atol=1e-15)
        nptest.assert_allclose(nodes[-1], 1.0, atol=1e-15)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_shape_and_dtype(self, dtype: npt.DTypeLike) -> None:
        """Output shape and dtype are correct."""
        n = 7
        nodes = get_modified_chebyshev_nodes_1d(n, dtype)
        assert nodes.shape == (n,)
        assert nodes.dtype == np.dtype(dtype)

    def test_two_points(self) -> None:
        """n_pts=2 gives [0, 1]."""
        nodes = get_modified_chebyshev_nodes_1d(2)
        nptest.assert_allclose(nodes, [0.0, 1.0], atol=1e-15)

    def test_three_points(self) -> None:
        """n_pts=3 gives [0, 0.5, 1]."""
        nodes = get_modified_chebyshev_nodes_1d(3)
        nptest.assert_allclose(nodes, [0.0, 0.5, 1.0], atol=1e-15)

    def test_symmetry(self) -> None:
        """Nodes are symmetric about 0.5: nodes[i] + nodes[n-1-i] == 1."""
        nodes = get_modified_chebyshev_nodes_1d(8)
        nptest.assert_allclose(nodes + nodes[::-1], 1.0, atol=1e-15)

    def test_monotonicity(self) -> None:
        """Nodes are strictly increasing."""
        nodes = get_modified_chebyshev_nodes_1d(10)
        assert np.all(np.diff(nodes) > 0)


class TestQuadratureRule:
    """Tests for the QuadratureRule value type."""

    def test_basic_properties(self) -> None:
        rule = QuadratureRule(points=[[0.25, 0.75], [0.5, 0.5]], weights=[0.4, 0.6])
        assert rule.ndim == 2
        assert rule.num_points == 2
        assert rule.points.shape == (2, 2)
        assert rule.weights.shape == (2,)

    def test_arrays_are_read_only(self) -> None:
        rule = QuadratureRule(points=[[0.5]], weights=[1.0])
        assert not rule.points.flags.writeable
        assert not rule.weights.flags.writeable

    def test_does_not_alias_input(self) -> None:
        pts = np.array([[0.5, 0.5]])
        rule = QuadratureRule(points=pts, weights=[1.0])
        pts[0, 0] = 0.1  # mutating the source must not affect the rule
        nptest.assert_array_equal(rule.points, [[0.5, 0.5]])

    def test_repr(self) -> None:
        assert repr(QuadratureRule([[0.5, 0.5]], [1.0])) == "QuadratureRule(ndim=2, num_points=1)"

    @pytest.mark.parametrize(
        ("points", "weights", "match"),
        [
            ([0.5, 0.5], [1.0], "points must be 2D"),
            ([[0.5]], [[1.0]], "weights must be 1D"),
            ([[0.5], [0.5]], [1.0], "must match the number of points"),
            (np.empty((0, 2)), [], "non-empty"),
            ([[np.inf, 0.5]], [1.0], "finite"),
            ([[0.5, 0.5]], [np.nan], "finite"),
            ([[-0.1, 0.5]], [1.0], "unit cube"),
            ([[0.5, 1.5]], [1.0], "unit cube"),
        ],
    )
    def test_validation(
        self,
        points: npt.ArrayLike,
        weights: npt.ArrayLike,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            QuadratureRule(points, weights)

    def test_endpoints_allowed(self) -> None:
        # Points exactly on the unit-cube boundary are valid (e.g. Lobatto).
        rule = QuadratureRule(points=[[0.0, 0.0], [1.0, 1.0]], weights=[0.5, 0.5])
        assert rule.num_points == 2


class TestTensorProductQuadrature:
    """Tests for tensor_product_quadrature."""

    def test_shape_and_count(self) -> None:
        rule = tensor_product_quadrature([get_gauss_legendre_1d(2), get_gauss_legendre_1d(3)])
        assert rule.ndim == 2
        assert rule.num_points == 6
        assert rule.points.shape == (6, 2)

    def test_c_order_last_axis_fastest(self) -> None:
        # Axis-0 nodes {0.25, 0.75}, axis-1 nodes {0.5}: C-order has axis 1 fastest.
        rule = tensor_product_quadrature(
            [(np.array([0.25, 0.75]), np.array([0.5, 0.5])), (np.array([0.5]), np.array([1.0]))]
        )
        nptest.assert_allclose(rule.points, [[0.25, 0.5], [0.75, 0.5]])
        nptest.assert_allclose(rule.weights, [0.5, 0.5])

    def test_weights_are_outer_product(self) -> None:
        rule = tensor_product_quadrature(
            [(np.array([0.5]), np.array([0.3])), (np.array([0.5]), np.array([0.4]))]
        )
        nptest.assert_allclose(rule.weights, [0.12])

    def test_single_axis(self) -> None:
        nodes, weights = get_gauss_legendre_1d(4)
        rule = tensor_product_quadrature([(nodes, weights)])
        assert rule.ndim == 1
        assert rule.num_points == 4
        nptest.assert_allclose(rule.points[:, 0], nodes)

    def test_integrates_polynomial(self) -> None:
        # int over [0,1]^2 of x^3 y = (1/4)(1/2) = 1/8.
        rule = tensor_product_quadrature([get_gauss_legendre_1d(2), get_gauss_legendre_1d(2)])
        val = float((rule.weights * rule.points[:, 0] ** 3 * rule.points[:, 1]).sum())
        nptest.assert_allclose(val, 1.0 / 8.0, rtol=1e-13)

    def test_empty_rules(self) -> None:
        with pytest.raises(ValueError, match="at least one axis"):
            tensor_product_quadrature([])

    def test_mismatched_nodes_weights(self) -> None:
        with pytest.raises(ValueError, match="matching non-empty"):
            tensor_product_quadrature([(np.array([0.5, 0.5]), np.array([1.0]))])


class TestGaussLegendreQuadrature:
    """Tests for gauss_legendre_quadrature."""

    def test_isotropic(self) -> None:
        rule = gauss_legendre_quadrature(3, 2)
        assert rule.ndim == 3
        assert rule.num_points == 8
        nptest.assert_allclose(rule.weights.sum(), 1.0, atol=1e-14)

    def test_anisotropic(self) -> None:
        rule = gauss_legendre_quadrature(2, [2, 4])
        assert rule.num_points == 8

    def test_exactness_degree(self) -> None:
        # n-point GL is exact to degree 2n-1; n=3 -> degree 5 per axis.
        rule = gauss_legendre_quadrature(2, 3)
        # int over [0,1]^2 of x^5 y^5 = (1/6)^2.
        val = float((rule.weights * rule.points[:, 0] ** 5 * rule.points[:, 1] ** 5).sum())
        nptest.assert_allclose(val, (1.0 / 6.0) ** 2, rtol=1e-13)

    def test_matches_manual_tensor_product(self) -> None:
        rule = gauss_legendre_quadrature(2, [2, 3])
        manual = tensor_product_quadrature([get_gauss_legendre_1d(2), get_gauss_legendre_1d(3)])
        nptest.assert_allclose(rule.points, manual.points)
        nptest.assert_allclose(rule.weights, manual.weights)

    @pytest.mark.parametrize(
        ("ndim", "npts", "match"),
        [
            (0, 2, "ndim must be >= 1"),
            (2, [2], "length-2 sequence"),
            (2, [2, 0], "must be >= 1"),
            (1, 0, "must be >= 1"),
        ],
    )
    def test_validation(self, ndim: int, npts: int | list[int], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            gauss_legendre_quadrature(ndim, npts)
